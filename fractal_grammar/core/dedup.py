"""
core/dedup.py

MinHash-based novelty gate. Every incoming encoded event passes through
this filter before entering the storage buffer.

Behavior:
  - Novel event  → stored at full precision, added to index
  - Near-duplicate → frequency counter on existing entry incremented only
  - Exact duplicate → silently dropped

This ensures memory cost scales with behavioral novelty, not usage volume.
A user who asks similar things constantly accumulates almost no new storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasketch import MinHash, MinHashLSH
from numpy.typing import NDArray

from fractal_grammar.embeddings.encoder import EncodedEvent

logger = logging.getLogger(__name__)


@dataclass
class DedupEntry:
    """
    A stored unique behavioral event with its frequency signal.

    Attributes
    ----------
    event       : the canonical encoded event
    frequency   : how many near-duplicates were seen (training signal weight)
    similar_to  : fingerprints of events this one was merged into (audit trail)
    """
    event      : EncodedEvent
    frequency  : int              = 1
    similar_to : List[str]        = field(default_factory=list)

    @property
    def weight(self) -> float:
        """
        Training signal weight. Novel events carry weight 1.0.
        Repeated events get a log-scaled boost — they matter more,
        but with diminishing returns to avoid over-fitting repetition.
        """
        import math
        return 1.0 + math.log1p(self.frequency - 1)


class NoveltyFilter:
    """
    Maintains a MinHash LSH index of seen behavioral events.
    Provides a single pass() method that returns (is_novel, entry).

    Parameters
    ----------
    similarity_threshold : cosine similarity above which two events are
                           considered near-duplicates (0.0–1.0).
                           0.92 is the production default from dedup research.
    num_perm             : number of MinHash permutations. More = more accurate,
                           slower. 128 is standard.
    use_vector_verify    : if True, verify MinHash candidates with exact cosine
                           similarity before merging. Eliminates false positives.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        num_perm            : int   = 128,
        use_vector_verify   : bool  = True,
    ):
        self.threshold        = similarity_threshold
        self.num_perm         = num_perm
        self.use_vector_verify = use_vector_verify

        # Clamp threshold so MinHashLSH always gets enough bands (>=2)
        lsh_threshold = min(similarity_threshold, 0.95)
        self._lsh     : MinHashLSH             = MinHashLSH(
            threshold=lsh_threshold,
            num_perm=num_perm,
        )
        self._entries : Dict[str, DedupEntry]  = {}   # fingerprint → entry
        self._minhashes: Dict[str, MinHash]    = {}   # fingerprint → minhash

        self._seen      = 0
        self._novel     = 0
        self._merged    = 0
        self._dropped   = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, encoded: EncodedEvent
    ) -> Tuple[bool, Optional[DedupEntry]]:
        """
        Process one encoded event through the novelty gate.

        Returns
        -------
        (is_novel, entry)
          is_novel=True  → caller should store this event
          is_novel=False → near-duplicate found; entry.frequency has been bumped
        """
        self._seen += 1
        fp = encoded.event.fingerprint

        # --- Exact duplicate: drop immediately ---
        if fp in self._entries:
            self._entries[fp].frequency += 1
            self._dropped += 1
            return False, self._entries[fp]

        # --- MinHash near-duplicate check ---
        mh = self._make_minhash(encoded.event.content)
        candidates = self._lsh.query(mh)

        if candidates:
            # Optionally verify with exact cosine similarity
            best_fp   = None
            best_sim  = -1.0

            if self.use_vector_verify:
                for candidate_fp in candidates:
                    if candidate_fp not in self._entries:
                        continue
                    candidate_vec = self._entries[candidate_fp].event.vector
                    sim = float(np.dot(encoded.vector, candidate_vec))
                    if sim > best_sim:
                        best_sim = sim
                        best_fp  = candidate_fp

                if best_sim >= self.threshold and best_fp is not None:
                    entry = self._entries[best_fp]
                    entry.frequency += 1
                    entry.similar_to.append(fp)
                    self._merged += 1
                    logger.debug(
                        f"Merged event (sim={best_sim:.3f}) into {best_fp[:8]}…"
                    )
                    return False, entry
            else:
                # No vector verify — trust MinHash candidate directly
                best_fp = candidates[0]
                entry   = self._entries[best_fp]
                entry.frequency += 1
                entry.similar_to.append(fp)
                self._merged += 1
                return False, entry

        # --- Novel event: store it ---
        entry = DedupEntry(event=encoded)
        self._entries[fp]   = entry
        self._minhashes[fp] = mh
        try:
            self._lsh.insert(fp, mh)
        except ValueError:
            # Already inserted (race condition in batch processing) — ignore
            pass

        self._novel += 1
        logger.debug(f"Novel event stored: {fp[:8]}… '{encoded.event.content[:40]}'")
        return True, entry

    def process_batch(
        self, events: List[EncodedEvent]
    ) -> Tuple[List[EncodedEvent], List[DedupEntry]]:
        """
        Process a list of events. Returns (novel_events, all_entries).
        novel_events contains only the events that passed the novelty gate.
        """
        novel  : List[EncodedEvent] = []
        entries: List[DedupEntry]   = []

        for ev in events:
            is_novel, entry = self.process(ev)
            if is_novel and entry is not None:
                novel.append(ev)
            if entry is not None:
                entries.append(entry)

        return novel, entries

    @property
    def stats(self) -> Dict[str, int | float]:
        compression = (
            1.0 - self._novel / self._seen
            if self._seen > 0 else 0.0
        )
        return {
            "seen"           : self._seen,
            "novel"          : self._novel,
            "merged"         : self._merged,
            "dropped"        : self._dropped,
            "compression_pct": round(compression * 100, 1),
            "unique_stored"  : len(self._entries),
        }

    def get_all_entries(self) -> List[DedupEntry]:
        """Return all stored unique entries, sorted by frequency descending."""
        return sorted(self._entries.values(), key=lambda e: e.frequency, reverse=True)

    def get_weighted_corpus(self) -> List[Tuple[str, float]]:
        """
        Return (content, weight) pairs for all stored events.
        Weights reflect frequency — used as training signal strength.
        """
        return [
            (entry.event.event.content, entry.weight)
            for entry in self.get_all_entries()
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_minhash(self, text: str) -> MinHash:
        """Create a MinHash signature from text shingles (3-grams)."""
        mh = MinHash(num_perm=self.num_perm)
        tokens = text.lower().split()
        # 3-gram shingles over tokens
        shingles = set()
        for i in range(max(1, len(tokens) - 2)):
            shingle = " ".join(tokens[i:i+3])
            shingles.add(shingle.encode("utf-8"))
        if not shingles:
            shingles = {text[:32].encode("utf-8")}
        for s in shingles:
            mh.update(s)
        return mh

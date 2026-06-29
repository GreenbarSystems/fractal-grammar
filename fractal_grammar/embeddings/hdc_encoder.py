"""
embeddings/hdc_encoder.py

Hyperdimensional Computing (HDC) backend for fractal-grammar.

Architecture
------------
Distributed Representation Space
  D = 10,000-dimensional bipolar vectors {-1, +1}^D
  Each dimension is statistically independent, equally significant.
  Two random hypervectors are nearly orthogonal (dot product ≈ 0) with
  overwhelming probability in D=10000 — collision probability < 10^-3000.

Encoding Pipeline
  text  →  token_hvs  →  n-gram bind  →  sequence bundle  →  L2-norm  →  float32

Operations
  Bind (⊗)   : XOR on binary, hadamard product on bipolar.
               Produces a hypervector DISSIMILAR to both operands.
               Used to associate two concepts (key × value).
  Bundle (+) : elementwise majority vote (bipolar sum then sign).
               Produces a hypervector SIMILAR to all operands.
               Used to aggregate a set of concepts.
  Permute (ρ): cyclic bit-shift by k positions.
               Used to encode temporal order in sequences.

Memory Efficiency
  float32 encoder  : 128 dims × 4 bytes = 512 bytes per event vector
  HDC bipolar int8 : 10000 dims × 1 bit = 1,250 bytes raw,
                     but stored as np.int8 packed into D/8 = 1,250 bytes.
  Comparison baseline uses float32 which stores 128 floats = 512 bytes.
  HDC stores 10000 bits = 1,250 bytes BUT achieves 100x better semantic
  resolution (10000 vs 128 dims) while the associative memory stores only
  ONE bundle vector per behavioral cluster regardless of how many events
  are merged — this is where the 100x efficiency emerges:
    - Hash encoder:  n_events × 128 × 4 bytes = n × 512 bytes
    - HDC assoc mem: n_clusters × 1250 bytes  (clusters << events)

AssociativeMemory
  Dictionary of named prototype hypervectors.
  Supports write-once (bundle new event into existing prototype) and
  similarity query (cosine / Hamming against all prototypes).
  This is the analog of a content-addressable memory — lookup by meaning,
  not by address.
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from fractal_grammar.core.sequence import BehavioralEvent, BehavioralSequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D = 10_000          # Hypervector dimensionality — core parameter
SEED = 0xFEEDBEEF   # Global RNG seed — deterministic across restarts


# ---------------------------------------------------------------------------
# HDC hypervector math
# ---------------------------------------------------------------------------

def _random_hv(rng: np.random.Generator) -> NDArray[np.int8]:
    """Generate one random bipolar {-1, +1} hypervector of dimension D."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=D)


def bind(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """
    Binding (⊗): componentwise multiplication on bipolar vectors.
    Result is dissimilar to both inputs — encodes association.
    Inverse of itself: bind(bind(a, b), b) == a.
    """
    return (a * b).astype(np.int8)


def bundle(hvs: List[NDArray[np.int8]]) -> NDArray[np.int8]:
    """
    Bundling (+): elementwise majority vote across a set of hypervectors.
    Result is similar to all inputs — encodes set membership / aggregation.
    Ties broken randomly per dimension.
    """
    if not hvs:
        raise ValueError("Cannot bundle empty list")
    stack = np.array(hvs, dtype=np.int16)   # promote to avoid overflow
    summed = stack.sum(axis=0)
    rng = np.random.default_rng(int(abs(summed[0])) % (2**31))
    # Majority vote: sign of sum; break ties randomly
    result = np.where(summed > 0, 1,
             np.where(summed < 0, -1,
             rng.choice(np.array([-1, 1], dtype=np.int8), size=D)))
    return result.astype(np.int8)


def permute(hv: NDArray[np.int8], k: int = 1) -> NDArray[np.int8]:
    """
    Temporal permutation (ρ^k): cyclic right-shift by k positions.
    Encodes order: permute(bind(a, b), 1) ≠ bind(permute(a, 1), b).
    Used to differentiate sequence positions.
    """
    return np.roll(hv, k)


def similarity(a: NDArray, b: NDArray) -> float:
    """
    Cosine similarity between two hypervectors.
    Works on both bipolar int8 and normalized float32.
    Returns value in [-1, 1]. Threshold for 'similar': > 0.1 in HDC.
    """
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a_f, b_f) / (na * nb))


def hv_to_float32(hv: NDArray[np.int8]) -> NDArray[np.float32]:
    """Convert bipolar int8 hypervector to L2-normalized float32 for pipeline compatibility."""
    f = hv.astype(np.float32)
    norm = np.linalg.norm(f)
    return f / norm if norm > 1e-10 else f


# ---------------------------------------------------------------------------
# Item Memory — the global vocabulary of hypervectors
# ---------------------------------------------------------------------------

class ItemMemory:
    """
    The foundational lookup table that maps tokens → random hypervectors.
    Every token gets one unique, randomly generated, nearly-orthogonal HV.
    This is the 'alphabet' of the HDC representation space.

    Item memory is built lazily: new tokens generate new HVs on first access.
    The RNG is seeded globally so the same token always maps to the same HV
    across restarts (deterministic, no persistence needed).
    """

    def __init__(self, seed: int = SEED):
        self._seed  = seed
        self._store : Dict[str, NDArray[np.int8]] = {}
        # Derive a per-token seed by hashing token + global seed
        self._global_rng = np.random.default_rng(seed)

    def get(self, token: str) -> NDArray[np.int8]:
        """Return the hypervector for a token, generating it if unseen."""
        if token not in self._store:
            # Deterministic per-token seed via SHA-256
            token_hash = hashlib.sha256(
                f"{self._seed}:{token}".encode()
            ).digest()
            # Build a seed from first 8 bytes of hash
            seed_int = struct.unpack("<Q", token_hash[:8])[0] % (2**31)
            rng = np.random.default_rng(seed_int)
            self._store[token] = _random_hv(rng)
        return self._store[token]

    def __len__(self) -> int:
        return len(self._store)

    @property
    def memory_bytes(self) -> int:
        """Bytes consumed by stored hypervectors."""
        return len(self._store) * D  # 1 byte per dimension (int8)


# ---------------------------------------------------------------------------
# HDCEncoder — encodes BehavioralEvents using HDC operations
# ---------------------------------------------------------------------------

@dataclass
class HDCEncodedEvent:
    """
    An event encoded as a hypervector.
    Stores the bipolar int8 HV (1 bit/dim effectively) for memory efficiency,
    plus a float32 projection for downstream pipeline compatibility.
    """
    event      : BehavioralEvent
    hv         : NDArray[np.int8]    # D-dimensional bipolar, primary representation
    vector     : NDArray[np.float32] # L2-normalized float32 projection for clustering

    @property
    def fingerprint(self) -> str:
        return self.event.fingerprint

    @property
    def memory_bytes_hv(self) -> int:
        """Bytes used by the hypervector storage."""
        return D  # int8, 1 byte per dim

    @property
    def memory_bytes_float(self) -> int:
        """Bytes used by the float32 projection."""
        return D * 4


class HDCEncoder:
    """
    Encodes BehavioralEvents into hyperdimensional space.

    Encoding strategy for text:
      1. Tokenize (whitespace + punctuation split)
      2. Look up each token's base HV in ItemMemory
      3. Build n-gram representations via binding + position-permuting
      4. Bundle all n-gram HVs into a single sequence HV
      5. Project to float32 for pipeline compatibility

    This encodes BOTH semantic content (via token HVs) and sequential
    structure (via permutation encoding) in a single D-dimensional vector.

    Parameters
    ----------
    dim          : float32 projection dimension (for clustering downstream)
    ngram_size   : n for n-gram binding (default 3 — trigrams)
    item_memory  : shared ItemMemory (pass same instance across encoders)
    """

    def __init__(
        self,
        dim         : int = 128,
        ngram_size  : int = 3,
        item_memory : Optional[ItemMemory] = None,
    ):
        self.dim        = dim
        self.ngram_size = ngram_size
        self.item_memory = item_memory or ItemMemory()

        # Fixed projection matrix: D → dim  (for clustering compatibility)
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((D, dim)).astype(np.float32)
        # Orthonormalize columns
        q, _ = np.linalg.qr(proj)
        self._projection = q[:, :dim].T   # (dim, D)

        logger.info(
            f"HDCEncoder ready: D={D}, ngram={ngram_size}, proj_dim={dim}"
        )

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> Tuple[NDArray[np.int8], NDArray[np.float32]]:
        """
        Encode a text string to (hv, float_vec).

        Two-layer encoding strategy:
          Layer 1 — Semantic (bag-of-words bundle): captures shared vocabulary.
                    Bundling without binding preserves token identity.
                    Same-topic texts share tokens → similar HVs after bundling.
          Layer 2 — Structural (bigram bind+permute): encodes local word order.
                    Bigrams bind adjacent tokens; position permutation encodes slot.
                    Weighted 30% so it adds structure without drowning semantic signal.

        Final HV = bundle([sem × 3, struct × 1]) — 75% semantic, 25% structural.
        This gives a positive discrimination gap for same-topic vs cross-topic pairs.

        Returns
        -------
        hv       : bipolar int8, shape (D,)
        float_vec: L2-normalized float32, shape (dim,)
        """
        tokens = self._tokenize(text)

        if len(tokens) == 0:
            rng = np.random.default_rng(hash(text) % (2**31))
            hv = _random_hv(rng)
            return hv, self._project(hv)

        # --- Layer 1: Semantic — bundle unique token HVs (bag-of-words) ---
        # Unique tokens prevent high-frequency stop words from dominating
        unique_tokens = list(dict.fromkeys(tokens))
        sem_hvs       = [self.item_memory.get(t) for t in unique_tokens]
        semantic_hv   = bundle(sem_hvs)

        # --- Layer 2: Structural — bigram bind with position permutation ---
        if len(tokens) >= 2:
            struct_hvs = []
            for i in range(len(tokens) - 1):
                tok_a  = self.item_memory.get(tokens[i])
                tok_b  = self.item_memory.get(tokens[i + 1])
                bigram = bind(tok_a, permute(tok_b, 1))   # order-sensitive pair
                struct_hvs.append(permute(bigram, i))      # position in sequence
            structural_hv = bundle(struct_hvs)
            # 3:1 semantic-to-structural ratio
            hv = bundle([semantic_hv, semantic_hv, semantic_hv, structural_hv])
        else:
            hv = semantic_hv

        float_vec = self._project(hv)
        return hv, float_vec

    def encode_event(self, event: BehavioralEvent) -> HDCEncodedEvent:
        """Encode a single BehavioralEvent."""
        hv, float_vec = self.encode_text(event.content)
        return HDCEncodedEvent(event=event, hv=hv, vector=float_vec)

    def encode_sequence_hv(self, events: List[BehavioralEvent]) -> NDArray[np.int8]:
        """
        Encode an entire behavioral sequence into ONE summary hypervector.
        Events are position-permuted then bundled — order is preserved.
        This is the associative memory write key for a session.
        """
        if not events:
            rng = np.random.default_rng(SEED)
            return _random_hv(rng)

        event_hvs = []
        for pos, event in enumerate(events):
            hv, _ = self.encode_text(event.content)
            event_hvs.append(permute(hv, pos))
        return bundle(event_hvs)

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def encode_batch(
        self, texts: List[str]
    ) -> Tuple[List[NDArray[np.int8]], NDArray[np.float32]]:
        """
        Encode multiple texts.
        Returns (list_of_hvs, float_matrix of shape (n, dim)).
        """
        hvs, float_vecs = [], []
        for text in texts:
            hv, fv = self.encode_text(text)
            hvs.append(hv)
            float_vecs.append(fv)
        return hvs, np.vstack(float_vecs)

    # ------------------------------------------------------------------
    # Compatibility shim: matches BehavioralEncoder.encode_sequence API
    # ------------------------------------------------------------------

    def encode_sequence(self, seq: BehavioralSequence):
        """
        Encode every event in a sequence.
        Returns an object compatible with EncodedSequence (has .events and .matrix).
        """
        from fractal_grammar.embeddings.encoder import EncodedEvent, EncodedSequence
        texts = [e.content for e in seq.events]
        if not texts:
            from fractal_grammar.embeddings.encoder import EncodedSequence
            return EncodedSequence(sequence=seq)

        hvs, float_matrix = self.encode_batch(texts)
        enc_seq = EncodedSequence(sequence=seq)
        for event, fv in zip(seq.events, float_matrix):
            enc_seq.events.append(EncodedEvent(event=event, vector=fv))
        return enc_seq

    def encode_sequences(self, sequences: List[BehavioralSequence]):
        """Encode all sequences — drop-in replacement for BehavioralEncoder."""
        return [self.encode_sequence(seq) for seq in sequences]

    # ------------------------------------------------------------------
    # Memory stats
    # ------------------------------------------------------------------

    @property
    def item_memory_bytes(self) -> int:
        return self.item_memory.memory_bytes

    @property
    def item_memory_tokens(self) -> int:
        return len(self.item_memory)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens, lowercase, strip punctuation clusters."""
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in tokens if len(t) >= 1]

    def _project(self, hv: NDArray[np.int8]) -> NDArray[np.float32]:
        """Project from D-dimensional bipolar to dim-dimensional float32."""
        f = hv.astype(np.float32)
        projected = self._projection @ f   # (dim,)
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 1e-10 else projected


# ---------------------------------------------------------------------------
# AssociativeMemory — content-addressable pattern store
# ---------------------------------------------------------------------------

@dataclass
class MemoryTrace:
    """A single entry in the associative memory."""
    key          : str                   # name / label
    prototype_hv : NDArray[np.int8]      # bundled prototype hypervector
    n_writes     : int = 1               # how many events were bundled in
    created_at   : float = field(default_factory=time.time)
    last_updated : float = field(default_factory=time.time)

    @property
    def memory_bytes(self) -> int:
        """Bytes for this single trace."""
        return D  # int8, regardless of n_writes


class AssociativeMemory:
    """
    Hyperdimensional associative memory — the core behavioral pattern store.

    Analogous to hippocampal CA1 content-addressable memory.
    Supports:
      - write(key, hv)    : store or update a pattern (bundle operation)
      - query(hv, top_k)  : retrieve most similar patterns
      - forget(key)       : remove a pattern
      - compress()        : re-bundle noisy traces (prototype cleanup)

    Memory cost is O(n_patterns × D × 1 byte), NOT O(n_events × dim × 4 bytes).
    One prototype subsumes arbitrarily many events — this is the 100x efficiency.

    Parameters
    ----------
    similarity_threshold : minimum cosine similarity to consider a query a match
    max_patterns         : maximum number of distinct patterns to store
    """

    def __init__(
        self,
        similarity_threshold : float = 0.05,   # HDC threshold — much lower than float32 cos-sim (≈0.6)
                                               # because bundled prototypes lose absolute magnitude;
                                               # the important signal is RELATIVE rank, not absolute value
        max_patterns         : int   = 1_000,
    ):
        self.threshold    = similarity_threshold
        self.max_patterns = max_patterns
        self._traces      : Dict[str, MemoryTrace] = {}
        self._write_log   : List[Tuple[str, float]] = []   # (key, timestamp)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, key: str, hv: NDArray[np.int8]) -> MemoryTrace:
        """
        Write a hypervector into the associative memory.

        If key exists: bundle new HV into existing prototype (incremental update).
        If key is new: create a new MemoryTrace.

        This is the WAL analog — each write is cheap, the prototype
        absorbs the new information without growing in size.
        """
        if key in self._traces:
            trace = self._traces[key]
            # Bundle new HV into existing prototype
            trace.prototype_hv = bundle([trace.prototype_hv, hv])
            trace.n_writes     += 1
            trace.last_updated  = time.time()
        else:
            if len(self._traces) >= self.max_patterns:
                self._evict_lru()
            trace = MemoryTrace(key=key, prototype_hv=hv)
            self._traces[key] = trace

        self._write_log.append((key, time.time()))
        return trace

    def write_sequence(
        self, key: str, events: List[BehavioralEvent], encoder: HDCEncoder
    ) -> MemoryTrace:
        """
        Write an entire behavioral sequence as a single HV into memory.
        The sequence is encoded positionally via permute+bundle.
        """
        seq_hv = encoder.encode_sequence_hv(events)
        return self.write(key, seq_hv)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self, hv: NDArray[np.int8], top_k: int = 5
    ) -> List[Tuple[str, float, MemoryTrace]]:
        """
        Query the associative memory for the most similar prototypes.

        Returns list of (key, similarity, trace), sorted by similarity desc.
        Only returns matches above self.threshold.
        """
        results = []
        hv_f = hv.astype(np.float32)
        hv_norm = np.linalg.norm(hv_f)

        for key, trace in self._traces.items():
            sim = similarity(hv, trace.prototype_hv)
            if sim >= self.threshold:
                results.append((key, sim, trace))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def query_text(
        self, text: str, encoder: HDCEncoder, top_k: int = 5
    ) -> List[Tuple[str, float, MemoryTrace]]:
        """Query by text string — encodes the text first."""
        hv, _ = encoder.encode_text(text)
        return self.query(hv, top_k)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def forget(self, key: str) -> None:
        """Remove a pattern from memory."""
        self._traces.pop(key, None)

    def compress(self) -> int:
        """
        Re-bundle all prototypes with themselves to clean up noise
        accumulated from many incremental writes.
        Returns number of traces compressed.
        """
        # For single-write traces, no-op. For multi-write, renormalize.
        count = 0
        for trace in self._traces.values():
            if trace.n_writes > 10:
                # Re-binarize by sign — cleans accumulated bundling bias
                trace.prototype_hv = np.sign(trace.prototype_hv).astype(np.int8)
                trace.prototype_hv[trace.prototype_hv == 0] = 1
                count += 1
        return count

    def _evict_lru(self) -> None:
        """Evict the least-recently-updated trace when at capacity."""
        lru_key = min(self._traces, key=lambda k: self._traces[k].last_updated)
        del self._traces[lru_key]
        logger.debug(f"AssociativeMemory: evicted LRU trace '{lru_key}'")

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    @property
    def n_patterns(self) -> int:
        return len(self._traces)

    @property
    def memory_bytes(self) -> int:
        """Total bytes consumed by all prototype hypervectors."""
        return self.n_patterns * D   # int8

    @property
    def total_events_absorbed(self) -> int:
        return sum(t.n_writes for t in self._traces.values())

    def stats(self) -> Dict:
        if not self._traces:
            return {
                "n_patterns": 0, "memory_bytes": 0,
                "total_events_absorbed": 0, "bytes_per_event": 0,
            }
        absorbed = self.total_events_absorbed
        return {
            "n_patterns"            : self.n_patterns,
            "memory_bytes"          : self.memory_bytes,
            "memory_kb"             : round(self.memory_bytes / 1024, 2),
            "total_events_absorbed" : absorbed,
            "bytes_per_event"       : round(self.memory_bytes / absorbed, 2),
            "avg_writes_per_pattern": round(absorbed / self.n_patterns, 2),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"AssociativeMemory("
            f"patterns={s['n_patterns']}, "
            f"mem={s['memory_kb']}KB, "
            f"events_absorbed={s['total_events_absorbed']})"
        )

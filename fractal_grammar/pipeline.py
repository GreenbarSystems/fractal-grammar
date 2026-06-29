"""
fractal_grammar/pipeline.py

Top-level FractalGrammarPipeline — the single entry point for the library.

Usage (minimal):

    from fractal_grammar import FractalGrammarPipeline, from_llm_log

    pipe = FractalGrammarPipeline()
    sequences = [from_llm_log(messages) for messages in my_logs]
    ruleset = pipe.run(sequences)
    ruleset.to_jsonl("training_data.jsonl")
    ruleset.save("grammar.json")
    print(pipe.report())

Full pipeline stages:
  1. Encode   — BehavioralSequences → EncodedSequences
  2. Filter   — NoveltyFilter deduplicates, builds weighted buffer
  3. Cluster  — HierarchicalClusterer builds multi-resolution tree
  4. Detect   — SelfSimilarityDetector finds fractal patterns
  5. Extract  — GrammarExtractor builds compressed GrammarRuleset
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fractal_grammar.core.sequence import BehavioralSequence
from fractal_grammar.core.dedup import DedupEntry, NoveltyFilter
from fractal_grammar.embeddings.encoder import BehavioralEncoder, Backend
from fractal_grammar.embeddings.hdc_encoder import HDCEncoder, AssociativeMemory
from fractal_grammar.clustering.hierarchical import (
    HierarchicalClusterTree, HierarchicalClusterer
)
from fractal_grammar.clustering.self_similarity import (
    SelfSimilarPattern, SelfSimilarityDetector, compute_corpus_hurst
)
from fractal_grammar.grammar.extractor import GrammarExtractor, GrammarRuleset

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Full configuration for the FractalGrammarPipeline.

    Sensible defaults work out of the box. Override for tuning.
    """
    # Encoder
    embedding_dim        : int   = 128
    embedding_backend    : Optional[Backend] = None   # None = auto-detect
    embedding_model      : str   = "all-MiniLM-L6-v2"
    use_hdc              : bool  = False  # True = use HDC backend + AssociativeMemory

    # Novelty filter
    dedup_threshold      : float = 0.92
    dedup_num_perm       : int   = 128
    dedup_vector_verify  : bool  = True

    # Clustering
    cluster_levels       : List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    cluster_metric       : str   = "euclidean"
    min_events_to_cluster: int   = 5

    # Self-similarity detection
    similarity_threshold : float = 0.6
    min_hurst            : float = 0.55
    min_pattern_depth    : int   = 2

    # Grammar extraction
    max_examples_per_rule: int   = 3
    include_noise_residuals: bool = True

    # Logging
    log_level            : int   = logging.INFO


class FractalGrammarPipeline:
    """
    End-to-end fractal grammar extraction pipeline.

    Parameters
    ----------
    config : PipelineConfig — all settings in one place
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logging.basicConfig(level=self.config.log_level)

        # HDC path: use HDCEncoder directly + AssociativeMemory
        if self.config.use_hdc:
            self._hdc_encoder       = HDCEncoder(dim=self.config.embedding_dim)
            self._assoc_memory      = AssociativeMemory()
            # Wrap HDCEncoder in a compatible shim for the pipeline stages
            self._encoder           = self._hdc_encoder
        else:
            self._hdc_encoder  = None
            self._assoc_memory = None
            self._encoder   = BehavioralEncoder(
                dim     = self.config.embedding_dim,
                backend = self.config.embedding_backend,
                model   = self.config.embedding_model,
            )
        self._filter    = NoveltyFilter(
            similarity_threshold = self.config.dedup_threshold,
            num_perm             = self.config.dedup_num_perm,
            use_vector_verify    = self.config.dedup_vector_verify,
        )
        self._clusterer = HierarchicalClusterer(
            levels     = self.config.cluster_levels,
            metric     = self.config.cluster_metric,
            min_events = self.config.min_events_to_cluster,
        )
        self._detector  = SelfSimilarityDetector(
            similarity_threshold = self.config.similarity_threshold,
            min_hurst            = self.config.min_hurst,
            min_depth            = self.config.min_pattern_depth,
        )
        self._extractor = GrammarExtractor(
            max_examples_per_rule = self.config.max_examples_per_rule,
            include_noise         = self.config.include_noise_residuals,
        )

        # Runtime state (populated by run())
        self._tree     : Optional[HierarchicalClusterTree] = None
        self._patterns : List[SelfSimilarPattern]          = []
        self._ruleset  : Optional[GrammarRuleset]          = None
        self._timings  : Dict[str, float]                  = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, sequences: List[BehavioralSequence]) -> GrammarRuleset:
        """
        Run the full pipeline on a list of BehavioralSequences.
        Returns a GrammarRuleset ready for export.
        """
        total_start = time.perf_counter()

        # --- Stage 1: Encode ---
        t0 = time.perf_counter()
        logger.info(f"Stage 1/5 — Encoding {len(sequences)} sequences…")
        encoded_seqs = self._encoder.encode_sequences(sequences)
        all_encoded  = [ev for seq in encoded_seqs for ev in seq.events]
        self._timings["encode"] = time.perf_counter() - t0
        logger.info(f"  → {len(all_encoded)} events encoded ({self._timings['encode']:.2f}s)")

        # --- Stage 2: Novelty filter ---
        t0 = time.perf_counter()
        logger.info("Stage 2/5 — Novelty filtering…")
        novel_events, all_entries = self._filter.process_batch(all_encoded)
        self._timings["filter"] = time.perf_counter() - t0
        stats = self._filter.stats
        logger.info(
            f"  → {stats['novel']} novel / {stats['seen']} seen "
            f"({stats['compression_pct']}% compressed) "
            f"({self._timings['filter']:.2f}s)"
        )

        if len(all_entries) < self.config.min_events_to_cluster:
            logger.warning(
                f"Only {len(all_entries)} unique events after dedup. "
                f"Need {self.config.min_events_to_cluster} to cluster. "
                f"Returning minimal ruleset."
            )
            return self._minimal_ruleset(all_entries)

        # Get unique stored entries (not duplicates)
        unique_entries = self._filter.get_all_entries()

        # --- Stage 3: Cluster ---
        t0 = time.perf_counter()
        logger.info(f"Stage 3/5 — Clustering {len(unique_entries)} unique events…")
        self._tree = self._clusterer.fit(unique_entries)
        self._timings["cluster"] = time.perf_counter() - t0
        logger.info(
            f"  → {self._tree.summary()} ({self._timings['cluster']:.2f}s)"
        )

        # --- HDC: populate AssociativeMemory from cluster labels ---
        if self._assoc_memory is not None and self._tree is not None:
            for entry in unique_entries:
                # Use cluster label as pattern key; fall back to fingerprint prefix
                cluster_key = getattr(entry.event.event, 'metadata', {}).get(
                    'cluster_label',
                    f"cluster_{hash(entry.event.event.fingerprint) % 64}"
                )
                hv, _ = self._hdc_encoder.encode_text(entry.event.event.content)
                self._assoc_memory.write(cluster_key, hv)

        # --- Stage 4: Self-similarity detection ---
        t0 = time.perf_counter()
        logger.info("Stage 4/5 — Detecting self-similar patterns…")
        self._patterns   = self._detector.detect(self._tree)
        corpus_hurst     = compute_corpus_hurst(self._tree)
        self._timings["detect"] = time.perf_counter() - t0
        logger.info(
            f"  → {len(self._patterns)} patterns, corpus H={corpus_hurst:.3f} "
            f"({self._timings['detect']:.2f}s)"
        )

        # --- Stage 5: Grammar extraction ---
        t0 = time.perf_counter()
        logger.info("Stage 5/5 — Extracting grammar ruleset…")
        self._ruleset = self._extractor.extract(
            tree         = self._tree,
            patterns     = self._patterns,
            corpus_hurst = corpus_hurst,
        )
        self._timings["extract"] = time.perf_counter() - t0
        self._timings["total"]   = time.perf_counter() - total_start
        logger.info(
            f"  → {self._ruleset.n_rules} rules, "
            f"{self._ruleset.n_residuals} residuals "
            f"({self._timings['extract']:.2f}s)"
        )
        logger.info(f"Pipeline complete in {self._timings['total']:.2f}s")

        return self._ruleset

    # ------------------------------------------------------------------
    # Incremental update (for continuous learning loop)
    # ------------------------------------------------------------------

    def update(self, new_sequences: List[BehavioralSequence]) -> GrammarRuleset:
        """
        Incrementally update an existing grammar with new behavioral data.
        Reuses the existing novelty filter index — new sequences are compared
        against all previously seen events.

        This is the WAL pattern: capture is continuous, compaction is periodic.
        Call run() initially, then update() for each new batch.
        """
        if self._tree is None:
            logger.info("No existing grammar — running full pipeline.")
            return self.run(new_sequences)

        logger.info(f"Incremental update: {len(new_sequences)} new sequences")

        # Encode and filter new sequences against existing index
        encoded_seqs = self._encoder.encode_sequences(new_sequences)
        all_encoded  = [ev for seq in encoded_seqs for ev in seq.events]
        novel, _     = self._filter.process_batch(all_encoded)

        if not novel:
            logger.info("No novel events in update batch — grammar unchanged.")
            return self._ruleset

        # Re-cluster and re-extract with the full updated entry set
        unique_entries = self._filter.get_all_entries()
        self._tree     = self._clusterer.fit(unique_entries)
        self._patterns = self._detector.detect(self._tree)
        corpus_hurst   = compute_corpus_hurst(self._tree)
        self._ruleset  = self._extractor.extract(
            self._tree, self._patterns, corpus_hurst
        )
        return self._ruleset

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Human-readable pipeline report."""
        if self._ruleset is None:
            return "Pipeline has not been run yet."

        s = self._ruleset.summary()
        lines = [
            "=" * 60,
            "  FRACTAL GRAMMAR EXTRACTION REPORT",
            "=" * 60,
            f"  Input events       : {s.get('total_input_events', '?')}",
            f"  Novel (stored)     : {s.get('events_covered', '?')}",
            f"  Dedup compression  : {self._filter.stats['compression_pct']}%",
            "",
            f"  Grammar rules      : {s['n_rules']}",
            f"    Strong  (H≥0.70) : {s['strong_rules']}",
            f"    Moderate(H≥0.55) : {s['moderate_rules']}",
            f"  Residuals          : {s['n_residuals']}",
            f"  Coverage           : {s.get('coverage_pct', '?')}%",
            "",
            f"  Corpus Hurst (H)   : {s['corpus_hurst']}",
            f"  Avg rule Hurst     : {s['avg_hurst']}",
            f"  Avg compression    : {s['avg_compression']}x",
            f"  Total compression  : {round(self._ruleset.total_compression, 2)}x",
            "",
            "  Timings:",
        ]
        for stage, t in self._timings.items():
            lines.append(f"    {stage:<12}: {t:.2f}s")

        if self._ruleset.rules:
            lines += ["", "  Top rules by Hurst:"]
            for rule in self._ruleset.rules[:5]:
                lines.append(
                    f"    [{rule.rule_id}] H={rule.hurst_exponent:.3f} "
                    f"d={rule.depth} "
                    f"'{rule.representative[:50]}'"
                )
        if self._assoc_memory is not None:
            am = self._assoc_memory.stats()
            mem_kb = round(am['memory_bytes'] / 1024, 2) if am['memory_bytes'] else 0
            lines += [
                "",
                "  AssociativeMemory (HDC):",
                f"    Patterns stored    : {am['n_patterns']}",
                f"    Memory used        : {mem_kb}KB",
                f"    Events absorbed    : {am['total_events_absorbed']}",
                f"    Bytes/event        : {am.get('bytes_per_event', 0)}",
            ]
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _minimal_ruleset(self, entries: List[DedupEntry]) -> GrammarRuleset:
        """Return a pass-through ruleset when corpus is too small to cluster."""
        from fractal_grammar.grammar.extractor import GrammarRuleset
        rs = GrammarRuleset(
            residuals    = entries,
            corpus_hurst = 0.5,
            stats        = {
                "total_input_events": len(entries),
                "events_covered"    : 0,
                "coverage_pct"      : 0.0,
                "n_residuals"       : len(entries),
                "n_rules"           : 0,
                "note"              : "Corpus too small to cluster",
            },
        )
        self._ruleset = rs
        return rs

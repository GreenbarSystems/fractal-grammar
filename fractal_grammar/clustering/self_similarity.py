"""
clustering/self_similarity.py

Fractal self-similarity detector.

Core insight from neuroscience: the brain encodes behavioral RULES in
the prefrontal cortex and behavioral EPISODES in the hippocampus.
Faster learning correlates with longer-duration fractal patterns —
the rule covers more ground than a single episode.

This module finds those rules: patterns that repeat across multiple
resolution levels of the cluster tree. A true behavioral rule is a
pattern whose structure is self-similar whether you look at it coarsely
(domain level) or finely (micro-behavior level).

A pattern is fractal if:
  1. It appears as a coherent cluster at multiple resolution levels
  2. Its children clusters are geometrically similar to the parent
     (centroid cosine similarity is high across levels)
  3. The structural variance within children is lower than between siblings
     (the children are compressed versions of the parent, not random splits)

The Hurst exponent (H) quantifies this:
  H > 0.7  → long-range correlation / fractal structure  (loop machine)
  H ≈ 0.5  → random / no structure
  H < 0.5  → anti-correlated / noise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from fractal_grammar.clustering.hierarchical import (
    Cluster, ClusterLevel, HierarchicalClusterTree
)

logger = logging.getLogger(__name__)


@dataclass
class SelfSimilarPattern:
    """
    A behavioral pattern that appears self-similarly across multiple
    resolution levels — a candidate grammar rule.

    Attributes
    ----------
    root_cluster    : the coarsest-level cluster this pattern originates from
    instances       : all clusters across levels that manifest this pattern
    hurst_exponent  : fractal dimension measure (>0.7 = strong self-similarity)
    compression_ratio: how much a grammar rule compresses this pattern
                       (members / levels that encode it)
    representative  : the best single example of this behavioral pattern
    rule_label      : auto-generated descriptive label
    """
    root_cluster    : Cluster
    instances       : List[Cluster]        = field(default_factory=list)
    hurst_exponent  : float                = 0.0
    compression_ratio: float               = 0.0
    representative  : str                  = ""
    rule_label      : str                  = ""
    child_similarity: List[float]          = field(default_factory=list)

    @property
    def depth(self) -> int:
        """How many resolution levels this pattern spans."""
        return len(set(c.level for c in self.instances))

    @property
    def total_members(self) -> int:
        return sum(c.size for c in self.instances)

    @property
    def is_strong(self) -> bool:
        """A strong fractal pattern: high Hurst, spans 3+ levels."""
        return self.hurst_exponent >= 0.7 and self.depth >= 3

    @property
    def is_moderate(self) -> bool:
        return self.hurst_exponent >= 0.55 and self.depth >= 2

    def to_dict(self) -> Dict:
        return {
            "rule_label"       : self.rule_label,
            "representative"   : self.representative,
            "hurst_exponent"   : round(self.hurst_exponent, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "depth"            : self.depth,
            "total_members"    : self.total_members,
            "is_strong"        : self.is_strong,
            "child_similarity" : [round(s, 4) for s in self.child_similarity],
        }


class SelfSimilarityDetector:
    """
    Traverses a HierarchicalClusterTree and identifies self-similar
    patterns across resolution levels.

    Parameters
    ----------
    similarity_threshold : minimum cosine similarity between parent and
                           child cluster centroids to count as self-similar
    min_hurst            : minimum Hurst exponent to report a pattern
    min_depth            : minimum number of levels a pattern must span
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        min_hurst           : float = 0.55,
        min_depth           : int   = 2,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_hurst            = min_hurst
        self.min_depth            = min_depth

    def detect(
        self, tree: HierarchicalClusterTree
    ) -> List[SelfSimilarPattern]:
        """
        Run full self-similarity detection over the cluster tree.
        Returns patterns sorted by Hurst exponent (strongest first).
        """
        if not tree.levels:
            return []

        patterns: List[SelfSimilarPattern] = []

        # Start from the coarsest level that actually has non-noise clusters
        root_level = None
        for lvl in tree.levels:
            if lvl.n_clusters > 0:
                root_level = lvl
                break

        if root_level is None:
            logger.warning("No non-noise clusters found at any level.")
            return []

        for root_cluster in root_level.clusters:
            if root_cluster.is_noise:
                continue

            pattern = self._trace_pattern(root_cluster, tree)
            if pattern is None:
                continue
            if pattern.depth < self.min_depth:
                continue
            if pattern.hurst_exponent < self.min_hurst:
                continue

            pattern.representative  = self._pick_representative(pattern)
            pattern.rule_label      = self._generate_label(pattern)
            pattern.compression_ratio = self._compute_compression(pattern)
            patterns.append(pattern)

        patterns.sort(key=lambda p: p.hurst_exponent, reverse=True)
        logger.info(
            f"Detected {len(patterns)} self-similar patterns "
            f"({sum(1 for p in patterns if p.is_strong)} strong, "
            f"{sum(1 for p in patterns if p.is_moderate and not p.is_strong)} moderate)"
        )
        return patterns

    # ------------------------------------------------------------------
    # Core tracing
    # ------------------------------------------------------------------

    def _trace_pattern(
        self,
        root: Cluster,
        tree: HierarchicalClusterTree,
    ) -> Optional[SelfSimilarPattern]:
        """
        Starting from a root cluster, trace its self-similar descendants
        through finer resolution levels.
        """
        instances      : List[Cluster] = [root]
        child_sims     : List[float]   = []
        current_cluster = root

        # Walk down through levels via children links,
        # starting from the level AFTER the root's level
        start_idx = root.level + 1
        for level_idx in range(start_idx, len(tree.levels)):
            level = tree.levels[level_idx]

            # Find children of current_cluster at this level.
            # First try parent_id link; fall back to member overlap.
            children = [
                c for c in level.clusters
                if not c.is_noise and c.parent_id == current_cluster.cluster_id
            ]
            if not children:
                # Fallback: find clusters whose members overlap with current
                current_fps = {e.event.fingerprint for e in current_cluster.members}
                children = [
                    c for c in level.clusters
                    if not c.is_noise and
                    any(e.event.fingerprint in current_fps for e in c.members)
                ]

            if not children:
                break

            # Pick the child most similar to the parent centroid
            sims = [current_cluster.self_similarity_to(ch) for ch in children]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            best_child = children[best_idx]

            if best_sim < self.similarity_threshold:
                break

            child_sims.append(best_sim)
            instances.append(best_child)
            current_cluster = best_child

        if len(instances) < 2:
            return None

        hurst = self._compute_hurst(instances, child_sims)

        return SelfSimilarPattern(
            root_cluster    = root,
            instances       = instances,
            hurst_exponent  = hurst,
            child_similarity= child_sims,
        )

    # ------------------------------------------------------------------
    # Hurst exponent estimation
    # ------------------------------------------------------------------

    def _compute_hurst(
        self,
        instances : List[Cluster],
        child_sims: List[float],
    ) -> float:
        """
        Estimate the Hurst exponent for this pattern.

        We use a simplified R/S analysis over the centroid similarity series.
        A full DFA (Detrended Fluctuation Analysis) would be more precise
        but requires longer series than we typically have.

        For short series we use the mean child similarity as a proxy,
        scaled to the [0, 1] range and mapped to [0.5, 1.0].
        """
        if not child_sims:
            return 0.5

        if len(child_sims) >= 4:
            return self._hurst_rs(child_sims)

        # Short series: scale mean similarity to Hurst range
        mean_sim = float(np.mean(child_sims))
        # High similarity → high Hurst (strong self-similarity)
        return 0.5 + 0.5 * mean_sim

    def _hurst_rs(self, series: List[float]) -> float:
        """
        R/S (Rescaled Range) Hurst exponent estimation.
        Works on similarity series of length >= 4.
        """
        x = np.array(series)
        n = len(x)
        if n < 4:
            return 0.5 + 0.5 * float(np.mean(x))

        try:
            mean = x.mean()
            deviation = np.cumsum(x - mean)
            R = deviation.max() - deviation.min()
            S = x.std(ddof=1)
            if S < 1e-10:
                return 0.8   # perfectly consistent → high Hurst
            rs = R / S
            # H = log(R/S) / log(n) — simplified single-scale estimate
            H = np.log(rs) / np.log(n) if rs > 0 else 0.5
            # Clamp to valid range
            return float(np.clip(H, 0.0, 1.0))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_representative(self, pattern: SelfSimilarPattern) -> str:
        """
        Pick the most representative text example for this pattern.
        Strategy: highest-weight member at the finest level.
        """
        finest = pattern.instances[-1]
        if not finest.members:
            return pattern.root_cluster.dominant_content
        best = max(finest.members, key=lambda e: e.weight)
        return best.event.event.content

    def _generate_label(self, pattern: SelfSimilarPattern) -> str:
        """
        Auto-generate a descriptive label.
        Format: DOMAIN:LEVEL_COUNT:STRENGTH
        """
        strength = (
            "strong"   if pattern.is_strong   else
            "moderate" if pattern.is_moderate else
            "weak"
        )
        domain_label = pattern.root_cluster.label
        return (
            f"{domain_label}_L{pattern.root_cluster.cluster_id}"
            f"_d{pattern.depth}_{strength}"
        )

    def _compute_compression(self, pattern: SelfSimilarPattern) -> float:
        """
        Compression ratio: how many member events does one grammar rule
        represent? Higher is better.
        Rule stores O(1) information. Members store O(N) information.
        """
        unique_levels = pattern.depth
        if unique_levels <= 1:
            return 1.0
        total = pattern.total_members
        # Rule = 1 centroid + depth similarity values
        rule_cost = 1 + unique_levels
        return total / rule_cost if rule_cost > 0 else 1.0


def compute_corpus_hurst(tree: HierarchicalClusterTree) -> float:
    """
    Compute a corpus-level Hurst exponent by analyzing the inter-cluster
    similarity series across all levels.

    A high corpus Hurst means the overall behavioral corpus is fractal —
    patterns at coarse scale predict patterns at fine scale.
    This is the property that makes grammar compression effective.
    """
    if len(tree.levels) < 2:
        return 0.5

    level_sims: List[float] = []
    for i in range(len(tree.levels) - 1):
        coarse = tree.levels[i]
        fine   = tree.levels[i + 1]

        if not coarse.clusters or not fine.clusters:
            continue

        # Average pairwise similarity between coarse and fine centroids
        coarse_list = [c.centroid for c in coarse.clusters if not c.is_noise]
        fine_list   = [c.centroid for c in fine.clusters   if not c.is_noise]
        if not coarse_list or not fine_list:
            continue
        coarse_centroids = np.vstack(coarse_list)
        fine_centroids   = np.vstack(fine_list)

        sim_matrix = coarse_centroids @ fine_centroids.T
        level_sims.append(float(sim_matrix.max(axis=1).mean()))

    if not level_sims:
        return 0.5

    # Apply R/S to level similarity series
    detector = SelfSimilarityDetector()
    return detector._hurst_rs(level_sims) if len(level_sims) >= 4 else (
        0.5 + 0.5 * float(np.mean(level_sims))
    )

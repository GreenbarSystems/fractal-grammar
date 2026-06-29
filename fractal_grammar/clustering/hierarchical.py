"""
clustering/hierarchical.py

Multi-resolution hierarchical clustering engine.

This is the core of the fractal approach. We cluster behavioral events
at multiple scales simultaneously — coarse (macro-patterns) down to fine
(micro-patterns) — then measure self-similarity across those scales.

The loop-machine analogy: the clustering finds all the loops of different
sizes, and the self-similarity detector finds which small loops are
components of the larger loops.

Architecture:
  Level 0 (coarsest) — major behavioral domains   (3–8 clusters)
  Level 1             — behavioral categories      (8–20 clusters)
  Level 2             — behavioral sub-patterns    (20–60 clusters)
  Level 3 (finest)    — near-atomic behaviors      (60–200 clusters)

Each level is an HDBSCAN clustering. HDBSCAN is chosen over K-Means because:
  - No need to pre-specify cluster count
  - Handles noise/outliers natively
  - Produces hierarchical structure internally
  - Density-aware: clusters emerge from behavioral density, not geometry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import hdbscan
from sklearn.preprocessing import normalize

from fractal_grammar.core.dedup import DedupEntry

logger = logging.getLogger(__name__)


# Resolution levels and their HDBSCAN minimum cluster sizes
RESOLUTION_LEVELS = {
    0: {"min_cluster_size": 10, "min_samples": 3,  "label": "domain"},
    1: {"min_cluster_size": 5,  "min_samples": 2,  "label": "category"},
    2: {"min_cluster_size": 3,  "min_samples": 1,  "label": "pattern"},
    3: {"min_cluster_size": 2,  "min_samples": 1,  "label": "micro"},
}


@dataclass
class Cluster:
    """
    A single behavioral cluster at one resolution level.

    Attributes
    ----------
    level       : resolution level (0=coarsest, 3=finest)
    cluster_id  : HDBSCAN-assigned ID (-1 = noise)
    label       : human-readable resolution name
    centroid    : mean vector of all member events
    members     : DedupEntries belonging to this cluster
    total_weight: sum of member frequencies (behavioral importance signal)
    children    : sub-clusters at the next finer resolution
    parent_id   : cluster_id of parent at coarser level (None at level 0)
    """
    level       : int
    cluster_id  : int
    label       : str
    centroid    : NDArray[np.float32]
    members     : List[DedupEntry]          = field(default_factory=list)
    total_weight: float                     = 0.0
    children    : List["Cluster"]           = field(default_factory=list)
    parent_id   : Optional[int]             = None

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == -1

    @property
    def dominant_content(self) -> str:
        """Text of the highest-weight member — the 'representative' example."""
        if not self.members:
            return ""
        best = max(self.members, key=lambda e: e.weight)
        return best.event.event.content

    def self_similarity_to(self, other: "Cluster") -> float:
        """Cosine similarity between this cluster's centroid and another's."""
        norm_a = np.linalg.norm(self.centroid)
        norm_b = np.linalg.norm(other.centroid)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(self.centroid, other.centroid) / (norm_a * norm_b))


@dataclass
class ClusterLevel:
    """All clusters at one resolution level."""
    level     : int
    label     : str
    clusters  : List[Cluster]          = field(default_factory=list)
    noise     : Optional[Cluster]      = None
    labels    : Optional[NDArray]      = None   # raw HDBSCAN label array

    @property
    def n_clusters(self) -> int:
        return len([c for c in self.clusters if not c.is_noise])

    def get(self, cluster_id: int) -> Optional[Cluster]:
        for c in self.clusters:
            if c.cluster_id == cluster_id:
                return c
        return None


@dataclass
class HierarchicalClusterTree:
    """
    Full multi-resolution cluster tree over a behavioral corpus.

    levels[0] = coarsest (domains)
    levels[3] = finest   (micro-behaviors)
    """
    levels : List[ClusterLevel]         = field(default_factory=list)
    matrix : Optional[NDArray]          = None   # (n_events, dim) input matrix
    entries: List[DedupEntry]           = field(default_factory=list)

    @property
    def total_events(self) -> int:
        return len(self.entries)

    @property
    def depth(self) -> int:
        return len(self.levels)

    def summary(self) -> Dict:
        return {
            "total_events": self.total_events,
            "levels": [
                {
                    "level"     : lvl.level,
                    "label"     : lvl.label,
                    "n_clusters": lvl.n_clusters,
                }
                for lvl in self.levels
            ]
        }


class HierarchicalClusterer:
    """
    Fits a multi-resolution cluster tree over a list of DedupEntries.

    Parameters
    ----------
    levels     : which resolution levels to compute (default: all 4)
    metric     : distance metric for HDBSCAN ('euclidean' or 'cosine')
    min_events : minimum events needed to attempt clustering
    """

    def __init__(
        self,
        levels    : List[int] = None,
        metric    : str       = "euclidean",
        min_events: int       = 5,
    ):
        self.levels     = levels or list(RESOLUTION_LEVELS.keys())
        self.metric     = metric
        self.min_events = min_events

    def fit(self, entries: List[DedupEntry]) -> HierarchicalClusterTree:
        """
        Build the full cluster tree from a list of DedupEntries.
        Entries should have passed through NoveltyFilter first.
        """
        if len(entries) < self.min_events:
            logger.warning(
                f"Only {len(entries)} events — need at least {self.min_events}. "
                f"Returning shallow tree."
            )

        # Build weighted matrix — frequency-weight each vector
        vectors = np.vstack([e.event.vector for e in entries]).astype(np.float32)
        weights = np.array([e.weight for e in entries], dtype=np.float32)

        # L2-normalize for cosine-aware clustering
        matrix  = normalize(vectors, norm="l2")

        tree = HierarchicalClusterTree(
            matrix=matrix,
            entries=entries,
        )

        prev_labels: Optional[NDArray] = None

        for level_idx in self.levels:
            cfg   = RESOLUTION_LEVELS[level_idx]
            label = cfg["label"]

            level_result = self._cluster_at_level(
                matrix      = matrix,
                entries     = entries,
                weights     = weights,
                level_idx   = level_idx,
                label       = label,
                min_cluster_size = max(
                    cfg["min_cluster_size"],
                    max(2, len(entries) // 30),  # adaptive floor
                ),
                min_samples      = cfg["min_samples"],
                prev_labels      = prev_labels,
            )

            tree.levels.append(level_result)
            prev_labels = level_result.labels
            logger.info(
                f"Level {level_idx} ({label}): "
                f"{level_result.n_clusters} clusters"
            )

        # Link children across levels
        self._link_hierarchy(tree)

        return tree

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cluster_at_level(
        self,
        matrix          : NDArray,
        entries         : List[DedupEntry],
        weights         : NDArray,
        level_idx       : int,
        label           : str,
        min_cluster_size: int,
        min_samples     : int,
        prev_labels     : Optional[NDArray],
    ) -> ClusterLevel:

        n = len(entries)

        # With very few events, fall back to a single cluster
        if n < min_cluster_size:
            centroid = matrix.mean(axis=0)
            single   = Cluster(
                level      = level_idx,
                cluster_id = 0,
                label      = label,
                centroid   = centroid,
                members    = entries,
                total_weight = float(weights.sum()),
            )
            lvl = ClusterLevel(
                level    = level_idx,
                label    = label,
                clusters = [single],
                labels   = np.zeros(n, dtype=int),
            )
            return lvl

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            min_samples      = min_samples,
            metric           = self.metric,
            cluster_selection_method = "eom",   # excess of mass — stable clusters
            prediction_data  = True,
        )

        try:
            raw_labels = clusterer.fit_predict(matrix)
        except Exception as e:
            logger.warning(f"HDBSCAN failed at level {level_idx}: {e}. Using single cluster.")
            raw_labels = np.zeros(n, dtype=int)

        unique_ids = sorted(set(raw_labels))
        clusters: List[Cluster] = []

        for cid in unique_ids:
            mask    = raw_labels == cid
            members = [entries[i] for i in range(n) if mask[i]]
            w       = weights[mask]
            vecs    = matrix[mask]
            # Frequency-weighted centroid
            centroid = (vecs * w[:, None]).sum(axis=0) / w.sum()

            cluster = Cluster(
                level        = level_idx,
                cluster_id   = cid,
                label        = label,
                centroid     = centroid.astype(np.float32),
                members      = members,
                total_weight = float(w.sum()),
                parent_id    = (
                    int(prev_labels[mask][0])
                    if prev_labels is not None and mask.any()
                    else None
                ),
            )
            clusters.append(cluster)

        noise = next((c for c in clusters if c.is_noise), None)

        return ClusterLevel(
            level    = level_idx,
            label    = label,
            clusters = clusters,
            noise    = noise,
            labels   = raw_labels,
        )

    def _link_hierarchy(self, tree: HierarchicalClusterTree) -> None:
        """
        Link clusters across adjacent levels so each cluster knows its
        children at the next finer resolution.
        """
        for i in range(len(tree.levels) - 1):
            coarse = tree.levels[i]
            fine   = tree.levels[i + 1]

            for fine_cluster in fine.clusters:
                if fine_cluster.parent_id is None:
                    continue
                parent = coarse.get(fine_cluster.parent_id)
                if parent is not None:
                    parent.children.append(fine_cluster)

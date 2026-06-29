"""
grammar/extractor.py

Grammar rule extractor and centroid synthesizer.

Takes the self-similar patterns detected by the SelfSimilarityDetector
and compresses them into a compact GrammarRuleset — the behavioral grammar.

Key outputs:
  1. GrammarRule   — one rule per self-similar pattern (stores the rule, not all instances)
  2. GrammarRuleset — the complete compressed behavioral grammar
  3. JSONL export  — LoRA-ready training data synthesized from the grammar

The compression logic:
  - Strong fractal patterns → single grammar rule replaces all member examples
  - Moderate patterns       → rule + 2-3 representative examples
  - Noise / weak patterns   → kept as-is (no compression benefit)

The synthesizer generates training examples from rules by:
  1. Taking the weighted centroid as the "canonical form"
  2. Sampling highest-weight members as positive examples
  3. Generating instruction/output pairs suitable for LoRA JSONL
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from fractal_grammar.clustering.self_similarity import SelfSimilarPattern
from fractal_grammar.clustering.hierarchical import (
    HierarchicalClusterTree, Cluster
)
from fractal_grammar.core.dedup import DedupEntry

logger = logging.getLogger(__name__)


@dataclass
class GrammarRule:
    """
    One compressed behavioral rule.

    A rule represents a self-similar behavioral pattern that recurs
    across multiple resolution scales. Instead of storing all N examples
    of this pattern, the grammar stores this one rule.

    Attributes
    ----------
    rule_id          : unique identifier
    label            : human-readable name from self-similarity detection
    centroid         : the behavioral center of this rule in vector space
    representative   : best single text example of this behavior
    examples         : top-K member examples (for moderate patterns)
    hurst_exponent   : fractal strength of this rule
    compression_ratio: N_members / rule_cost (how much this rule compresses)
    total_weight     : sum of member frequencies — behavioral importance
    depth            : how many resolution levels this pattern spans
    metadata         : arbitrary additional context
    """
    rule_id          : str
    label            : str
    centroid         : NDArray[np.float32]
    representative   : str
    examples         : List[str]              = field(default_factory=list)
    hurst_exponent   : float                  = 0.0
    compression_ratio: float                  = 0.0
    total_weight     : float                  = 0.0
    depth            : int                    = 1
    metadata         : Dict[str, Any]         = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "rule_id"          : self.rule_id,
            "label"            : self.label,
            "representative"   : self.representative,
            "examples"         : self.examples,
            "hurst_exponent"   : round(self.hurst_exponent, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "total_weight"     : round(self.total_weight, 3),
            "depth"            : self.depth,
            "metadata"         : self.metadata,
        }

    def to_jsonl_training_pair(
        self,
        system_prompt: str = "You are a personalized AI assistant.",
    ) -> List[Dict]:
        """
        Export this rule as LoRA-ready JSONL training pairs.

        Format: instruction/output pairs weighted by rule importance.
        Returns multiple pairs (one per example), each tagged with weight.
        """
        pairs = []

        # Primary pair from representative
        pairs.append({
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": self.representative},
                {"role": "assistant", "content": f"[Rule: {self.label}]"},
            ],
            "weight"    : round(self.total_weight, 3),
            "rule_id"   : self.rule_id,
            "hurst"     : round(self.hurst_exponent, 4),
        })

        # Additional pairs from examples (lower weight)
        for i, example in enumerate(self.examples[:3]):
            pairs.append({
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": example},
                    {"role": "assistant", "content": f"[Rule: {self.label}]"},
                ],
                "weight"  : round(self.total_weight * 0.5, 3),
                "rule_id" : self.rule_id,
                "hurst"   : round(self.hurst_exponent, 4),
            })

        return pairs


@dataclass
class GrammarRuleset:
    """
    The complete compressed behavioral grammar for one user/corpus.

    Contains:
      - rules       : the compressed behavioral grammar (fractal patterns)
      - residuals   : events that didn't compress (noise, weak patterns)
      - corpus_hurst: overall corpus fractal dimension
      - stats       : compression statistics
    """
    rules         : List[GrammarRule]     = field(default_factory=list)
    residuals     : List[DedupEntry]      = field(default_factory=list)
    corpus_hurst  : float                 = 0.0
    stats         : Dict[str, Any]        = field(default_factory=dict)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def n_residuals(self) -> int:
        return len(self.residuals)

    @property
    def total_compression(self) -> float:
        """
        Overall compression ratio.
        How many input events does the grammar represent per rule?
        """
        total_members = sum(r.compression_ratio * (1 + r.depth)
                            for r in self.rules)
        if self.n_rules == 0:
            return 1.0
        return total_members / self.n_rules

    def summary(self) -> Dict:
        return {
            "n_rules"          : self.n_rules,
            "n_residuals"      : self.n_residuals,
            "corpus_hurst"     : round(self.corpus_hurst, 4),
            "total_compression": round(self.total_compression, 2),
            "strong_rules"     : sum(1 for r in self.rules if r.hurst_exponent >= 0.7),
            "moderate_rules"   : sum(1 for r in self.rules
                                     if 0.55 <= r.hurst_exponent < 0.7),
            **self.stats,
        }

    def to_jsonl(
        self,
        path         : str,
        include_residuals: bool = True,
        system_prompt: str = "You are a personalized AI assistant.",
    ) -> int:
        """
        Export the full grammar as a LoRA-ready JSONL training file.
        Returns the number of training pairs written.
        """
        pairs: List[Dict] = []

        for rule in self.rules:
            pairs.extend(rule.to_jsonl_training_pair(system_prompt))

        if include_residuals:
            for entry in self.residuals:
                pairs.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": entry.event.event.content},
                        {"role": "assistant", "content": "[Residual]"},
                    ],
                    "weight"  : round(entry.weight, 3),
                    "rule_id" : "residual",
                    "hurst"   : 0.0,
                })

        # Sort by weight descending — most important patterns first
        pairs.sort(key=lambda p: p["weight"], reverse=True)

        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        logger.info(f"Wrote {len(pairs)} training pairs to {path}")
        return len(pairs)

    def save(self, path: str) -> None:
        """Save the full ruleset as JSON (without numpy arrays)."""
        import numpy as np

        def _json_safe(obj):
            """Recursively convert numpy scalars to native Python types."""
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        data = {
            "corpus_hurst" : self.corpus_hurst,
            "stats"        : self.stats,
            "rules"        : [_json_safe(r.to_dict()) for r in self.rules],
            "residuals"    : [
                {
                    "content"  : e.event.event.content,
                    "weight"   : round(e.weight, 3),
                    "frequency": e.frequency,
                }
                for e in self.residuals
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Grammar ruleset saved to {path}")


class GrammarExtractor:
    """
    Converts a HierarchicalClusterTree + self-similar patterns into
    a compact GrammarRuleset.

    Parameters
    ----------
    max_examples_per_rule : how many member examples to keep per rule
                            (beyond the representative). Strong rules
                            need fewer; moderate need more context.
    include_noise         : whether to add noise-cluster events to residuals
    """

    def __init__(
        self,
        max_examples_per_rule: int  = 3,
        include_noise        : bool = True,
    ):
        self.max_examples_per_rule = max_examples_per_rule
        self.include_noise         = include_noise

    def extract(
        self,
        tree    : HierarchicalClusterTree,
        patterns: List[SelfSimilarPattern],
        corpus_hurst: float = 0.0,
    ) -> GrammarRuleset:
        """
        Build the GrammarRuleset from the cluster tree and detected patterns.
        """
        ruleset = GrammarRuleset(corpus_hurst=corpus_hurst)

        # Track which entries have been covered by a rule
        covered_fingerprints: set = set()

        # --- Build rules from strong / moderate patterns ---
        for i, pattern in enumerate(patterns):
            rule = self._pattern_to_rule(pattern, rule_id=f"R{i:04d}")
            if rule is None:
                continue
            ruleset.rules.append(rule)

            # Mark all members across all instances as covered
            for cluster in pattern.instances:
                for entry in cluster.members:
                    covered_fingerprints.add(entry.event.fingerprint)

        # --- Collect residuals: events not covered by any rule ---
        for entry in tree.entries:
            fp = entry.event.fingerprint
            if fp not in covered_fingerprints:
                ruleset.residuals.append(entry)

        # --- Add noise cluster events to residuals if requested ---
        if self.include_noise:
            for level in tree.levels:
                if level.noise and level.noise.members:
                    for entry in level.noise.members:
                        fp = entry.event.fingerprint
                        if fp not in covered_fingerprints:
                            ruleset.residuals.append(entry)
                            covered_fingerprints.add(fp)

        # --- Deduplicate residuals ---
        seen_fps: set = set()
        unique_residuals = []
        for entry in ruleset.residuals:
            fp = entry.event.fingerprint
            if fp not in seen_fps:
                unique_residuals.append(entry)
                seen_fps.add(fp)
        ruleset.residuals = sorted(
            unique_residuals, key=lambda e: e.weight, reverse=True
        )

        # --- Stats ---
        total_events = tree.total_events
        covered      = len(covered_fingerprints)
        ruleset.stats = {
            "total_input_events"  : total_events,
            "events_covered"      : covered,
            "coverage_pct"        : round(covered / total_events * 100, 1) if total_events else 0,
            "n_residuals"         : len(ruleset.residuals),
            "n_rules"             : ruleset.n_rules,
            "avg_hurst"           : round(
                np.mean([r.hurst_exponent for r in ruleset.rules]).item()
                if ruleset.rules else 0.0, 4
            ),
            "avg_compression"     : round(
                np.mean([r.compression_ratio for r in ruleset.rules]).item()
                if ruleset.rules else 1.0, 2
            ),
        }

        logger.info(
            f"Grammar extracted: {ruleset.n_rules} rules, "
            f"{ruleset.n_residuals} residuals, "
            f"{ruleset.stats['coverage_pct']}% coverage, "
            f"corpus H={corpus_hurst:.3f}"
        )
        return ruleset

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pattern_to_rule(
        self,
        pattern: SelfSimilarPattern,
        rule_id: str,
    ) -> Optional[GrammarRule]:
        """Convert one self-similar pattern into one GrammarRule."""

        if not pattern.instances:
            return None

        # Centroid: weighted average across all instances
        centroids = np.vstack([c.centroid for c in pattern.instances])
        weights   = np.array([c.total_weight for c in pattern.instances])
        if weights.sum() < 1e-10:
            weights = np.ones(len(centroids))
        centroid  = (centroids * weights[:, None]).sum(axis=0) / weights.sum()

        # Examples: top-K members by weight from finest level
        finest  = pattern.instances[-1]
        members = sorted(finest.members, key=lambda e: e.weight, reverse=True)

        # Strong rules need fewer examples (the rule is self-sufficient)
        k = (
            1 if pattern.is_strong   else
            2 if pattern.is_moderate else
            self.max_examples_per_rule
        )
        examples = [m.event.event.content for m in members[1:k+1]]

        total_weight = sum(c.total_weight for c in pattern.instances)

        return GrammarRule(
            rule_id          = rule_id,
            label            = pattern.rule_label,
            centroid         = centroid.astype(np.float32),
            representative   = pattern.representative,
            examples         = examples,
            hurst_exponent   = pattern.hurst_exponent,
            compression_ratio= pattern.compression_ratio,
            total_weight     = total_weight,
            depth            = pattern.depth,
            metadata         = {
                "child_similarity": pattern.child_similarity,
                "root_cluster_id" : pattern.root_cluster.cluster_id,
                "n_instances"     : len(pattern.instances),
            },
        )

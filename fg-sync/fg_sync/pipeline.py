"""
fg_sync/pipeline.py
-------------------
Reads incremental conversation captures from capture.jsonl,
runs them through the fractal-grammar extraction pipeline,
and writes a new ruleset.json.

Integration with fractal-grammar library (v0.2.0+):
  - MinHash LSH novelty gate (cosine dissimilarity > threshold)
  - HDC encoding (D=10,000, seed=0xFEEDBEEF)
  - HDBSCAN clustering
  - Grammar pattern extraction
  - AssociativeMemory write/persist

Known limitations (v0.1.0):
  - AssociativeMemory cluster keys are hash-derived, not real HDBSCAN labels
  - Clustering runs on float32 projected vectors, not HDC-space
  - These are tracked in KNOWN_LIMITATIONS.md — v0.3.0 target
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fg_sync.config import PipelineConfig, RulesetConfig

logger = logging.getLogger("fg_sync.pipeline")

# Cursor file — stores byte offset into capture.jsonl for incremental reads
CURSOR_FILE = Path.home() / ".fg-sync" / "cursor.json"
ASSOC_MEMORY_FILE = Path.home() / ".fg-sync" / "assoc_memory.pkl"


# ---------------------------------------------------------------------------
# Capture reader
# ---------------------------------------------------------------------------

def read_incremental(capture_path: Path) -> tuple[list[dict], int]:
    """
    Read new records from capture.jsonl since the last cursor position.

    Returns
    -------
    records : list[dict]
        New capture records since last run.
    new_offset : int
        New byte offset to persist in cursor.json.
    """
    cursor = _load_cursor()
    offset = cursor.get("offset", 0)
    new_records: list[dict] = []

    if not capture_path.exists():
        return [], offset

    with open(capture_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                new_records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed capture line")
        new_offset = f.tell()

    logger.info("Read %d new capture records (offset %d → %d)", len(new_records), offset, new_offset)
    return new_records, new_offset


def _load_cursor() -> dict:
    if CURSOR_FILE.exists():
        try:
            return json.loads(CURSOR_FILE.read_text())
        except Exception:
            pass
    return {"offset": 0}


def _save_cursor(offset: int):
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(json.dumps({"offset": offset, "ts": datetime.now(timezone.utc).isoformat()}))


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def extract_events(records: list[dict]) -> list[str]:
    """
    Flatten capture records into a list of text events for the pipeline.
    Each user message is one event.
    """
    events = []
    for rec in records:
        for msg in rec.get("messages", []):
            if msg.get("role") == "user":
                text = msg.get("content", "").strip()
                if text and len(text) > 10:
                    events.append(text)
    return events


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    capture_path: Path,
    pipeline_cfg: PipelineConfig,
    ruleset_cfg: RulesetConfig,
    dry_run: bool = False,
) -> dict | None:
    """
    Run the full fractal-grammar extraction pipeline and return the new ruleset dict.
    Returns None if skipped (not enough events, no new data, etc.).
    """
    t0 = time.monotonic()

    # 1. Read incremental captures
    new_records, new_offset = read_incremental(capture_path)

    if not new_records:
        logger.info("No new captures — pipeline skipped")
        return None

    events = extract_events(new_records)
    logger.info("Extracted %d text events from %d capture records", len(events), len(new_records))

    if len(events) < pipeline_cfg.min_events_to_run:
        logger.info(
            "Only %d events — need %d to run pipeline. Skipping.",
            len(events), pipeline_cfg.min_events_to_run
        )
        return None

    # 2. Run fractal-grammar pipeline
    try:
        ruleset = _run_fractal_grammar(events, pipeline_cfg, ruleset_cfg, new_records)
    except ImportError as e:
        logger.error(
            "fractal-grammar library not installed. Install with: pip install fractal-grammar\n"
            "Or add fractal_grammar/ to your Python path.\nError: %s", e
        )
        return None
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return None

    if ruleset is None:
        return None

    # 3. Persist cursor and ruleset
    if not dry_run:
        _save_cursor(new_offset)
        ruleset_cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        ruleset_cfg.output_path.write_text(
            json.dumps(ruleset, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(
            "Ruleset written to %s (%d rules, %.1fs)",
            ruleset_cfg.output_path,
            len(ruleset.get("behavioral_rules", [])),
            time.monotonic() - t0,
        )
    else:
        logger.info("[DRY RUN] Would write ruleset with %d rules", len(ruleset.get("behavioral_rules", [])))

    return ruleset


def _run_fractal_grammar(
    events: list[str],
    pipeline_cfg: PipelineConfig,
    ruleset_cfg: RulesetConfig,
    raw_records: list[dict],
) -> dict | None:
    """
    Internal: call fractal-grammar library, build ruleset dict.
    Falls back gracefully if library or optional deps are missing.
    """
    from fractal_grammar.pipeline import FractalGrammarPipeline, PipelineConfig as FGConfig

    fg_config = FGConfig(
        novelty_threshold=pipeline_cfg.novelty_threshold,
        min_cluster_size=pipeline_cfg.min_cluster_size,
        use_hdc=pipeline_cfg.use_hdc,
        hdc_dimensions=pipeline_cfg.hdc_dimensions,
        hdc_seed=pipeline_cfg.hdc_seed,
    )
    pipeline = FractalGrammarPipeline(config=fg_config)

    # Load existing AssociativeMemory if present
    if ASSOC_MEMORY_FILE.exists():
        try:
            with open(ASSOC_MEMORY_FILE, "rb") as f:
                pipeline.assoc_memory = pickle.load(f)
            logger.info("Loaded existing AssociativeMemory from %s", ASSOC_MEMORY_FILE)
        except Exception as e:
            logger.warning("Could not load AssociativeMemory: %s — starting fresh", e)

    result = pipeline.run(events)

    if result is None or not result.clusters:
        logger.info("Pipeline produced no clusters — not enough signal")
        return None

    # Persist updated AssociativeMemory
    if hasattr(pipeline, "assoc_memory") and pipeline.assoc_memory is not None:
        ASSOC_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ASSOC_MEMORY_FILE, "wb") as f:
            pickle.dump(pipeline.assoc_memory, f)

    # Build ruleset dict
    ruleset = _build_ruleset(result, raw_records, pipeline_cfg, ruleset_cfg)
    return ruleset


def _build_ruleset(
    pipeline_result: Any,
    raw_records: list[dict],
    pipeline_cfg: PipelineConfig,
    ruleset_cfg: RulesetConfig,
) -> dict:
    """Convert fractal-grammar pipeline result into fg-sync ruleset schema."""
    now = datetime.now(timezone.utc)
    decay_cutoff = now - timedelta(days=ruleset_cfg.decay_days)

    # Aggregate metadata from raw records
    models_seen = list({r.get("model", "unknown") for r in raw_records})
    session_count = len({r.get("session_id") for r in raw_records})
    event_count = sum(1 for r in raw_records for m in r.get("messages", []) if m.get("role") == "user")

    # Build user profile from cluster data
    all_top_terms: list[str] = []
    behavioral_rules: list[dict] = []

    clusters = getattr(pipeline_result, "clusters", [])
    grammar_rules = getattr(pipeline_result, "grammar_rules", {})
    timestamps_by_cluster = _map_timestamps(raw_records, clusters)

    for i, cluster in enumerate(clusters[:ruleset_cfg.max_rules]):
        cluster_id = getattr(cluster, "label", f"cluster_{i}")
        cluster_terms = getattr(cluster, "top_terms", [])
        cluster_size = getattr(cluster, "size", 0)
        centroid_repr = getattr(cluster, "centroid_hex", "")

        # Get grammar rules for this cluster
        rules_for_cluster = grammar_rules.get(str(cluster_id), [])
        pattern_summary = _summarize_cluster(cluster_terms, rules_for_cluster)

        # Timestamps
        ts_data = timestamps_by_cluster.get(str(cluster_id), {})
        first_seen = ts_data.get("first", now).isoformat()
        last_seen = ts_data.get("last", now).isoformat()
        last_dt = ts_data.get("last", now)

        # Recency score: 1.0 if seen today, decays linearly to 0.1 at decay_days
        age_days = max(0, (now - last_dt).days)
        recency_score = max(0.1, 1.0 - (age_days / max(ruleset_cfg.decay_days, 1)) * (1 - 0.1))

        # Weight: cluster size (normalized) × recency
        max_size = max((getattr(c, "size", 1) for c in clusters), default=1)
        size_weight = cluster_size / max(max_size, 1)
        weight = round(ruleset_cfg.recency_weight * recency_score + (1 - ruleset_cfg.recency_weight) * size_weight, 3)

        all_top_terms.extend(cluster_terms[:3])

        behavioral_rules.append({
            "id": f"rule_{i+1:03d}",
            "cluster_label": _slugify(cluster_terms[0] if cluster_terms else f"cluster_{i}"),
            "weight": round(weight, 3),
            "recency_score": round(recency_score, 3),
            "pattern_summary": pattern_summary,
            "trigger_terms": cluster_terms[:8],
            "grammar_rules": rules_for_cluster[:5],
            "hdc_centroid_hex": centroid_repr[:16] if centroid_repr else "",
            "first_seen": first_seen,
            "last_seen": last_seen,
            "event_count": cluster_size,
        })

    # Sort by weight descending
    behavioral_rules.sort(key=lambda r: r["weight"], reverse=True)

    # Build prompt prefix
    prompt_prefix = _build_prompt_prefix(behavioral_rules, ruleset_cfg.max_prompt_tokens)

    # User profile
    # Deduplicate top terms
    seen: set[str] = set()
    unique_terms = [t for t in all_top_terms if not (t in seen or seen.add(t))]

    # Extract bigrams from top terms
    bigrams = [[unique_terms[j], unique_terms[j+1]] for j in range(0, min(len(unique_terms)-1, 6), 2)]

    user_profile = {
        "primary_domains": unique_terms[:5],
        "top_bigrams": bigrams,
        "session_count": session_count,
        "models_seen": models_seen,
    }

    return {
        "schema_version": "1.0",
        "generated_at": now.isoformat(),
        "model": models_seen[0] if models_seen else "unknown",
        "event_count": event_count,
        "session_count": session_count,
        "user_profile": user_profile,
        "behavioral_rules": behavioral_rules,
        "prompt_prefix": prompt_prefix,
    }


def _map_timestamps(raw_records: list[dict], clusters: list) -> dict[str, dict]:
    """Best-effort: map cluster IDs to first/last seen timestamps from raw records."""
    # Without true label assignment per record, we approximate using record timestamps
    # This is noted as a known limitation — v0.3.0 will do per-record label assignment
    ts_map: dict[str, dict] = {}
    if not raw_records:
        return ts_map

    all_ts = []
    for r in raw_records:
        try:
            all_ts.append(datetime.fromisoformat(r["ts"].replace("Z", "+00:00")))
        except Exception:
            pass

    if not all_ts:
        return ts_map

    first_ts = min(all_ts)
    last_ts = max(all_ts)

    for i, cluster in enumerate(clusters):
        cluster_id = str(getattr(cluster, "label", f"cluster_{i}"))
        ts_map[cluster_id] = {"first": first_ts, "last": last_ts}

    return ts_map


def _summarize_cluster(terms: list[str], rules: list) -> str:
    """Generate a one-sentence pattern summary from top terms and grammar rules."""
    if not terms:
        return "Unclassified behavioral cluster."
    term_str = ", ".join(terms[:4])
    if rules:
        rule_str = rules[0] if isinstance(rules[0], str) else str(rules[0])
        return f"Recurring pattern around {term_str}. Primary grammar rule: {rule_str[:120]}."
    return f"Recurring behavioral pattern centered on: {term_str}."


def _build_prompt_prefix(rules: list[dict], max_tokens: int) -> str:
    """
    Build a compact system prompt prefix from behavioral rules.
    Truncates by weight order to stay within token budget.
    Rough token estimate: 1 token ≈ 4 chars.
    """
    char_budget = max_tokens * 4
    lines = [
        "## Behavioral Context (fg-sync)\n",
        "This session includes a personalized behavioral context derived from your local conversation history.\n\n",
    ]

    for rule in rules:
        line = f"- [{rule['cluster_label']} w={rule['weight']:.2f}] {rule['pattern_summary']}\n"
        lines.append(line)
        if sum(len(l) for l in lines) > char_budget:
            lines.pop()  # remove the one that pushed over budget
            break

    lines.append("\n---\n")
    return "".join(lines)


def _slugify(text: str) -> str:
    """Convert a term to a valid identifier slug."""
    import re
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:40]

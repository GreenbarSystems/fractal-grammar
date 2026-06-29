"""
fg_sync/metrics.py
------------------
Local performance metrics suite for fg-sync.

Measures five metrics comparing baseline Ollama sessions (no fg-sync)
against fg-sync sessions (ruleset injected):

  M1 — Prompt Clarification Rate
       % of user turns that are follow-up corrections ("I meant...", "no, ...")
  M2 — System Prompt Token Overhead
       Tokens consumed by injected prefix vs. zero (baseline) or naive prompt
  M3 — Ruleset Storage Footprint
       ruleset.json size vs. raw capture.jsonl size (compression ratio)
  M4 — Response Re-Roll Rate
       % of assistant responses the user regenerated (detected via repeated
       assistant turns on same prompt)
  M5 — Context Re-Establishment Cost
       Tokens spent in first user turn re-explaining background context
       (detected via heuristic keyword matching)

Usage:
  from fg_sync.metrics import MetricsCollector
  collector = MetricsCollector(metrics_path, capture_path, ruleset_path)
  collector.record_session(session_record)
  print(collector.compare())
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Clarification signals — user correcting the model
CLARIFICATION_PATTERNS = re.compile(
    r"\b(I meant|no[,.]?\s|actually[,.]?\s|not what I asked|that'?s? (not|wrong)|"
    r"let me clarify|clarification|to clarify|I said|I was asking|I mean)\b",
    re.IGNORECASE,
)

# Context re-establishment signals — user re-explaining themselves
CONTEXT_REESTABLISH_PATTERNS = re.compile(
    r"\b(I am a|I'?m a|I work|my (project|company|product|goal|background)|"
    r"as I mentioned|as a reminder|to recap|for context|just so you know|"
    r"to give you context)\b",
    re.IGNORECASE,
)

# Approximate tokens from text (1 token ≈ 4 chars)
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Session analyzer
# ---------------------------------------------------------------------------

def analyze_session(record: dict, injected: bool = False, prefix_chars: int = 0) -> dict:
    """
    Analyze a single capture record and return a metrics snapshot.

    Parameters
    ----------
    record : dict
        A single record from capture.jsonl.
    injected : bool
        Whether fg-sync was active for this session.
    prefix_chars : int
        Length of injected prompt prefix in characters (0 if not injected).

    Returns
    -------
    dict
        Metrics snapshot for this session.
    """
    messages = record.get("messages", [])
    user_turns = [m for m in messages if m.get("role") == "user"]
    assistant_turns = [m for m in messages if m.get("role") == "assistant"]

    # M1: Clarification rate
    clarification_count = sum(
        1 for m in user_turns
        if CLARIFICATION_PATTERNS.search(m.get("content", ""))
    )
    clarification_rate = clarification_count / max(len(user_turns), 1)

    # M2: System prompt token overhead
    tokens_prefix = _approx_tokens(" " * prefix_chars) if prefix_chars > 0 else 0

    # M4: Re-roll rate — detect repeated consecutive assistant turns
    # Heuristic: if two assistant messages follow each other without user in between
    reroll_count = 0
    prev_role = None
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and prev_role == "assistant":
            reroll_count += 1
        prev_role = role
    reroll_rate = reroll_count / max(len(assistant_turns), 1)

    # M5: Context re-establishment cost (first user turn only)
    context_reestablishment_tokens = 0
    if user_turns:
        first_content = user_turns[0].get("content", "")
        if CONTEXT_REESTABLISH_PATTERNS.search(first_content):
            context_reestablishment_tokens = _approx_tokens(first_content)

    # Total tokens
    total_tokens = record.get("tokens_prompt", 0) + record.get("tokens_completion", 0)
    if total_tokens == 0:
        # Estimate from content
        all_content = " ".join(m.get("content", "") for m in messages)
        total_tokens = _approx_tokens(all_content)

    return {
        "ts": record.get("ts", datetime.now(timezone.utc).isoformat()),
        "session_id": record.get("session_id", ""),
        "model": record.get("model", "unknown"),
        "mode": "fg_sync" if injected else "baseline",
        "fg_injected": injected,
        # M1
        "clarification_count": clarification_count,
        "clarification_rate": round(clarification_rate, 4),
        # M2
        "tokens_prefix": tokens_prefix,
        # M3 — computed at storage level, not per session
        "tokens_prompt": record.get("tokens_prompt", 0),
        "tokens_completion": record.get("tokens_completion", 0),
        "total_tokens": total_tokens,
        # M4
        "reroll_count": reroll_count,
        "reroll_rate": round(reroll_rate, 4),
        # M5
        "context_reestablishment_tokens": context_reestablishment_tokens,
        "user_turn_count": len(user_turns),
        "assistant_turn_count": len(assistant_turns),
    }


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Records and compares session metrics across baseline and fg-sync modes.

    Parameters
    ----------
    metrics_path : Path
        Path to metrics.jsonl (append-only).
    capture_path : Path
        Path to capture.jsonl.
    ruleset_path : Path
        Path to ruleset.json.
    """

    def __init__(self, metrics_path: Path, capture_path: Path, ruleset_path: Path):
        self.metrics_path = Path(metrics_path).expanduser()
        self.capture_path = Path(capture_path).expanduser()
        self.ruleset_path = Path(ruleset_path).expanduser()

    def record_session(self, record: dict, injected: bool = False, prefix_chars: int = 0):
        """Analyze a session record and append to metrics.jsonl."""
        snapshot = analyze_session(record, injected=injected, prefix_chars=prefix_chars)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        return snapshot

    def load_all(self) -> list[dict]:
        """Load all metric records from metrics.jsonl."""
        if not self.metrics_path.exists():
            return []
        records = []
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def compare(self) -> dict:
        """
        Compute M1–M5 comparison between baseline and fg-sync sessions.

        Returns
        -------
        dict
            Comparison table with baseline vs fg-sync averages and delta.
        """
        all_records = self.load_all()
        if not all_records:
            return {"error": "No metric records found. Run `fg-sync sync` and use the proxy first."}

        baseline = [r for r in all_records if r.get("mode") == "baseline"]
        fgsync = [r for r in all_records if r.get("mode") == "fg_sync"]

        def avg(records: list[dict], key: str) -> float:
            vals = [r[key] for r in records if key in r]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        def pct_delta(base: float, fg: float) -> str:
            if base == 0:
                return "N/A"
            delta = (fg - base) / base * 100
            sign = "+" if delta > 0 else ""
            return f"{sign}{delta:.1f}%"

        metrics = {
            "M1_clarification_rate": {
                "baseline": avg(baseline, "clarification_rate"),
                "fg_sync": avg(fgsync, "clarification_rate"),
            },
            "M2_tokens_prefix_overhead": {
                "baseline": 0,
                "fg_sync": avg(fgsync, "tokens_prefix"),
            },
            "M4_reroll_rate": {
                "baseline": avg(baseline, "reroll_rate"),
                "fg_sync": avg(fgsync, "reroll_rate"),
            },
            "M5_context_reestablishment_tokens": {
                "baseline": avg(baseline, "context_reestablishment_tokens"),
                "fg_sync": avg(fgsync, "context_reestablishment_tokens"),
            },
        }

        # Add deltas
        for key, vals in metrics.items():
            vals["delta"] = pct_delta(vals["baseline"], vals["fg_sync"])

        result = {
            "session_counts": {"baseline": len(baseline), "fg_sync": len(fgsync)},
            "metrics": metrics,
        }

        # M3: Storage footprint
        result["M3_storage"] = self.storage_report()

        return result

    def storage_report(self) -> dict:
        """
        M3: Compare raw capture.jsonl size vs ruleset.json + AssociativeMemory.
        """
        from fg_sync.pipeline import ASSOC_MEMORY_FILE

        capture_bytes = self.capture_path.stat().st_size if self.capture_path.exists() else 0
        ruleset_bytes = self.ruleset_path.stat().st_size if self.ruleset_path.exists() else 0
        assoc_bytes = ASSOC_MEMORY_FILE.stat().st_size if ASSOC_MEMORY_FILE.exists() else 0
        total_fg = ruleset_bytes + assoc_bytes

        ratio = round(capture_bytes / max(total_fg, 1), 1)

        return {
            "capture_jsonl_bytes": capture_bytes,
            "ruleset_json_bytes": ruleset_bytes,
            "assoc_memory_bytes": assoc_bytes,
            "total_fg_sync_bytes": total_fg,
            "compression_ratio": f"{ratio}:1",
            "capture_mb": round(capture_bytes / 1_048_576, 3),
            "fg_sync_kb": round(total_fg / 1024, 1),
        }

    def render_table(self) -> str:
        """Render comparison as a human-readable ASCII table."""
        data = self.compare()

        if "error" in data:
            return data["error"]

        lines = [
            "",
            "┌─────────────────────────────────────────────────────────────┐",
            "│                  fg-sync Metrics Comparison                 │",
            "├────────────────────────────┬──────────────┬─────────────────┤",
            "│ Metric                     │   Baseline   │   fg-sync       │",
            "├────────────────────────────┼──────────────┼─────────────────┤",
        ]

        labels = {
            "M1_clarification_rate":               "M1 Clarification Rate",
            "M2_tokens_prefix_overhead":           "M2 Prefix Token Overhead",
            "M4_reroll_rate":                      "M4 Re-Roll Rate",
            "M5_context_reestablishment_tokens":   "M5 Context Re-Est. Tokens",
        }

        for key, label in labels.items():
            vals = data["metrics"].get(key, {})
            b = vals.get("baseline", 0)
            f = vals.get("fg_sync", 0)
            delta = vals.get("delta", "N/A")
            lines.append(
                f"│ {label:<26} │ {str(b):<12} │ {str(f):<7} ({delta:>6}) │"
            )

        lines.append("├────────────────────────────┴──────────────┴─────────────────┤")

        # M3 storage
        m3 = data.get("M3_storage", {})
        lines.append(f"│ M3 Storage: capture={m3.get('capture_mb',0):.2f}MB  fg-sync={m3.get('fg_sync_kb',0):.1f}KB  ratio={m3.get('compression_ratio','?')} │")

        counts = data.get("session_counts", {})
        lines.append(f"│ Sessions: baseline={counts.get('baseline',0)}  fg-sync={counts.get('fg_sync',0):<30} │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        return "\n".join(lines)

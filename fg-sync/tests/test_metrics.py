"""Tests for the metrics collector — M1–M5 analysis."""
import json
import tempfile
from pathlib import Path
import pytest
from fg_sync.metrics import analyze_session, MetricsCollector

BASELINE_RECORD = {
    "ts": "2026-06-20T10:00:00+00:00",
    "session_id": "sess-001",
    "model": "llama3.2:3b",
    "messages": [
        {"role": "user", "content": "I am a developer building an ERP system. What is a journal entry?"},
        {"role": "assistant", "content": "A journal entry records a financial transaction..."},
        {"role": "user", "content": "I meant double-entry, not single entry. Try again."},
        {"role": "assistant", "content": "Double-entry accounting means..."},
    ],
    "tokens_prompt": 120,
    "tokens_completion": 80,
    "fg_injected": False,
}

FGSYNC_RECORD = {
    "ts": "2026-06-28T10:00:00+00:00",
    "session_id": "sess-002",
    "model": "llama3.2:3b",
    "messages": [
        {"role": "user", "content": "What is a journal entry?"},
        {"role": "assistant", "content": "A double-entry journal records a financial transaction..."},
    ],
    "tokens_prompt": 220,  # higher due to prefix
    "tokens_completion": 80,
    "fg_injected": True,
}


def test_analyze_session_baseline():
    result = analyze_session(BASELINE_RECORD, injected=False)
    assert result["mode"] == "baseline"
    assert result["fg_injected"] is False
    # M1: second user turn contains clarification signal
    assert result["clarification_count"] >= 1
    assert result["clarification_rate"] > 0
    # M5: first user turn contains context re-establishment
    assert result["context_reestablishment_tokens"] > 0


def test_analyze_session_fgsync():
    result = analyze_session(FGSYNC_RECORD, injected=True, prefix_chars=400)
    assert result["mode"] == "fg_sync"
    assert result["tokens_prefix"] == 100  # 400 chars / 4
    # No clarification needed
    assert result["clarification_count"] == 0
    # No context re-establishment (no "I am a developer" in first message)
    assert result["context_reestablishment_tokens"] == 0


def test_metrics_collector_record_and_compare(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    capture_path = tmp_path / "capture.jsonl"
    ruleset_path = tmp_path / "ruleset.json"

    # Write some dummy capture and ruleset files for M3
    capture_path.write_text(json.dumps(BASELINE_RECORD) + "\n" + json.dumps(FGSYNC_RECORD) + "\n")
    ruleset_path.write_text(json.dumps({"schema_version": "1.0", "behavioral_rules": []}))

    collector = MetricsCollector(
        metrics_path=metrics_path,
        capture_path=capture_path,
        ruleset_path=ruleset_path,
    )

    # Record one baseline, one fg-sync
    collector.record_session(BASELINE_RECORD, injected=False)
    collector.record_session(FGSYNC_RECORD, injected=True, prefix_chars=400)

    all_records = collector.load_all()
    assert len(all_records) == 2
    assert all_records[0]["mode"] == "baseline"
    assert all_records[1]["mode"] == "fg_sync"

    comparison = collector.compare()
    assert "metrics" in comparison
    assert "M1_clarification_rate" in comparison["metrics"]
    assert "M3_storage" in comparison


def test_render_table_no_records(tmp_path):
    collector = MetricsCollector(
        metrics_path=tmp_path / "metrics.jsonl",
        capture_path=tmp_path / "capture.jsonl",
        ruleset_path=tmp_path / "ruleset.json",
    )
    output = collector.render_table()
    assert "No metric records" in output


def test_storage_report_missing_files(tmp_path):
    collector = MetricsCollector(
        metrics_path=tmp_path / "metrics.jsonl",
        capture_path=tmp_path / "capture.jsonl",
        ruleset_path=tmp_path / "ruleset.json",
    )
    report = collector.storage_report()
    assert report["capture_jsonl_bytes"] == 0
    assert report["ruleset_json_bytes"] == 0
    assert "compression_ratio" in report

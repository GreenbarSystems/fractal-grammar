"""Tests for pipeline utilities — incremental reader, event extractor, cursor tracking."""
import json
import tempfile
from pathlib import Path
import pytest
from fg_sync.pipeline import (
    read_incremental,
    extract_events,
    _load_cursor,
    _save_cursor,
    CURSOR_FILE,
)


SAMPLE_RECORDS = [
    {
        "ts": "2026-06-28T01:00:00+00:00",
        "session_id": "abc",
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": "How do I implement double-entry accounting in PostgreSQL?"},
            {"role": "assistant", "content": "You would create a ledger table..."},
        ],
        "tokens_prompt": 50,
        "tokens_completion": 80,
        "fg_injected": False,
    },
    {
        "ts": "2026-06-28T02:00:00+00:00",
        "session_id": "def",
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": "What is HDBSCAN and how does it compare to k-means?"},
            {"role": "assistant", "content": "HDBSCAN is a density-based clustering..."},
        ],
        "tokens_prompt": 30,
        "tokens_completion": 60,
        "fg_injected": False,
    },
]


def _write_capture(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_read_incremental_from_zero(tmp_path, monkeypatch):
    capture = tmp_path / "capture.jsonl"
    _write_capture(capture, SAMPLE_RECORDS)

    # Monkeypatch cursor file location
    cursor_file = tmp_path / "cursor.json"
    monkeypatch.setattr("fg_sync.pipeline.CURSOR_FILE", cursor_file)

    records, offset = read_incremental(capture)
    assert len(records) == 2
    assert offset > 0


def test_read_incremental_second_run(tmp_path, monkeypatch):
    """Second read after cursor advance should return empty."""
    capture = tmp_path / "capture.jsonl"
    _write_capture(capture, SAMPLE_RECORDS)

    cursor_file = tmp_path / "cursor.json"
    monkeypatch.setattr("fg_sync.pipeline.CURSOR_FILE", cursor_file)

    # First read
    records1, offset1 = read_incremental(capture)
    assert len(records1) == 2

    # Simulate cursor save
    cursor_file.write_text(json.dumps({"offset": offset1}))

    # Second read — no new data
    records2, offset2 = read_incremental(capture)
    assert len(records2) == 0


def test_read_incremental_new_append(tmp_path, monkeypatch):
    """After first read, appending a new record should return only the new record."""
    capture = tmp_path / "capture.jsonl"
    _write_capture(capture, SAMPLE_RECORDS)

    cursor_file = tmp_path / "cursor.json"
    monkeypatch.setattr("fg_sync.pipeline.CURSOR_FILE", cursor_file)

    records1, offset1 = read_incremental(capture)
    cursor_file.write_text(json.dumps({"offset": offset1}))

    # Append a new record
    new_record = {
        "ts": "2026-06-28T03:00:00+00:00",
        "session_id": "ghi",
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "What is hyperdimensional computing?"}],
        "tokens_prompt": 20,
        "tokens_completion": 50,
        "fg_injected": True,
    }
    with open(capture, "a") as f:
        f.write(json.dumps(new_record) + "\n")

    records2, offset2 = read_incremental(capture)
    assert len(records2) == 1
    assert records2[0]["session_id"] == "ghi"


def test_extract_events():
    events = extract_events(SAMPLE_RECORDS)
    assert len(events) == 2
    assert "double-entry" in events[0].lower() or "postgresql" in events[0].lower()


def test_extract_events_filters_short_messages():
    records = [
        {
            "messages": [
                {"role": "user", "content": "hi"},           # too short
                {"role": "user", "content": "ok"},           # too short
                {"role": "user", "content": "What is HDC?"},  # borderline
                {"role": "assistant", "content": "HDC is..."},
            ]
        }
    ]
    events = extract_events(records)
    # Short messages filtered (len <= 10)
    assert all(len(e) > 10 for e in events)


def test_read_incremental_missing_file(tmp_path, monkeypatch):
    cursor_file = tmp_path / "cursor.json"
    monkeypatch.setattr("fg_sync.pipeline.CURSOR_FILE", cursor_file)
    records, offset = read_incremental(tmp_path / "nonexistent.jsonl")
    assert records == []
    assert offset == 0

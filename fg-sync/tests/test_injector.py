"""Tests for the Injector — ruleset loading and prompt injection."""
import json
import tempfile
from pathlib import Path
import pytest
from fg_sync.injector import Injector

SAMPLE_RULESET = {
    "schema_version": "1.0",
    "generated_at": "2026-06-28T02:00:00Z",
    "event_count": 500,
    "session_count": 12,
    "behavioral_rules": [
        {
            "id": "rule_001",
            "cluster_label": "technical_queries",
            "weight": 0.91,
            "recency_score": 0.88,
            "pattern_summary": "User asks technical questions.",
            "trigger_terms": ["code", "architecture"],
            "event_count": 80,
        }
    ],
    "prompt_prefix": "## Behavioral Context (fg-sync)\nUser is a technical founder.\n---\n",
}


def _write_ruleset(tmp_path: Path) -> Path:
    ruleset_path = tmp_path / "ruleset.json"
    ruleset_path.write_text(json.dumps(SAMPLE_RULESET), encoding="utf-8")
    return ruleset_path


def test_injector_loads_ruleset(tmp_path):
    ruleset_path = _write_ruleset(tmp_path)
    inj = Injector(ruleset_path=ruleset_path)
    assert inj.is_active()


def test_injector_no_file(tmp_path):
    inj = Injector(ruleset_path=tmp_path / "nonexistent.json")
    assert not inj.is_active()


def test_inject_api_chat_no_existing_system(tmp_path):
    ruleset_path = _write_ruleset(tmp_path)
    inj = Injector(ruleset_path=ruleset_path)

    body = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = inj.inject(body, "/api/chat")
    assert "system" in result
    assert result["system"].startswith("## Behavioral Context (fg-sync)")
    # Messages array untouched
    assert result["messages"] == body["messages"]


def test_inject_preserves_existing_system(tmp_path):
    ruleset_path = _write_ruleset(tmp_path)
    inj = Injector(ruleset_path=ruleset_path)

    body = {
        "model": "llama3.2:3b",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = inj.inject(body, "/api/chat")
    assert "## Behavioral Context (fg-sync)" in result["system"]
    assert "You are a helpful assistant." in result["system"]
    # Prefix comes before existing system
    assert result["system"].index("## Behavioral Context") < result["system"].index("You are a helpful assistant.")


def test_inject_no_ruleset_passthrough(tmp_path):
    inj = Injector(ruleset_path=tmp_path / "nonexistent.json")
    body = {"model": "llama3.2:3b", "messages": []}
    result = inj.inject(body, "/api/chat")
    assert result == body  # unchanged


def test_injector_status_active(tmp_path):
    ruleset_path = _write_ruleset(tmp_path)
    inj = Injector(ruleset_path=ruleset_path)
    s = inj.status()
    assert s["active"]
    assert s["rule_count"] == 1
    assert s["event_count"] == 500


def test_injector_status_inactive(tmp_path):
    inj = Injector(ruleset_path=tmp_path / "missing.json")
    s = inj.status()
    assert not s["active"]


def test_injector_reload(tmp_path):
    ruleset_path = tmp_path / "ruleset.json"
    inj = Injector(ruleset_path=ruleset_path)
    assert not inj.is_active()

    # Write ruleset
    _write_ruleset(tmp_path)
    inj.reload()
    assert inj.is_active()


def test_token_budget_enforcement(tmp_path):
    # Create a ruleset with a very long prefix
    ruleset = dict(SAMPLE_RULESET)
    ruleset["prompt_prefix"] = "A" * 5000  # way over any budget
    ruleset_path = tmp_path / "ruleset.json"
    ruleset_path.write_text(json.dumps(ruleset))

    inj = Injector(ruleset_path=ruleset_path, max_prompt_tokens=100)
    # Should be active but truncated
    assert inj.is_active()
    body = inj.inject({"messages": []}, "/api/chat")
    assert len(body["system"]) <= 100 * 4 + 10  # budget + small suffix

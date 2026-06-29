"""Tests for config loader."""
import tempfile
from pathlib import Path
import pytest
from fg_sync.config import load_config, FgSyncConfig


SAMPLE_TOML = """
[proxy]
listen_port = 11436
ollama_port = 11434

[pipeline]
schedule = "0 3 * * *"
min_events_to_run = 25
hdc_dimensions = 10000
hdc_seed = 4277009102
use_hdc = true

[ruleset]
max_rules = 15
max_prompt_tokens = 300

[source]
type = "proxy"
"""


def test_load_defaults():
    cfg = load_config(path=None)  # no file — pure defaults
    assert cfg.proxy.listen_port == 11435
    assert cfg.proxy.ollama_port == 11434
    assert cfg.pipeline.hdc_dimensions == 10000
    assert cfg.pipeline.hdc_seed == 0xFEEDBEEF
    assert cfg.pipeline.assoc_memory_threshold == 0.05
    assert cfg.ruleset.max_prompt_tokens == 400
    assert cfg.source.type == "proxy"


def test_load_from_toml():
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(SAMPLE_TOML)
        tmp_path = f.name

    cfg = load_config(path=tmp_path)
    assert cfg.proxy.listen_port == 11436
    assert cfg.pipeline.schedule == "0 3 * * *"
    assert cfg.pipeline.min_events_to_run == 25
    assert cfg.ruleset.max_rules == 15
    assert cfg.ruleset.max_prompt_tokens == 300


def test_ensure_dirs(tmp_path):
    cfg = load_config()
    cfg.proxy.capture_path = tmp_path / "capture.jsonl"
    cfg.ruleset.output_path = tmp_path / "ruleset.json"
    cfg.metrics.metrics_path = tmp_path / "metrics.jsonl"
    cfg.ensure_dirs()
    assert (tmp_path).exists()

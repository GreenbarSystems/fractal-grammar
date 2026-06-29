"""
fg_sync/config.py
-----------------
Configuration loader for fg-sync.toml.
Uses Python 3.11 stdlib tomllib; falls back to tomli for 3.10.
All paths are expanded (~ resolved) on load.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError("Install tomli for Python < 3.11: pip install tomli") from e

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
FG_SYNC_HOME = Path.home() / ".fg-sync"
DEFAULT_CONFIG_PATH = Path.home() / ".fg-sync" / "fg-sync.toml"
FALLBACK_CONFIG_PATH = Path("fg-sync.toml")


# ---------------------------------------------------------------------------
# Dataclasses — one per TOML section
# ---------------------------------------------------------------------------

@dataclass
class ProxyConfig:
    listen_port: int = 11435
    ollama_port: int = 11434
    ollama_host: str = "127.0.0.1"
    capture_path: Path = FG_SYNC_HOME / "capture.jsonl"

    def __post_init__(self):
        self.capture_path = Path(self.capture_path).expanduser()


@dataclass
class PipelineConfig:
    # Cron expression (UTC)
    schedule: str = "0 2 * * *"
    # Minimum captures before pipeline runs
    min_events_to_run: int = 50
    # MinHash novelty gate — cosine dissimilarity threshold
    novelty_threshold: float = 0.92
    # HDC settings (must match fractal-grammar library)
    hdc_dimensions: int = 10_000
    hdc_seed: int = 0xFEEDBEEF
    # HDBSCAN
    min_cluster_size: int = 5
    # AssociativeMemory retrieval threshold
    assoc_memory_threshold: float = 0.05
    # Whether to use HDC backend (vs HashProjection fallback)
    use_hdc: bool = True


@dataclass
class RulesetConfig:
    output_path: Path = FG_SYNC_HOME / "ruleset.json"
    # Hard cap on number of injected behavioral rules
    max_rules: int = 20
    # Token budget for the generated system prompt prefix
    max_prompt_tokens: int = 400
    # Weight multiplier for recent patterns
    recency_weight: float = 0.7
    # Days after which a pattern starts decaying in weight
    decay_days: int = 30

    def __post_init__(self):
        self.output_path = Path(self.output_path).expanduser()


@dataclass
class MetricsConfig:
    enabled: bool = True
    metrics_path: Path = FG_SYNC_HOME / "metrics.jsonl"
    # Sessions to use as pre-fg-sync baseline
    baseline_session_count: int = 10

    def __post_init__(self):
        self.metrics_path = Path(self.metrics_path).expanduser()


@dataclass
class SourceConfig:
    # "proxy" (default) | "openwebui"
    type: str = "proxy"
    # Only used when type = "openwebui"
    openwebui_db: Path = Path.home() / ".local/share/open-webui/webui.db"

    def __post_init__(self):
        self.openwebui_db = Path(self.openwebui_db).expanduser()


@dataclass
class FgSyncConfig:
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ruleset: RulesetConfig = field(default_factory=RulesetConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    source: SourceConfig = field(default_factory=SourceConfig)

    # Path this config was loaded from (not in TOML)
    _config_path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False, compare=False)

    def ensure_dirs(self):
        """Create ~/.fg-sync and any parent dirs needed."""
        FG_SYNC_HOME.mkdir(parents=True, exist_ok=True)
        self.proxy.capture_path.parent.mkdir(parents=True, exist_ok=True)
        self.ruleset.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        (FG_SYNC_HOME / "logs").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _expand(value):
    """Recursively expand ~ in string values."""
    if isinstance(value, str):
        return os.path.expanduser(value)
    return value


def load_config(path: Path | str | None = None) -> FgSyncConfig:
    """
    Load fg-sync.toml from:
      1. Explicit path argument
      2. ~/.fg-sync/fg-sync.toml
      3. ./fg-sync.toml (CWD fallback)
      4. Built-in defaults (no file needed)
    """
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    candidates += [DEFAULT_CONFIG_PATH, FALLBACK_CONFIG_PATH]

    raw: dict = {}
    config_path = DEFAULT_CONFIG_PATH

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            config_path = candidate
            break

    def section(key: str) -> dict:
        return raw.get(key, {})

    def _path(d: dict, key: str, default: Path) -> Path:
        return Path(d.get(key, default)).expanduser()

    # Proxy
    p = section("proxy")
    proxy = ProxyConfig(
        listen_port=p.get("listen_port", 11435),
        ollama_port=p.get("ollama_port", 11434),
        ollama_host=p.get("ollama_host", "127.0.0.1"),
        capture_path=_path(p, "capture_path", FG_SYNC_HOME / "capture.jsonl"),
    )

    # Pipeline
    pl = section("pipeline")
    pipeline = PipelineConfig(
        schedule=pl.get("schedule", "0 2 * * *"),
        min_events_to_run=pl.get("min_events_to_run", 50),
        novelty_threshold=pl.get("novelty_threshold", 0.92),
        hdc_dimensions=pl.get("hdc_dimensions", 10_000),
        hdc_seed=pl.get("hdc_seed", 0xFEEDBEEF),
        min_cluster_size=pl.get("min_cluster_size", 5),
        assoc_memory_threshold=pl.get("assoc_memory_threshold", 0.05),
        use_hdc=pl.get("use_hdc", True),
    )

    # Ruleset
    rs = section("ruleset")
    ruleset = RulesetConfig(
        output_path=_path(rs, "output_path", FG_SYNC_HOME / "ruleset.json"),
        max_rules=rs.get("max_rules", 20),
        max_prompt_tokens=rs.get("max_prompt_tokens", 400),
        recency_weight=rs.get("recency_weight", 0.7),
        decay_days=rs.get("decay_days", 30),
    )

    # Metrics
    m = section("metrics")
    metrics = MetricsConfig(
        enabled=m.get("enabled", True),
        metrics_path=_path(m, "metrics_path", FG_SYNC_HOME / "metrics.jsonl"),
        baseline_session_count=m.get("baseline_session_count", 10),
    )

    # Source
    s = section("source")
    source = SourceConfig(
        type=s.get("type", "proxy"),
        openwebui_db=_path(
            s, "openwebui_db",
            Path.home() / ".local/share/open-webui/webui.db"
        ),
    )

    cfg = FgSyncConfig(
        proxy=proxy,
        pipeline=pipeline,
        ruleset=ruleset,
        metrics=metrics,
        source=source,
        _config_path=config_path,
    )
    return cfg

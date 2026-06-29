"""
fg_sync/daemon.py
-----------------
fg-sync daemon: cron scheduler + pipeline orchestrator.

Runs the fractal-grammar extraction pipeline on a configurable schedule
(default: 2am UTC nightly, defined in fg-sync.toml [pipeline].schedule).

Lifecycle:
  1. Load config
  2. Start APScheduler with configured cron
  3. On each tick: read new captures, run pipeline, write ruleset.json
  4. Signal injector to hot-reload via SIGHUP (or direct method call)
  5. Block until KeyboardInterrupt / SIGTERM

Can also be invoked in one-shot mode (run_once=True) for:
  - `fg-sync sync` command
  - systemd timer / launchd one-shot execution
  - Testing
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from fg_sync.config import FgSyncConfig, load_config
from fg_sync.injector import Injector
from fg_sync.pipeline import run_pipeline
from fg_sync.metrics import MetricsCollector

logger = logging.getLogger("fg_sync.daemon")


class FgSyncDaemon:
    """
    Orchestrates the fg-sync pipeline on a cron schedule.

    Parameters
    ----------
    config : FgSyncConfig
    injector : Injector
        The injector instance shared with the proxy (hot-reload on pipeline completion).
    """

    def __init__(self, config: FgSyncConfig, injector: Injector):
        self.config = config
        self.injector = injector
        self._scheduler = BlockingScheduler(timezone="UTC")
        self._metrics = MetricsCollector(
            metrics_path=config.metrics.metrics_path,
            capture_path=config.proxy.capture_path,
            ruleset_path=config.ruleset.output_path,
        )

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline_once(self, dry_run: bool = False) -> bool:
        """
        Execute the fractal-grammar pipeline synchronously.

        Returns
        -------
        bool
            True if pipeline ran and produced output, False if skipped.
        """
        logger.info("Starting pipeline run (dry_run=%s)", dry_run)

        # Determine capture source
        capture_path = self._resolve_capture_path()

        ruleset = run_pipeline(
            capture_path=capture_path,
            pipeline_cfg=self.config.pipeline,
            ruleset_cfg=self.config.ruleset,
            dry_run=dry_run,
        )

        if ruleset is None:
            logger.info("Pipeline run complete — no output generated")
            return False

        # Hot-reload the injector
        if not dry_run:
            reloaded = self.injector.reload()
            if reloaded:
                logger.info("Injector reloaded with new ruleset")
            else:
                logger.warning("Injector reload failed — ruleset.json may be invalid")

        return True

    def _resolve_capture_path(self) -> Path:
        """Return the appropriate capture data path based on source config."""
        source_type = self.config.source.type

        if source_type == "openwebui":
            # For Open WebUI source, first export to a temp JSONL in the fg-sync home
            return self._export_openwebui_to_jsonl()
        else:
            # Default: proxy capture.jsonl
            return self.config.proxy.capture_path

    def _export_openwebui_to_jsonl(self) -> Path:
        """
        Read Open WebUI SQLite and append new records to capture.jsonl.
        Returns capture.jsonl path for the pipeline.
        """
        from fg_sync.sources.openwebui import OpenWebUISource
        from fg_sync.pipeline import _load_cursor, _save_cursor
        import aiofiles
        import json as _json
        from pathlib import Path as _Path

        capture_path = self.config.proxy.capture_path
        cursor = _load_cursor()
        since_ts = cursor.get("openwebui_updated_at")

        source = OpenWebUISource(
            db_path=self.config.source.openwebui_db,
            since_ts=since_ts,
        )

        new_records = list(source.read())
        if not new_records:
            logger.info("OpenWebUI source: no new records since last sync")
            return capture_path

        # Append to capture.jsonl synchronously
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(capture_path, "a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")

        # Update cursor with latest updated_at from Open WebUI
        latest_ts = source.latest_updated_at()
        if latest_ts:
            raw_cursor = {"offset": cursor.get("offset", 0), "openwebui_updated_at": latest_ts}
            from fg_sync.pipeline import CURSOR_FILE
            CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
            CURSOR_FILE.write_text(_json.dumps(raw_cursor))

        logger.info("OpenWebUI: appended %d records to capture.jsonl", len(new_records))
        return capture_path

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def start(self):
        """Start the blocking scheduler. Blocks until SIGTERM/SIGINT."""
        schedule = self.config.pipeline.schedule

        # Parse cron expression (5-field: min hour dom month dow)
        try:
            parts = schedule.strip().split()
            if len(parts) != 5:
                raise ValueError(f"Expected 5-field cron, got {len(parts)} fields: {schedule!r}")
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
                timezone="UTC",
            )
        except Exception as e:
            logger.error("Invalid cron schedule %r: %s — falling back to nightly 2am UTC", schedule, e)
            trigger = CronTrigger(hour=2, minute=0, timezone="UTC")

        self._scheduler.add_job(
            self.run_pipeline_once,
            trigger=trigger,
            id="fg_sync_pipeline",
            name="Fractal Grammar Pipeline",
            max_instances=1,  # prevent overlap if pipeline is slow
            coalesce=True,    # if missed, run once immediately on next opportunity
        )

        logger.info("Daemon scheduled: %r (UTC). Next run at startup if past due.", schedule)

        # Register shutdown handler
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("fg-sync daemon stopping")
        finally:
            self._scheduler.shutdown(wait=False)

    def _handle_sigterm(self, signum, frame):
        logger.info("SIGTERM received — shutting down scheduler")
        self._scheduler.shutdown(wait=False)
        sys.exit(0)


# ---------------------------------------------------------------------------
# Standalone entry point (used by `fg-sync sync` one-shot mode)
# ---------------------------------------------------------------------------

def run_once(config_path: str | None = None, dry_run: bool = False) -> bool:
    """
    One-shot pipeline execution — no scheduler.

    Parameters
    ----------
    config_path : str | None
        Optional path to fg-sync.toml.
    dry_run : bool
        If True, do not write ruleset or advance cursor.

    Returns
    -------
    bool
        True if pipeline produced output.
    """
    cfg = load_config(config_path)
    cfg.ensure_dirs()

    injector = Injector(
        ruleset_path=cfg.ruleset.output_path,
        max_prompt_tokens=cfg.ruleset.max_prompt_tokens,
    )

    daemon = FgSyncDaemon(config=cfg, injector=injector)
    return daemon.run_pipeline_once(dry_run=dry_run)

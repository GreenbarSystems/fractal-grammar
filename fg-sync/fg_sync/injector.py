"""
fg_sync/injector.py
-------------------
Reads ruleset.json and injects the prompt_prefix into Ollama API requests.

Injection targets:
  POST /api/chat  → request_body["system"] field (prepended)
  POST /api/generate → request_body["system"] field (prepended)

The injector is reload-safe: call reload() or send SIGHUP to the proxy process
to pick up a freshly written ruleset.json without restarting.

The user's original system prompt (if any) is preserved — prefix is prepended,
not replaced. The messages[] array is never modified.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from pathlib import Path

logger = logging.getLogger("fg_sync.injector")


class Injector:
    """
    Thread-safe ruleset loader and system prompt injector.

    Parameters
    ----------
    ruleset_path : Path
        Path to ruleset.json written by the pipeline.
    max_prompt_tokens : int
        Hard cap on prefix token budget (enforced at write time in pipeline;
        this is a sanity cap on read).
    """

    def __init__(self, ruleset_path: Path, max_prompt_tokens: int = 400):
        self.ruleset_path = Path(ruleset_path).expanduser()
        self.max_prompt_tokens = max_prompt_tokens
        self._lock = threading.RLock()
        self._ruleset: dict | None = None
        self._prompt_prefix: str | None = None
        self._active: bool = False

        # Load on init if file exists
        self._try_load()

        # Register SIGHUP for hot reload (Unix only)
        try:
            signal.signal(signal.SIGHUP, self._handle_sighup)
            logger.debug("SIGHUP handler registered for ruleset hot-reload")
        except (AttributeError, OSError):
            # Windows or restricted environment
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, request_body: dict, endpoint: str) -> dict:
        """
        Inject the behavioral system prompt prefix into an Ollama request body.

        Parameters
        ----------
        request_body : dict
            Parsed JSON body from POST /api/chat or /api/generate.
        endpoint : str
            The request path ("/api/chat" or "/api/generate").

        Returns
        -------
        dict
            Modified request body (copy — original not mutated).
        """
        with self._lock:
            if not self._active or not self._prompt_prefix:
                return request_body  # no ruleset — passthrough unchanged

        # Work on a shallow copy to avoid mutating caller's dict
        body = dict(request_body)
        prefix = self._prompt_prefix

        existing_system = body.get("system", "")

        if existing_system:
            # Prepend, preserve user's system prompt
            body["system"] = prefix + "\n" + existing_system
        else:
            body["system"] = prefix

        logger.debug(
            "Injected %d-char prefix into %s (existing system: %s)",
            len(prefix), endpoint, "yes" if existing_system else "no"
        )
        return body

    def reload(self) -> bool:
        """
        Reload ruleset.json from disk.

        Returns
        -------
        bool
            True if successfully loaded, False if file missing or invalid.
        """
        return self._try_load()

    def is_active(self) -> bool:
        """Return True if a valid ruleset is loaded and injection is active."""
        with self._lock:
            return self._active

    def status(self) -> dict:
        """Return a status dict for `fg-sync status` command."""
        with self._lock:
            if not self._active or not self._ruleset:
                return {
                    "active": False,
                    "ruleset_path": str(self.ruleset_path),
                    "message": "No ruleset loaded — run `fg-sync sync` to generate one.",
                }
            rules = self._ruleset.get("behavioral_rules", [])
            return {
                "active": True,
                "ruleset_path": str(self.ruleset_path),
                "generated_at": self._ruleset.get("generated_at", "unknown"),
                "event_count": self._ruleset.get("event_count", 0),
                "session_count": self._ruleset.get("session_count", 0),
                "rule_count": len(rules),
                "top_rules": [
                    {"label": r["cluster_label"], "weight": r["weight"]}
                    for r in rules[:5]
                ],
                "prompt_prefix_chars": len(self._prompt_prefix or ""),
            }

    def get_ruleset(self) -> dict | None:
        """Return the raw ruleset dict."""
        with self._lock:
            return self._ruleset

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_load(self) -> bool:
        """Attempt to load ruleset.json. Thread-safe."""
        if not self.ruleset_path.exists():
            with self._lock:
                self._active = False
                self._ruleset = None
                self._prompt_prefix = None
            logger.debug("Ruleset file not found at %s", self.ruleset_path)
            return False

        try:
            raw = self.ruleset_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load ruleset: %s", e)
            return False

        # Validate schema version
        if data.get("schema_version") != "1.0":
            logger.warning(
                "Ruleset schema_version=%r — expected '1.0'. Proceeding anyway.",
                data.get("schema_version")
            )

        prefix = data.get("prompt_prefix", "")
        if not prefix:
            logger.warning("Ruleset loaded but prompt_prefix is empty — injection will be a no-op")

        # Enforce token budget (4 chars ≈ 1 token)
        char_limit = self.max_prompt_tokens * 4
        if len(prefix) > char_limit:
            logger.warning(
                "prompt_prefix (%d chars) exceeds budget (%d chars) — truncating",
                len(prefix), char_limit
            )
            prefix = prefix[:char_limit] + "\n---\n"

        with self._lock:
            self._ruleset = data
            self._prompt_prefix = prefix
            self._active = bool(prefix)

        logger.info(
            "Ruleset loaded: %d rules, %d events, prefix=%d chars",
            len(data.get("behavioral_rules", [])),
            data.get("event_count", 0),
            len(prefix),
        )
        return True

    def _handle_sighup(self, signum, frame):
        """SIGHUP handler — hot reload ruleset from disk."""
        logger.info("SIGHUP received — reloading ruleset")
        self.reload()

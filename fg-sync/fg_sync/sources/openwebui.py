"""
fg_sync/sources/openwebui.py
-----------------------------
Path B: Read conversation history directly from Open WebUI's SQLite database.

Use when the user already runs Open WebUI and does NOT want to change their
client port from 11434 to 11435.

Open WebUI stores chats in:
  ~/.local/share/open-webui/webui.db   (Linux/macOS)
  %APPDATA%\open-webui\webui.db        (Windows)

Table: chat
  id TEXT PRIMARY KEY
  user_id TEXT
  title TEXT
  chat JSON  -- contains {"messages": [...]} in Open WebUI format
  created_at INTEGER (unix timestamp)
  updated_at INTEGER (unix timestamp)

Usage:
  from fg_sync.sources.openwebui import OpenWebUISource
  source = OpenWebUISource(db_path, since_ts=cursor_ts)
  for record in source.read():
      # record is a standard fg-sync capture dict
      ...
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("fg_sync.sources.openwebui")


class OpenWebUISource:
    """
    Read conversation records from Open WebUI's SQLite database.

    Parameters
    ----------
    db_path : Path
        Path to webui.db
    since_ts : float | None
        Unix timestamp — only return chats updated after this time.
        Pass None to return all chats.
    """

    def __init__(self, db_path: Path, since_ts: float | None = None):
        self.db_path = Path(db_path).expanduser()
        self.since_ts = since_ts

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Open WebUI database not found at {self.db_path}\n"
                "If you are not using Open WebUI, run fg-sync with --source proxy instead."
            )

    def read(self) -> Iterator[dict]:
        """
        Yield fg-sync capture-compatible dicts from webui.db.

        Yields
        ------
        dict with keys: ts, session_id, model, messages, fg_injected
        """
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        try:
            if self.since_ts is not None:
                rows = conn.execute(
                    "SELECT id, chat, updated_at FROM chat WHERE updated_at > ? ORDER BY updated_at ASC",
                    (int(self.since_ts),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, chat, updated_at FROM chat ORDER BY updated_at ASC"
                ).fetchall()

            logger.info("OpenWebUISource: found %d chats to process", len(rows))

            for row in rows:
                record = self._parse_row(row)
                if record:
                    yield record

        finally:
            conn.close()

    def _parse_row(self, row: sqlite3.Row) -> dict | None:
        """Parse a single webui.db chat row into fg-sync capture format."""
        try:
            chat_data = json.loads(row["chat"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping chat %s — invalid JSON", row["id"])
            return None

        raw_messages = chat_data.get("messages", [])
        if not raw_messages:
            return None

        # Normalize Open WebUI message format → fg-sync format
        messages = []
        for msg in raw_messages:
            role = msg.get("role", "")
            # Open WebUI stores content as string or list of content blocks
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": str(content)})

        if len(messages) < 2:
            return None

        # Extract model from chat metadata
        model = chat_data.get("models", ["unknown"])[0] if chat_data.get("models") else "unknown"
        if isinstance(model, dict):
            model = model.get("id", "unknown")

        ts = datetime.fromtimestamp(row["updated_at"], tz=timezone.utc).isoformat()

        return {
            "ts": ts,
            "session_id": str(row["id"]),
            "model": model,
            "endpoint": "/api/chat",
            "messages": messages,
            "tokens_prompt": 0,       # not stored by Open WebUI
            "tokens_completion": 0,   # not stored by Open WebUI
            "duration_ms": 0,
            "fg_injected": False,
            "source": "openwebui",
        }

    def latest_updated_at(self) -> float | None:
        """Return the highest updated_at timestamp in the database."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT MAX(updated_at) as max_ts FROM chat").fetchone()
            return float(row[0]) if row and row[0] else None
        finally:
            conn.close()

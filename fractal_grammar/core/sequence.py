"""
core/sequence.py

Universal sequence interface. Accepts any timestamped behavioral input —
LLM interaction logs, event streams, click sequences, or arbitrary JSON.
All inputs normalize to a BehavioralEvent and a BehavioralSequence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class InputType(Enum):
    LLM_INTERACTION = "llm_interaction"   # {role, content} pairs
    EVENT_STREAM    = "event_stream"       # {event_type, payload} dicts
    RAW_TEXT        = "raw_text"           # plain string
    CUSTOM          = "custom"             # caller-supplied embedding


@dataclass
class BehavioralEvent:
    """
    Atomic unit of behavior. Everything normalizes here before
    entering the embedding layer.

    Attributes
    ----------
    content     : the primary text or label for this event
    metadata    : arbitrary key-value context (role, event_type, etc.)
    timestamp   : unix float; defaults to now
    source_type : which InputType produced this event
    raw         : original input preserved for traceability
    """
    content     : str
    metadata    : Dict[str, Any]   = field(default_factory=dict)
    timestamp   : float            = field(default_factory=time.time)
    source_type : InputType        = InputType.RAW_TEXT
    raw         : Any              = field(default=None, repr=False)

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content"     : self.content,
            "metadata"    : self.metadata,
            "timestamp"   : self.timestamp,
            "source_type" : self.source_type.value,
            "fingerprint" : self.fingerprint,
        }


@dataclass
class BehavioralSequence:
    """
    An ordered collection of BehavioralEvents representing one session
    or logical unit of behavior (a conversation, a task, a workflow).

    Attributes
    ----------
    events      : ordered list of events
    session_id  : caller-supplied identifier
    metadata    : session-level context
    """
    events     : List[BehavioralEvent] = field(default_factory=list)
    session_id : str                   = field(default_factory=lambda: str(time.time()))
    metadata   : Dict[str, Any]        = field(default_factory=dict)

    def append(self, event: BehavioralEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id" : self.session_id,
            "metadata"   : self.metadata,
            "events"     : [e.to_dict() for e in self.events],
        }


# ---------------------------------------------------------------------------
# Ingestion helpers — convert common formats to BehavioralSequence
# ---------------------------------------------------------------------------

def from_llm_log(
    messages  : List[Dict[str, str]],
    session_id: Optional[str] = None,
    metadata  : Optional[Dict[str, Any]] = None,
) -> BehavioralSequence:
    """
    Ingest an OpenAI-style message list.

    Example input:
        [{"role": "user", "content": "How do I reverse a list?"},
         {"role": "assistant", "content": "Use list[::-1] or reversed()."}]
    """
    seq = BehavioralSequence(
        session_id=session_id or str(time.time()),
        metadata=metadata or {},
    )
    for msg in messages:
        role    = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not content.strip():
            continue
        event = BehavioralEvent(
            content=content,
            metadata={"role": role},
            source_type=InputType.LLM_INTERACTION,
            raw=msg,
        )
        seq.append(event)
    return seq


def from_event_stream(
    events    : List[Dict[str, Any]],
    content_key: str = "label",
    session_id: Optional[str] = None,
    metadata  : Optional[Dict[str, Any]] = None,
) -> BehavioralSequence:
    """
    Ingest a list of event dicts.

    Example input:
        [{"event_type": "click", "label": "Submit button", "ts": 1700000000},
         {"event_type": "nav",   "label": "Dashboard",     "ts": 1700000005}]
    """
    seq = BehavioralSequence(
        session_id=session_id or str(time.time()),
        metadata=metadata or {},
    )
    for ev in events:
        content = str(ev.get(content_key, ev.get("event_type", str(ev))))
        ts      = float(ev.get("ts", ev.get("timestamp", time.time())))
        event   = BehavioralEvent(
            content=content,
            metadata={k: v for k, v in ev.items() if k not in (content_key, "ts", "timestamp")},
            timestamp=ts,
            source_type=InputType.EVENT_STREAM,
            raw=ev,
        )
        seq.append(event)
    return seq


def from_raw_texts(
    texts     : List[str],
    session_id: Optional[str] = None,
    metadata  : Optional[Dict[str, Any]] = None,
) -> BehavioralSequence:
    """Ingest a list of plain strings as sequential events."""
    seq = BehavioralSequence(
        session_id=session_id or str(time.time()),
        metadata=metadata or {},
    )
    for text in texts:
        if text.strip():
            seq.append(BehavioralEvent(
                content=text,
                source_type=InputType.RAW_TEXT,
                raw=text,
            ))
    return seq


def from_jsonl(path: str, mode: str = "auto") -> List[BehavioralSequence]:
    """
    Load a JSONL file where each line is either:
      - An LLM log:    {"session_id": "...", "messages": [...]}
      - An event list: {"session_id": "...", "events": [...]}
      - A raw text:    {"session_id": "...", "text": "..."}

    mode = "auto" detects format per line.
    """
    sequences: List[BehavioralSequence] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("session_id")
            meta = obj.get("metadata", {})

            if "messages" in obj:
                sequences.append(from_llm_log(obj["messages"], sid, meta))
            elif "events" in obj:
                sequences.append(from_event_stream(obj["events"], session_id=sid, metadata=meta))
            elif "text" in obj:
                texts = obj["text"] if isinstance(obj["text"], list) else [obj["text"]]
                sequences.append(from_raw_texts(texts, sid, meta))
    return sequences

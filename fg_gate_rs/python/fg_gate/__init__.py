"""
fg_gate — FBG v0.3.0 binary encoding Python package
====================================================
Drop-in replacement for the Python reference encoder (fg_binary_encoding/encoder.py).

Usage::

    from fg_gate import encode_rules, decode_rules, rebuild_gate, promote_gate

    # Encode Layer 0 rules
    rules = [
        {"lhs": "context", "rhs": "tone", "support": 10, "confidence": 0.91,
         "layer": 0, "session_count": 8, "ttl": None},
    ]
    blob = encode_rules(rules, layer_id=0)  # → bytes

    # Decode
    decoded_rules, layer_id = decode_rules(blob)

    # Rebuild ephemeral gate from session event log
    events = [(0, "context", "tone"), (1, "context", "tone"), (2, "direct", "brief")]
    gate_entries = rebuild_gate(events)    # list of dicts, gate NOT persisted

    # Full promotion pipeline (gate + layer classification)
    result = promote_gate(events, l0_threshold=5, l1_threshold=2)
    # result.keys() = ["layer0", "layer1", "layer2", "gate", "stats"]

Layer IDs::

    LAYER_DOMAIN     = 0   (Domain — rules in >= 5 sessions)
    LAYER_BEHAVIORAL = 1   (Behavioral — rules in 2-4 sessions)
    LAYER_SESSION    = 2   (Session — last-session only, TTL=1)
"""

from fg_gate._fg_gate import (   # noqa: F401
    encode_rules,
    decode_rules,
    rebuild_gate,
    promote_gate,
    fg_version,
    LAYER_DOMAIN,
    LAYER_BEHAVIORAL,
    LAYER_SESSION,
    MAGIC,
    VERSION,
)

__version__ = "0.3.0"
__all__ = [
    "encode_rules",
    "decode_rules",
    "rebuild_gate",
    "promote_gate",
    "fg_version",
    "LAYER_DOMAIN",
    "LAYER_BEHAVIORAL",
    "LAYER_SESSION",
    "MAGIC",
    "VERSION",
]

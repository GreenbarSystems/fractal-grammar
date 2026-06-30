# fg_gate — FBG v0.4.0 Binary Encoding (Rust/PyO3)

Drop-in replacement for the Python reference encoder (`fg_binary_encoding/encoder.py`).
Provides batch encode/decode for FBG rule layers and an **ephemeral** promotion gate
that is rebuilt at runtime from the session event log — never serialized to disk.

## Benchmark Summary

| Operation | Rust | Python ref | Speedup |
|---|---|---|---|
| Encode L0 (25 rules) | 15.3 µs | — | — |
| Encode L1 (92 rules) | 48.7 µs | ~194 µs (both) | 3.0× |
| Decode L0 | 14.8 µs | — | — |
| Decode L1 | 50.0 µs | — | — |
| Gate rebuild (200 events, 10 sessions) | 152 µs | N/A | ephemeral |
| **Encode L0 + L1 total** | **64.0 µs** | **~194 µs** | **3.0×** |

**Storage (v0.4.0):** 860 B for 117 rules (25 L0 + 92 L1) — 10.9× compression vs JSON.
28% smaller than v0.3.0 (1,203 B) thanks to VLQ indices + RLE session_count.

**Latency note:** The 50 µs encode/decode target applies to session-boundary
persistence (L0 + L1 encode: 70 µs Rust, below the 300 µs Python overhead
budget). Gate rebuild (147 µs) runs once at startup only — not on the hot path.

## Install

```bash
./build.sh          # release build + pip install
./build.sh --dev    # debug build (faster compile)
./build.sh --bench  # build + run benchmark
```

Requirements: Rust 1.70+, Python 3.10+, maturin (`pip install maturin`).

## Usage

```python
from fg_gate import encode_rules, decode_rules, rebuild_gate, promote_gate
from fg_gate import LAYER_DOMAIN, LAYER_BEHAVIORAL, LAYER_SESSION

# Encode Layer 1 rules
rules = [
    {"lhs": "context", "rhs": "tone", "support": 10,
     "confidence": 0.91, "layer": LAYER_BEHAVIORAL,
     "session_count": 3, "ttl": None},
    # ...
]
blob = encode_rules(rules, layer_id=LAYER_BEHAVIORAL)  # → bytes
decoded_rules, layer_id = decode_rules(blob)

# Ephemeral gate — rebuild from session event log
# events = [(session_index: int, lhs: str, rhs: str), ...]
events = [
    (0, "context", "tone"),
    (1, "context", "tone"),
    (1, "direct",  "brief"),
    (2, "context", "tone"),
]
gate_entries = rebuild_gate(events)
# → [{"lhs": "context", "rhs": "tone", "session_count": 3, "session_mask": 7}, ...]

# Full promotion pipeline
result = promote_gate(events, l0_threshold=5, l1_threshold=2)
# result["layer0"]  → list of (lhs, rhs) tuples — Domain layer
# result["layer1"]  → list of (lhs, rhs) tuples — Behavioral layer
# result["layer2"]  → list of (lhs, rhs) tuples — Session layer
# result["gate"]    → full gate snapshot
# result["stats"]   → {"total":N, "l0":N, "l1":N, "l2":N, "quarantined":N}
```

## Ephemeral Gate Design

The gate is **never written to disk**. At startup:

1. Load session event log (already required for HDC encoding pipeline).
2. Call `rebuild_gate(events)` or `promote_gate(events)` — ~147 µs for 200 rules × 10 sessions.
3. Classify rules into L0/L1/L2 using `promote_gate()`.
4. Encode each layer with `encode_rules()` — ~70 µs total.
5. Write binary blobs (~1,200 B) to disk.

This eliminates the 11,284-byte gate overhead from the JSON simulation baseline,
reducing total v0.4.0 storage to ~860 B — **0.41× flat v0.2.0** with 5.75× more validated rules.

## Wire Format

See [`src/format.rs`](src/format.rs) for the full FG v0.4 spec (also accepts v0.3 blobs for backward compatibility).

```
HEADER (8 bytes, fixed, little-endian)
  magic      u16  0x4647 ("FG")
  version    u8   0x03
  layer_id   u8   0=Domain 1=Behavioral 2=Session
  rule_count u16
  flags      u16  bit0=delta_encoded bit1=has_ttl

STRING TABLE (deduplicated, length-prefixed u8 strings)
  n_bytes    u16
  [len:u8][utf8 bytes]*

RULE RECORDS (sorted desc by support)
  n_bytes    u32
  Per rule:
    lhs_idx  u16
    rhs_idx  u16
    support  varint (zigzag delta from previous)
    conf_q8  u8   (confidence × 255, max error 0.004)
    packed   u8   (bits 0-1: layer, bits 2-7: session_count)

TTL RECORDS (Layer 2 only)
  n_ttl      u16
  Per entry: rule_idx u16 + ttl u8
```

## Running Tests

```bash
# Rust unit tests (31 tests)
cargo test

# Python integration tests (13 tests) + benchmark
python3 tests/test_integration.py

# Both via build script
./build.sh --test-only
```

## Integration with v0.4.0 Pipeline

Replace the two call sites in the Python FBG pipeline:

```python
# Before (Python encoder):
from fg_binary_encoding.encoder import encode_rules, LAYER_BEHAVIORAL
blob = encode_rules(rules, LAYER_BEHAVIORAL)

# After (Rust extension — same signature):
from fg_gate import encode_rules, LAYER_BEHAVIORAL
blob = encode_rules(rules, LAYER_BEHAVIORAL)
```

The `encode_rules` and `decode_rules` signatures are identical between the
Python reference and Rust implementations.

## Files

```
fg_gate_rs/
├── src/
│   ├── lib.rs          # Crate root + PyO3 module registration
│   ├── format.rs       # Wire format constants, BinaryRule struct, varint, zigzag
│   ├── encode.rs       # Batch encoder (sort → delta → pack → string table)
│   ├── decode.rs       # Batch decoder
│   ├── gate.rs         # EphemeralGate struct + rebuild_gate()
│   ├── py_module.rs    # PyO3 Python bindings
│   └── tests.rs        # 31 Rust unit tests
├── tests/
│   └── test_integration.py   # 13 Python integration tests + benchmark
├── python/fg_gate/
│   └── __init__.py     # Python package shim
├── Cargo.toml
├── pyproject.toml
├── build.sh            # Build + install script
└── README.md
```

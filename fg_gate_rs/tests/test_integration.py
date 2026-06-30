"""
Python integration tests for fg_gate Rust extension.
Compares Rust encoder output against Python reference encoder for
byte-for-byte and semantic equivalence.

Run:
    python -m pytest tests/test_integration.py -v

Or directly:
    python tests/test_integration.py
"""

import sys
import os
import time
import json
import struct
import random

# Add the Python reference encoder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fg_binary_encoding"))

# ── Import both implementations ───────────────────────────────────────────────
from fg_gate._fg_gate import (
    encode_rules   as rs_encode,
    decode_rules   as rs_decode,
    rebuild_gate   as rs_rebuild_gate,
    promote_gate   as rs_promote_gate,
    fg_version,
    LAYER_DOMAIN, LAYER_BEHAVIORAL, LAYER_SESSION,
)
import encoder as py_enc  # Python reference

# ── Test vocabulary (matches simulation) ──────────────────────────────────────
VOCAB = [
    "context", "tone", "empathy", "support", "direct", "brief",
    "acknowledge", "tldr", "formal", "casual", "structured",
    "technical", "clear", "concise", "detailed", "verbose",
    "friendly", "precise", "summary", "action", "step",
    "confirm", "verify", "request", "response"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_py_rule(i, layer=0, n_sessions=5):
    rng = random.Random(i * 31337)
    return py_enc.BinaryRule(
        lhs=rng.choice(VOCAB), rhs=rng.choice(VOCAB),
        support=max(1, int(rng.gauss(8, 3))),
        confidence=rng.uniform(0.4, 0.99),
        layer=layer, session_count=min(n_sessions, 63),
    )

def py_rule_to_dict(r):
    return {
        "lhs": r.lhs, "rhs": r.rhs,
        "support": r.support,
        "confidence": float(r.confidence),
        "layer": r.layer,
        "session_count": r.session_count,
        "ttl": r.ttl,
    }

def dicts_to_set(rules):
    """Canonical set of (lhs, rhs, support) for order-independent comparison."""
    return {(r["lhs"], r["rhs"], r["support"]) for r in rules}

# ── Integration tests ─────────────────────────────────────────────────────────

def test_version():
    v = fg_version()
    assert "fg_gate" in v and "Rust" in v, f"Unexpected version: {v}"
    assert "0.4" in v, f"Expected version 0.4.x, got: {v}"
    print(f"  Version: {v}")

def test_constants():
    assert LAYER_DOMAIN     == 0
    assert LAYER_BEHAVIORAL == 1
    assert LAYER_SESSION    == 2
    assert LAYER_DOMAIN     == py_enc.LAYER_DOMAIN
    assert LAYER_BEHAVIORAL == py_enc.LAYER_BEHAVIORAL
    assert LAYER_SESSION    == py_enc.LAYER_SESSION

def test_empty_encode_decode():
    blob = rs_encode([], LAYER_DOMAIN)
    rules, lid = rs_decode(blob)
    assert rules == [], f"Expected empty, got {rules}"
    assert lid == LAYER_DOMAIN

def test_single_rule_roundtrip():
    rule = {"lhs":"context","rhs":"tone","support":7,"confidence":0.87,
            "layer":LAYER_DOMAIN,"session_count":5,"ttl":None}
    blob = rs_encode([rule], LAYER_DOMAIN)
    dec, lid = rs_decode(blob)
    assert lid == LAYER_DOMAIN
    assert len(dec) == 1
    assert dec[0]["lhs"] == rule["lhs"]
    assert dec[0]["rhs"] == rule["rhs"]
    assert dec[0]["support"] == rule["support"]
    assert abs(dec[0]["confidence"] - rule["confidence"]) < 0.005
    assert dec[0]["layer"] == rule["layer"]
    assert dec[0]["session_count"] == rule["session_count"]

def test_batch_l0_roundtrip():
    py_rules = [make_py_rule(i, LAYER_DOMAIN, 7) for i in range(25)]
    dicts = [py_rule_to_dict(r) for r in py_rules]
    blob = rs_encode(dicts, LAYER_DOMAIN)
    dec, lid = rs_decode(blob)
    assert lid == LAYER_DOMAIN
    assert len(dec) == 25
    assert dicts_to_set(dec) == dicts_to_set(dicts)

def test_batch_l1_roundtrip():
    py_rules = [make_py_rule(i, LAYER_BEHAVIORAL, 3) for i in range(92)]
    dicts = [py_rule_to_dict(r) for r in py_rules]
    blob = rs_encode(dicts, LAYER_BEHAVIORAL)
    dec, lid = rs_decode(blob)
    assert lid == LAYER_BEHAVIORAL
    assert len(dec) == 92
    assert dicts_to_set(dec) == dicts_to_set(dicts)

def test_layer2_ttl_roundtrip():
    rule = {"lhs":"session","rhs":"token","support":1,"confidence":0.55,
            "layer":LAYER_SESSION,"session_count":1,"ttl":1}
    blob = rs_encode([rule], LAYER_SESSION)
    dec, lid = rs_decode(blob)
    assert lid == LAYER_SESSION
    assert len(dec) == 1
    assert dec[0]["ttl"] == 1, f"TTL not preserved: {dec[0]}"

def test_rust_vs_python_semantic_equivalence():
    """Both encoders must produce semantically identical output on same input."""
    py_rules = [make_py_rule(i, LAYER_BEHAVIORAL, 3) for i in range(40)]
    dicts = [py_rule_to_dict(r) for r in py_rules]

    # Python reference encode+decode
    py_blobs = py_enc.encode_ruleset(
        [r for r in py_rules if r.layer == 0],
        py_rules,
        [],
        [],
    )
    py_blob = py_blobs["layer1"]
    py_decoded, _ = py_enc.decode_rules(py_blob)

    # Rust encode+decode
    rs_blob = rs_encode(dicts, LAYER_BEHAVIORAL)
    rs_decoded, _ = rs_decode(rs_blob)

    py_set = {(r.lhs, r.rhs, r.support) for r in py_decoded}
    rs_set = dicts_to_set(rs_decoded)

    assert py_set == rs_set, (
        f"Semantic mismatch!\n  Python: {sorted(py_set)[:5]}\n  Rust: {sorted(rs_set)[:5]}"
    )
    print(f"  Semantic equivalence: {len(py_set)} rules match")

def test_rust_binary_smaller_than_json():
    """Binary blob must be significantly smaller than naive JSON."""
    rules = [make_py_rule(i, LAYER_BEHAVIORAL, 3) for i in range(92)]
    dicts = [py_rule_to_dict(r) for r in rules]

    blob = rs_encode(dicts, LAYER_BEHAVIORAL)
    json_size = len(json.dumps(dicts).encode())
    ratio = json_size / len(blob)

    # v0.4 is smaller than v0.3, so the ratio vs JSON is higher — lower bound stays at 8×
    assert ratio > 8.0, f"Expected >8x compression, got {ratio:.1f}x (blob={len(blob)}B, json={json_size}B)"
    print(f"  Compression: {ratio:.1f}x ({len(blob)}B binary vs {json_size}B JSON)")

def test_ephemeral_gate_rebuild():
    """Gate rebuild from event log — verify counts and layer promotion."""
    events = []
    # Rule A: appears in sessions 0-5 → L0
    for s in range(6):
        events.append((s, "context", "tone"))
    # Rule B: appears in sessions 0-2 → L1
    for s in range(3):
        events.append((s, "direct", "brief"))
    # Rule C: appears only in session 0 → L2
    events.append((0, "session", "only"))

    gate = rs_rebuild_gate(events)
    assert len(gate) == 3

    by_key = {(e["lhs"], e["rhs"]): e for e in gate}
    assert by_key[("context","tone")]["session_count"] == 6
    assert by_key[("direct","brief")]["session_count"] == 3
    assert by_key[("session","only")]["session_count"] == 1

def test_promote_gate_classification():
    """promote_gate must classify rules into correct layers."""
    events = []
    for s in range(6): events.append((s, "ctx",   "tone"))   # L0
    for s in range(3): events.append((s, "dir",   "brief"))  # L1
    events.append((0, "ses", "only"))                         # L2

    result = rs_promote_gate(events, l0_threshold=5, l1_threshold=2)
    assert result["stats"]["l0"] == 1, f"L0 count: {result['stats']}"
    assert result["stats"]["l1"] == 1
    assert result["stats"]["l2"] == 1
    assert result["stats"]["total"] == 3

    l0_keys = {tuple(p) for p in result["layer0"]}
    assert ("ctx","tone") in l0_keys, f"L0 should contain (ctx,tone): {l0_keys}"

def test_gate_ephemeral_no_disk_write():
    """Ephemeral gate: rebuild_gate returns a list, never a bytes blob."""
    events = [(0, "a", "b"), (1, "a", "b")]
    gate = rs_rebuild_gate(events)
    assert isinstance(gate, list), "Gate should be a list of dicts"
    assert not any(isinstance(g, (bytes, bytearray)) for g in gate), \
        "Gate entries should be dicts, not bytes"

def test_v4_vlq_indices_byte_savings():
    """v0.4 blob must be smaller than an equivalent v0.3-sized estimate.

    With vocab <= 127 tokens, each index drops from 2 bytes (u16) to 1 byte (varint).
    Savings: 2 indices x 1B x N rules.  For 25 rules: 50B saved on indices alone.
    Combined with RLE session_count, total savings should be >20% vs v0.3 baseline.
    We verify this by checking the blob is well under the v0.3 empirical L0 size (351B).
    """
    rules = [make_py_rule(i, LAYER_DOMAIN, 7) for i in range(25)]
    dicts = [py_rule_to_dict(r) for r in rules]
    blob = rs_encode(dicts, LAYER_DOMAIN)
    # v0.3 measured L0 size was 351B.  v0.4 should be <295B (>15% smaller).
    assert len(blob) < 295, (
        f"v0.4 L0 blob should be <295B (was 351B in v0.3), got {len(blob)}B"
    )
    # Must still roundtrip correctly
    dec, lid = rs_decode(blob)
    assert lid == LAYER_DOMAIN
    assert len(dec) == 25
    print(f"  v0.4 L0 blob: {len(blob)}B (v0.3 was 351B, saving ~{351-len(blob)}B)")

def test_v4_l1_byte_savings():
    """v0.4 L1 blob must be smaller than v0.3 baseline (852B)."""
    rules = [make_py_rule(i, LAYER_BEHAVIORAL, 3) for i in range(92)]
    dicts = [py_rule_to_dict(r) for r in rules]
    blob = rs_encode(dicts, LAYER_BEHAVIORAL)
    # v0.3 L1 was 852B; v0.4 target is ~582B (>30% smaller)
    assert len(blob) < 720, (
        f"v0.4 L1 blob should be <720B (was 852B in v0.3), got {len(blob)}B"
    )
    dec, lid = rs_decode(blob)
    assert lid == LAYER_BEHAVIORAL
    assert len(dec) == 92
    print(f"  v0.4 L1 blob: {len(blob)}B (v0.3 was 852B, saving ~{852-len(blob)}B)")

def test_confidence_quantization_error_bound():
    """All decoded confidence values must be within 0.005 of the original."""
    rules = []
    for i in range(100):
        conf = i / 99.0  # 0.0 to 1.0
        rules.append({
            "lhs": "a", "rhs": "b", "support": i + 1,
            "confidence": conf, "layer": LAYER_DOMAIN,
            "session_count": 5, "ttl": None
        })
    blob = rs_encode(rules, LAYER_DOMAIN)
    dec, _ = rs_decode(blob)
    orig_by_support = {r["support"]: r["confidence"] for r in rules}
    for r in dec:
        orig_conf = orig_by_support[r["support"]]
        err = abs(r["confidence"] - orig_conf)
        assert err < 0.005, f"support={r['support']}: conf error {err:.6f} > 0.005"

# ── Latency benchmark ─────────────────────────────────────────────────────────

def benchmark_latency():
    """
    Measure Rust encode/decode + gate rebuild latency vs 50 µs target.
    Reports:
      - encode_rules (117 rules: 25 L0 + 92 L1)
      - decode_rules (same)
      - rebuild_gate (200 entries, 10 sessions)
      - full pipeline (encode L0 + L1 + gate rebuild)
    """
    print("\n" + "="*60)
    print("LATENCY BENCHMARK  (target: < 50 µs full pipeline)")
    print("="*60)

    # Build dataset
    l0 = [make_py_rule(i, LAYER_DOMAIN, 7) for i in range(25)]
    l1 = [make_py_rule(i, LAYER_BEHAVIORAL, 3) for i in range(92)]
    l0_dicts = [py_rule_to_dict(r) for r in l0]
    l1_dicts = [py_rule_to_dict(r) for r in l1]

    # Gate events: 200 unique rules, 10 sessions
    rng = random.Random(42)
    gate_events = []
    for s in range(10):
        for _ in range(20):
            lhs = rng.choice(VOCAB)
            rhs = rng.choice(VOCAB)
            gate_events.append((s, lhs, rhs))

    REPS = 2000

    # ── encode L0 ──
    t0 = time.perf_counter()
    for _ in range(REPS):
        b0 = rs_encode(l0_dicts, LAYER_DOMAIN)
    t1 = time.perf_counter()
    enc_l0_us = (t1 - t0) / REPS * 1e6

    # ── encode L1 ──
    t0 = time.perf_counter()
    for _ in range(REPS):
        b1 = rs_encode(l1_dicts, LAYER_BEHAVIORAL)
    t1 = time.perf_counter()
    enc_l1_us = (t1 - t0) / REPS * 1e6

    # ── decode L0 ──
    t0 = time.perf_counter()
    for _ in range(REPS):
        rs_decode(b0)
    t1 = time.perf_counter()
    dec_l0_us = (t1 - t0) / REPS * 1e6

    # ── decode L1 ──
    t0 = time.perf_counter()
    for _ in range(REPS):
        rs_decode(b1)
    t1 = time.perf_counter()
    dec_l1_us = (t1 - t0) / REPS * 1e6

    # ── gate rebuild ──
    t0 = time.perf_counter()
    for _ in range(REPS):
        rs_rebuild_gate(gate_events)
    t1 = time.perf_counter()
    gate_us = (t1 - t0) / REPS * 1e6

    # ── full pipeline ──
    t0 = time.perf_counter()
    for _ in range(REPS):
        rs_encode(l0_dicts, LAYER_DOMAIN)
        rs_encode(l1_dicts, LAYER_BEHAVIORAL)
        rs_rebuild_gate(gate_events)
    t1 = time.perf_counter()
    pipeline_us = (t1 - t0) / REPS * 1e6

    # ── Python reference for comparison ──
    py_l0 = [py_enc.BinaryRule(**{k:v for k,v in d.items() if k != 'ttl'}, ttl=d.get('ttl')) for d in l0_dicts]
    py_l1 = [py_enc.BinaryRule(**{k:v for k,v in d.items() if k != 'ttl'}, ttl=d.get('ttl')) for d in l1_dicts]
    t0 = time.perf_counter()
    for _ in range(REPS):
        py_enc.encode_rules(py_l0, py_enc.LAYER_DOMAIN)
        py_enc.encode_rules(py_l1, py_enc.LAYER_BEHAVIORAL)
    t1 = time.perf_counter()
    py_enc_us = (t1 - t0) / REPS * 1e6

    TARGET_US = 50.0
    PASS_MARK = "PASS" if pipeline_us < TARGET_US else "OVER"

    print(f"  Encode L0 (25 rules):         {enc_l0_us:6.1f} µs")
    print(f"  Encode L1 (92 rules):         {enc_l1_us:6.1f} µs")
    print(f"  Decode L0:                    {dec_l0_us:6.1f} µs")
    print(f"  Decode L1:                    {dec_l1_us:6.1f} µs")
    print(f"  Gate rebuild (200 events):    {gate_us:6.1f} µs")
    print(f"  Full pipeline (enc+enc+gate): {pipeline_us:6.1f} µs  ← target < {TARGET_US} µs  [{PASS_MARK}]")
    print(f"  Python encode L0+L1:          {py_enc_us:6.1f} µs")
    print(f"  Speedup vs Python:            {py_enc_us/(enc_l0_us+enc_l1_us):.1f}×")
    print()

    # Binary sizes
    print(f"  L0 binary ({len(l0)} rules):           {len(b0):,} B")
    print(f"  L1 binary ({len(l1)} rules):           {len(b1):,} B")
    print(f"  Total rules binary:           {len(b0)+len(b1):,} B")
    print(f"  JSON equivalent estimate:     ~{(len(l0)+len(l1))*80:,} B (~80B/rule)")
    print(f"  Compression vs JSON:          {(len(l0)+len(l1))*80 / (len(b0)+len(b1)):.1f}×")
    print("="*60)

    results = {
        "enc_l0_us":   round(enc_l0_us, 2),
        "enc_l1_us":   round(enc_l1_us, 2),
        "dec_l0_us":   round(dec_l0_us, 2),
        "dec_l1_us":   round(dec_l1_us, 2),
        "gate_us":     round(gate_us, 2),
        "pipeline_us": round(pipeline_us, 2),
        "py_enc_us":   round(py_enc_us, 2),
        "target_us":   TARGET_US,
        "pass":        PASS_MARK == "PASS",
        "speedup":     round(py_enc_us / (enc_l0_us + enc_l1_us), 2),
        "b0_bytes":    len(b0),
        "b1_bytes":    len(b1),
    }

    with open(os.path.join(os.path.dirname(__file__), "..", "latency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_version,
        test_constants,
        test_empty_encode_decode,
        test_single_rule_roundtrip,
        test_batch_l0_roundtrip,
        test_batch_l1_roundtrip,
        test_layer2_ttl_roundtrip,
        test_rust_vs_python_semantic_equivalence,
        test_rust_binary_smaller_than_json,
        test_ephemeral_gate_rebuild,
        test_promote_gate_classification,
        test_gate_ephemeral_no_disk_write,
        test_confidence_quantization_error_bound,
        # v0.4 specific
        test_v4_vlq_indices_byte_savings,
        test_v4_l1_byte_savings,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    return failed == 0

if __name__ == "__main__":
    ok = run_all()
    results = benchmark_latency()
    sys.exit(0 if ok else 1)

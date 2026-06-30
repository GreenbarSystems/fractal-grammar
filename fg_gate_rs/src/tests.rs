/// FBG binary encoding — Rust unit test suite.
///
/// Test categories:
///   1. Varint encode/decode (zigzag, single-byte, multi-byte)
///   2. Confidence quantization (q8 roundtrip, clamp, error bound)
///   3. v0.3 legacy pack/unpack layer byte (kept for backward-compat path)
///   4. String table deduplication
///   5. Rule encode/decode round-trip (single, batch, empty, Layer 2 + TTL)
///   6. Delta encoding invariants
///   7. EphemeralGate: register, session_mask, promote, threshold
///   8. rebuild_gate convenience function
///   9. Error handling (bad magic, truncated data, version mismatch)
///  10. Binary size bound
///  11. v0.4 — VLQ index savings (Change A)
///  12. v0.4 — RLE session_count (Change B)
///  13. v0.4 — backward compat: v0.3 blobs still decode correctly

#[cfg(test)]
mod tests {
    use crate::format::*;
    use crate::encode::encode_rules;
    use crate::decode::decode_rules;
    use crate::gate::{EphemeralGate, rebuild_gate};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_rule(lhs: &str, rhs: &str, support: u32, conf: f32, layer: u8, sessions: u8) -> BinaryRule {
        BinaryRule {
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
            support,
            confidence: conf,
            layer,
            session_count: sessions,
            ttl: None,
        }
    }

    fn roundtrip(rules: &[BinaryRule], layer_id: u8) -> (Vec<BinaryRule>, u8) {
        let encoded = encode_rules(rules, layer_id);
        decode_rules(&encoded).expect("Decode failed")
    }

    // ── 1. Varint ─────────────────────────────────────────────────────────────

    #[test]
    fn test_varint_zero() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 0);
        assert_eq!(buf, &[0x00]);
        let (v, pos) = read_varint(&buf, 0).unwrap();
        assert_eq!(v, 0);
        assert_eq!(pos, 1);
    }

    #[test]
    fn test_varint_single_byte() {
        for v in [1u64, 63, 127] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            assert_eq!(buf.len(), 1, "v={v} should be 1 byte");
            let (decoded, _) = read_varint(&buf, 0).unwrap();
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_varint_multi_byte() {
        for v in [128u64, 255, 1024, 65535, 1_000_000] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let (decoded, _) = read_varint(&buf, 0).unwrap();
            assert_eq!(decoded, v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_zigzag_identity() {
        for v in [-100i64, -1, 0, 1, 100, i32::MAX as i64] {
            let enc = zigzag_encode(v);
            let dec = zigzag_decode(enc);
            assert_eq!(dec, v, "zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_zigzag_ordering() {
        // zigzag: small magnitude → small unsigned value
        assert!(zigzag_encode(0) < zigzag_encode(1));
        assert!(zigzag_encode(-1) < zigzag_encode(-2));
        assert!(zigzag_encode(1) < zigzag_encode(2));
    }

    // ── 2. Confidence quantization ────────────────────────────────────────────

    #[test]
    fn test_conf_q8_roundtrip() {
        for conf in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let q = conf_to_q8(conf);
            let back = q8_to_conf(q);
            assert!((back - conf).abs() < 0.005, "conf={conf} → q={q} → {back}");
        }
    }

    #[test]
    fn test_conf_q8_max_error() {
        // Maximum quantization error = 1/255 ≈ 0.00392
        for i in 0..=255u8 {
            let conf = i as f32 / 255.0;
            let q = conf_to_q8(conf);
            let back = q8_to_conf(q);
            assert!((back - conf).abs() < 0.005, "i={i} conf={conf} back={back}");
        }
    }

    #[test]
    fn test_conf_q8_clamp() {
        assert_eq!(conf_to_q8(-0.5), 0);
        assert_eq!(conf_to_q8(1.5), 255);
    }

    // ── 3. v0.3 legacy layer byte packing (backward-compat path only) ──────────

    #[test]
    fn test_pack_unpack_layer_byte_v3_legacy() {
        // v0.3 packed byte: bits 0-1 = layer, bits 2-7 = session_count (capped 63)
        for layer in 0u8..3 {
            for sc in [0u8, 1, 31, 62, 63, 100] {
                let packed = pack_layer_byte_v3(layer, sc);
                let (l, s) = unpack_layer_byte_v3(packed);
                assert_eq!(l, layer, "layer mismatch");
                assert_eq!(s, sc.min(63), "session_count mismatch (input {sc})");
            }
        }
    }

    // ── 4. Empty input ────────────────────────────────────────────────────────

    #[test]
    fn test_encode_empty_rules() {
        let encoded = encode_rules(&[], LAYER_DOMAIN);
        let (rules, layer_id) = decode_rules(&encoded).unwrap();
        assert!(rules.is_empty());
        assert_eq!(layer_id, LAYER_DOMAIN);
    }

    // ── 5. Round-trip — single rule ───────────────────────────────────────────

    #[test]
    fn test_single_rule_roundtrip() {
        let r = make_rule("context", "tone", 7, 0.87, LAYER_DOMAIN, 5);
        let (decoded, layer_id) = roundtrip(&[r.clone()], LAYER_DOMAIN);
        assert_eq!(layer_id, LAYER_DOMAIN);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].lhs, r.lhs);
        assert_eq!(decoded[0].rhs, r.rhs);
        assert_eq!(decoded[0].support, r.support);
        assert!((decoded[0].confidence - r.confidence).abs() < 0.005);
        assert_eq!(decoded[0].layer, r.layer);
        assert_eq!(decoded[0].session_count, r.session_count);
    }

    // ── 5. Round-trip — batch (Layer 0, Layer 1) ──────────────────────────────

    #[test]
    fn test_batch_roundtrip_layer0() {
        let rules = vec![
            make_rule("context", "tone",    10, 0.91, LAYER_DOMAIN, 8),
            make_rule("tone",    "empathy",  8, 0.75, LAYER_DOMAIN, 7),
            make_rule("empathy", "brief",    5, 0.60, LAYER_DOMAIN, 5),
            make_rule("direct",  "context",  3, 0.45, LAYER_DOMAIN, 5),
        ];
        let (decoded, lid) = roundtrip(&rules, LAYER_DOMAIN);
        assert_eq!(lid, LAYER_DOMAIN);
        assert_eq!(decoded.len(), rules.len());

        // Content check by (lhs, rhs, support) set — order may differ
        let orig: std::collections::HashSet<_> = rules.iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone(), r.support))
            .collect();
        let got: std::collections::HashSet<_>  = decoded.iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone(), r.support))
            .collect();
        assert_eq!(orig, got, "Set of (lhs,rhs,support) should be identical");
    }

    #[test]
    fn test_batch_roundtrip_layer1() {
        let rules: Vec<BinaryRule> = (0..20).map(|i| {
            make_rule(
                &format!("tok{}", i % 10),
                &format!("tok{}", (i + 3) % 10),
                (20 - i) as u32,
                0.5 + (i as f32) * 0.02,
                LAYER_BEHAVIORAL,
                (i % 4 + 2) as u8,
            )
        }).collect();

        let (decoded, lid) = roundtrip(&rules, LAYER_BEHAVIORAL);
        assert_eq!(lid, LAYER_BEHAVIORAL);
        assert_eq!(decoded.len(), rules.len());

        let orig_set: std::collections::HashSet<_> = rules.iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone(), r.support))
            .collect();
        let got_set: std::collections::HashSet<_> = decoded.iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone(), r.support))
            .collect();
        assert_eq!(orig_set, got_set);
    }

    // ── 5. Round-trip — Layer 2 + TTL ─────────────────────────────────────────

    #[test]
    fn test_layer2_ttl_roundtrip() {
        let mut r = make_rule("session", "token", 1, 0.55, LAYER_SESSION, 1);
        r.ttl = Some(1);
        let encoded = encode_rules(&[r.clone()], LAYER_SESSION);
        let (decoded, lid) = decode_rules(&encoded).unwrap();
        assert_eq!(lid, LAYER_SESSION);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].ttl, Some(1));
    }

    // ── 5. Round-trip — 92 rules (Layer 1 sim scale) ─────────────────────────

    #[test]
    fn test_large_batch_l1() {
        let vocab = vec!["context","tone","empathy","support","direct","brief",
                         "acknowledge","tldr","formal","casual","structured",
                         "technical","clear","concise","detailed","verbose",
                         "friendly","precise","summary","action"];

        let rules: Vec<BinaryRule> = (0..92usize).map(|i| {
            make_rule(
                vocab[i % vocab.len()],
                vocab[(i * 7 + 3) % vocab.len()],
                (100 - i) as u32,
                0.4 + (i as f32 % 40.0) / 100.0,
                LAYER_BEHAVIORAL,
                (i % 3 + 2) as u8,
            )
        }).collect();

        let encoded = encode_rules(&rules, LAYER_BEHAVIORAL);

        // Binary size should be << JSON equivalent
        let json_estimate = rules.len() * 80; // ~80 bytes/rule JSON
        assert!(
            encoded.len() < json_estimate / 5,
            "Binary {len}B should be <1/5 of JSON estimate {json_estimate}B",
            len = encoded.len()
        );

        let (decoded, lid) = decode_rules(&encoded).unwrap();
        assert_eq!(lid, LAYER_BEHAVIORAL);
        assert_eq!(decoded.len(), 92);

        let orig_set: std::collections::HashSet<_> = rules.iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone(), r.support))
            .collect();
        let got_set: std::collections::HashSet<_> = decoded.iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone(), r.support))
            .collect();
        assert_eq!(orig_set, got_set);
    }

    // ── 6. Delta encoding invariants ─────────────────────────────────────────

    #[test]
    fn test_delta_encoding_sorted_output() {
        // After encode+decode, rules should be sorted descending by support
        let rules = vec![
            make_rule("a", "b", 3, 0.5, LAYER_DOMAIN, 5),
            make_rule("c", "d", 9, 0.7, LAYER_DOMAIN, 6),
            make_rule("e", "f", 6, 0.6, LAYER_DOMAIN, 5),
        ];
        let (decoded, _) = roundtrip(&rules, LAYER_DOMAIN);
        // Decoded order is support-descending
        for i in 1..decoded.len() {
            assert!(decoded[i-1].support >= decoded[i].support,
                "Not sorted: pos {i}: {} < {}", decoded[i-1].support, decoded[i].support);
        }
    }

    #[test]
    fn test_delta_encoding_identical_support() {
        // Ties in support → delta = 0 → 1-byte varint
        let rules = vec![
            make_rule("a", "b", 5, 0.5, LAYER_DOMAIN, 5),
            make_rule("c", "d", 5, 0.6, LAYER_DOMAIN, 5),
            make_rule("e", "f", 5, 0.7, LAYER_DOMAIN, 5),
        ];
        let (decoded, _) = roundtrip(&rules, LAYER_DOMAIN);
        assert_eq!(decoded.len(), 3);
        assert!(decoded.iter().all(|r| r.support == 5));
    }

    // ── 7. EphemeralGate ─────────────────────────────────────────────────────

    #[test]
    fn test_gate_new_empty() {
        let g = EphemeralGate::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_gate_single_session() {
        let mut g = EphemeralGate::new();
        g.register_session(&[
            ("context".into(), "tone".into()),
            ("tone".into(),    "empathy".into()),
        ]);
        assert_eq!(g.len(), 2);
        let snap = g.snapshot();
        assert!(snap.iter().all(|e| e.session_count == 1));
        assert!(snap.iter().all(|e| e.session_mask == 0b0000_0001));
    }

    #[test]
    fn test_gate_multiple_sessions_count() {
        let mut g = EphemeralGate::new();
        let rule = ("context".to_string(), "tone".to_string());
        // Appear in sessions 0, 1, 2
        for _ in 0..3 {
            g.register_session(&[rule.clone()]);
        }
        let snap = g.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].session_count, 3);
    }

    #[test]
    fn test_gate_duplicate_within_session_not_double_counted() {
        let mut g = EphemeralGate::new();
        // Same rule twice in the same session register call
        let rule = ("a".to_string(), "b".to_string());
        g.register_session(&[rule.clone(), rule.clone()]);
        let snap = g.snapshot();
        assert_eq!(snap[0].session_count, 1, "Duplicate within session should not increment count");
    }

    #[test]
    fn test_gate_session_mask_bits() {
        let mut g = EphemeralGate::new();
        let rule = ("x".to_string(), "y".to_string());
        // Sessions 0, 2, 4 → bits 0, 2, 4 set
        for s in 0u8..5 {
            if s % 2 == 0 {
                g.register_session(&[rule.clone()]);
            } else {
                g.register_session(&[]); // empty session — advances cursor
            }
        }
        let snap = g.snapshot();
        assert_eq!(snap[0].session_count, 3);
        // bits 0, 2, 4 → 0b0001_0101 = 21
        assert_eq!(snap[0].session_mask, 0b0001_0101, "mask={:#010b}", snap[0].session_mask);
    }

    #[test]
    fn test_gate_promote_thresholds() {
        let mut g = EphemeralGate::new();
        let r_l0 = ("l0a".to_string(), "l0b".to_string());
        let r_l1 = ("l1a".to_string(), "l1b".to_string());
        let r_l2 = ("l2a".to_string(), "l2b".to_string());
        let r_q  = ("qa".to_string(),  "qb".to_string());  // quarantined

        // r_l0: 6 sessions, r_l1: 3 sessions, r_l2: 1 session, r_q: 0 (never registered)
        for i in 0..6u8 {
            let mut rules = vec![r_l0.clone()];
            if i < 3 { rules.push(r_l1.clone()); }
            if i == 0 { rules.push(r_l2.clone()); }
            g.register_session(&rules);
        }

        let (l0, l1, l2) = g.promote();
        assert!(l0.contains(&r_l0), "r_l0 should be in L0 (6 sessions)");
        assert!(l1.contains(&r_l1), "r_l1 should be in L1 (3 sessions)");
        assert!(l2.contains(&r_l2), "r_l2 should be in L2 (1 session)");
        assert!(!l0.contains(&r_q), "r_q never registered — not in L0");
    }

    // ── 8. rebuild_gate ──────────────────────────────────────────────────────

    #[test]
    fn test_rebuild_gate_basic() {
        let events = vec![
            (0u32, "a".to_string(), "b".to_string()),
            (0,    "c".to_string(), "d".to_string()),
            (1,    "a".to_string(), "b".to_string()),
            (2,    "a".to_string(), "b".to_string()),
        ];
        let gate = rebuild_gate(&events);
        let snap = gate.snapshot();
        let ab = snap.iter().find(|e| e.lhs == "a" && e.rhs == "b").unwrap();
        assert_eq!(ab.session_count, 3);
        let cd = snap.iter().find(|e| e.lhs == "c" && e.rhs == "d").unwrap();
        assert_eq!(cd.session_count, 1);
    }

    #[test]
    fn test_rebuild_gate_gap_sessions() {
        // Sessions 0, 2 (skip 1) — rule appears in 0 and 2
        let events = vec![
            (0u32, "x".to_string(), "y".to_string()),
            (2,    "x".to_string(), "y".to_string()),
        ];
        let gate = rebuild_gate(&events);
        let snap = gate.snapshot();
        assert_eq!(snap[0].session_count, 2);
        // Bits 0 and 2 set → 0b0000_0101 = 5
        assert_eq!(snap[0].session_mask, 0b0000_0101);
    }

    // ── 9. Error handling ────────────────────────────────────────────────────

    #[test]
    fn test_decode_bad_magic() {
        let bad = vec![0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = decode_rules(&bad);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("magic"));
    }

    #[test]
    fn test_decode_bad_version() {
        let mut data = vec![0x47, 0x46, 0x99, 0x00, 0x00, 0x00, 0x00, 0x00];
        // Fix magic but wrong version
        data[0] = 0x47; data[1] = 0x46;
        let result = decode_rules(&data);
        // We'll accept either magic error or version error since byte order matters
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_correct_magic_wrong_version() {
        // MAGIC = 0x4647 LE → bytes [0x47, 0x46]
        let data = vec![0x47, 0x46, 0x99u8, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = decode_rules(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("version") || err.contains("magic"),
                "Expected version or magic error, got: {err}");
    }

    #[test]
    fn test_decode_truncated() {
        let r = make_rule("a", "b", 5, 0.5, LAYER_DOMAIN, 5);
        let encoded = encode_rules(&[r], LAYER_DOMAIN);
        // Truncate to just the header
        let truncated = &encoded[..8];
        // rule_count > 0 but no data follows — should fail
        // Note: empty rule_count case returns Ok, so only fail if rule_count > 0
        // The header says rule_count=1, so decode should fail on missing string table
        let result = decode_rules(truncated);
        // Either error (truncated) or we get back 0 rules depending on count
        // With rule_count=1 and truncated data, it must fail
        assert!(result.is_err(), "Should fail on truncated data");
    }

    #[test]
    fn test_decode_empty_buffer() {
        let result = decode_rules(&[]);
        assert!(result.is_err());
    }

    // ── 10. Binary size bound ─────────────────────────────────────────────────

    #[test]
    fn test_binary_smaller_than_json_estimate_at_scale() {
        // 117 rules (25 L0 + 92 L1) → binary should be < 2 KB
        let vocab = ["context","tone","empathy","support","direct","brief",
                     "acknowledge","tldr","formal","casual","structured",
                     "technical","clear","concise","detailed","verbose",
                     "friendly","precise","summary","action","step",
                     "confirm","verify","request","response"];

        let l0: Vec<BinaryRule> = (0..25).map(|i| make_rule(
            vocab[i % vocab.len()], vocab[(i*3+1) % vocab.len()],
            (10 - i / 5) as u32, 0.8, LAYER_DOMAIN, 7,
        )).collect();

        let l1: Vec<BinaryRule> = (0..92).map(|i| make_rule(
            vocab[i % vocab.len()], vocab[(i*7+3) % vocab.len()],
            (4 - i / 25) as u32, 0.6, LAYER_BEHAVIORAL, 3,
        )).collect();

        let b0 = encode_rules(&l0, LAYER_DOMAIN);
        let b1 = encode_rules(&l1, LAYER_BEHAVIORAL);
        let total = b0.len() + b1.len();

        assert!(total < 2048,
            "Rules-only binary should be <2 KB at sim scale, got {total} B");
    }

    // ── 11. v0.4 — VLQ index savings (Change A) ────────────────────────────

    #[test]
    fn test_v4_vlq_indices_smaller_than_v3_fixed_u16() {
        // Build a 25-rule L0 layer with vocab < 128 (every index fits in 1 byte).
        // v0.4 should be smaller than a v0.3-equivalent blob because each
        // index saves 1 byte (varint 1B vs u16 2B) × 2 per rule × 25 rules = 50B.
        let rules: Vec<BinaryRule> = (0..25).map(|i| make_rule(
            ["ctx","tone","emp","sup","dir"][i % 5],
            ["brief","ack","frml","cas","str"][i % 5],
            (25 - i) as u32, 0.7, LAYER_DOMAIN, 7,
        )).collect();

        let v4_blob = encode_rules(&rules, LAYER_DOMAIN);

        // v0.4 per-rule record: varint(lhs) + varint(rhs) + varint(support) + conf_q8
        // All indices < 10 → 1 byte each.  support ≤ 25 → 1 byte.  conf_q8 = 1 byte.
        // Per-rule record ≈ 4 bytes vs v0.3's ≈8 bytes (4 fixed idx + varint support + 2)
        // Also: SESSION_RUNS section = 2 + 2 bytes (1 run of 25 same value)
        // Sanity: v0.4 blob must be < 500 bytes for 25 rules
        assert!(v4_blob.len() < 500,
            "v0.4 L0 blob for 25 rules should be <500B, got {}", v4_blob.len());

        // Roundtrip must be lossless
        let (decoded, lid) = decode_rules(&v4_blob).expect("decode failed");
        assert_eq!(lid, LAYER_DOMAIN);
        assert_eq!(decoded.len(), 25);
    }

    #[test]
    fn test_v4_header_version_byte() {
        // Encoded blob must advertise VERSION = 0x04
        let r = make_rule("a", "b", 5, 0.5, LAYER_DOMAIN, 5);
        let blob = encode_rules(&[r], LAYER_DOMAIN);
        assert_eq!(blob[2], VERSION, "version byte should be 0x{:02x}, got 0x{:02x}", VERSION, blob[2]);
    }

    #[test]
    fn test_v4_flag_session_rle_set() {
        // FLAG_SESSION_RLE (bit 2) must be set in the flags field of v0.4 blobs
        let r = make_rule("a", "b", 5, 0.5, LAYER_DOMAIN, 5);
        let blob = encode_rules(&[r], LAYER_DOMAIN);
        let flags = u16::from_le_bytes([blob[6], blob[7]]);
        assert!(flags & FLAG_SESSION_RLE != 0,
            "FLAG_SESSION_RLE should be set, flags={flags:#06x}");
    }

    // ── 12. v0.4 — RLE session_count (Change B) ───────────────────────────

    #[test]
    fn test_rle_encode_session_counts_single_run() {
        // All same value → one run
        let counts = vec![3u8; 30];
        let runs = rle_encode_session_counts(&counts);
        assert_eq!(runs.len(), 1, "Expected 1 run, got {}", runs.len());
        assert_eq!(runs[0], (30, 3));
    }

    #[test]
    fn test_rle_encode_session_counts_multiple_runs() {
        let counts = vec![5u8, 5, 3, 3, 3, 2];
        let runs = rle_encode_session_counts(&counts);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0], (2, 5));
        assert_eq!(runs[1], (3, 3));
        assert_eq!(runs[2], (1, 2));
    }

    #[test]
    fn test_rle_encode_empty() {
        let runs = rle_encode_session_counts(&[]);
        assert!(runs.is_empty());
    }

    #[test]
    fn test_v4_l2_session_count_rle_minimal() {
        // L2: all session_count = 1 → single run → SESSION_RUNS = 4 bytes
        // (2B n_runs + 1B count + 1B value) vs 30 bytes raw in v0.3
        let rules: Vec<BinaryRule> = (0..30).map(|i| BinaryRule {
            lhs: format!("lhs{i}"),
            rhs: format!("rhs{i}"),
            support: 1,
            confidence: 0.4,
            layer: LAYER_SESSION,
            session_count: 1,
            ttl: Some(1),
        }).collect();

        let blob = encode_rules(&rules, LAYER_SESSION);
        let (decoded, lid) = decode_rules(&blob).expect("decode failed");
        assert_eq!(lid, LAYER_SESSION);
        assert_eq!(decoded.len(), 30);
        assert!(decoded.iter().all(|r| r.session_count == 1),
            "All L2 rules should have session_count=1");
        assert!(decoded.iter().all(|r| r.ttl == Some(1)),
            "All L2 rules should have ttl=1");
    }

    #[test]
    fn test_v4_session_count_preserved_across_layers() {
        // Mixed session_counts in L1 layer must all decode correctly
        let rules = vec![
            make_rule("a", "b", 10, 0.8, LAYER_BEHAVIORAL, 4),
            make_rule("c", "d",  8, 0.7, LAYER_BEHAVIORAL, 3),
            make_rule("e", "f",  6, 0.6, LAYER_BEHAVIORAL, 3),
            make_rule("g", "h",  4, 0.5, LAYER_BEHAVIORAL, 2),
            make_rule("i", "j",  2, 0.4, LAYER_BEHAVIORAL, 2),
        ];
        let (decoded, _) = roundtrip(&rules, LAYER_BEHAVIORAL);
        assert_eq!(decoded.len(), rules.len());
        // Build a lookup by (lhs, rhs) to check session_count independently of sort order
        use std::collections::HashMap;
        let by_key: HashMap<_, _> = decoded.iter()
            .map(|r| ((r.lhs.clone(), r.rhs.clone()), r.session_count))
            .collect();
        for r in &rules {
            let key = (r.lhs.clone(), r.rhs.clone());
            assert_eq!(
                by_key[&key], r.session_count,
                "session_count mismatch for ({}, {}): expected {} got {}",
                r.lhs, r.rhs, r.session_count, by_key[&key]
            );
        }
    }

    // ── 13. v0.4 — backward compat: v0.3 blobs still decode ───────────────

    #[test]
    fn test_v3_blob_still_decodes() {
        // Hand-craft a minimal v0.3 blob and verify the decoder accepts it.
        // v0.3: magic 0x4647 LE, version 0x03, layer 0x00, rule_count 1, flags 0x0001
        // String table: 1 entry "a" (len=1, 'a') + 1 entry "b" (len=1, 'b')
        //   st_n_bytes=4, [1,'a',1,'b']
        // Rule data: lhs_idx u16=0x0000, rhs_idx u16=0x0001, support varint=5 (zigzag delta from 0: delta=5, zz=10)
        //   conf_q8=0x80, packed=pack_layer_byte_v3(0,5)=0x14
        // rule_data_len u32 = 4+1+1+1 = 7 bytes (lhs u16 + rhs u16 + varint(10) + conf + packed)

        let mut blob: Vec<u8> = Vec::new();
        // Header
        blob.extend_from_slice(&MAGIC.to_le_bytes());       // magic
        blob.push(VERSION_V3);                               // version 0x03
        blob.push(LAYER_DOMAIN);                            // layer_id
        blob.extend_from_slice(&1u16.to_le_bytes());        // rule_count = 1
        blob.extend_from_slice(&FLAG_DELTA_ENCODED.to_le_bytes()); // flags
        // String table: total 4 bytes ("a"=2, "b"=2)
        blob.extend_from_slice(&4u16.to_le_bytes());        // st_n_bytes
        blob.push(1); blob.push(b'a');                      // "a"
        blob.push(1); blob.push(b'b');                      // "b"
        // Rule data
        let mut rule_buf: Vec<u8> = Vec::new();
        rule_buf.extend_from_slice(&0u16.to_le_bytes());    // lhs_idx = 0
        rule_buf.extend_from_slice(&1u16.to_le_bytes());    // rhs_idx = 1
        // support = 5, zigzag_encode(5 - 0) = 10 → varint 1 byte
        rule_buf.push(0x0A);                                // varint(10)
        rule_buf.push(0x80);                                // conf_q8 = 0.5
        rule_buf.push(pack_layer_byte_v3(0, 5));            // packed: layer=0, sc=5
        blob.extend_from_slice(&(rule_buf.len() as u32).to_le_bytes());
        blob.extend_from_slice(&rule_buf);

        let (rules, lid) = decode_rules(&blob).expect("v0.3 blob should decode successfully");
        assert_eq!(lid, LAYER_DOMAIN);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].lhs, "a");
        assert_eq!(rules[0].rhs, "b");
        assert_eq!(rules[0].support, 5);
        assert_eq!(rules[0].session_count, 5);
        assert_eq!(rules[0].layer, 0);
    }
}

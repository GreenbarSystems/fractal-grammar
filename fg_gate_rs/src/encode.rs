/// Batch encoder — rules → FG v0.4 binary blob.
///
/// v0.4 changes vs v0.3:
///   Change A: lhs_idx / rhs_idx encoded as varint (was fixed u16).
///             Saves ~1 byte per index when vocab ≤ 127 tokens (typical).
///   Change B: session_count extracted into a SESSION_RUNS RLE section that
///             precedes the rule records.  Layer bits removed from per-rule byte
///             (layer_id already lives in the header).  Per-rule payload
///             shrinks from 2 trailing bytes to 1 (conf_q8 only).

use crate::format::*;

// ── String table ──────────────────────────────────────────────────────────────

/// Deduplicating string pool.  Strings stored in insertion order;
/// lookup is O(n) but n is typically <50 tokens for FBG rulesets.
struct StringTable {
    strings: Vec<String>,
}

impl StringTable {
    fn new() -> Self { Self { strings: Vec::new() } }

    /// Intern a string, returning its index.
    fn intern(&mut self, s: &str) -> u64 {
        if let Some(i) = self.strings.iter().position(|x| x == s) {
            return i as u64;
        }
        let idx = self.strings.len() as u64;
        self.strings.push(s.to_string());
        idx
    }

    fn get_idx(&self, s: &str) -> u64 {
        self.strings.iter().position(|x| x == s)
            .expect("String not in table") as u64
    }

    /// Serialize to bytes: u16 total_byte_len, then (u8 len, utf8 bytes)*.
    fn serialize(&self) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        for s in &self.strings {
            let bytes = s.as_bytes();
            assert!(bytes.len() <= 255, "Token too long: {}", s);
            data.push(bytes.len() as u8);
            data.extend_from_slice(bytes);
        }
        let mut out = Vec::with_capacity(2 + data.len());
        let len = data.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&data);
        out
    }
}

// ── Rule encoder ──────────────────────────────────────────────────────────────

/// Encode a slice of rules for one layer to FG v0.4 binary.
///
/// Rules are sorted descending by support before encoding (required for
/// delta compression on support values).  Input slice is not mutated.
pub fn encode_rules(rules: &[BinaryRule], layer_id: u8) -> Vec<u8> {
    if rules.is_empty() {
        // Empty payload: header only
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.push(VERSION);
        out.push(layer_id);
        out.extend_from_slice(&0u16.to_le_bytes()); // rule_count = 0
        out.extend_from_slice(&0u16.to_le_bytes()); // flags = 0
        return out;
    }

    // Sort a local index by support descending (stable, preserves ties)
    let mut order: Vec<usize> = (0..rules.len()).collect();
    order.sort_by(|&a, &b| rules[b].support.cmp(&rules[a].support));

    // Build string table (insertion order follows the sorted rule sequence)
    let mut st = StringTable::new();
    for &i in &order {
        st.intern(&rules[i].lhs);
        st.intern(&rules[i].rhs);
    }
    let st_bytes = st.serialize();

    // Determine flags
    let has_ttl = layer_id == LAYER_SESSION
        && rules.iter().any(|r| r.ttl.is_some());
    let flags: u16 = FLAG_DELTA_ENCODED | FLAG_SESSION_RLE
        | if has_ttl { FLAG_HAS_TTL } else { 0 };

    // ── SESSION_RUNS section (Change B) ──────────────────────────────────────
    // Collect session_counts in support-sorted order
    let session_counts_sorted: Vec<u8> = order.iter()
        .map(|&i| rules[i].session_count)
        .collect();
    let runs = rle_encode_session_counts(&session_counts_sorted);
    let mut session_rle_bytes: Vec<u8> = Vec::new();
    write_session_rle(&mut session_rle_bytes, &runs);

    // ── Rule records (Change A + streamlined per-rule byte) ──────────────────
    let mut rule_buf: Vec<u8> = Vec::new();
    let mut prev_support: i64 = 0;

    for &i in &order {
        let r = &rules[i];

        // Change A: lhs_idx and rhs_idx as varint (not u16)
        let lhs_idx = st.get_idx(&r.lhs);
        let rhs_idx = st.get_idx(&r.rhs);
        write_varint(&mut rule_buf, lhs_idx);
        write_varint(&mut rule_buf, rhs_idx);

        // support — zigzag delta from previous (sorted desc → delta ≤ 0 usually)
        let delta = r.support as i64 - prev_support;
        write_varint(&mut rule_buf, zigzag_encode(delta));
        prev_support = r.support as i64;

        // conf_q8 only — session_count moved to SESSION_RUNS, layer in header
        rule_buf.push(conf_to_q8(r.confidence));
    }

    // TTL records (Layer 2 only)
    let mut ttl_buf: Vec<u8> = Vec::new();
    if has_ttl {
        let ttl_entries: Vec<(u16, u8)> = order
            .iter()
            .enumerate()
            .filter_map(|(enc_idx, &orig_idx)| {
                rules[orig_idx].ttl.map(|t| (enc_idx as u16, t))
            })
            .collect();
        ttl_buf.extend_from_slice(&(ttl_entries.len() as u16).to_le_bytes());
        for (rule_idx, ttl) in &ttl_entries {
            ttl_buf.extend_from_slice(&rule_idx.to_le_bytes());
            ttl_buf.push(*ttl);
        }
    }

    // ── Assemble final output ─────────────────────────────────────────────────
    let rule_count = rules.len() as u16;
    let mut out: Vec<u8> = Vec::with_capacity(
        8 + st_bytes.len() + session_rle_bytes.len() + 4 + rule_buf.len() + ttl_buf.len()
    );

    // Header
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.push(VERSION);
    out.push(layer_id);
    out.extend_from_slice(&rule_count.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());

    // String table
    out.extend_from_slice(&st_bytes);

    // SESSION_RUNS section (Change B)
    out.extend_from_slice(&session_rle_bytes);

    // Rule data (prefixed with u32 byte length)
    out.extend_from_slice(&(rule_buf.len() as u32).to_le_bytes());
    out.extend_from_slice(&rule_buf);

    // TTL data (if any)
    out.extend_from_slice(&ttl_buf);

    out
}

/// Convenience wrapper: returns bytes directly.
pub fn encode_rules_bytes(rules: &[BinaryRule], layer_id: u8) -> Vec<u8> {
    encode_rules(rules, layer_id)
}

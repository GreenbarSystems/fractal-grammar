/// FG wire format constants and core structs.
///
/// VERSION HISTORY
/// ───────────────
/// 0x03  (v0.3.0)  Fixed u16 indices; layer+session_count packed byte.
/// 0x04  (v0.4.0)  Change A: lhs/rhs indices are now varint (not u16).
///                 Change B: session_count stored in an RLE section that
///                           precedes the rule records; layer bits removed
///                           from the per-rule byte (layer_id is in header).
///
/// FILE LAYOUT (little-endian throughout)
/// ──────────────────────────────────────
/// HEADER          8 bytes  fixed
/// STR_TABLE       variable  deduplicated, length-prefixed u8 strings
/// SESSION_RUNS    variable  RLE section for session_count (v0.4+)
/// RULE_DATA       variable  delta-encoded rule records
/// TTL_DATA        optional  Layer 2 only
///
/// HEADER
///   [0..1]  magic      : u16 = 0x4647  ('F','G')
///   [2]     version    : u8  = 0x04
///   [3]     layer_id   : u8  (0=Domain 1=Behavioral 2=Session 0xFF=reserved)
///   [4..5]  rule_count : u16
///   [6..7]  flags      : u16
///             bit 0 = DELTA_ENCODED    (support stored as zigzag delta)
///             bit 1 = HAS_TTL         (Layer 2 only — TTL records appended)
///             bit 2 = SESSION_RLE     (session_count stored in RLE section)
///
/// STRING TABLE  (unchanged from v0.3)
///   [0..1]  n_bytes    : u16  (total byte length of all strings + length prefixes)
///   For each string entry:
///     [0]     len      : u8
///     [1..N]  utf8 bytes
///
/// SESSION_RUNS section (v0.4+, present when FLAG_SESSION_RLE is set)
///   [0..1]  n_runs     : u16
///   For each run (rules sorted descending by support first):
///     [0]   count      : u8   (number of consecutive rules with this session_count)
///     [1]   value      : u8   (session_count value, capped at 255)
///
/// RULE RECORDS  (after SESSION_RUNS)
///   [0..3]  n_bytes    : u32  (byte length of all packed rule records)
///   For each rule (sorted descending by support before encoding):
///     [0..N]  lhs_idx  : varint   ← was u16 in v0.3 (Change A)
///     [N..M]  rhs_idx  : varint   ← was u16 in v0.3 (Change A)
///     [M..P]  support  : varint   (zigzag delta if DELTA_ENCODED)
///     [P]     conf_q8  : u8       (confidence * 255.0 + 0.5) as u8
///     NOTE: session_count is no longer stored per-rule (it is in SESSION_RUNS)
///           layer is no longer stored per-rule (it is in the header layer_id)
///
/// TTL RECORDS  (Layer 2 only, appended after rule records if HAS_TTL)
///   [0..1]  n_ttl      : u16
///   For each TTL entry:
///     [0..1]  rule_idx : u16
///     [2]     ttl      : u8   (sessions remaining, relative offset)

pub const MAGIC:   u16 = 0x4647;
pub const VERSION: u8  = 0x04;
pub const VERSION_V3: u8 = 0x03; // accepted for backward-compatible decode

pub const LAYER_DOMAIN:     u8 = 0;
pub const LAYER_BEHAVIORAL: u8 = 1;
pub const LAYER_SESSION:    u8 = 2;

pub const FLAG_DELTA_ENCODED: u16 = 0x0001;
pub const FLAG_HAS_TTL:       u16 = 0x0002;
pub const FLAG_SESSION_RLE:   u16 = 0x0004; // v0.4+

/// A decoded rule as a plain Rust struct.
/// This is the canonical in-memory representation; Python sees it as a dict.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryRule {
    pub lhs:           String,
    pub rhs:           String,
    pub support:       u32,
    /// Confidence in [0.0, 1.0]. Quantized to u8 on the wire (max error ~0.004).
    pub confidence:    f32,
    /// 0 = Domain, 1 = Behavioral, 2 = Session
    pub layer:         u8,
    /// Number of sessions this rule appeared in (capped at 255)
    pub session_count: u8,
    /// Layer 2 TTL — None unless layer == 2
    pub ttl:           Option<u8>,
}

/// File-level header (8 bytes).
#[derive(Debug, Clone)]
pub struct FgHeader {
    pub version:    u8,
    pub layer_id:   u8,
    pub rule_count: u16,
    pub flags:      u16,
}

impl FgHeader {
    pub fn is_delta(&self) -> bool { self.flags & FLAG_DELTA_ENCODED != 0 }
    pub fn has_ttl(&self) -> bool  { self.flags & FLAG_HAS_TTL        != 0 }
    pub fn has_session_rle(&self) -> bool { self.flags & FLAG_SESSION_RLE != 0 }
}

/// Zigzag-encode a signed integer → unsigned.
/// Maps: 0→0, -1→1, 1→2, -2→3, 2→4 …
#[inline]
pub fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Zigzag-decode unsigned → signed.
#[inline]
pub fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Write a variable-length unsigned integer (little-endian, 7 bits per byte,
/// continuation bit = 0x80).
pub fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Read a varint from a byte slice starting at `pos`. Returns (value, new_pos).
/// Returns Err if the buffer is truncated or varint exceeds 10 bytes.
pub fn read_varint(data: &[u8], pos: usize) -> Result<(u64, usize), String> {
    let mut result: u64 = 0;
    let mut shift:  u32 = 0;
    let mut p = pos;
    loop {
        if p >= data.len() {
            return Err(format!("Truncated varint at pos {p}"));
        }
        let byte = data[p] as u64;
        p += 1;
        result |= (byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, p));
        }
        shift += 7;
        if shift >= 70 {
            return Err("Varint overflow (>10 bytes)".to_string());
        }
    }
}

/// Quantize confidence [0.0, 1.0] → u8.
#[inline]
pub fn conf_to_q8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Dequantize u8 → f32 confidence.
#[inline]
pub fn q8_to_conf(q: u8) -> f32 {
    q as f32 / 255.0
}

// ── v0.3 legacy helpers (used only by the backward-compatible decoder path) ───

/// v0.3: Pack layer (0-2, 2 bits) and session_count (0-63, 6 bits) into one byte.
#[allow(dead_code)]
#[inline]
pub fn pack_layer_byte_v3(layer: u8, session_count: u8) -> u8 {
    (layer & 0x03) | ((session_count.min(63) & 0x3F) << 2)
}

/// v0.3: Unpack a layer byte → (layer, session_count).
#[inline]
pub fn unpack_layer_byte_v3(b: u8) -> (u8, u8) {
    (b & 0x03, (b >> 2) & 0x3F)
}

// ── v0.4 RLE helpers ──────────────────────────────────────────────────────────

/// Encode a sequence of session_count values (in rule-sorted order) as
/// (count: u8, value: u8) run-length pairs.  Returns the run vector.
pub fn rle_encode_session_counts(counts: &[u8]) -> Vec<(u8, u8)> {
    if counts.is_empty() { return Vec::new(); }
    let mut runs: Vec<(u8, u8)> = Vec::new();
    let mut cur_val = counts[0];
    let mut cur_run: u8 = 1;
    for &v in &counts[1..] {
        if v == cur_val && cur_run < 255 {
            cur_run += 1;
        } else {
            runs.push((cur_run, cur_val));
            cur_val = v;
            cur_run = 1;
        }
    }
    runs.push((cur_run, cur_val));
    runs
}

/// Serialize RLE runs to bytes: u16 n_runs, then (count u8, value u8)*.
pub fn write_session_rle(buf: &mut Vec<u8>, runs: &[(u8, u8)]) {
    let n = runs.len() as u16;
    buf.extend_from_slice(&n.to_le_bytes());
    for (count, value) in runs {
        buf.push(*count);
        buf.push(*value);
    }
}

/// Read the SESSION_RUNS section from `data` starting at `pos`.
/// Returns (session_counts_in_rule_order, new_pos).
pub fn read_session_rle(data: &[u8], pos: usize, rule_count: usize)
    -> Result<(Vec<u8>, usize), String>
{
    if pos + 2 > data.len() {
        return Err(format!("SESSION_RUNS header truncated at pos {pos}"));
    }
    let n_runs = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
    let mut p = pos + 2;

    let mut counts: Vec<u8> = Vec::with_capacity(rule_count);
    for run_i in 0..n_runs {
        if p + 2 > data.len() {
            return Err(format!("SESSION_RUNS run {run_i} truncated at pos {p}"));
        }
        let count = data[p] as usize;
        let value = data[p + 1];
        p += 2;
        for _ in 0..count {
            counts.push(value);
        }
    }

    if counts.len() != rule_count {
        return Err(format!(
            "SESSION_RUNS decoded {} counts but expected {rule_count}",
            counts.len()
        ));
    }
    Ok((counts, p))
}

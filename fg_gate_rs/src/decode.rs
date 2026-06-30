/// Batch decoder — FG binary blob → rules.
///
/// Supports both v0.3 (fixed u16 indices, packed layer byte) and
/// v0.4 (varint indices, SESSION_RUNS RLE section).  Version is
/// determined from the header byte at offset 2.

use crate::format::*;

// ── String table deserialization ──────────────────────────────────────────────

fn read_string_table(data: &[u8], pos: usize) -> Result<(Vec<String>, usize), String> {
    if pos + 2 > data.len() {
        return Err(format!("String table header truncated at pos {pos}"));
    }
    let st_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
    let mut p = pos + 2;
    let end = p + st_len;

    if end > data.len() {
        return Err(format!("String table data truncated: need {end}, have {}", data.len()));
    }

    let mut strings: Vec<String> = Vec::new();
    while p < end {
        let slen = data[p] as usize;
        p += 1;
        if p + slen > end {
            return Err(format!("String entry truncated at pos {p}"));
        }
        let s = std::str::from_utf8(&data[p..p + slen])
            .map_err(|e| format!("Invalid UTF-8 in string table: {e}"))?
            .to_string();
        strings.push(s);
        p += slen;
    }
    Ok((strings, end))
}

// ── Public decoder ────────────────────────────────────────────────────────────

/// Decode a FG binary blob (v0.3 or v0.4) into a Vec of BinaryRule + the layer_id.
pub fn decode_rules(data: &[u8]) -> Result<(Vec<BinaryRule>, u8), String> {
    if data.len() < 8 {
        return Err(format!("Buffer too short for header: {} bytes", data.len()));
    }

    let magic      = u16::from_le_bytes([data[0], data[1]]);
    let version    = data[2];
    let layer_id   = data[3];
    let rule_count = u16::from_le_bytes([data[4], data[5]]) as usize;
    let flags      = u16::from_le_bytes([data[6], data[7]]);

    if magic != MAGIC {
        return Err(format!("Bad magic: {:#06x} (expected {:#06x})", magic, MAGIC));
    }
    if version != VERSION && version != VERSION_V3 {
        return Err(format!(
            "Unsupported version: {version:#04x} (supported: {VERSION:#04x}, {VERSION_V3:#04x})"
        ));
    }

    if rule_count == 0 {
        return Ok((Vec::new(), layer_id));
    }

    let delta_encoded  = flags & FLAG_DELTA_ENCODED != 0;
    let has_ttl        = flags & FLAG_HAS_TTL       != 0;
    let has_session_rle = flags & FLAG_SESSION_RLE  != 0;

    // String table
    let (strings, mut pos) = read_string_table(data, 8)?;

    // ── SESSION_RUNS section (v0.4 only) ──────────────────────────────────────
    let session_counts: Vec<u8> = if has_session_rle {
        let (counts, new_pos) = read_session_rle(data, pos, rule_count)?;
        pos = new_pos;
        counts
    } else {
        // v0.3 path: session_count is inside the per-rule packed byte — handled below
        Vec::new()
    };

    // ── Rule data ─────────────────────────────────────────────────────────────
    if pos + 4 > data.len() {
        return Err("Rule data length field truncated".to_string());
    }
    let rule_data_len = u32::from_le_bytes([
        data[pos], data[pos+1], data[pos+2], data[pos+3]
    ]) as usize;
    pos += 4;

    if pos + rule_data_len > data.len() {
        return Err(format!(
            "Rule data truncated: need {} bytes from pos {pos}, have {}",
            rule_data_len, data.len()
        ));
    }

    let rule_end  = pos + rule_data_len;
    let rule_data = &data[pos..rule_end];
    pos = rule_end;

    let mut rules: Vec<BinaryRule> = Vec::with_capacity(rule_count);
    let mut rp: usize = 0;
    let mut prev_support: i64 = 0;

    for rule_i in 0..rule_count {
        // ── lhs_idx / rhs_idx ─────────────────────────────────────────────────
        let (lhs_idx, rhs_idx) = if has_session_rle {
            // v0.4: both indices are varint
            let (li, p1) = read_varint(rule_data, rp)?;
            let (ri, p2) = read_varint(rule_data, p1)?;
            rp = p2;
            (li as usize, ri as usize)
        } else {
            // v0.3: both indices are fixed u16
            if rp + 4 > rule_data.len() {
                return Err(format!("v0.3 rule record truncated at offset {rp}"));
            }
            let li = u16::from_le_bytes([rule_data[rp], rule_data[rp+1]]) as usize;
            let ri = u16::from_le_bytes([rule_data[rp+2], rule_data[rp+3]]) as usize;
            rp += 4;
            (li, ri)
        };

        if lhs_idx >= strings.len() || rhs_idx >= strings.len() {
            return Err(format!(
                "String index out of range: lhs={lhs_idx} rhs={rhs_idx} table_len={}",
                strings.len()
            ));
        }

        // ── support (varint, possibly delta-zigzag) ───────────────────────────
        let (zz, new_rp) = read_varint(rule_data, rp)?;
        rp = new_rp;

        let support: u32 = if delta_encoded {
            let delta = zigzag_decode(zz);
            let s = prev_support + delta;
            if s < 0 {
                return Err(format!("Negative support after delta decode at rule {rule_i}: {s}"));
            }
            prev_support = s;
            s as u32
        } else {
            prev_support = zz as i64;
            zz as u32
        };

        // ── conf_q8 + optional packed byte (v0.3 only) ───────────────────────
        if rp >= rule_data.len() {
            return Err(format!("conf_q8 truncated at rule {rule_i}, rp={rp}"));
        }
        let conf_q8 = rule_data[rp];
        rp += 1;

        let (session_count, layer) = if has_session_rle {
            // v0.4: session_count from RLE section; layer from header
            (session_counts[rule_i], layer_id)
        } else {
            // v0.3: packed byte follows conf_q8
            if rp >= rule_data.len() {
                return Err(format!("v0.3 packed byte truncated at rule {rule_i}"));
            }
            let packed = rule_data[rp];
            rp += 1;
            let (l, sc) = unpack_layer_byte_v3(packed);
            (sc, l)
        };

        let confidence = q8_to_conf(conf_q8);

        rules.push(BinaryRule {
            lhs: strings[lhs_idx].clone(),
            rhs: strings[rhs_idx].clone(),
            support,
            confidence,
            layer,
            session_count,
            ttl: None,
        });
    }

    // ── TTL records ───────────────────────────────────────────────────────────
    if has_ttl {
        if pos + 2 > data.len() {
            return Err("TTL count truncated".to_string());
        }
        let n_ttl = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
        pos += 2;
        for _ in 0..n_ttl {
            if pos + 3 > data.len() {
                return Err("TTL record truncated".to_string());
            }
            let rule_idx = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
            let ttl_val  = data[pos + 2];
            pos += 3;
            if rule_idx < rules.len() {
                rules[rule_idx].ttl = Some(ttl_val);
            }
        }
    }
    let _ = pos;

    Ok((rules, layer_id))
}

/// Convenience wrapper matching the encode API.
pub fn decode_rules_bytes(data: &[u8]) -> Result<(Vec<BinaryRule>, u8), String> {
    decode_rules(data)
}

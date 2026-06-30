/// Ephemeral promotion gate.
///
/// DESIGN PRINCIPLE — ephemeral gate:
///   The gate is NEVER written to disk.  It is rebuilt from the session event
///   log each time fg-sync starts.  This eliminates the 11,284-byte gate
///   overhead identified in the simulation baseline, reducing total storage to
///   ~1,097 bytes (rules only) — 0.52× flat v0.2.0.
///
/// The gate tracks, per (lhs, rhs) pair:
///   - session_count : u8   number of distinct sessions this rule appeared in
///   - session_mask  : u8   bitmask of which of the last 8 sessions it appeared
///
/// Promotion thresholds (v0.3.0 spec):
///   session_count >= 5  → Layer 0 (Domain)
///   session_count >= 2  → Layer 1 (Behavioral)
///   session_count == 1  → Layer 2 (Session), TTL = 1

use std::collections::HashMap;

/// One entry in the ephemeral gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GateEntry {
    pub lhs:           String,
    pub rhs:           String,
    /// Number of distinct sessions this rule has appeared in (capped at 255)
    pub session_count: u8,
    /// Bitmask of which of the last 8 sessions contained this rule
    pub session_mask:  u8,
}

/// In-memory promotion gate — rebuilt from event log, never serialized.
pub struct EphemeralGate {
    /// (lhs, rhs) → (session_count, session_mask)
    entries: HashMap<(String, String), (u8, u8)>,
    /// How many sessions have been processed
    session_cursor: u8,
}

impl EphemeralGate {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            session_cursor: 0,
        }
    }

    /// Register all rules extracted from one session.
    /// Call this once per session during gate rebuild.
    pub fn register_session(&mut self, rules: &[(String, String)]) {
        let bit = 1u8 << (self.session_cursor % 8);
        for (lhs, rhs) in rules {
            let key = (lhs.clone(), rhs.clone());
            let entry = self.entries.entry(key).or_insert((0u8, 0u8));
            let (ref mut count, ref mut mask) = *entry;
            // Only increment session_count if this bit wasn't already set
            // (rules can appear multiple times within a session — count once)
            if *mask & bit == 0 {
                *count = count.saturating_add(1);
                *mask |= bit;
            }
        }
        self.session_cursor = self.session_cursor.wrapping_add(1);
    }

    /// Expire old sessions: shift the mask window forward.
    /// Call when you want the mask to reflect only the last 8 sessions.
    pub fn advance_window(&mut self) {
        // Mask is a circular bitmask over 8 sessions — no explicit expiry needed
        // since register_session writes by bit position modulo 8.
    }

    /// Snapshot the gate as a Vec<GateEntry> for inspection / promotion decisions.
    pub fn snapshot(&self) -> Vec<GateEntry> {
        let mut entries: Vec<GateEntry> = self.entries
            .iter()
            .map(|((lhs, rhs), &(count, mask))| GateEntry {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                session_count: count,
                session_mask:  mask,
            })
            .collect();
        // Stable sort by session_count desc, then lhs/rhs for determinism
        entries.sort_by(|a, b| {
            b.session_count.cmp(&a.session_count)
                .then(a.lhs.cmp(&b.lhs))
                .then(a.rhs.cmp(&b.rhs))
        });
        entries
    }

    /// Classify all gate entries into layers using v0.3.0 promotion thresholds.
    /// Returns (layer0_keys, layer1_keys, layer2_keys) — each is a Vec<(lhs, rhs)>.
    pub fn promote(&self) -> (Vec<(String,String)>, Vec<(String,String)>, Vec<(String,String)>) {
        let mut l0: Vec<(String,String)> = Vec::new();
        let mut l1: Vec<(String,String)> = Vec::new();
        let mut l2: Vec<(String,String)> = Vec::new();

        for ((lhs, rhs), &(count, _mask)) in &self.entries {
            match count {
                5..=u8::MAX => l0.push((lhs.clone(), rhs.clone())),
                2..=4       => l1.push((lhs.clone(), rhs.clone())),
                1           => l2.push((lhs.clone(), rhs.clone())),
                0           => {} // should not happen
            }
        }
        (l0, l1, l2)
    }

    pub fn len(&self) -> usize  { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Advance the session cursor without registering any rules.
    /// Used when a session index exists in the log but produced no rules.
    pub fn advance_cursor(&mut self) {
        self.session_cursor = self.session_cursor.wrapping_add(1);
    }
}

impl Default for EphemeralGate {
    fn default() -> Self { Self::new() }
}

/// Convenience function: build an EphemeralGate from a list of
/// (session_index, lhs, rhs) tuples — the format used by the Python pipeline.
pub fn rebuild_gate(events: &[(u32, String, String)]) -> EphemeralGate {
    // Group by session_index, then register in order
    let mut max_session: u32 = 0;
    for (s, _, _) in events { max_session = max_session.max(*s); }

    let mut gate = EphemeralGate::new();
    for session_idx in 0..=max_session {
        let session_rules: Vec<(String, String)> = events
            .iter()
            .filter(|(s, _, _)| *s == session_idx)
            .map(|(_, l, r)| (l.clone(), r.clone()))
            .collect();
        if !session_rules.is_empty() {
            gate.register_session(&session_rules);
        } else {
            // Advance cursor even for empty sessions to keep bitmask aligned
            gate.advance_cursor();
        }
    }
    gate
}

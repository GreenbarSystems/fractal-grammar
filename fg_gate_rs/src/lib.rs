// fg_gate — FBG v0.3.0 Binary Encoding Extension
// ================================================
// Provides batch encode/decode for:
//   - Rule records (layers 0, 1, 2) with bit-packed fields, delta-varint support, confidence q8
//   - Ephemeral gate rebuild from session history (no persistence — gate is never written to disk)
//
// Wire format: see format.rs for full spec.
// PyO3 module: see py_module.rs for Python bindings.

mod format;
mod encode;
mod decode;
mod gate;
mod py_module;
#[cfg(test)]
mod tests;

pub use format::{BinaryRule, FgHeader, MAGIC, VERSION};
pub use encode::{encode_rules, encode_rules_bytes};
pub use decode::{decode_rules, decode_rules_bytes};
pub use gate::{EphemeralGate, GateEntry, rebuild_gate};

use pyo3::prelude::*;

/// fg_gate — FBG binary encoding extension (v0.3.0)
#[pymodule]
fn _fg_gate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_module::register(m)
}

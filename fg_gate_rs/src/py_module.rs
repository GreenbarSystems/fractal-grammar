/// PyO3 Python bindings for the FBG binary encoding extension.
///
/// All functions use Python dicts as the interchange format so the Rust
/// extension is a drop-in replacement for the Python encoder — no new
/// Python types, no structural changes to the calling code.
///
/// Exported Python functions:
///   encode_rules(rules: list[dict], layer_id: int) -> bytes
///   decode_rules(data: bytes) -> tuple[list[dict], int]
///   rebuild_gate(events: list[tuple[int, str, str]]) -> list[dict]
///   promote_gate(events: list[tuple[int, str, str]]) -> dict
///   fg_version() -> str

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::format::{BinaryRule, LAYER_DOMAIN, LAYER_BEHAVIORAL, LAYER_SESSION};
use crate::encode::encode_rules as rs_encode;
use crate::decode::decode_rules as rs_decode;
use crate::gate::{rebuild_gate as rs_rebuild, EphemeralGate};

// ── Dict ↔ BinaryRule conversions ─────────────────────────────────────────────

fn dict_to_rule(py: Python<'_>, d: &Bound<'_, PyDict>) -> PyResult<BinaryRule> {
    let get = |key: &str| -> PyResult<PyObject> {
        d.get_item(key)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
            .map(|v| v.into())
    };

    let lhs: String        = get("lhs")?.extract(py)?;
    let rhs: String        = get("rhs")?.extract(py)?;
    let support: u32       = get("support")?.extract(py)?;
    let confidence: f32    = get("confidence")?.extract(py)?;
    let layer: u8          = get("layer")?.extract(py)?;
    let session_count: u8  = get("session_count")?.extract(py)?;
    let ttl: Option<u8>    = d.get_item("ttl")?
        .and_then(|v| if v.is_none() { None } else { Some(v) })
        .map(|v| v.extract::<u8>())
        .transpose()?;

    Ok(BinaryRule { lhs, rhs, support, confidence, layer, session_count, ttl })
}

fn rule_to_dict<'py>(py: Python<'py>, r: &BinaryRule) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("lhs",           &r.lhs)?;
    d.set_item("rhs",           &r.rhs)?;
    d.set_item("support",       r.support)?;
    d.set_item("confidence",    r.confidence)?;
    d.set_item("layer",         r.layer)?;
    d.set_item("session_count", r.session_count)?;
    match r.ttl {
        Some(t) => d.set_item("ttl", t)?,
        None    => d.set_item("ttl", py.None())?,
    }
    Ok(d)
}

// ── Python-exported functions ─────────────────────────────────────────────────

/// Encode a list of rule dicts to FG v0.3 binary.
///
/// Args:
///     rules (list[dict]): each dict must have keys:
///         lhs (str), rhs (str), support (int), confidence (float),
///         layer (int 0-2), session_count (int 0-63)
///         ttl (int | None) — optional, used for layer 2 only
///     layer_id (int): 0=Domain, 1=Behavioral, 2=Session
///
/// Returns:
///     bytes: FG v0.3 binary blob
#[pyfunction]
pub fn encode_rules<'py>(
    py: Python<'py>,
    rules: &Bound<'_, PyList>,
    layer_id: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let rs_rules: Vec<BinaryRule> = rules
        .iter()
        .map(|item| {
            let d = item.downcast::<PyDict>()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    "Each rule must be a dict"
                ))?;
            dict_to_rule(py, d)
        })
        .collect::<PyResult<_>>()?;

    let bytes = rs_encode(&rs_rules, layer_id);
    Ok(PyBytes::new(py, &bytes))
}

/// Decode FG v0.3 binary to a list of rule dicts.
///
/// Args:
///     data (bytes): FG v0.3 binary blob
///
/// Returns:
///     tuple[list[dict], int]: (rules, layer_id)
///
/// Raises:
///     ValueError: on bad magic, unsupported version, or truncated data
#[pyfunction]
pub fn decode_rules<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<Bound<'py, PyTuple>> {
    let (rules, layer_id) = rs_decode(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let py_rules = PyList::new(
        py,
        rules.iter()
            .map(|r| rule_to_dict(py, r))
            .collect::<PyResult<Vec<_>>>()?,
    )?;

    let result = PyTuple::new(py, &[py_rules.into_any(), layer_id.into_pyobject(py)?.into_any()])?;
    Ok(result)
}

/// Rebuild the ephemeral promotion gate from a session event log.
///
/// The gate is never persisted to disk — it is rebuilt at startup from the
/// existing session event log.  This eliminates the ~11 KB gate overhead in
/// the JSON baseline.
///
/// Args:
///     events: list of (session_index: int, lhs: str, rhs: str) tuples
///             representing every rule pair seen in every session.
///
/// Returns:
///     list[dict]: gate snapshot, each dict has:
///         lhs (str), rhs (str), session_count (int), session_mask (int)
///         sorted by session_count descending.
#[pyfunction]
pub fn rebuild_gate<'py>(
    py: Python<'py>,
    events: &Bound<'_, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let rs_events: Vec<(u32, String, String)> = events
        .iter()
        .map(|item| {
            let t = item.downcast::<PyTuple>()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    "Each event must be a (session_idx, lhs, rhs) tuple"
                ))?;
            if t.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Event tuple must have 3 elements: (session_idx, lhs, rhs)"
                ));
            }
            let session_idx: u32 = t.get_item(0)?.extract()?;
            let lhs: String      = t.get_item(1)?.extract()?;
            let rhs: String      = t.get_item(2)?.extract()?;
            Ok((session_idx, lhs, rhs))
        })
        .collect::<PyResult<_>>()?;

    let gate = rs_rebuild(&rs_events);
    let snapshot = gate.snapshot();

    let py_entries = PyList::new(
        py,
        snapshot.iter().map(|e| {
            let d = PyDict::new(py);
            d.set_item("lhs",           &e.lhs).unwrap();
            d.set_item("rhs",           &e.rhs).unwrap();
            d.set_item("session_count", e.session_count).unwrap();
            d.set_item("session_mask",  e.session_mask).unwrap();
            d
        }).collect::<Vec<_>>(),
    )?;

    Ok(py_entries)
}

/// Rebuild the gate AND return per-layer promotion results.
///
/// Args:
///     events: list of (session_index, lhs, rhs) tuples
///     l0_threshold (int): min sessions for Domain layer (default 5)
///     l1_threshold (int): min sessions for Behavioral layer (default 2)
///
/// Returns:
///     dict with keys:
///         "layer0": list of (lhs, rhs) str tuples
///         "layer1": list of (lhs, rhs) str tuples
///         "layer2": list of (lhs, rhs) str tuples
///         "gate":   full gate snapshot as list[dict]
///         "stats":  dict with "total", "l0", "l1", "l2", "quarantined"
#[pyfunction]
#[pyo3(signature = (events, l0_threshold=5, l1_threshold=2))]
pub fn promote_gate<'py>(
    py: Python<'py>,
    events: &Bound<'_, PyList>,
    l0_threshold: u8,
    l1_threshold: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let rs_events: Vec<(u32, String, String)> = events
        .iter()
        .map(|item| {
            let t = item.downcast::<PyTuple>()?;
            let session_idx: u32 = t.get_item(0)?.extract()?;
            let lhs: String      = t.get_item(1)?.extract()?;
            let rhs: String      = t.get_item(2)?.extract()?;
            Ok((session_idx, lhs, rhs))
        })
        .collect::<PyResult<_>>()?;

    // Build gate with custom thresholds
    let mut gate = EphemeralGate::new();
    let max_session = rs_events.iter().map(|(s,_,_)| *s).max().unwrap_or(0);
    for session_idx in 0..=max_session {
        let session_rules: Vec<(String, String)> = rs_events.iter()
            .filter(|(s,_,_)| *s == session_idx)
            .map(|(_,l,r)| (l.clone(), r.clone()))
            .collect();
        if !session_rules.is_empty() {
            gate.register_session(&session_rules);
        } else {
            gate.advance_cursor();
        }
    }

    let snapshot = gate.snapshot();

    // Classify with supplied thresholds
    let mut l0_pairs: Vec<Bound<'py, PyTuple>> = Vec::new();
    let mut l1_pairs: Vec<Bound<'py, PyTuple>> = Vec::new();
    let mut l2_pairs: Vec<Bound<'py, PyTuple>> = Vec::new();
    let mut quarantined: usize = 0;

    for e in &snapshot {
        match e.session_count {
            n if n >= l0_threshold                           => l0_pairs.push(PyTuple::new(py, &[&e.lhs, &e.rhs])?),
            n if n >= l1_threshold && n < l0_threshold      => l1_pairs.push(PyTuple::new(py, &[&e.lhs, &e.rhs])?),
            1                                                => l2_pairs.push(PyTuple::new(py, &[&e.lhs, &e.rhs])?),
            _                                                => { quarantined += 1; }
        }
    }

    // Gate snapshot list
    let py_gate = PyList::new(
        py,
        snapshot.iter().map(|e| {
            let d = PyDict::new(py);
            d.set_item("lhs",           &e.lhs).unwrap();
            d.set_item("rhs",           &e.rhs).unwrap();
            d.set_item("session_count", e.session_count).unwrap();
            d.set_item("session_mask",  e.session_mask).unwrap();
            d
        }).collect::<Vec<_>>(),
    )?;

    // Stats
    let stats = PyDict::new(py);
    stats.set_item("total",       snapshot.len())?;
    stats.set_item("l0",          l0_pairs.len())?;
    stats.set_item("l1",          l1_pairs.len())?;
    stats.set_item("l2",          l2_pairs.len())?;
    stats.set_item("quarantined", quarantined)?;

    let result = PyDict::new(py);
    result.set_item("layer0", PyList::new(py, l0_pairs)?)?;
    result.set_item("layer1", PyList::new(py, l1_pairs)?)?;
    result.set_item("layer2", PyList::new(py, l2_pairs)?)?;
    result.set_item("gate",   py_gate)?;
    result.set_item("stats",  stats)?;

    Ok(result)
}

/// Return the extension version string.
#[pyfunction]
pub fn fg_version() -> &'static str {
    "fg_gate 0.4.0 (Rust)"
}

// ── Module registration ───────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_rules, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rules, m)?)?;
    m.add_function(wrap_pyfunction!(rebuild_gate, m)?)?;
    m.add_function(wrap_pyfunction!(promote_gate, m)?)?;
    m.add_function(wrap_pyfunction!(fg_version,   m)?)?;

    // Expose layer constants
    m.add("LAYER_DOMAIN",     LAYER_DOMAIN)?;
    m.add("LAYER_BEHAVIORAL", LAYER_BEHAVIORAL)?;
    m.add("LAYER_SESSION",    LAYER_SESSION)?;
    m.add("MAGIC",            crate::format::MAGIC)?;
    m.add("VERSION",          crate::format::VERSION)?;

    Ok(())
}

# Known Limitations — fg-sync v0.1.0

These are documented honestly because power users trust projects that don't hide their rough edges.

## L1 — AssociativeMemory cluster keys are hash-derived, not real HDBSCAN labels

**What this means**: After HDBSCAN clustering, each cluster's centroid is keyed in AssociativeMemory using a hash derived from the projected vector, not the actual HDBSCAN integer label. The two systems (clustering, AM) are not tightly coupled end-to-end.

**Impact**: Retrieval still works (the AM finds nearest neighbors by cosine similarity), but cluster provenance is approximate. A single behavioral event may be stored under a slightly different key than its true cluster centroid.

**Fix target**: v0.3.0 — per-record HDBSCAN label assignment fed directly into AssociativeMemory.write().

---

## L2 — Clustering runs on float32 projected vectors, not HDC-space

**What this means**: The pipeline projects events into float32 space (via HashProjection or sentence-transformer embeddings) for HDBSCAN, then separately encodes into HDC bipolar hypervectors for AssociativeMemory. These are two separate representations — HDBSCAN does not operate in the HDC space.

**Impact**: Cluster boundaries found by HDBSCAN do not perfectly align with HDC cosine neighborhoods. This is suboptimal — true HDC-space clustering would use Hamming distance or bipolar cosine directly.

**Fix target**: v0.3.0 — HDBSCAN replaced with HDC-native nearest-neighbor clustering.

---

## L3 — Encoding speed: 750 ev/s (HDC) vs 13,000 ev/s (HashProjection)

**What this means**: The two-layer HDC encoder (75% semantic bundle + 25% bigram bind) runs at ~750 events/second on a CPU. HashProjection runs at ~13,000 ev/s — 17x faster.

**Impact**: At the default min_events_to_run=50, the pipeline completes in under 0.1s. At n=10,000 events (stress test scale), pipeline takes ~13 seconds. Nightly batch is fine. Real-time encoding is not viable.

**Workaround**: `use_hdc = false` in fg-sync.toml switches to HashProjection (faster, but no AssociativeMemory support — all metrics except M3 still work).

---

## L4 — Pattern retrieval accuracy: 75% on 4-cluster held-out queries

**What this means**: In stress testing, AssociativeMemory correctly retrieved the right cluster for 3/4 held-out topic queries. The 4th topic (cross-topic similarity) was confused at the default threshold=0.05.

**Impact**: 1 in 4 behavioral clusters may be slightly mischaracterized in the generated ruleset. In practice, misidentified patterns contribute low-weight rules that the token budget cap naturally suppresses.

**Workaround**: Lower `assoc_memory_threshold` to 0.03 for more selective retrieval (higher precision, lower recall). Raise to 0.08 for more recall (may introduce noise).

**Fix target**: v0.2.0 — improved two-layer encoding ratio tuning; v0.3.0 — per-cluster threshold calibration.

---

## L5 — Windows support is untested

**What this means**: fg-proxy and the CLI are written for macOS/Linux. Windows paths (`%LOCALAPPDATA%\Ollama\server.log`, `%APPDATA%\open-webui`) are handled in config but the proxy and daemon have not been tested on Windows.

**Known Windows-specific issues**:
- SIGHUP (hot-reload) is not available on Windows — injector must be restarted to pick up new ruleset
- launchd/systemd service files are Unix-only (`contrib/fg-sync.plist`, `contrib/fg-sync.service`)

**Workaround**: Windows users should use `fg-sync sync` manually or schedule via Windows Task Scheduler.

**Fix target**: v0.2.0 — Windows CI pipeline + SIGHUP fallback via file watcher.

---

## L6 — Response quality improvement is not yet measured

**What this means**: M1–M5 metrics measure proxy behavior and storage compression. They do not measure whether the injected behavioral context actually improves Ollama response quality, reduces hallucinations, or increases task success rate.

**Why**: Automated response quality evaluation requires a ground truth dataset and an eval harness — neither is included in v0.1.0.

**Fix target**: v0.2.0 — reference eval harness using ROUGE/BERTScore on re-rolled vs accepted responses.

---

*Last updated: 2026-06-28 | fg-sync v0.1.0*

# fg-sync

**Persistent behavioral memory for local LLMs. No cloud. No fine-tuning.**

fg-sync is a CLI sidecar for [Ollama](https://github.com/ollama/ollama) that builds a compressed behavioral ruleset from your local conversation history and injects it as a system prompt prefix on every request — giving your local model persistent memory of *how* you interact without touching model weights.

```
Client → fg-proxy :11435 → Ollama :11434
         (captures)
capture.jsonl → fractal-grammar pipeline → ruleset.json → system prompt prefix
```

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![GitHub Stars](https://img.shields.io/github/stars/ryandmoore1976/fractal-grammar)](https://github.com/ryandmoore1976/fractal-grammar)

---

## Why fg-sync?

Every Ollama session starts cold. You re-explain your stack, your domain, your preferred response style — every time. Naive solutions (pasting a system prompt, RAG over chat history) either consume your context window or require external infrastructure.

fg-sync compresses your behavioral patterns using **fractal grammar extraction** + **hyperdimensional computing (HDC)** into a minimal ruleset that fits in ~400 tokens. The model learns *your behavioral grammar*, not a transcript of your conversations.

### Measured results (from stress testing on fractal-grammar v0.2.0)

| Metric | Value |
|---|---|
| AssociativeMemory footprint at n=10,000 events | **39 KB** (flat — O(n_clusters), not O(n_events)) |
| Compression ratio vs raw conversation history | **~82:1** |
| Crossover point (where HDC beats flat storage) | ~300–400 events |
| Encoding speed | 750 events/sec |
| Pattern retrieval accuracy (held-out queries) | 75% (4-cluster, tuned threshold=0.05) |

> **What you can claim**: storage is ~82x smaller than raw JSONL; AssociativeMemory footprint is flat regardless of conversation count.  
> **What is not yet measured**: response quality improvement, hallucination rate reduction. These are on the roadmap.

---

## How It Works

### The capture problem (and why `~/.ollama/logs/server.log` doesn't help)

`~/.ollama/logs/server.log` is a **diagnostic log** — it contains HTTP metadata, model load events, and errors. It does not contain conversation payloads. fg-sync solves this with a thin HTTP proxy:

**Path A (default)**: fg-proxy intercepts every `POST /api/chat` call on port 11435 before forwarding to Ollama on 11434. Zero dependencies. Works with any Ollama client.

**Path B**: Read directly from Open WebUI's SQLite database at `~/.local/share/open-webui/webui.db`. No port change needed if you already run Open WebUI.

### The pipeline

```
capture.jsonl
    │
    ▼
MinHash LSH novelty gate (cosine dissimilarity threshold = 0.92)
    │  deduplicated events
    ▼
HDC encoder (D=10,000, seed=0xFEEDBEEF)
    │  bipolar hypervectors
    │  75% semantic bundle + 25% structural bigram bind
    ▼
HDBSCAN clustering (min_cluster_size=5)
    │  behavioral clusters
    ▼
Fractal grammar extraction
    │  behavioral rules per cluster
    ▼
AssociativeMemory write (threshold=0.05)
    │
    ▼
ruleset.json → system prompt prefix (≤400 tokens)
```

### System prompt injection

fg-proxy reads `ruleset.json` at startup and on every `SIGHUP`. On each `POST /api/chat`, it prepends the `prompt_prefix` to the `system` field. Your existing system prompt (if any) is preserved — the prefix is prepended, never replaced. The `messages[]` array is never modified.

---

## Installation

```bash
# One-line install (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/ryandmoore1976/fractal-grammar/main/fg-sync/install.sh | bash

# Or via pip
pip install fg-sync[pipeline]
```

Requires Python 3.11+. Ollama must be running separately.

---

## Quick Start

```bash
# 1. Initialize config
fg-sync init

# 2. Start proxy + scheduler (blocks — use screen/tmux or system service)
fg-sync run

# 3. Point your Ollama client at port 11435 instead of 11434
#    Example with curl:
curl http://localhost:11435/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "hello"}]
}'

# 4. After 50+ conversations, run a manual sync (or wait for nightly 2am UTC)
fg-sync sync

# 5. Check state
fg-sync status

# 6. View performance metrics
fg-sync metrics compare
fg-sync metrics storage
```

---

## Configuration (fg-sync.toml)

Default location: `~/.fg-sync/fg-sync.toml`

```toml
[proxy]
listen_port = 11435       # point your client here
ollama_port = 11434       # where Ollama is running

[pipeline]
schedule = "0 2 * * *"           # nightly 2am UTC
min_events_to_run = 50           # minimum captures before pipeline runs
novelty_threshold = 0.92         # MinHash dedup aggressiveness
hdc_dimensions = 10000           # HDC dimensionality (do not change after first sync)
hdc_seed = 4277009102            # 0xFEEDBEEF — HDC random seed
min_cluster_size = 5             # HDBSCAN parameter
assoc_memory_threshold = 0.05    # AssociativeMemory retrieval threshold

[ruleset]
max_prompt_tokens = 400          # system prompt token budget
recency_weight = 0.7             # weight for recent patterns
decay_days = 30                  # pattern decay period

[source]
type = "proxy"                   # "proxy" (default) or "openwebui"
```

---

## CLI Reference

```
fg-sync run                   Start proxy + scheduler daemon
fg-sync run --no-proxy        Run scheduler only (no HTTP proxy)
fg-sync run --source openwebui  Use Open WebUI DB as data source

fg-sync sync                  One-shot pipeline run
fg-sync sync --dry-run        Preview pipeline without writing files

fg-sync status                Show current ruleset and injector state

fg-sync metrics compare       M1–M5 comparison: baseline vs fg-sync
fg-sync metrics storage       M3 storage footprint + compression ratio

fg-sync export --format txt   Output system prompt prefix as plain text
fg-sync export --format json  Output full ruleset.json

fg-sync init                  Generate default config
fg-sync reset                 Clear all fg-sync state (capture, ruleset, memory)
```

---

## Ruleset Schema (ruleset.json)

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-06-28T02:00:00Z",
  "event_count": 847,
  "session_count": 23,
  "behavioral_rules": [
    {
      "id": "rule_001",
      "cluster_label": "accounting_erp_queries",
      "weight": 0.91,
      "recency_score": 0.88,
      "pattern_summary": "...",
      "trigger_terms": ["journal entry", "AP", "ERP"],
      "hdc_centroid_hex": "a3f2...d901",
      "first_seen": "2026-05-01T00:00:00Z",
      "last_seen": "2026-06-27T18:32:00Z",
      "event_count": 142
    }
  ],
  "prompt_prefix": "## Behavioral Context (fg-sync)\n..."
}
```

---

## Performance Metrics (M1–M5)

fg-sync includes a built-in metrics suite to demonstrate measurable improvement over baseline Ollama sessions.

| ID | Metric | How Measured |
|---|---|---|
| M1 | Clarification Rate | % of user turns correcting the model |
| M2 | Prefix Token Overhead | Tokens consumed by injected prefix |
| M3 | Storage Compression | ruleset.json + assoc_memory.pkl vs raw capture.jsonl |
| M4 | Re-Roll Rate | % of assistant responses regenerated |
| M5 | Context Re-Establishment Cost | Tokens spent re-explaining background per session |

```bash
fg-sync metrics storage
# ┌─────────────────────────────────────────┐
# │ capture.jsonl   →   4.200 MB            │
# │ ruleset.json    →   12 KB               │
# │ assoc_memory    →   39 KB               │
# │ Compression     →   82:1                │
# └─────────────────────────────────────────┘
```

---

## System Services

### macOS (launchd)
```bash
cp contrib/fg-sync.plist ~/Library/LaunchAgents/com.fractalgrammar.fg-sync.plist
launchctl load ~/Library/LaunchAgents/com.fractalgrammar.fg-sync.plist
```

### Linux (systemd)
```bash
mkdir -p ~/.config/systemd/user
cp contrib/fg-sync.service ~/.config/systemd/user/fg-sync.service
systemctl --user enable --now fg-sync
```

---

## Known Limitations (v0.1.0)

1. **AssociativeMemory cluster keys are hash-derived, not real HDBSCAN labels** — per-record label assignment is planned for v0.3.0
2. **Clustering runs on float32 projected vectors, not HDC-space** — full HDC-space clustering is a v0.3.0 target
3. **Encoding speed**: 750 ev/s (HDC) vs 13,000 ev/s (HashProjection) — suitable for nightly batch, not real-time
4. **Pattern retrieval accuracy**: 75% on 4-topic held-out queries — requires ~5-10 writes per cluster topic to stabilize
5. **Windows**: `%LOCALAPPDATA%\Ollama\server.log` path supported; proxy tested on macOS/Linux only

---

## Architecture

See [FG_SYNC_ARCHITECTURE.md](../FG_SYNC_ARCHITECTURE.md) for the full component specification.

---

## Roadmap

- **v0.1.0** — Core: proxy, pipeline integration, CLI, metrics (this release)
- **v0.2.0** — Per-record HDBSCAN label assignment; HDC-space clustering
- **v0.3.0** — Windows support; multi-model ruleset routing
- **Pro** — Dashboard UI, multi-model routing, GitHub Sponsors

---

## Related

- [fractal-grammar](https://github.com/ryandmoore1976/fractal-grammar) — the underlying extraction library
- [Ollama](https://github.com/ollama/ollama) — local LLM runner
- [Open WebUI](https://github.com/open-webui/open-webui) — supported as alternative data source

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Built by [Ryan Moore](https://github.com/ryandmoore1976) | Tempe, Arizona

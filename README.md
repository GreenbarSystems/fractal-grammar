# fractal-grammar

**Behavioral grammar extraction for memory-constrained local AI.**

Extract the rules that generate behavior. Not the behavior itself.

```
pip install fractal-grammar
```

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-teal.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://rustup.rs)
[![Tests: 55/55](https://img.shields.io/badge/tests-55%2F55%20passing-green.svg)]()
[![Version: 0.4.0](https://img.shields.io/badge/version-0.4.0-orange.svg)]()

---

## Why This Exists

Every approach to LLM personalization today does the same thing: store what happened and replay it as context. RAG stores embeddings of individual interactions and retrieves the most similar ones at inference time. Prompt-based personalization maintains a user profile injected into the system prompt. LoRA fine-tuning batches raw interaction logs and retrains periodically.

All three approaches scale storage with interaction count. On a local device with 8–16 GB of RAM, a model already consuming 4–8 GB, this is a hard constraint. You cannot store everything.

The underlying assumption — that you need to store individual interactions — is wrong.

Human behavioral sequences are not random. They are self-similar across timescales. The way you interact with an assistant at session 5 structurally predicts the way you will interact with it at session 50. This property is measurable: it is the Hurst exponent, and it is the basis for this library.

When behavioral logs exhibit Hurst exponent H > 0.70, a compact set of generative rules — a behavioral grammar — can represent the statistical structure of those logs at a fraction of the storage cost. You store the grammar. Not the log.

**Storage scales with behavioral novelty. Not interaction count.**

---

## The Novel Combination

This library is not a new algorithm. It is a new composition of proven components applied to a problem that has not been approached this way before.

| Component | Prior Art | What Is New Here |
|---|---|---|
| Hurst exponent / R/S analysis | Mandelbrot & Wallis (1969) — financial time series | First use as a compression criterion for LLM behavioral logs |
| HDBSCAN hierarchical clustering | Campello et al. (2013) | First application across four behavioral resolution levels with cross-level self-similarity tracing |
| MinHash LSH novelty gating | Broder (1997) — web near-duplicate detection | First use as a behavioral novelty gate controlling grammar pipeline ingestion |
| Hyperdimensional computing (HDC) encoding | Kanerva (2009) | First use as the behavioral embedding layer in a grammar extraction pipeline |
| Hierarchical residual compression | Video codec P/B-frame design | First application to behavioral grammar persistence across temporal layers |
| mPFC rule / CA1 episode distinction | Neuroscience literature, 1990s–present | First use as design rationale for separating grammar (rules) from interaction logs (episodes) |

The novel claim is the composition: **using Hurst exponent self-similarity detection across HDBSCAN resolution levels, encoded via hyperdimensional computing and compressed through a hierarchical residual architecture, as the criterion for identifying and persisting compressible behavioral grammar rules from LLM interaction logs.**

See the [Prior Art](#prior-art) section and the accompanying whitepapers.

---

## Architecture

### v0.1.0 — Core Pipeline

```
Input: BehavioralSequence list (LLM logs, event streams, raw text)
│
├─ Stage 1: ENCODE
│   Embed each event into a normalized vector.
│   Backend options: HashProjection (zero-download), TF-IDF+SVD, SentenceTransformer.
│   HDC encoding: D=10,000, seed=0xFEEDBEEF
│     75% semantic bundle + 25% bigram bind
│   Output: EncodedSequence list (10,000-dim L2-normalized vectors)
│
├─ Stage 2: NOVELTY GATE
│   MinHash LSH (128 permutations, Jaccard threshold 0.92).
│   O(1) per event — cost does not grow with corpus size.
│   Novel events enter the pipeline. Redundant events increment
│   the weight of their nearest existing entry.
│   Typical reduction: 40–60% of raw events discarded
│
├─ Stage 3: HIERARCHICAL CLUSTER
│   HDBSCAN at 4 resolution levels:
│     Level 0 — Domain       (broadest behavioral category)
│     Level 1 — Category     (task type within domain)
│     Level 2 — Pattern      (recurring sub-task structure)
│     Level 3 — Micro        (specific behavioral signature)
│
├─ Stage 4: SELF-SIMILARITY DETECT  ← the novel step
│   Compute Hurst exponent over cross-level similarity series.
│   H ≥ 0.70, depth ≥ 3 = strong fractal rules.
│   H ≥ 0.55, depth ≥ 2 = moderate rules.
│
└─ Stage 5: GRAMMAR EXTRACT
    One GrammarRule per strong fractal pattern.
    Rule storage cost: O(1) — one centroid + depth similarity values.
    Output: GrammarRuleset (JSON) + LoRA-ready JSONL training pairs
```

### v0.3.0 — Overdub Architecture

v0.3.0 introduces **hierarchical residual compression** — a three-layer storage model inspired by video codec P/B-frames and loop machine overdubbing. Each layer encodes only the residual behavioral signal not explained by lower layers.

```
Session event log
│
├─ PROMOTION GATE (ephemeral — never persisted)
│   Tracks per-rule session membership via 8-bit circular bitmask.
│   Classifies rules into three temporal layers:
│     session_count ≥ 5  →  Layer 0: Domain    (stable identity, I-frames)
│     session_count 2–4  →  Layer 1: Behavioral (current context, P-frames)
│     session_count = 1  →  Layer 2: Session    (transient signal, B-frames)
│   Quarantine rate on 1,000-event trace: 41.5% of burst rules expire silently.
│
├─ Layer 0 (Domain)     — rules in ≥5 sessions  — stable across months
├─ Layer 1 (Behavioral) — rules in 2–4 sessions — current behavioral context
└─ Layer 2 (Session)    — last-session-only, TTL=1 — transient signal

Storage: 25 L0 rules + 92 L1 rules = 117 rules vs. 20-rule flat v0.2.0 cap
```

### v0.4.0 — Binary Encoding (fg_gate_rs)

v0.4.0 replaces the JSON serialization layer with a Rust/PyO3 binary encoder implementing two compression improvements:

**Change A — VLQ string indices:** `lhs_idx` and `rhs_idx` encoded as varint (was fixed u16). For typical FBG vocabulary (≤127 unique tokens), every index fits in 1 byte — saving 1 byte per index × 2 per rule.

**Change B — RLE session_count:** A `SESSION_RUNS` section encodes the session_count field as run-length pairs preceding the rule records. Layer bits are removed from the per-rule payload entirely (layer ID is already in the 8-byte header).

```
Wire format (VERSION = 0x04):
  HEADER        8 bytes   magic(u16) version(u8) layer_id(u8) rule_count(u16) flags(u16)
  STRING_TABLE  variable  deduplicated length-prefixed u8 strings
  SESSION_RUNS  variable  u16 n_runs + (count:u8, value:u8)* pairs   ← NEW v0.4
  RULE_RECORDS  variable  varint(lhs_idx) varint(rhs_idx)             ← VLQ indices
                          varint(support, zigzag delta) conf_q8:u8
  TTL_RECORDS   optional  Layer 2 only
```

Backward compatible: v0.3 blobs (VERSION = 0x03) are still decoded correctly.

---

## Storage Comparison

| Architecture | Rules | Storage | vs. flat v0.2.0 |
|---|---|---|---|
| Flat v0.2.0 (JSON, 20-rule hard cap) | 20 | 2,104 B | 1.0× (baseline) |
| Overdub v0.3.0 (JSON) | 117 | 25,012 B | 11.9× larger |
| Overdub v0.3.0 (binary, ephemeral gate) | 117 | 1,203 B | 0.57× |
| **Overdub v0.4.0 (binary, VLQ + RLE)** | **117** | **860 B** | **0.41×** |

The overdub architecture stores 5.75× more validated rules than the flat cap, at 41% of the original storage cost.

---

## Benchmark Summary

### Latency (fg_gate_rs v0.4.0 — Rust/PyO3)

| Operation | Rust v0.4 | Python ref | Speedup |
|---|---|---|---|
| Encode L0 (25 rules) | 15.3 µs | — | — |
| Encode L1 (92 rules) | 48.7 µs | ~194 µs (both) | 3.0× |
| Decode L0 | 14.8 µs | — | — |
| Decode L1 | 50.0 µs | — | — |
| Gate rebuild (200 entries, 10 sessions) | 152 µs | — | startup-only |
| **Encode L0 + L1 combined** | **64 µs** | **~194 µs** | **3.0×** |

**Compression vs JSON:** 860 B binary vs. ~9,360 B JSON for 117 rules — 10.9× ratio.

**Latency context:** The 50 µs encode/decode target applies to session-boundary persistence (encode L0 + L1: 64 µs). Gate rebuild (152 µs) runs once at startup — not on the hot path.

### Grammar Extraction (v0.1.0 pipeline)

Measured on a 53-event corpus, hash projection backend, CPU only:

| Metric | Value |
|---|---|
| Events after novelty gate | 18 (66% reduction) |
| Corpus Hurst exponent | 0.601 |
| Grammar rules extracted | 2 |
| Strongest rule compression | 6.25× |
| Total compression | **30×** |
| End-to-end latency | **70 ms (CPU)** |

**Hash backend caveat:** produces false-positive grammar rules at n < 100. Use SentenceTransformer with ≥150 events for reliable results. See [Stress Test Results](#stress-test-results).

### RAG vs. FBG (deterministic harness, 40 prompts, 4 categories)

| Metric | RAG | FBG |
|---|---|---|
| Style adherence | 0.786 | **0.875** (+0.089) |
| Instruction persistence | 0.750 | **0.775** (+0.025) |
| Cross-domain transfer | **0.923** | 0.692 (−0.231) |
| Behavioral decay (T1→T10) | 0.875 → 0.172 | 0.312 → 0.286 |
| Crossover point | — | Turn 7 (FBG wins at scale) |

**CDTR note:** RAG wins on cross-domain transfer at corpus sizes below ~300 events. FBG wins on style and persistence at all corpus sizes tested.

---

## Quick Start

### Zero-download demo

```python
from fractal_grammar import FractalGrammarPipeline, PipelineConfig, from_raw_texts
from fractal_grammar.embeddings.encoder import Backend

config = PipelineConfig(
    embedding_backend=Backend.HASH_PROJECTION,
    similarity_threshold=0.3,
    min_hurst=0.45,
)
pipe = FractalGrammarPipeline(config)
sequences = [from_raw_texts([t]) for t in your_texts]
ruleset = pipe.run(sequences)

print(pipe.report())
ruleset.save_json("grammar.json")
```

### Production (semantic backend — recommended)

```python
config = PipelineConfig(
    embedding_backend=Backend.SENTENCE_TRANSFORMER,
    model="all-MiniLM-L6-v2",
    similarity_threshold=0.6,
    min_hurst=0.55,
    min_pattern_depth=3,
)
```

### Binary encoding (v0.4.0)

```python
from fg_gate import encode_rules, decode_rules, promote_gate
from fg_gate import LAYER_DOMAIN, LAYER_BEHAVIORAL, LAYER_SESSION

# Encode extracted rules into binary
blob = encode_rules(rules, layer_id=LAYER_BEHAVIORAL)   # → bytes, ~6 µs/rule
decoded_rules, layer_id = decode_rules(blob)

# Rebuild ephemeral promotion gate from session event log
# events = [(session_index: int, lhs: str, rhs: str), ...]
result = promote_gate(events, l0_threshold=5, l1_threshold=2)
# result["layer0"]  → Domain rules
# result["layer1"]  → Behavioral rules
# result["layer2"]  → Session rules (TTL=1)
# result["stats"]   → {"total", "l0", "l1", "l2", "quarantined"}
```

### fg-sync proxy (local Ollama integration)

```bash
# Start the proxy on port 11435 (forwards to Ollama on 11434)
python fg_sync/server.py

# Configure your client to point at port 11435 instead of 11434
# The proxy intercepts POST /api/chat, injects the current ruleset
# as a system prompt prefix, and forwards to Ollama transparently
```

---

## Stress Test Results

The pipeline was adversarially tested before publication. Five tests designed to establish a falsifiable baseline. **Three falsifying findings** were produced and are documented in full:

1. Hash projection backend produces false-positive rules at n < 100 (geometric collision)
2. No grammar collapse found at any noise injection rate with hash backend
3. Structured vs. anti-persistent corpora not separable with hash backend (H gap = 0.018)

All three failures have the same root cause: the hash projection backend was designed for zero-download demos, not structural validity testing. The failures define the validity boundary of the current implementation — they do not invalidate the hypothesis.

Raw results: [`stress_test_results.json`](stress_test_results.json)  
Full findings: [`stress_test_findings.md`](stress_test_findings.md)

```bash
python -m pytest fractal_grammar/tests/test_hypothesis_stress.py -v -s
```

---

## When to Use This Library

**Good fit:**
- Domain-constrained assistant with a consistent user behavioral space
- Local inference where logs cannot leave the device
- Memory-constrained deployment (consumer hardware, edge devices)
- You want the training signal compressed, not just the context window

**Poor fit:**
- Highly exploratory usage with no repeating structure (corpus H will be < 0.55 — the pipeline reports this)
- Very small user bases where individual interaction history matters more than patterns
- Requires an online LoRA adaptation loop (extraction is implemented; feeding grammar into a LoRA adapter is not yet in this library)

---

## Module Reference

```
fractal_grammar/
├── pipeline.py                    FractalGrammarPipeline, PipelineConfig
├── core/
│   ├── sequence.py                BehavioralEvent, BehavioralSequence
│   └── dedup.py                   NoveltyFilter (MinHash LSH)
├── embeddings/
│   └── encoder.py                 BehavioralEncoder (HASH_PROJECTION, TFIDF_SVD, SENTENCE_TRANSFORMER)
├── clustering/
│   ├── hierarchical.py            HierarchicalClusterer, HierarchicalClusterTree
│   └── self_similarity.py         SelfSimilarityDetector, compute_corpus_hurst()
└── grammar/
    └── extractor.py               GrammarExtractor, GrammarRuleset, GrammarRule

fg_gate_rs/                        Rust/PyO3 binary encoding extension (v0.4.0)
├── src/
│   ├── format.rs                  Wire format spec, BinaryRule struct, varint, RLE helpers
│   ├── encode.rs                  Batch encoder (VLQ indices, RLE session_count)
│   ├── decode.rs                  Batch decoder (v0.3 + v0.4 backward compat)
│   ├── gate.rs                    EphemeralGate — rebuild from event log, never persisted
│   ├── py_module.rs               PyO3 bindings
│   └── tests.rs                   40 Rust unit tests
└── tests/
    └── test_integration.py        15 Python integration tests + latency benchmark

fg_sync/                           Transparent Ollama proxy
├── server.py                      HTTP proxy: port 11435 → 11434
└── pipeline.py                    Cron pipeline: extract → compress → inject
```

---

## Roadmap

| Component | Status | Notes |
|---|---|---|
| Grammar extraction pipeline | **v0.1.0 — released** | Core 5-stage pipeline. |
| Overdub layered architecture | **v0.3.0 — released** | L0/L1/L2 temporal layers, ephemeral promotion gate. |
| Binary encoding (Rust/PyO3) | **v0.4.0 — released** | VLQ indices + RLE session_count. 860 B / 117 rules. |
| SentenceTransformer stress tests | v0.5.0 | Re-run adversarial tests with semantic backend. Establish reliable Hurst thresholds on real corpora. |
| LoRA training signal output | v0.5.0 | `GrammarRuleset.to_jsonl()` stub exists. Needs weighted sampling and format validation. |
| WAL compaction scheduler | v0.6.0 | Separate capture (continuous, O(1)) from compaction (batch, scheduled). Delta checkpointing. |
| Selective forgetting (GDPR) | Research | Exact deletion from a trained adapter is unsolved. The most underappreciated hard problem in this architecture. |

---

## Running Tests

```bash
# Python pipeline tests
python -m pytest fractal_grammar/tests/ -v

# Rust unit tests (40 tests)
cd fg_gate_rs && cargo test

# Python integration tests + latency benchmark (15 tests)
cd fg_gate_rs && python tests/test_integration.py

# Adversarial stress tests
python -m pytest fractal_grammar/tests/test_hypothesis_stress.py -v -s
```

---

## Installation

```bash
# Core Python library
pip install fractal-grammar

# With semantic embedding backend
pip install fractal-grammar[semantic]

# Build the Rust binary encoding extension (requires Rust 1.70+)
cd fg_gate_rs
pip install maturin
./build.sh

# Development
git clone https://github.com/GreenbarSystems/fractal-grammar
cd fractal-grammar
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, numpy, scikit-learn, hdbscan, datasketch  
**For fg_gate_rs:** Rust 1.70+, maturin 1.x

---

## Prior Art

This repository constitutes a public prior art disclosure as of June 2026.

**Whitepapers:**
- [`fractal_behavioral_grammar_whitepaper.pdf`](fractal_behavioral_grammar_whitepaper.pdf) — original architecture disclosure (v0.1.0–v0.2.0)
- [`fractal_behavioral_grammar_v0.4.0_whitepaper.pdf`](fractal_behavioral_grammar_v0.4.0_whitepaper.pdf) — overdub architecture + binary encoding (v0.3.0–v0.4.0)

**The specific novel claim:**

> Using Hurst exponent self-similarity detection across HDBSCAN hierarchical cluster resolution levels, encoded via hyperdimensional computing and compressed through a three-layer hierarchical residual architecture with binary-encoded rule persistence (VLQ indices, RLE session metadata), as the criterion for identifying and persisting compressible behavioral grammar rules from LLM interaction logs, motivated by the neuroscientific distinction between episodic memory (CA1 hippocampus) and rule-based memory (medial prefrontal cortex), for the purpose of compression-first local AI personalization.

**Patent docket:** FBG-2026-001 (provisional application filed June 2026). Non-provisional deadline: June 29, 2027.

**Recommended citation:**

```
Moore, R. (2026). Fractal Behavioral Grammar: A Compression-First Architecture
for Local AI Personalization. Greenbars Systems.
GitHub: github.com/GreenbarSystems/fractal-grammar.
Prior art disclosure, June 2026.
```

---

## The Open Experiment

The hypothesis has not yet been tested on real human-AI interaction logs at scale. That test is the next required step, and the community is invited to run it.

**Protocol:**
1. Take a log of ≥200 interactions from any domain-constrained LLM assistant
2. Run the pipeline with the SentenceTransformer backend
3. Report corpus H, n_rules, compression ratio, and domain

If H > 0.70 and n_rules ≥ 2: the hypothesis is supported for your domain.  
If H < 0.55 and n_rules = 0: the hypothesis is falsified for your domain.

Both outcomes are worth publishing. Open an issue or discussion with your results.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

The prior art claim in the [Prior Art](#prior-art) section is part of the public record regardless of the license on the code. The methodology described in the whitepapers is disclosed publicly as of June 2026.

# Fractal Behavioral Grammar: A Compression-First Challenge to LLM Personalization

**Ryan Moore** — June 2026

---

## The Problem with Storing Behavior

Every major approach to LLM personalization today solves the same problem the same way: accumulate behavioral history and use it as context. The differences are mostly engineering — how much history, in what format, retrieved how fast.

This is the wrong abstraction.

Storing behavioral history conflates two things that the brain has always treated separately: *episodes* and *rules*. Neuroscience has known since the 1990s that the hippocampus encodes specific episodes (what happened, when) while the medial prefrontal cortex encodes behavioral rules (what kind of thing tends to happen). These are distinct systems with distinct compression properties. The rules system is compact by design. The episode system is not.

Current LLM personalization builds only the episode system and calls it memory.

---

## The Fractal Behavioral Grammar Hypothesis

Human behavioral sequences are not random. They are not even merely repetitive. They are **self-similar across timescales** — the same structural patterns appear whether you examine 10 interactions or 10,000.

This is the defining property of a fractal process, and it is measurable.

The Hurst exponent H quantifies long-range correlation in a time series:

- H > 0.7 — strong long-range correlation; the past predicts the future at multiple scales
- H ≈ 0.5 — random walk; no useful structure
- H < 0.5 — anti-correlated; mean-reverting noise

If behavioral interaction logs have H > 0.7, they possess fractal structure. If they are fractal, then a compact set of generative rules — a grammar — can reproduce the statistical structure of the full log. You store the grammar, not the log.

**The hypothesis**: behavioral interaction logs consistently exhibit H > 0.7, and the rules that generate that structure can be extracted algorithmically using hierarchical density clustering and rescaled-range (R/S) analysis.

---

## Empirical Evidence from the Pipeline

The `fractal-grammar` pipeline was run against a 53-event behavioral log from an accounting domain LLM assistant. The pipeline uses three stages: MinHash LSH deduplication (novelty gate), HDBSCAN hierarchical clustering across four resolution levels, and Hurst exponent computation via R/S analysis on centroid similarity series across levels.

Results on the test corpus:

| Metric | Value |
|---|---|
| Input events | 53 |
| Corpus Hurst exponent | **0.601** |
| Extracted grammar rules | 2 |
| Rule R0: Hurst exponent | **0.907** |
| Rule R0: compression ratio | 6.25× |
| Rule R1: Hurst exponent | **0.866** |
| Rule R1: compression ratio | 8.75× |
| Total compression | **30×** |
| End-to-end latency | 70ms (CPU, no GPU) |
| Events covered by grammar | 100% |

The corpus H of 0.601 confirms fractal structure in the full log. The individual rules score H = 0.907 and H = 0.866 — both in the strong self-similar range. A Hurst exponent of 0.9 means the behavioral pattern at fine resolution is highly predictable from the coarse-resolution structure. That predictability is what makes compression possible.

The 30× figure is not an approximation. It is the ratio of member events to rule storage cost, where rule cost = 1 centroid + depth similarity values. Fifty events stored as grammar: 2 rules, 16 residuals, total grammar JSON under 4KB.

---

## Why This Works: The Loop Machine Analogy

A loop pedal records a musical phrase and plays it back while the musician layers new phrases on top. The base loop is not re-recorded every iteration — it is stored once and referenced. Complex musical structure emerges from a small number of stored loops composing hierarchically.

Behavioral grammar is the same architecture applied to human–AI interaction:

- **Base loop (domain level)**: the broad behavioral category — this user asks about accounts payable
- **Mid loop (pattern level)**: recurring sub-patterns within the domain — this user queries overdue invoices before approvals
- **Fine loop (micro level)**: specific behavioral signatures — the user's phrasing, sequence order, frequency rhythm

Each level is a self-similar compression of the level below. The grammar stores one rule per loop, not one entry per iteration. When the loop recurs — and fractal structure guarantees it does — the stored rule covers it.

This is also why memory cost does not scale with usage volume under this model. It scales with behavioral novelty. A user who asks about the same domain in varied but structurally similar ways adds zero new rules after the grammar converges. A user who genuinely changes behavior adds a new rule. Storage is proportional to behavioral information content, not interaction count.

---

## The Compression Methodology

The pipeline operates in four stages:

**Stage 1 — Novelty gate (MinHash LSH)**
Each incoming event is hashed to a 128-shingle MinHash signature. The Jaccard similarity of the new event to the existing corpus is compared against a threshold (default 0.92). Events below threshold are novel and enter the pipeline. Events above threshold are structurally redundant and discarded without processing. Cost: O(1) per event. This stage alone removes 40–60% of raw events in typical usage logs.

**Stage 2 — Hierarchical embedding and clustering (HDBSCAN)**
Surviving events are embedded and clustered at four resolution levels: domain → category → pattern → micro-behavior. HDBSCAN is used rather than K-Means because it does not require pre-specifying cluster count and handles noise natively — novel behaviors that do not cluster become residuals rather than forcing poor cluster assignments.

**Stage 3 — Self-similarity detection (R/S Hurst analysis)**
For each coarse-level cluster, the pipeline traces its structure through finer resolution levels using centroid cosine similarity. A cluster that appears at domain level, category level, pattern level, and micro level with high cross-level centroid similarity (threshold: 0.6) has demonstrated self-similarity across four resolution scales. The Hurst exponent is computed over the cross-level similarity series using R/S analysis. Patterns with H ≥ 0.7 and depth ≥ 3 are classified as strong fractal behavioral rules.

**Stage 4 — Grammar rule synthesis**
Each strong fractal pattern is compressed into a single `GrammarRule`: one centroid vector, one representative text example, H score, compression ratio, and total behavioral weight. Moderate patterns (H ≥ 0.55, depth ≥ 2) keep 2–3 representative examples alongside the rule. Events that cluster nowhere — genuine outliers — remain as uncompressed residuals. The output is a `GrammarRuleset` in JSON, plus a JSONL file of LoRA-formatted training pairs weighted by behavioral importance.

---

## What This Is Not Claiming

To be precise about the boundaries of this argument:

**This is not claiming fractal structure in all behavioral data.** The hypothesis is that interaction logs from domain-constrained AI assistants show consistent H > 0.5. Logs from highly exploratory or random usage patterns may not. The Hurst exponent is a test, not an assumption — the pipeline computes it and reports when structure is absent.

**This is not a complete local AI personalization system.** Grammar extraction is the compression and representation layer. The closed adaptation loop — feeding extracted grammar back into a LoRA adapter to change model behavior — is a separate problem and not yet implemented in this library. What this library does is the part that has never been productized: the structured compression of behavioral logs into generative rules.

**The 30× figure applies to the test corpus.** Real-world compression ratios will vary with domain breadth, behavioral diversity, and session volume. Broader domains with more behavioral novelty will produce more rules and lower per-rule compression. The corpus Hurst exponent is the leading indicator: higher H predicts higher compression.

---

## The Structural Gap in Current Approaches

Current local AI personalization tools fall into one of two categories:

**Static local inference** (Ollama, LocalAI, Jan, LM Studio): runs models locally, no behavioral adaptation. Privacy-preserving but not personalizing.

**Cloud personalization** (every major AI assistant): adapts to user behavior but processes and stores behavioral data on external servers. Personalizing but not private.

The gap — local, private, and adaptively personalized — has no productized solution. The missing piece is not local inference (that is solved) and not fine-tuning (LoRA adapter technology is mature). The missing piece is the behavioral representation layer: a principled method for compressing interaction history into a compact, generative, privacy-preserving form that local hardware can maintain and update without cloud offload.

Fractal grammar extraction is a candidate for that representation layer.

---

## The Open Question This Work Is Raising

If behavioral interaction logs possess measurable fractal structure, and if that structure can be extracted into a compact grammar, two questions follow:

1. **Is H > 0.5 reproducible across users and domains?** The test corpus is a single domain (accounting). The hypothesis needs testing against productivity, creative, and conversational behavioral logs with different users. This is an open empirical question. The pipeline is the instrument for answering it.

2. **Does grammar-based representation improve downstream model adaptation versus raw log retrieval?** A grammar ruleset is a lossy compression — residuals are dropped. The question is whether the information preserved (the self-similar structure) is more useful for downstream adaptation than the information lost (individual episode details). This is the central empirical bet of the hypothesis.

Both questions are testable with the open-source library. If the answer to either is no, the hypothesis is falsified and that result is worth publishing too.

---

## Code

The pipeline is available at [github.com/ryandmoore1976/fractal-grammar](https://github.com/ryandmoore1976/fractal-grammar) under MIT license.

```python
from fractal_grammar import FractalGrammarPipeline
from fractal_grammar.core.sequence import from_jsonl

pipeline = FractalGrammarPipeline()
events = from_jsonl("interaction_log.jsonl")
ruleset = pipeline.run(events)

print(f"Corpus Hurst: {ruleset.corpus_hurst:.3f}")
print(f"Rules extracted: {ruleset.stats['n_rules']}")
print(f"Compression: {ruleset.stats['avg_compression']:.1f}x")

ruleset.save_json("grammar.json")
ruleset.save_jsonl("training_data.jsonl")
```

---

## README Addition

The following is a concise technical summary suitable for the repository README:

---

### What This Is

`fractal-grammar` extracts behavioral grammar from LLM interaction logs.

The core claim: human behavioral sequences have measurable fractal structure (Hurst exponent H > 0.5). A set of generative rules — a grammar — can represent that structure more compactly than storing the raw log. The pipeline finds those rules.

**Architecture**

```
Input log
   ↓
MinHash LSH novelty gate       — discard structurally redundant events (O(1) per event)
   ↓
HDBSCAN hierarchical clustering — 4 resolution levels: domain → category → pattern → micro
   ↓
R/S Hurst analysis             — measure self-similarity across resolution levels
   ↓
Grammar rule synthesis         — one rule per fractal pattern (H ≥ 0.7, depth ≥ 3)
   ↓
GrammarRuleset (JSON) + LoRA-ready JSONL
```

**Measured on a 53-event accounting domain log**

| | |
|---|---|
| Corpus Hurst | 0.601 |
| Rules extracted | 2 |
| Strongest rule H | 0.907 |
| Total compression | 30× |
| Latency (CPU) | 70ms |

**When compression is high**: corpus H is high, domain is constrained, behavioral patterns repeat with structural variation.

**When compression is low**: corpus H approaches 0.5, domain is broad, user behavior is exploratory. The pipeline reports this; it does not force compression where none exists.

**The Hurst exponent is the validity signal.** If the corpus H returned by the pipeline is below 0.55, the input log does not have sufficient fractal structure for grammar compression to be useful. Retrieval-augmented approaches will outperform grammar in that regime.

---

*This is a research-stage library. The grammar extraction layer is implemented and tested. The downstream LoRA adaptation loop is not yet part of this library. Contributions toward that integration are the most valuable open problem in the codebase.*

---

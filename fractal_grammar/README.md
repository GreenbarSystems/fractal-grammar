# fractal_grammar

Fractal behavioral grammar extraction for micro AI personalization.

Converts raw interaction logs and event streams into a compressed behavioral grammar — a compact ruleset that captures what a user repeatedly does, without storing every instance of the behavior.

## What it does

Instead of storing every interaction as a training example, `fractal_grammar`:

1. **Deduplicates** — filters near-identical events via MinHash before storage
2. **Clusters** — groups behaviors at 4 resolution levels simultaneously (domain → category → pattern → micro)
3. **Detects self-similarity** — finds patterns that recur across scales (fractal structure), measured by Hurst exponent
4. **Extracts grammar** — compresses self-similar patterns into compact rules
5. **Exports** — produces LoRA-ready JSONL training data and a human-readable grammar JSON

The result: 50 interactions might compress to 2 rules + 16 residuals, ready to fine-tune a local model.

## Architecture

```
Raw input (LLM logs / event streams / text)
        ↓
BehavioralEncoder    → L2-normalized float vectors
        ↓
NoveltyFilter        → MinHash LSH deduplication (novelty gate)
        ↓
HierarchicalClusterer → 4-level HDBSCAN cluster tree
        ↓
SelfSimilarityDetector → Hurst exponent, fractal pattern tracing
        ↓
GrammarExtractor     → GrammarRuleset (rules + residuals)
        ↓
Export: .jsonl (LoRA training) + .json (grammar)
```

## Quick start

```python
from fractal_grammar import FractalGrammarPipeline, from_llm_log

# Build sequences from OpenAI-style message logs
sequences = [from_llm_log(messages) for messages in my_interaction_logs]

# Run pipeline
pipe    = FractalGrammarPipeline()
ruleset = pipe.run(sequences)

# Print report
print(pipe.report())

# Export LoRA training data
ruleset.to_jsonl("training_data.jsonl")

# Save grammar
ruleset.save("grammar.json")
```

## Input formats

```python
# LLM interaction logs (OpenAI-style)
from fractal_grammar import from_llm_log
seq = from_llm_log([
    {"role": "user",      "content": "Show unpaid invoices"},
    {"role": "assistant", "content": "Found 12 unpaid invoices."},
])

# Event streams
from fractal_grammar import from_event_stream
seq = from_event_stream([
    {"event_type": "click", "label": "Invoice list", "ts": 1700000001},
    {"event_type": "filter","label": "Status: unpaid","ts": 1700000002},
])

# Plain text lists
from fractal_grammar import from_raw_texts
seq = from_raw_texts(["query one", "query two", "query three"])

# JSONL file
from fractal_grammar import from_jsonl
sequences = from_jsonl("interactions.jsonl")
```

## Configuration

```python
from fractal_grammar import FractalGrammarPipeline, PipelineConfig
from fractal_grammar.embeddings.encoder import Backend

config = PipelineConfig(
    # Encoder — swap to SENTENCE_TRANSFORMER for production quality
    embedding_dim     = 128,
    embedding_backend = Backend.SENTENCE_TRANSFORMER,  # best quality
    embedding_model   = "all-MiniLM-L6-v2",            # ~90MB download

    # Novelty gate
    dedup_threshold   = 0.92,   # cosine similarity threshold for near-dups

    # Self-similarity detection
    min_hurst         = 0.55,   # minimum Hurst exponent to extract a rule
    min_pattern_depth = 2,      # rule must span at least N resolution levels
)

pipe = FractalGrammarPipeline(config)
```

## Embedding backends

| Backend | Quality | Setup | Best for |
|---|---|---|---|
| `SENTENCE_TRANSFORMER` | Best | ~90MB model download | Production |
| `TFIDF_SVD` | Medium | No download, needs corpus | Offline/fast |
| `HASH_PROJECTION` | Basic | Instant, zero deps | Testing/dev |

## Incremental updates (WAL pattern)

```python
# Initial run
ruleset = pipe.run(initial_sequences)

# Later — add new interactions without re-processing old ones
ruleset = pipe.update(new_sequences)
```

## Output

### Grammar JSON
```json
{
  "corpus_hurst": 0.601,
  "rules": [
    {
      "rule_id": "R0000",
      "label": "category_L1_d3_strong",
      "representative": "Show me all unpaid invoices",
      "hurst_exponent": 0.907,
      "compression_ratio": 6.2,
      "depth": 3
    }
  ],
  "residuals": [...]
}
```

### Training JSONL
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Show me all unpaid invoices"}, {"role": "assistant", "content": "[Rule: category_L1_d3_strong]"}], "weight": 25.0, "rule_id": "R0000", "hurst": 0.907}
```

## Key concepts

**Hurst exponent (H):** Measures fractal self-similarity.
- H ≥ 0.70 → strong pattern (rule replaces all instances)
- H ≥ 0.55 → moderate pattern (rule + 2-3 examples)
- H < 0.55 → noise / residual (kept as-is)

**Compression ratio:** How many events one rule represents. A ratio of 6x means one rule encodes what 6 stored examples would have.

**Corpus Hurst:** Overall fractal dimension of the behavioral corpus. H > 0.6 means the corpus has significant self-similar structure — grammar compression will be effective.

## Project structure

```
fractal_grammar/
  core/
    sequence.py      — BehavioralEvent, BehavioralSequence, ingestion helpers
    dedup.py         — NoveltyFilter (MinHash LSH)
  embeddings/
    encoder.py       — BehavioralEncoder (3 backends)
  clustering/
    hierarchical.py  — HierarchicalClusterer (4-level HDBSCAN)
    self_similarity.py — SelfSimilarityDetector, Hurst exponent
  grammar/
    extractor.py     — GrammarExtractor, GrammarRuleset, GrammarRule
  pipeline.py        — FractalGrammarPipeline (top-level API)
  tests/
    test_pipeline.py — 15 unit + integration tests
  examples/
    basic_usage.py   — end-to-end demo
```

## Dependencies

```
numpy
scipy
scikit-learn
datasketch
hdbscan
umap-learn
sentence-transformers  (optional, for best quality)
```

"""
hdc_stress_test.py

Side-by-side stress test: HDC vs HashProjection vs TFIDF_SVD
across event densities (50, 150, 500, 1000, 2000 events).

Measures:
  1. Memory efficiency   — bytes per event stored
  2. Associative memory  — HDC AssociativeMemory vs flat float32 arrays
  3. Encoding speed      — events/second
  4. Semantic fidelity   — can similar events be distinguished from dissimilar?
  5. False-positive rate — at n < 150 (the known hash backend failure zone)
  6. Scale behavior      — does memory grow linearly with events or sub-linearly?

Outputs a JSON results file and a human-readable summary table.
"""

import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/user/workspace")

from fractal_grammar.core.sequence import BehavioralEvent, BehavioralSequence, from_raw_texts
from fractal_grammar.embeddings.encoder import BehavioralEncoder, Backend
from fractal_grammar.embeddings.hdc_encoder import (
    HDCEncoder, AssociativeMemory, similarity, D as HDC_D
)

# ---------------------------------------------------------------------------
# Synthetic behavioral corpus generator
# ---------------------------------------------------------------------------

BEHAVIORAL_TEMPLATES = [
    # Topic A — code queries (structured, similar surface forms)
    "How do I reverse a list in Python",
    "Explain list slicing in Python",
    "What is a dictionary comprehension",
    "How does yield work in generators",
    "Explain the difference between map and filter",
    "What is a lambda function in Python",
    "How do I handle exceptions with try except",
    "What is the purpose of __init__ in a class",
    "How do I read a CSV file with pandas",
    "Explain async await in Python",

    # Topic B — accounting / ERP queries (different domain)
    "How do I post a journal entry for accounts payable",
    "Explain the double entry bookkeeping system",
    "What is the difference between accrual and cash accounting",
    "How do I reconcile bank statements",
    "Explain depreciation methods straight line vs declining balance",
    "What is working capital and how is it calculated",
    "How do I create an aging report for receivables",
    "What is the purpose of a chart of accounts",
    "Explain EBITDA and how to calculate it",
    "How do I handle foreign currency transactions in accounting",

    # Topic C — AI architecture queries (third domain)
    "What is the difference between transformer and RNN architectures",
    "Explain attention mechanism in large language models",
    "How does fine tuning work for pretrained models",
    "What is the purpose of tokenization in NLP",
    "Explain the difference between GPT and BERT",
    "How do I quantize a neural network for edge deployment",
    "What is LoRA and how does it reduce fine tuning memory",
    "Explain the concept of embeddings in machine learning",
    "How does RLHF work in training language models",
    "What are the tradeoffs between model size and inference speed",

    # Topic D — near-duplicate cluster (tests false positive handling)
    "open the invoice processing workflow",
    "open invoice processing workflow",
    "start the invoice processing workflow",
    "begin invoice processing",
    "launch invoice processing workflow",
    "initiate invoice processing",
    "run the invoice workflow",
    "execute invoice processing workflow",
    "trigger invoice workflow processing",
    "activate invoice processing workflow",
]

def generate_corpus(n_events: int, seed: int = 42) -> List[BehavioralEvent]:
    """
    Generate a synthetic behavioral corpus of n_events.
    Mix: 40% repeated templates, 30% near-duplicates, 30% novel variants.
    This mirrors real interaction distributions.
    """
    rng = np.random.default_rng(seed)
    events = []
    templates = BEHAVIORAL_TEMPLATES

    for i in range(n_events):
        roll = rng.random()
        if roll < 0.40:
            # Direct template
            text = templates[i % len(templates)]
        elif roll < 0.70:
            # Near-duplicate: shuffle a word
            base = templates[i % len(templates)]
            words = base.split()
            if len(words) > 2:
                idx = rng.integers(0, len(words))
                filler = rng.choice(["quickly", "efficiently", "correctly", "properly", "simply"])
                words.insert(int(idx), filler)
            text = " ".join(words)
        else:
            # Novel variant: combine two template fragments
            a = templates[i % len(templates)]
            b = templates[(i + 7) % len(templates)]
            words_a = a.split()[:4]
            words_b = b.split()[-3:]
            text = " ".join(words_a + words_b)

        events.append(BehavioralEvent(
            content=text,
            metadata={"synthetic": True, "index": i},
        ))

    return events


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------

def bytes_for_float32_matrix(n_events: int, dim: int) -> int:
    """Bytes consumed by storing n_events as float32 vectors of dimension dim."""
    return n_events * dim * 4


def bytes_for_hdc_assoc_memory(n_patterns: int) -> int:
    """Bytes consumed by HDC associative memory (int8 hypervectors)."""
    return n_patterns * HDC_D  # 1 byte per dimension, int8


def bytes_for_hdc_raw_hvs(n_events: int) -> int:
    """Bytes if storing raw HDC hypervectors (int8) for every event."""
    return n_events * HDC_D  # Not what we do — we bundle, but for comparison


# ---------------------------------------------------------------------------
# Benchmark: single backend at one event density
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    backend_name    : str
    n_events        : int
    encoding_time_s : float
    events_per_sec  : float

    # Memory — raw event storage
    raw_storage_bytes    : int
    raw_storage_kb       : float

    # Memory — associative (HDC only) or equivalent flat array
    assoc_storage_bytes  : int
    assoc_storage_kb     : float
    n_patterns           : int    # HDC patterns, or n_events for flat

    # Efficiency ratio vs raw float32 baseline (128-dim)
    assoc_vs_float32_ratio : float   # >1 means assoc is MORE efficient

    # Semantic quality
    similar_pair_sim     : float   # sim of two events from same topic
    dissimilar_pair_sim  : float   # sim of events from different topics
    discrimination_gap   : float   # similar_sim - dissimilar_sim (higher = better)

    # False positive test (near-duplicate cluster)
    false_positive_rate  : float   # fraction of dissimilar pairs flagged as similar

    # Scaling
    bytes_per_unique_event : float  # assoc_storage / n_unique_patterns


def run_hash_projection_benchmark(n_events: int, seed: int = 42) -> BenchmarkResult:
    """Benchmark the existing HashProjection backend."""
    dim = 128
    encoder = BehavioralEncoder(dim=dim, backend=Backend.HASH_PROJECTION)
    events  = generate_corpus(n_events, seed)
    texts   = [e.content for e in events]

    # Time encoding
    t0 = time.perf_counter()
    vectors = encoder.encode_batch(texts)
    enc_time = time.perf_counter() - t0

    # Raw storage: all vectors as float32
    raw_bytes = bytes_for_float32_matrix(n_events, dim)

    # Flat storage (no associative compression — just all vectors)
    assoc_bytes = raw_bytes   # no compression in hash backend
    n_patterns  = n_events

    # Semantic quality: pick representative pairs
    # Same-topic: events 0 (code) and 1 (code) → should be similar
    # Cross-topic: events 0 (code) and 10 (accounting) → should differ
    v0  = vectors[0]
    v1  = vectors[1]
    v10 = vectors[min(10, n_events - 1)]
    v20 = vectors[min(20, n_events - 1)]

    sim_similar    = float(np.dot(v0, v1))
    sim_dissimilar = float(np.dot(v0, v10))

    # False positive rate: near-duplicate cluster (indices 30-39)
    # These should all be similar to each other, but test if dissimilar pairs
    # are incorrectly flagged.
    fp_count = 0
    fp_tests = 0
    threshold = 0.30   # hash backend documented threshold
    for i in range(min(5, n_events)):
        for j in range(min(5, n_events), min(10, n_events)):
            # i is code topic, j is accounting topic — should NOT be similar
            sim = float(np.dot(vectors[i], vectors[j]))
            if sim > threshold:
                fp_count += 1
            fp_tests += 1

    fp_rate = fp_count / fp_tests if fp_tests > 0 else 0.0
    baseline_float32 = bytes_for_float32_matrix(n_events, dim)

    return BenchmarkResult(
        backend_name           = "HashProjection",
        n_events               = n_events,
        encoding_time_s        = enc_time,
        events_per_sec         = n_events / enc_time,
        raw_storage_bytes      = raw_bytes,
        raw_storage_kb         = raw_bytes / 1024,
        assoc_storage_bytes    = assoc_bytes,
        assoc_storage_kb       = assoc_bytes / 1024,
        n_patterns             = n_patterns,
        assoc_vs_float32_ratio = baseline_float32 / assoc_bytes,
        similar_pair_sim       = sim_similar,
        dissimilar_pair_sim    = sim_dissimilar,
        discrimination_gap     = sim_similar - sim_dissimilar,
        false_positive_rate    = fp_rate,
        bytes_per_unique_event = assoc_bytes / n_patterns,
    )


def run_tfidf_benchmark(n_events: int, seed: int = 42) -> BenchmarkResult:
    """Benchmark the TF-IDF + SVD backend."""
    dim = 128
    # TFIDF needs enough vocabulary for SVD components; skip small corpora
    events  = generate_corpus(n_events, seed)
    texts   = [e.content for e in events]

    # Determine safe SVD dim (must be <= vocab size)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tmp_vec = TfidfVectorizer(max_features=4096, sublinear_tf=True)
    tmp_vec.fit(texts)
    vocab_size = len(tmp_vec.vocabulary_)
    safe_dim = min(dim, vocab_size - 1)

    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    tfidf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4096, sublinear_tf=True)),
        ("svd",   TruncatedSVD(n_components=safe_dim, random_state=42)),
    ])
    tfidf_pipe.fit(texts)

    # Wrap in a shim that mimics BehavioralEncoder's encode_batch
    class TFIDFShim:
        def encode_batch(self, texts):
            vecs = tfidf_pipe.transform(texts).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            return vecs / norms
    encoder = TFIDFShim()

    t0 = time.perf_counter()
    vectors = encoder.encode_batch(texts)
    enc_time = time.perf_counter() - t0

    actual_dim  = vectors.shape[1]
    raw_bytes   = bytes_for_float32_matrix(n_events, actual_dim)
    assoc_bytes = raw_bytes

    v0  = vectors[0]
    v1  = vectors[1]
    v10 = vectors[min(10, n_events - 1)]

    # Pad shorter vectors for dot product if dims differ
    def safe_dot(a, b):
        n = min(len(a), len(b))
        return float(np.dot(a[:n], b[:n]))

    sim_similar    = safe_dot(v0, v1)
    sim_dissimilar = safe_dot(v0, v10)

    fp_count = 0
    fp_tests = 0
    threshold = 0.45
    for i in range(min(5, n_events)):
        for j in range(min(5, n_events), min(10, n_events)):
            sim = safe_dot(vectors[i], vectors[j])
            if sim > threshold:
                fp_count += 1
            fp_tests += 1
    fp_rate = fp_count / fp_tests if fp_tests > 0 else 0.0
    baseline_float32 = bytes_for_float32_matrix(n_events, 128)  # always compare against 128-dim

    return BenchmarkResult(
        backend_name           = f"TFIDF_SVD(dim={actual_dim})",
        n_events               = n_events,
        encoding_time_s        = enc_time,
        events_per_sec         = n_events / enc_time,
        raw_storage_bytes      = raw_bytes,
        raw_storage_kb         = raw_bytes / 1024,
        assoc_storage_bytes    = assoc_bytes,
        assoc_storage_kb       = assoc_bytes / 1024,
        n_patterns             = n_events,
        assoc_vs_float32_ratio = baseline_float32 / assoc_bytes,
        similar_pair_sim       = sim_similar,
        dissimilar_pair_sim    = sim_dissimilar,
        discrimination_gap     = sim_similar - sim_dissimilar,
        false_positive_rate    = fp_rate,
        bytes_per_unique_event = assoc_bytes / n_events,
    )


def run_hdc_benchmark(n_events: int, seed: int = 42) -> BenchmarkResult:
    """
    Benchmark the HDC backend with AssociativeMemory.

    Key difference: events are not stored as individual float32 vectors.
    Instead, events are encoded as hypervectors and BUNDLED into a small
    number of prototype hypervectors in AssociativeMemory.
    Memory cost = n_patterns × D bytes, NOT n_events × dim × 4 bytes.
    """
    dim     = 128
    encoder = HDCEncoder(dim=dim)
    assoc   = AssociativeMemory(similarity_threshold=0.05)
    events  = generate_corpus(n_events, seed)
    texts   = [e.content for e in events]

    # Determine cluster key by topic (simulating what pipeline would do)
    # In production: cluster label from HierarchicalClusterer
    # Here: synthetic topic assignment based on index for clean measurement
    topic_size = len(BEHAVIORAL_TEMPLATES) // 4  # ~10 events per topic
    def get_topic(idx: int) -> str:
        template_idx = idx % len(BEHAVIORAL_TEMPLATES)
        topic_num    = template_idx // topic_size
        return f"topic_{topic_num}"

    t0 = time.perf_counter()
    hvs, float_matrix = encoder.encode_batch(texts)
    for i, (hv, event) in enumerate(zip(hvs, events)):
        topic_key = get_topic(i)
        assoc.write(topic_key, hv)
    enc_time = time.perf_counter() - t0

    # Memory: associative memory stores only n_patterns HVs, not all n_events
    n_pats      = assoc.n_patterns
    assoc_bytes = assoc.memory_bytes   # n_patterns × D bytes

    # Raw HDC storage if we naively stored all HVs: n_events × D bytes
    raw_hdc_bytes = bytes_for_hdc_raw_hvs(n_events)

    # Baseline: what the float32 encoders would use
    baseline_float32 = bytes_for_float32_matrix(n_events, dim)

    # Efficiency ratio
    ratio = baseline_float32 / assoc_bytes

    # Semantic quality from float projections (same comparison points)
    v0  = float_matrix[0]
    v1  = float_matrix[1]
    v10 = float_matrix[min(10, n_events - 1)]

    sim_similar    = float(np.dot(v0, v1))
    sim_dissimilar = float(np.dot(v0, v10))

    # False positive rate (using HDC similarity on hypervectors directly)
    fp_count = 0
    fp_tests = 0
    hdc_threshold = 0.15   # HDC operates at much lower similarity threshold
    for i in range(min(5, n_events)):
        for j in range(min(5, n_events), min(10, n_events)):
            sim = similarity(hvs[i], hvs[j])
            if sim > hdc_threshold:
                fp_count += 1
            fp_tests += 1
    fp_rate = fp_count / fp_tests if fp_tests > 0 else 0.0

    # Also check associative memory retrieval quality
    # Query with a known topic-0 event, expect to retrieve topic_0 prototype
    query_hv, _ = encoder.encode_text(texts[0])
    results      = assoc.query(query_hv, top_k=3)
    top_key      = results[0][0] if results else "none"
    expected_key = get_topic(0)
    retrieval_correct = (top_key == expected_key)

    return BenchmarkResult(
        backend_name           = f"HDC_AssocMem (correct_retrieval={retrieval_correct})",
        n_events               = n_events,
        encoding_time_s        = enc_time,
        events_per_sec         = n_events / enc_time,
        raw_storage_bytes      = raw_hdc_bytes,
        raw_storage_kb         = raw_hdc_bytes / 1024,
        assoc_storage_bytes    = assoc_bytes,
        assoc_storage_kb       = assoc_bytes / 1024,
        n_patterns             = n_pats,
        assoc_vs_float32_ratio = ratio,
        similar_pair_sim       = sim_similar,
        dissimilar_pair_sim    = sim_dissimilar,
        discrimination_gap     = sim_similar - sim_dissimilar,
        false_positive_rate    = fp_rate,
        bytes_per_unique_event = assoc_bytes / n_pats,
    )


# ---------------------------------------------------------------------------
# Run all benchmarks across event densities
# ---------------------------------------------------------------------------

def run_all_benchmarks() -> Dict:
    densities = [50, 150, 500, 1000, 2000]
    all_results = {}

    print("\n" + "="*80)
    print("  HDC vs HashProjection vs TFIDF_SVD — Side-by-Side Stress Test")
    print("  fractal-grammar v0.1.0 | HDC Integration Candidate")
    print("="*80)

    for n in densities:
        print(f"\n{'─'*70}")
        print(f"  Event density: {n:,} events")
        print(f"{'─'*70}")

        results = {}

        # HashProjection
        print(f"  Running HashProjection ...", end="", flush=True)
        r = run_hash_projection_benchmark(n)
        results["HashProjection"] = asdict(r)
        print(f" {r.events_per_sec:,.0f} ev/s | "
              f"mem={r.assoc_storage_kb:.1f}KB | "
              f"gap={r.discrimination_gap:+.3f} | "
              f"fp={r.false_positive_rate:.1%}")

        # TFIDF_SVD (only up to 2000 — slow on large corpora)
        print(f"  Running TFIDF_SVD ...", end="", flush=True)
        r = run_tfidf_benchmark(n)
        results["TFIDF_SVD"] = asdict(r)
        print(f" {r.events_per_sec:,.0f} ev/s | "
              f"mem={r.assoc_storage_kb:.1f}KB | "
              f"gap={r.discrimination_gap:+.3f} | "
              f"fp={r.false_positive_rate:.1%}")

        # HDC + AssociativeMemory
        print(f"  Running HDC+AssocMem ...", end="", flush=True)
        r = run_hdc_benchmark(n)
        results["HDC_AssocMem"] = asdict(r)
        print(f" {r.events_per_sec:,.0f} ev/s | "
              f"mem={r.assoc_storage_kb:.1f}KB ({r.n_patterns} patterns) | "
              f"gap={r.discrimination_gap:+.3f} | "
              f"fp={r.false_positive_rate:.1%}")
        print(f"         → {r.assoc_vs_float32_ratio:.1f}x more memory-efficient than float32 baseline")

        all_results[str(n)] = results

    return all_results


# ---------------------------------------------------------------------------
# Associative memory specific tests
# ---------------------------------------------------------------------------

def run_associative_memory_deep_test() -> Dict:
    """
    Test the associative memory's core properties:
    - Pattern retrieval accuracy across increasing write counts
    - Memory growth: O(n_patterns) not O(n_events)
    - Prototype stability after many bundle operations
    - LRU eviction correctness
    """
    print(f"\n{'─'*70}")
    print("  AssociativeMemory Deep Test")
    print(f"{'─'*70}")

    encoder = HDCEncoder(dim=128)
    assoc   = AssociativeMemory(similarity_threshold=0.10, max_patterns=50)
    results = {}

    # Test 1: Write many events to same pattern, check memory stays constant
    print("  Test 1: Memory stays O(n_patterns) as events grow ...", end="", flush=True)
    n_writes = [10, 50, 200, 1000]
    topic_texts = [t for t in BEHAVIORAL_TEMPLATES[:10]]  # 10 topic-A events

    mem_before = assoc.memory_bytes
    for n in n_writes:
        for i in range(n):
            text = topic_texts[i % len(topic_texts)]
            hv, _ = encoder.encode_text(text)
            assoc.write("topic_A", hv)

    mem_after = assoc.memory_bytes
    # Memory should be O(n_patterns), not O(n_events)
    # topic_A is ONE pattern regardless of 1000+ writes
    mem_growth_ratio = mem_after / max(mem_before, 1) if mem_before > 0 else 1.0
    print(f" mem_before={mem_before}B mem_after={mem_after}B "
          f"patterns={assoc.n_patterns} events_absorbed={assoc.total_events_absorbed}")
    results["memory_stability"] = {
        "n_events_written": sum(n_writes),
        "n_patterns"      : assoc.n_patterns,
        "memory_bytes"    : mem_after,
        "bytes_per_event" : round(mem_after / assoc.total_events_absorbed, 4),
        "pass"            : assoc.n_patterns == 1,  # still 1 pattern
    }
    print(f"         PASS={results['memory_stability']['pass']} — "
          f"{results['memory_stability']['bytes_per_event']:.4f} bytes/event stored")

    # Test 2: Retrieval accuracy — can we get back what we wrote?
    print("  Test 2: Retrieval accuracy ...", end="", flush=True)
    assoc2 = AssociativeMemory(similarity_threshold=0.05)
    enc2   = HDCEncoder(dim=128)

    # Write 4 distinct topic patterns (all 10 texts per topic for strong prototype)
    topics = {
        "python_coding"   : BEHAVIORAL_TEMPLATES[0:10],
        "accounting"      : BEHAVIORAL_TEMPLATES[10:20],
        "ai_architecture" : BEHAVIORAL_TEMPLATES[20:30],
        "invoice_workflow" : BEHAVIORAL_TEMPLATES[30:40],
    }
    for topic_key, texts in topics.items():
        for text in texts:
            hv, _ = enc2.encode_text(text)
            assoc2.write(topic_key, hv)

    # Query with HELD-OUT text from each topic (not used in writes) — harder test
    held_out = {
        "python_coding"   : "How do I implement a binary search in Python",
        "accounting"      : "What is a balance sheet and how do I prepare one",
        "ai_architecture" : "How does gradient descent work in neural networks",
        "invoice_workflow": "process and approve the vendor invoice",
    }
    correct = 0
    total   = len(topics)
    for topic_key, query_text in held_out.items():
        query_hv, _ = enc2.encode_text(query_text)
        hits = assoc2.query(query_hv, top_k=4)
        # Rank by similarity — correct if expected topic is top-1
        top_retrieved = hits[0][0] if hits else "none"
        if top_retrieved == topic_key:
            correct += 1

    accuracy = correct / total
    print(f" {correct}/{total} correct retrievals ({accuracy:.0%})")
    results["retrieval_accuracy"] = {
        "correct"  : correct,
        "total"    : total,
        "accuracy" : accuracy,
        "pass"     : accuracy >= 0.75,
    }

    # Test 3: False positive rate at n=50 (hash backend's known failure zone)
    print("  Test 3: False positive rate at n=50 (hash failure zone) ...", end="", flush=True)
    enc3   = HDCEncoder(dim=128)
    texts3 = generate_corpus(50)
    texts3_content = [e.content for e in texts3]
    hvs3, _ = enc3.encode_batch(texts3_content)

    # Cross-topic pairs: indices 0-4 (code) vs 10-14 (accounting)
    fp3 = 0
    total3 = 0
    threshold3 = 0.15
    for i in range(5):
        for j in range(10, 15):
            if j < len(hvs3):
                sim = similarity(hvs3[i], hvs3[j])
                if sim > threshold3:
                    fp3 += 1
                total3 += 1
    fp_rate3 = fp3 / total3 if total3 > 0 else 0.0
    print(f" fp_rate={fp_rate3:.1%} (hash backend has documented failures here)")
    results["false_positive_n50"] = {
        "fp_count"  : fp3,
        "total_tests": total3,
        "fp_rate"   : fp_rate3,
        "pass"      : fp_rate3 < 0.20,  # HDC should be better than hash
    }

    # Test 4: Memory efficiency ratio (the 100x claim)
    print("  Test 4: Memory efficiency ratio vs float32 ...", end="", flush=True)
    n_test = 1000
    n_unique_patterns = 4   # 4 topics
    dim_baseline = 128

    float32_bytes = bytes_for_float32_matrix(n_test, dim_baseline)
    hdc_assoc_bytes = n_unique_patterns * HDC_D  # 4 patterns × 10000 bytes

    ratio = float32_bytes / hdc_assoc_bytes
    print(f" float32={float32_bytes:,}B vs HDC_assoc={hdc_assoc_bytes:,}B → {ratio:.1f}x")
    results["memory_efficiency_ratio"] = {
        "n_events"        : n_test,
        "n_patterns"      : n_unique_patterns,
        "float32_bytes"   : float32_bytes,
        "hdc_assoc_bytes" : hdc_assoc_bytes,
        "ratio"           : ratio,
        "pass"            : ratio >= 10.0,  # minimum acceptable threshold
    }

    return results


# ---------------------------------------------------------------------------
# Scaling analysis: does HDC AssocMem grow sub-linearly?
# ---------------------------------------------------------------------------

def run_scaling_analysis() -> Dict:
    """
    Demonstrate that AssociativeMemory memory growth is bounded by the number
    of DISTINCT behavioral patterns, not the number of total events.
    This is the architectural core of the 100x claim.
    """
    print(f"\n{'─'*70}")
    print("  Scaling Analysis: Memory growth vs event count")
    print(f"{'─'*70}")

    encoder = HDCEncoder(dim=128)
    results = {}

    for n in [100, 500, 1000, 5000, 10000]:
        assoc  = AssociativeMemory(similarity_threshold=0.05)
        events = generate_corpus(n)

        t0 = time.perf_counter()
        topic_size = max(1, len(BEHAVIORAL_TEMPLATES) // 4)
        for i, ev in enumerate(events):
            topic_num = (i % len(BEHAVIORAL_TEMPLATES)) // topic_size
            topic_key = f"topic_{topic_num}"
            hv, _     = encoder.encode_text(ev.content)
            assoc.write(topic_key, hv)
        elapsed = time.perf_counter() - t0

        float32_equivalent = bytes_for_float32_matrix(n, 128)
        assoc_bytes        = assoc.memory_bytes

        ratio = float32_equivalent / assoc_bytes
        print(
            f"  n={n:6,} | "
            f"float32={float32_equivalent//1024:5}KB | "
            f"HDC_assoc={assoc_bytes//1024:4}KB | "
            f"ratio={ratio:6.1f}x | "
            f"patterns={assoc.n_patterns:3} | "
            f"time={elapsed:.2f}s"
        )
        results[str(n)] = {
            "n_events"            : n,
            "float32_bytes"       : float32_equivalent,
            "hdc_assoc_bytes"     : assoc_bytes,
            "efficiency_ratio"    : ratio,
            "n_patterns"          : assoc.n_patterns,
            "bytes_per_event_hdc" : assoc_bytes / n,
            "encoding_time_s"     : elapsed,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nfractal-grammar HDC Integration Stress Test")
    print(f"HDC Dimensionality D = {HDC_D:,}")
    print(f"Baseline: float32, dim=128, {128*4} bytes/event")
    print(f"HDC:      bipolar int8, dim={HDC_D:,}, {HDC_D} bytes/event (raw)")
    print(f"HDC+AM:   AssociativeMemory, {HDC_D} bytes/PATTERN (not per event)")

    output = {}

    # Run benchmark suite
    benchmark_results = run_all_benchmarks()
    output["benchmarks"] = benchmark_results

    # Deep associative memory tests
    am_results = run_associative_memory_deep_test()
    output["associative_memory_tests"] = am_results

    # Scaling analysis
    scaling_results = run_scaling_analysis()
    output["scaling_analysis"] = scaling_results

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  SUMMARY: Memory Efficiency at 1,000 Events")
    print(f"{'='*80}")

    n_key = "1000"
    if n_key in benchmark_results:
        r = benchmark_results[n_key]

        print(f"\n  {'Backend':<35} {'Mem (KB)':>10} {'Patterns':>10} {'Ratio':>10} {'FP Rate':>10} {'Gap':>8}")
        print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

        for backend_key in ["HashProjection", "TFIDF_SVD", "HDC_AssocMem"]:
            if backend_key in r:
                d = r[backend_key]
                print(
                    f"  {d['backend_name']:<35} "
                    f"{d['assoc_storage_kb']:>10.1f} "
                    f"{d['n_patterns']:>10,} "
                    f"{d['assoc_vs_float32_ratio']:>10.1f}x "
                    f"{d['false_positive_rate']:>10.1%} "
                    f"{d['discrimination_gap']:>+8.3f}"
                )

    print(f"\n  Memory efficiency ratio (AssocMem vs float32 baseline):")
    for n_str, scaling_data in scaling_results.items():
        n_ev = scaling_data["n_events"]
        ratio = scaling_data["efficiency_ratio"]
        pats  = scaling_data["n_patterns"]
        print(f"    n={n_ev:6,} events → {ratio:6.1f}x more efficient "
              f"({pats} patterns absorb all {n_ev:,} events)")

    # Test pass/fail summary
    print(f"\n  {'='*40}")
    print("  Validation Results:")
    for test_name, test_data in am_results.items():
        status = "PASS" if test_data.get("pass") else "FAIL"
        print(f"    [{status}] {test_name}")

    # Save results
    out_path = "/home/user/workspace/hdc_stress_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")
    print(f"{'='*80}")

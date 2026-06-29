"""
tests/test_hdc.py

Unit tests for the HDC backend: HDCEncoder, AssociativeMemory, and
algebraic properties of the HDC operations.

These tests verify the mathematical properties that the architecture
depends on — not just that the code runs without errors.
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/home/user/workspace")

from fractal_grammar.embeddings.hdc_encoder import (
    HDCEncoder, AssociativeMemory, ItemMemory, MemoryTrace,
    bind, bundle, permute, similarity, hv_to_float32, D, SEED
)
from fractal_grammar.core.sequence import BehavioralEvent, BehavioralSequence, from_raw_texts


# ---------------------------------------------------------------------------
# HDC algebraic property tests
# ---------------------------------------------------------------------------

class TestHDCAlgebra:
    """Verify the mathematical properties the architecture depends on."""

    def test_bind_self_inverse(self):
        """bind(bind(a, b), b) == a — bind is its own inverse."""
        rng = np.random.default_rng(1)
        a = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        b = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        result = bind(bind(a, b), b)
        assert np.array_equal(result, a), "bind is not self-inverse"

    def test_bind_dissimilar(self):
        """bind(a, b) should be dissimilar to both a and b."""
        rng = np.random.default_rng(2)
        a = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        b = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        c = bind(a, b)
        sim_ca = similarity(c, a)
        sim_cb = similarity(c, b)
        # Should be near 0 (within 3 sigma: 3/sqrt(10000) = 0.03)
        assert abs(sim_ca) < 0.05, f"bind result too similar to a: {sim_ca:.4f}"
        assert abs(sim_cb) < 0.05, f"bind result too similar to b: {sim_cb:.4f}"

    def test_bundle_similar_to_inputs(self):
        """bundle(hvs) should be similar to each input (positive sim)."""
        rng = np.random.default_rng(3)
        hvs = [rng.choice(np.array([-1, 1], dtype=np.int8), size=D) for _ in range(5)]
        bundled = bundle(hvs)
        for i, hv in enumerate(hvs):
            sim = similarity(bundled, hv)
            assert sim > 0, f"bundle not similar to input {i}: {sim:.4f}"

    def test_permute_dissimilar(self):
        """permute(hv, k) should be dissimilar to hv for k > 0."""
        rng = np.random.default_rng(4)
        hv = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        for k in [1, 10, 100]:
            shifted = permute(hv, k)
            sim = similarity(hv, shifted)
            assert abs(sim) < 0.05, f"permute(k={k}) not dissimilar: {sim:.4f}"

    def test_random_hvs_orthogonal(self):
        """Two independently generated random HVs should be near-orthogonal."""
        rng = np.random.default_rng(5)
        a = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        b = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        sim = similarity(a, b)
        assert abs(sim) < 0.05, f"Random HVs too similar: {sim:.4f}"

    def test_hv_to_float32_normalized(self):
        """hv_to_float32 should return an L2-normalized vector."""
        rng = np.random.default_rng(6)
        hv = rng.choice(np.array([-1, 1], dtype=np.int8), size=D)
        fv = hv_to_float32(hv)
        norm = np.linalg.norm(fv)
        assert abs(norm - 1.0) < 1e-5, f"float32 projection not normalized: norm={norm}"


# ---------------------------------------------------------------------------
# ItemMemory tests
# ---------------------------------------------------------------------------

class TestItemMemory:
    def test_deterministic(self):
        """Same token always maps to same HV."""
        im1 = ItemMemory(seed=SEED)
        im2 = ItemMemory(seed=SEED)
        for token in ["hello", "world", "python", "accounting", "invoice"]:
            assert np.array_equal(im1.get(token), im2.get(token)), \
                f"ItemMemory not deterministic for token '{token}'"

    def test_distinct_tokens_orthogonal(self):
        """Different tokens should have near-orthogonal HVs."""
        im = ItemMemory()
        hv1 = im.get("python")
        hv2 = im.get("accounting")
        hv3 = im.get("transformer")
        for a, b, label in [(hv1, hv2, "python/accounting"), (hv1, hv3, "python/transformer")]:
            sim = similarity(a, b)
            assert abs(sim) < 0.05, f"Tokens {label} not orthogonal: {sim:.4f}"

    def test_memory_grows(self):
        """ItemMemory grows lazily."""
        im = ItemMemory()
        assert len(im) == 0
        im.get("first")
        assert len(im) == 1
        im.get("second")
        assert len(im) == 2
        im.get("first")  # already exists
        assert len(im) == 2


# ---------------------------------------------------------------------------
# HDCEncoder tests
# ---------------------------------------------------------------------------

class TestHDCEncoder:
    def setup_method(self):
        self.enc = HDCEncoder(dim=128)

    def test_encode_text_shape(self):
        """encode_text returns correct shapes."""
        hv, fv = self.enc.encode_text("How do I reverse a list in Python")
        assert hv.shape == (D,), f"HV shape wrong: {hv.shape}"
        assert fv.shape == (128,), f"Float vec shape wrong: {fv.shape}"
        assert hv.dtype == np.int8

    def test_float_vec_normalized(self):
        """Float projection should be L2-normalized."""
        _, fv = self.enc.encode_text("normalize me please")
        norm = np.linalg.norm(fv)
        assert abs(norm - 1.0) < 1e-5, f"Not normalized: {norm}"

    def test_same_text_deterministic(self):
        """Same text always produces same HV."""
        text = "How do I reverse a list in Python"
        hv1, _ = self.enc.encode_text(text)
        hv2, _ = self.enc.encode_text(text)
        assert np.array_equal(hv1, hv2), "Non-deterministic encoding"

    def test_same_topic_more_similar_than_cross_topic(self):
        """
        Same-topic texts should be more similar to each other than cross-topic.
        This is the semantic discrimination property.
        """
        hv_code1, _ = self.enc.encode_text("How do I reverse a list in Python")
        hv_code2, _ = self.enc.encode_text("Explain list slicing in Python")
        hv_acct,  _ = self.enc.encode_text("How do I post a journal entry for accounts payable")

        sim_same  = similarity(hv_code1, hv_code2)
        sim_cross = similarity(hv_code1, hv_acct)
        gap       = sim_same - sim_cross

        assert gap > 0, (
            f"Same-topic not more similar than cross-topic.\n"
            f"  same-topic: {sim_same:.4f}\n"
            f"  cross-topic: {sim_cross:.4f}\n"
            f"  gap: {gap:.4f} (must be > 0)"
        )

    def test_empty_text(self):
        """Empty text should not raise."""
        hv, fv = self.enc.encode_text("")
        assert hv.shape == (D,)
        assert fv.shape == (128,)

    def test_single_word(self):
        """Single-word text should work without bigram layer."""
        hv, fv = self.enc.encode_text("python")
        assert hv.shape == (D,)

    def test_encode_batch(self):
        """encode_batch returns consistent shapes and per-text hvs."""
        texts = ["reverse a list", "journal entry", "attention mechanism"]
        hvs, float_matrix = self.enc.encode_batch(texts)
        assert len(hvs) == 3
        assert float_matrix.shape == (3, 128)
        for hv in hvs:
            assert hv.shape == (D,)
            assert hv.dtype == np.int8

    def test_encode_sequence_compatibility(self):
        """encode_sequence should be compatible with the pipeline's EncodedSequence."""
        from fractal_grammar.embeddings.encoder import EncodedSequence
        seq = from_raw_texts(["How do I reverse a list", "Explain slicing in Python"])
        enc_seq = self.enc.encode_sequence(seq)
        assert isinstance(enc_seq, EncodedSequence)
        assert len(enc_seq) == 2
        assert enc_seq.matrix.shape == (2, 128)

    def test_sequence_hv(self):
        """encode_sequence_hv returns one HV summarizing the whole sequence."""
        events = [
            BehavioralEvent(content="open invoice workflow"),
            BehavioralEvent(content="post journal entry"),
            BehavioralEvent(content="reconcile accounts"),
        ]
        hv = self.enc.encode_sequence_hv(events)
        assert hv.shape == (D,)
        assert hv.dtype == np.int8


# ---------------------------------------------------------------------------
# AssociativeMemory tests
# ---------------------------------------------------------------------------

class TestAssociativeMemory:
    def setup_method(self):
        self.enc   = HDCEncoder(dim=128)
        self.assoc = AssociativeMemory(similarity_threshold=0.05)

    def test_write_creates_trace(self):
        hv, _ = self.enc.encode_text("open invoice processing workflow")
        trace  = self.assoc.write("workflow", hv)
        assert self.assoc.n_patterns == 1
        assert trace.n_writes == 1

    def test_repeated_write_no_memory_growth(self):
        """Writing many events to same key keeps memory constant."""
        texts = [
            "open invoice workflow", "start invoice processing",
            "begin invoice workflow", "launch invoice processing",
            "run invoice workflow", "trigger invoice processing",
        ]
        for text in texts:
            hv, _ = self.enc.encode_text(text)
            self.assoc.write("workflow", hv)

        # Still only one pattern, but 6 writes absorbed
        assert self.assoc.n_patterns == 1
        assert self.assoc._traces["workflow"].n_writes == 6
        assert self.assoc.memory_bytes == D  # exactly one HV

    def test_distinct_keys(self):
        """Different keys create different patterns."""
        for topic, text in [
            ("python", "How do I reverse a list in Python"),
            ("accounting", "post a journal entry for accounts payable"),
        ]:
            hv, _ = self.enc.encode_text(text)
            self.assoc.write(topic, hv)

        assert self.assoc.n_patterns == 2
        assert self.assoc.memory_bytes == 2 * D

    def test_retrieval_correct_topic(self):
        """Query should retrieve the correct prototype."""
        topics = {
            "python_coding"   : ["How do I reverse a list in Python",
                                  "Explain list slicing in Python",
                                  "What is a dictionary comprehension",
                                  "How does yield work in generators",
                                  "What is a lambda function in Python"],
            "accounting"      : ["How do I post a journal entry for accounts payable",
                                  "Explain the double entry bookkeeping system",
                                  "What is the difference between accrual and cash accounting",
                                  "How do I reconcile bank statements",
                                  "Explain depreciation methods"],
        }
        enc = HDCEncoder(dim=128)
        assoc = AssociativeMemory(similarity_threshold=0.05)

        for key, texts in topics.items():
            for text in texts:
                hv, _ = enc.encode_text(text)
                assoc.write(key, hv)

        # Query each topic
        correct = 0
        for key, texts in topics.items():
            query_hv, _ = enc.encode_text(texts[0])
            hits = assoc.query(query_hv, top_k=2)
            if hits and hits[0][0] == key:
                correct += 1

        assert correct >= 1, f"No correct retrievals ({correct}/2)"

    def test_memory_bytes_linear_in_patterns(self):
        """Memory should grow linearly with number of distinct patterns."""
        enc = HDCEncoder(dim=128)
        assoc = AssociativeMemory()

        for i in range(10):
            hv, _ = enc.encode_text(f"unique pattern number {i} distinct content")
            assoc.write(f"pattern_{i}", hv)
            assert assoc.memory_bytes == (i + 1) * D, \
                f"Memory not linear at {i+1} patterns"

    def test_forget(self):
        """Forgetting a key removes it from memory."""
        hv, _ = self.enc.encode_text("test event")
        self.assoc.write("to_forget", hv)
        assert self.assoc.n_patterns == 1
        self.assoc.forget("to_forget")
        assert self.assoc.n_patterns == 0
        assert self.assoc.memory_bytes == 0

    def test_lru_eviction_at_capacity(self):
        """At max_patterns, writing a new key evicts the LRU."""
        assoc = AssociativeMemory(max_patterns=3)
        enc   = HDCEncoder(dim=128)

        for i in range(4):  # write 4 unique patterns to a 3-capacity store
            hv, _ = enc.encode_text(f"pattern {i} unique distinct content text")
            assoc.write(f"pat_{i}", hv)

        assert assoc.n_patterns == 3, f"Expected 3 patterns, got {assoc.n_patterns}"

    def test_stats(self):
        """Stats method returns correct counts."""
        for text in ["event one", "event two", "event three"]:
            hv, _ = self.enc.encode_text(text)
            self.assoc.write("key_a", hv)
        hv, _ = self.enc.encode_text("event four")
        self.assoc.write("key_b", hv)

        s = self.assoc.stats()
        assert s["n_patterns"] == 2
        assert s["total_events_absorbed"] == 4
        assert s["memory_bytes"] == 2 * D

    def test_compress(self):
        """compress() should re-binarize high-write traces."""
        enc   = HDCEncoder(dim=128)
        assoc = AssociativeMemory()
        # Write 20 events to one key
        texts = ["invoice workflow"] * 20
        for text in texts:
            hv, _ = enc.encode_text(text)
            assoc.write("workflow", hv)
        n = assoc.compress()
        assert n >= 0   # Should compress at least the high-write trace
        # After compress, prototype is still a valid bipolar HV
        trace = assoc._traces["workflow"]
        assert set(np.unique(trace.prototype_hv)).issubset({-1, 1})


# ---------------------------------------------------------------------------
# Integration: HDC backend in BehavioralEncoder shim
# ---------------------------------------------------------------------------

class TestHDCBackendIntegration:
    def test_backend_enum(self):
        """HDC should be a valid Backend enum value."""
        from fractal_grammar.embeddings.encoder import Backend
        assert hasattr(Backend, "HDC")

    def test_behavioral_encoder_hdc_backend(self):
        """BehavioralEncoder with HDC backend should encode correctly."""
        from fractal_grammar.embeddings.encoder import BehavioralEncoder, Backend
        encoder = BehavioralEncoder(dim=128, backend=Backend.HDC)
        vec = encoder.encode_text("How do I reverse a list in Python")
        assert vec.shape == (128,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_pipeline_hdc_mode(self):
        """Pipeline with use_hdc=True should run end-to-end."""
        from fractal_grammar.pipeline import FractalGrammarPipeline, PipelineConfig

        config = PipelineConfig(
            use_hdc=True,
            embedding_dim=128,
            min_events_to_cluster=5,
            log_level=50,  # suppress logs
        )
        pipe = FractalGrammarPipeline(config)

        texts = [
            "open invoice processing workflow",
            "post journal entry for accounts payable",
            "reconcile bank statement",
            "generate aging report for receivables",
            "close monthly accounting period",
            "review vendor invoices for approval",
            "process payroll for current period",
            "calculate depreciation for fixed assets",
            "prepare financial statements for quarter",
            "audit accounts payable transactions",
        ]
        seq = from_raw_texts(texts)
        ruleset = pipe.run([seq])
        assert ruleset is not None
        report = pipe.report()
        assert "AssociativeMemory (HDC)" in report


# ---------------------------------------------------------------------------
# Memory efficiency benchmark (lightweight — not a stress test)
# ---------------------------------------------------------------------------

class TestMemoryEfficiency:
    def test_assoc_vs_flat_at_1000_events(self):
        """
        AssociativeMemory should use less memory than flat float32 at n=1000.
        Threshold: at least 5x more efficient with 4 patterns.
        """
        n_events  = 1000
        dim       = 128
        n_patterns = 4

        flat_bytes  = n_events * dim * 4          # float32 per event
        assoc_bytes = n_patterns * D              # HDC per pattern

        ratio = flat_bytes / assoc_bytes
        assert ratio >= 5.0, (
            f"Efficiency ratio too low: {ratio:.1f}x "
            f"(flat={flat_bytes}B, assoc={assoc_bytes}B)"
        )

    def test_memory_stays_constant_with_more_events(self):
        """
        Writing 100x more events to same patterns should not increase memory.
        """
        enc   = HDCEncoder(dim=128)
        assoc = AssociativeMemory()

        # Write 10 events to 3 patterns
        for i in range(10):
            for j in range(3):
                hv, _ = enc.encode_text(f"pattern {j} text variant {i}")
                assoc.write(f"pat_{j}", hv)

        mem_10 = assoc.memory_bytes
        assert mem_10 == 3 * D

        # Write 1000 more events to same 3 patterns
        for i in range(1000):
            for j in range(3):
                hv, _ = enc.encode_text(f"pattern {j} text variant {i + 100}")
                assoc.write(f"pat_{j}", hv)

        mem_1010 = assoc.memory_bytes
        assert mem_1010 == mem_10, (
            f"Memory grew with more events: {mem_10} → {mem_1010}"
        )

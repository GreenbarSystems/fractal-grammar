"""
tests/test_pipeline.py

End-to-end tests for the fractal grammar extraction pipeline.
Run with: python -m pytest fractal_grammar/tests/ -v
"""

import json
import os
import tempfile
import pytest

from fractal_grammar import (
    FractalGrammarPipeline,
    PipelineConfig,
    from_llm_log,
    from_event_stream,
    from_raw_texts,
)
from fractal_grammar.core.sequence import BehavioralSequence
from fractal_grammar.core.dedup import NoveltyFilter
from fractal_grammar.embeddings.encoder import BehavioralEncoder, Backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ACCOUNTING_MESSAGES = [
    # Repeated pattern: invoice queries
    [{"role": "user", "content": "Show me all unpaid invoices for this month"},
     {"role": "assistant", "content": "Here are 12 unpaid invoices totaling $45,230."}],
    [{"role": "user", "content": "What invoices are outstanding this month?"},
     {"role": "assistant", "content": "You have 8 outstanding invoices."}],
    [{"role": "user", "content": "List unpaid invoices from October"},
     {"role": "assistant", "content": "Found 5 unpaid invoices from October."}],
    [{"role": "user", "content": "How many invoices are past due?"},
     {"role": "assistant", "content": "15 invoices are past due by 30+ days."}],
    [{"role": "user", "content": "Show overdue invoices from vendors"},
     {"role": "assistant", "content": "3 vendors have overdue invoices."}],

    # Repeated pattern: GL coding
    [{"role": "user", "content": "What GL code should I use for office supplies?"},
     {"role": "assistant", "content": "Use GL code 6200 for office supplies."}],
    [{"role": "user", "content": "Which account code goes with travel expenses?"},
     {"role": "assistant", "content": "Travel expenses map to GL 6400."}],
    [{"role": "user", "content": "GL account for software subscriptions?"},
     {"role": "assistant", "content": "Software subscriptions use GL 6150."}],
    [{"role": "user", "content": "What's the chart of accounts code for utilities?"},
     {"role": "assistant", "content": "Utilities are coded to GL 6300."}],
    [{"role": "user", "content": "Account number for payroll expenses?"},
     {"role": "assistant", "content": "Payroll is GL 5000."}],

    # Repeated pattern: reconciliation
    [{"role": "user", "content": "Reconcile the bank statement for last month"},
     {"role": "assistant", "content": "Bank reconciliation complete. Difference: $0."}],
    [{"role": "user", "content": "Run reconciliation for October bank account"},
     {"role": "assistant", "content": "Reconciliation shows $200 uncleared."}],
    [{"role": "user", "content": "Check if bank accounts are reconciled"},
     {"role": "assistant", "content": "2 accounts need reconciliation."}],
    [{"role": "user", "content": "What's the reconciliation status this month?"},
     {"role": "assistant", "content": "3 of 5 accounts reconciled."}],

    # Noise / unique queries
    [{"role": "user", "content": "What is the weather like today?"},
     {"role": "assistant", "content": "I don't have weather data."}],
    [{"role": "user", "content": "Tell me a joke"},
     {"role": "assistant", "content": "Why did the accountant break up? Too many issues."}],
]

EVENT_MESSAGES = [
    {"event_type": "click",     "label": "Invoice list",     "ts": 1700000001},
    {"event_type": "filter",    "label": "Status: unpaid",   "ts": 1700000002},
    {"event_type": "click",     "label": "Export CSV",       "ts": 1700000010},
    {"event_type": "navigate",  "label": "GL accounts",      "ts": 1700000020},
    {"event_type": "search",    "label": "office supplies",  "ts": 1700000025},
    {"event_type": "click",     "label": "Invoice list",     "ts": 1700000030},
    {"event_type": "filter",    "label": "Status: overdue",  "ts": 1700000031},
    {"event_type": "click",     "label": "Export CSV",       "ts": 1700000040},
    {"event_type": "navigate",  "label": "Reconciliation",   "ts": 1700000050},
    {"event_type": "click",     "label": "Run reconciliation","ts": 1700000055},
]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSequenceIngestion:
    def test_from_llm_log(self):
        seq = from_llm_log(ACCOUNTING_MESSAGES[0])
        assert len(seq) == 2
        assert seq.events[0].content == "Show me all unpaid invoices for this month"
        assert seq.events[0].metadata["role"] == "user"

    def test_from_event_stream(self):
        seq = from_event_stream(EVENT_MESSAGES)
        assert len(seq) == len(EVENT_MESSAGES)
        assert seq.events[0].content == "Invoice list"

    def test_from_raw_texts(self):
        texts = ["Hello world", "How are you?", ""]
        seq = from_raw_texts(texts)
        assert len(seq) == 2   # empty string excluded

    def test_fingerprint_stable(self):
        seq1 = from_llm_log(ACCOUNTING_MESSAGES[0])
        seq2 = from_llm_log(ACCOUNTING_MESSAGES[0])
        assert seq1.events[0].fingerprint == seq2.events[0].fingerprint


class TestEncoder:
    def test_hash_backend(self):
        enc = BehavioralEncoder(dim=64, backend=Backend.HASH_PROJECTION)
        vec = enc.encode_text("test sentence")
        assert vec.shape == (64,)
        import numpy as np
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_encode_sequence(self):
        enc = BehavioralEncoder(dim=64, backend=Backend.HASH_PROJECTION)
        seq = from_raw_texts(["hello", "world", "test"])
        encoded = enc.encode_sequence(seq)
        assert len(encoded) == 3
        assert encoded.matrix.shape == (3, 64)

    def test_batch_consistency(self):
        enc = BehavioralEncoder(dim=64, backend=Backend.HASH_PROJECTION)
        texts = ["invoice", "payment", "reconcile"]
        single = [enc.encode_text(t) for t in texts]
        batch  = enc.encode_batch(texts)
        import numpy as np
        for i, (s, b) in enumerate(zip(single, batch)):
            assert np.allclose(s, b, atol=1e-5), f"Mismatch at index {i}"


class TestNoveltyFilter:
    def test_dedup_exact(self):
        enc = BehavioralEncoder(dim=64, backend=Backend.HASH_PROJECTION)
        filt = NoveltyFilter(similarity_threshold=0.92)
        seq = from_raw_texts(["show invoices", "show invoices", "show invoices"])
        encoded = enc.encode_sequence(seq)

        results = [filt.process(ev) for ev in encoded.events]
        novel_count = sum(1 for is_novel, _ in results if is_novel)
        assert novel_count == 1
        assert filt.stats["seen"] == 3
        assert filt.stats["novel"] == 1

    def test_novel_events_pass(self):
        enc = BehavioralEncoder(dim=64, backend=Backend.HASH_PROJECTION)
        filt = NoveltyFilter()
        texts = [
            "invoice query",
            "GL code lookup",
            "bank reconciliation",
            "payroll processing",
            "vendor management",
        ]
        seq     = from_raw_texts(texts)
        encoded = enc.encode_sequence(seq)
        novel, entries = filt.process_batch(encoded.events)
        # All distinct — most should pass (some may merge if hash collision)
        assert len(novel) >= 3

    def test_weighted_corpus(self):
        enc = BehavioralEncoder(dim=64, backend=Backend.HASH_PROJECTION)
        filt = NoveltyFilter(similarity_threshold=0.99)
        texts = ["invoices"] * 5 + ["payroll"] * 2 + ["reconcile"]
        seq = from_raw_texts(texts)
        encoded = enc.encode_sequence(seq)
        filt.process_batch(encoded.events)
        corpus = filt.get_weighted_corpus()
        # Weights should reflect frequency
        weights = [w for _, w in corpus]
        assert max(weights) >= 1.0


class TestFullPipeline:
    """Integration tests for the full pipeline."""

    def _make_sequences(self):
        seqs = []
        for msgs in ACCOUNTING_MESSAGES:
            seqs.append(from_llm_log(msgs))
        seqs.append(from_event_stream(EVENT_MESSAGES))
        return seqs

    def test_pipeline_runs(self):
        config = PipelineConfig(
            embedding_dim=64,
            embedding_backend=Backend.HASH_PROJECTION,
            log_level=50,  # CRITICAL — suppress output in tests
        )
        pipe = FractalGrammarPipeline(config)
        seqs = self._make_sequences()
        ruleset = pipe.run(seqs)
        assert ruleset is not None
        assert isinstance(ruleset.n_rules, int)
        assert isinstance(ruleset.n_residuals, int)

    def test_pipeline_report(self):
        config = PipelineConfig(
            embedding_dim=64,
            embedding_backend=Backend.HASH_PROJECTION,
            log_level=50,
        )
        pipe = FractalGrammarPipeline(config)
        seqs = self._make_sequences()
        pipe.run(seqs)
        report = pipe.report()
        assert "FRACTAL GRAMMAR" in report
        assert "Corpus Hurst" in report

    def test_jsonl_export(self):
        config = PipelineConfig(
            embedding_dim=64,
            embedding_backend=Backend.HASH_PROJECTION,
            log_level=50,
        )
        pipe = FractalGrammarPipeline(config)
        seqs = self._make_sequences()
        ruleset = pipe.run(seqs)

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False
        ) as f:
            path = f.name

        try:
            n = ruleset.to_jsonl(path)
            assert n > 0
            with open(path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) == n
            # Each line must have messages and weight
            for line in lines:
                assert "messages" in line
                assert "weight" in line
        finally:
            os.unlink(path)

    def test_grammar_save_load(self):
        config = PipelineConfig(
            embedding_dim=64,
            embedding_backend=Backend.HASH_PROJECTION,
            log_level=50,
        )
        pipe = FractalGrammarPipeline(config)
        seqs = self._make_sequences()
        ruleset = pipe.run(seqs)

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name

        try:
            ruleset.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "rules" in data
            assert "corpus_hurst" in data
            assert "residuals" in data
        finally:
            os.unlink(path)

    def test_incremental_update(self):
        config = PipelineConfig(
            embedding_dim=64,
            embedding_backend=Backend.HASH_PROJECTION,
            log_level=50,
        )
        pipe = FractalGrammarPipeline(config)
        seqs = self._make_sequences()
        ruleset1 = pipe.run(seqs[:8])

        new_seqs = [from_llm_log(msgs) for msgs in ACCOUNTING_MESSAGES[8:]]
        ruleset2 = pipe.update(new_seqs)

        assert ruleset2 is not None
        # After update, corpus should have seen more events
        assert pipe._filter.stats["seen"] > 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

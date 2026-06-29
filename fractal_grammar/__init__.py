"""
fractal_grammar
===============
Fractal behavioral grammar extraction for micro AI personalization.

Core API
--------
FractalGrammarPipeline  — end-to-end pipeline
PipelineConfig          — configuration dataclass

Ingestion helpers
-----------------
from_llm_log            — OpenAI-style message lists
from_event_stream       — timestamped event dicts
from_raw_texts          — plain string lists
from_jsonl              — JSONL file loader

Output
------
GrammarRuleset          — compressed grammar (save / to_jsonl)
GrammarRule             — individual behavioral rule
"""

from fractal_grammar.pipeline import FractalGrammarPipeline, PipelineConfig
from fractal_grammar.core.sequence import (
    BehavioralEvent,
    BehavioralSequence,
    from_llm_log,
    from_event_stream,
    from_raw_texts,
    from_jsonl,
)
from fractal_grammar.grammar.extractor import GrammarRuleset, GrammarRule

__all__ = [
    "FractalGrammarPipeline",
    "PipelineConfig",
    "BehavioralEvent",
    "BehavioralSequence",
    "from_llm_log",
    "from_event_stream",
    "from_raw_texts",
    "from_jsonl",
    "GrammarRuleset",
    "GrammarRule",
]

__version__ = "0.1.0"

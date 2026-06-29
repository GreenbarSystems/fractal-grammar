"""
examples/basic_usage.py

Working demonstration of the fractal grammar extraction pipeline.

Run with:
    python fractal_grammar/examples/basic_usage.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fractal_grammar import (
    FractalGrammarPipeline,
    PipelineConfig,
    from_llm_log,
    from_event_stream,
    from_raw_texts,
)
from fractal_grammar.embeddings.encoder import Backend


# ---------------------------------------------------------------------------
# Sample behavioral data — simulates a user's interaction history
# ---------------------------------------------------------------------------

# LLM interaction logs — 3 dominant behavioral patterns:
#   1. Invoice/AP queries
#   2. GL code lookups
#   3. Reconciliation requests
# Plus some noise

INTERACTION_LOGS = [
    # --- Invoice pattern (high frequency) ---
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
    [{"role": "user", "content": "Which invoices need approval?"},
     {"role": "assistant", "content": "4 invoices are pending your approval."}],
    [{"role": "user", "content": "Get me the AP aging report"},
     {"role": "assistant", "content": "AP aging: $12K current, $8K 30-day, $3K 60-day."}],

    # --- GL code pattern (medium frequency) ---
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
    [{"role": "user", "content": "Which GL code for equipment depreciation?"},
     {"role": "assistant", "content": "Equipment depreciation uses GL 7100."}],

    # --- Reconciliation pattern ---
    [{"role": "user", "content": "Reconcile the bank statement for last month"},
     {"role": "assistant", "content": "Bank reconciliation complete. Difference: $0."}],
    [{"role": "user", "content": "Run reconciliation for October bank account"},
     {"role": "assistant", "content": "Reconciliation shows $200 uncleared."}],
    [{"role": "user", "content": "Check if bank accounts are reconciled"},
     {"role": "assistant", "content": "2 accounts need reconciliation."}],
    [{"role": "user", "content": "What's the reconciliation status this month?"},
     {"role": "assistant", "content": "3 of 5 accounts reconciled."}],
    [{"role": "user", "content": "Are all accounts balanced and reconciled?"},
     {"role": "assistant", "content": "Main operating account has a $50 discrepancy."}],

    # --- Noise: one-off queries ---
    [{"role": "user", "content": "What time zone is the server in?"},
     {"role": "assistant", "content": "The server runs UTC."}],
    [{"role": "user", "content": "How do I export data to Excel?"},
     {"role": "assistant", "content": "Click Export → Excel from the toolbar."}],
]

# Event stream — UI click patterns
EVENT_LOGS = [
    {"event_type": "click",    "label": "Invoice list",       "ts": 1700000001},
    {"event_type": "filter",   "label": "Status: unpaid",     "ts": 1700000002},
    {"event_type": "click",    "label": "Export CSV",         "ts": 1700000010},
    {"event_type": "navigate", "label": "GL accounts",        "ts": 1700000020},
    {"event_type": "search",   "label": "office supplies",    "ts": 1700000025},
    {"event_type": "click",    "label": "Invoice list",       "ts": 1700000030},
    {"event_type": "filter",   "label": "Status: overdue",    "ts": 1700000031},
    {"event_type": "click",    "label": "Export CSV",         "ts": 1700000040},
    {"event_type": "navigate", "label": "Reconciliation",     "ts": 1700000050},
    {"event_type": "click",    "label": "Run reconciliation", "ts": 1700000055},
    {"event_type": "navigate", "label": "Invoice list",       "ts": 1700000060},
    {"event_type": "filter",   "label": "Vendor: Acme Corp",  "ts": 1700000061},
    {"event_type": "click",    "label": "Approve invoice",    "ts": 1700000065},
]


def main():
    print("\n" + "=" * 60)
    print("  FRACTAL GRAMMAR EXTRACTION — DEMO")
    print("=" * 60)

    # --- Build input sequences ---
    sequences = []
    for msgs in INTERACTION_LOGS:
        sequences.append(from_llm_log(msgs))
    sequences.append(from_event_stream(EVENT_LOGS))

    print(f"\nInput: {len(sequences)} sequences")
    print(f"       {sum(len(s) for s in sequences)} total events")

    # --- Configure pipeline ---
    # Using HASH_PROJECTION backend for zero-download demo.
    # Swap to SENTENCE_TRANSFORMER for production quality.
    config = PipelineConfig(
        embedding_dim            = 128,
        embedding_backend        = Backend.HASH_PROJECTION,
        dedup_threshold          = 0.88,   # slightly looser for demo data
        min_events_to_cluster    = 4,
        similarity_threshold     = 0.3,    # lower for hash backend (no semantic similarity)
        min_hurst                = 0.45,
        min_pattern_depth        = 2,
    )

    # --- Run pipeline ---
    pipe    = FractalGrammarPipeline(config)
    ruleset = pipe.run(sequences)

    # --- Print report ---
    print("\n" + pipe.report())

    # --- Show extracted rules ---
    if ruleset.rules:
        print("\nExtracted Grammar Rules:")
        print("-" * 60)
        for rule in ruleset.rules:
            print(f"\n  [{rule.rule_id}] {rule.label}")
            print(f"  Hurst:       {rule.hurst_exponent:.3f}  "
                  f"({'strong' if rule.hurst_exponent >= 0.7 else 'moderate'})")
            print(f"  Compression: {rule.compression_ratio:.1f}x")
            print(f"  Weight:      {rule.total_weight:.1f}")
            print(f"  Depth:       {rule.depth} levels")
            print(f"  Example:     '{rule.representative[:70]}'")
            if rule.examples:
                print(f"  Also:        '{rule.examples[0][:70]}'")
    else:
        print("\nNo strong patterns detected — all events stored as residuals.")
        print(f"({ruleset.n_residuals} residuals retained)")

    # --- Export outputs ---
    out_dir = "/home/user/workspace"

    jsonl_path   = f"{out_dir}/training_data.jsonl"
    grammar_path = f"{out_dir}/grammar.json"

    n_pairs = ruleset.to_jsonl(jsonl_path)
    ruleset.save(grammar_path)

    print(f"\nOutputs written:")
    print(f"  Training JSONL : {jsonl_path}  ({n_pairs} pairs)")
    print(f"  Grammar JSON   : {grammar_path}")

    # --- Show a sample training pair ---
    if ruleset.rules:
        print("\nSample training pair (first rule):")
        pairs = ruleset.rules[0].to_jsonl_training_pair()
        import json
        print(json.dumps(pairs[0], indent=2))


if __name__ == "__main__":
    main()

"""CLI entry point for Ledger (M1 slice).

Usage:
    python run.py                         # runs on the flagship lending dataset
    python run.py path/to/data.csv        # runs on your own dataset
    python run.py data.csv "why are losses rising?"   # with a leadership question

Set ANTHROPIC_API_KEY to enable the natural-language layer (narratives, Q&A);
without it, the deterministic data-science core still runs end to end.
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from ledger.config import SETTINGS
from ledger.graph import run_analysis
from ledger.nodes.qa import answer_question
from ledger.report import render_report

load_dotenv()


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_dataset
    question = sys.argv[2] if len(sys.argv) > 2 else None

    mode = "LLM-enabled" if SETTINGS.has_api_key else "deterministic (no API key)"
    print(f"\n=== Ledger — running in {mode} mode ===")
    print(f"Dataset: {dataset}\n")

    state = run_analysis(dataset, question)

    print("--- DATA PROFILE ---")
    print(state.profile.quality_summary, "\n")

    print("--- PLAN ---")
    for step in state.plan:
        print(f"  • {step}")
    print()

    print("--- FINDINGS ---")
    for f in state.findings:
        print(f"  [{f.confidence:>6}] {f.claim}")
        for lim in f.limitations:
            print(f"           ⚠ {lim}")
    print()

    if state.projections:
        print("--- PROJECTIONS (with 80% intervals + backtest) ---")
        for p in state.projections:
            bt = p.backtest_error
            print(f"  {p.metric} @ {p.horizon}: {p.point:.1%} "
                  f"[{p.lower:.1%}, {p.upper:.1%}]  "
                  f"(backtest MAE {bt.get('MAE')}, beats_naive={bt.get('beats_naive')})")
        print()

    print(f"--- VALIDATOR --- overall confidence: {state.overall_confidence}")
    for g in state.guardrails:
        print(f"  • {g}")
    print()

    print("--- EXECUTIVE SUMMARY (BLUF) ---")
    print(state.executive_summary, "\n")

    print("--- CHARTS ---")
    for fig in state.figures:
        print(f"  • {fig}")
    print()

    # demo the Part-3 Q&A layer with a couple of grounded questions
    print("--- Q&A (Part 3 demo) ---")
    for q in ["What is the strongest driver of the outcome?",
              "Which model gave the highest accuracy?"]:
        print(f"  Q: {q}")
        print(f"  A: {answer_question(state, q)}\n")

    # M4: write the executive report
    report_path = render_report(state)
    print(f"--- REPORT --- wrote {report_path}  (open in a browser; Print → Save as PDF)")


if __name__ == "__main__":
    main()

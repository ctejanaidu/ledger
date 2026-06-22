"""Known-answer evals (SPEC §10) — prove the agent finds the TRUTH, not vibes.

The lending dataset (seed=42) has a known structure: default rate ~25%, risk
rises monotonically with `grade` (E worst), and there's a recent small-business
loss spike. These tests assert the deterministic core recovers that ground truth.

Runs WITHOUT pytest:  python -m evals.test_analyst
Runs WITH pytest too:  pytest evals/
"""
from __future__ import annotations

from ledger.config import SETTINGS
from ledger.graph import run_analysis

_STATE = None


def _state():
    global _STATE
    if _STATE is None:
        _STATE = run_analysis(SETTINGS.default_dataset)
    return _STATE


def test_overall_rate_recovered():
    """Agent's reported default rate matches the known ~25%."""
    s = _state()
    rate_finding = next(f for f in s.findings if "Overall" in f.claim)
    rate = rate_finding.evidence["rate"]
    assert 0.20 <= rate <= 0.30, f"expected ~0.25, got {rate}"


def test_grade_is_strongest_driver():
    """Agent identifies `grade` as the strongest categorical driver."""
    s = _state()
    assert any("grade" in f.claim and "driver" in f.claim for f in s.findings), \
        "agent did not flag grade as the strongest driver"


def test_grade_E_is_worst():
    """Within the grade breakdown, E has the highest default rate."""
    s = _state()
    driver = next(f for f in s.findings if "driver" in f.claim)
    by_grade = driver.evidence.get("by_grade", {})
    assert by_grade, "no per-grade breakdown stored"
    worst = max(by_grade, key=by_grade.get)
    assert worst == "E", f"expected E worst, got {worst} ({by_grade})"


def test_confidence_is_calibrated():
    """High-n findings get high confidence (knows what it does know)."""
    s = _state()
    overall = next(f for f in s.findings if "Overall" in f.claim)
    assert overall.confidence == "high", overall.confidence


def test_qa_reframes_accuracy_to_right_metric():
    """Asked about 'accuracy', Q&A answers with the selected model AND the right metric."""
    from ledger.nodes.qa import answer_question
    s = _state()
    ans = answer_question(s, "Which model gave the highest accuracy?").lower()
    assert s.model_leaderboard.selected.lower() in ans
    assert "roc-auc" in ans or "pr-auc" in ans  # reframed away from raw accuracy


def test_modeler_runs_and_selects():
    """Part 1 modeler produces a leaderboard and selects one of its candidates."""
    lb = _state().model_leaderboard
    assert lb is not None and lb.target == "default"
    names = {r.name for r in lb.candidates} | ({"Ensemble"} if lb.ensemble else set())
    assert lb.selected in names


def test_modeler_uses_roc_auc_on_balanced_target():
    """Lending (~25% positive) is balanced -> selection metric is ROC-AUC, not PR-AUC."""
    lb = _state().model_leaderboard
    assert lb.candidates[0].primary_metric == "ROC-AUC"


def test_modeler_beats_random_baseline():
    """The selected model must beat a coin flip (ROC-AUC > 0.5) by a clear margin."""
    lb = _state().model_leaderboard
    selected = next(r for r in lb.candidates + ([lb.ensemble] if lb.ensemble else [])
                    if r.name == lb.selected)
    assert selected.test_score > 0.6, selected.test_score


def test_diagnostician_decomposes_change():
    """Root-cause: a rate/mix decomposition finding with both effects is produced."""
    s = _state()
    dec = next((f for f in s.findings if "decomposition" in f.method.lower()), None)
    assert dec is not None
    assert "rate_effect" in dec.evidence and "mix_effect" in dec.evidence


def test_forecaster_intervals_and_backtest():
    """Every projection has a valid interval and a reported backtest error."""
    s = _state()
    assert s.projections, "no projections produced"
    for p in s.projections:
        assert p.lower <= p.point <= p.upper, (p.lower, p.point, p.upper)
        assert "MAE" in p.backtest_error and "beats_naive" in p.backtest_error


def test_validator_bans_overclaims_and_sets_confidence():
    """Validator emits the anti-overclaim guardrail and an aggregate confidence."""
    s = _state()
    assert s.overall_confidence in {"high", "medium", "low"}
    assert any("overclaiming words" in g for g in s.guardrails)
    assert any("recompute" in g.lower() for g in s.guardrails)


# --- regression coverage (the modeler must handle continuous targets too) ---
_REG_STATE = None


def _reg_state():
    """A regression dataset with continuous, all-unique features (the case that
    previously broke: KFold vs StratifiedKFold, and floats wrongly flagged as IDs)."""
    global _REG_STATE
    if _REG_STATE is None:
        import os
        import tempfile

        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 1500
        x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "target": 3 * x1 - 2 * x2 + rng.normal(0, 0.5, n)})
        path = os.path.join(tempfile.gettempdir(), "ledger_reg_eval.csv")
        df.to_csv(path, index=False)
        _REG_STATE = run_analysis(path)
    return _REG_STATE


def test_modeler_handles_regression():
    """Continuous target -> regression task selected by RMSE (not a classification metric)."""
    lb = _reg_state().model_leaderboard
    assert lb is not None and lb.task_type == "regression"
    assert lb.candidates[0].primary_metric == "RMSE"


def test_regression_uses_continuous_features():
    """Continuous all-unique features must NOT be dropped as IDs — the model must fit well."""
    lb = _reg_state().model_leaderboard
    selected = next(r for r in lb.candidates + ([lb.ensemble] if lb.ensemble else [])
                    if r.name == lb.selected)
    assert selected.all_metrics["R2"] > 0.8, selected.all_metrics


def test_explicit_target_selection():
    """A target that auto-detection can't name (e.g. 'house_price') still models when
    the user picks it explicitly."""
    import os
    import tempfile

    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 1200
    sqft, beds = rng.normal(2000, 500, n), rng.integers(1, 6, n)
    df = pd.DataFrame({"sqft": sqft, "beds": beds,
                       "house_price": 100 * sqft + 5000 * beds + rng.normal(0, 8000, n)})
    path = os.path.join(tempfile.gettempdir(), "ledger_target_eval.csv")
    df.to_csv(path, index=False)
    auto = run_analysis(path)                       # nothing auto-detected
    assert auto.model_leaderboard is None
    picked = run_analysis(path, target="house_price")  # explicit override
    assert picked.model_leaderboard is not None
    assert picked.model_leaderboard.task_type == "regression"


def test_user_technique_added_to_panel():
    """A valid user-requested technique is trained and appears in the leaderboard."""
    s = run_analysis(SETTINGS.default_dataset, target="default", user_technique="Decision Tree")
    names = [r.name for r in s.model_leaderboard.candidates]
    assert any("Decision Tree" in n for n in names), names


def test_user_technique_mismatch_is_reported():
    """A technique that can't train (regression model on a classification task) is reported,
    not silently dropped."""
    s = run_analysis(SETTINGS.default_dataset, target="default", user_technique="Linear Regression")
    note = (s.model_leaderboard.user_technique_note or "").lower()
    assert "could not be trained" in note


def test_text_binary_target_is_analyzed():
    """A Yes/No (text) target is detected, encoded to 0/1, and analyzed — not bounced to
    'no binary target detected'."""
    import os
    import tempfile

    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(2)
    n = 800
    x = rng.normal(0, 1, n)
    churn = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-x)), "Yes", "No")
    path = os.path.join(tempfile.gettempdir(), "ledger_textbin_eval.csv")
    pd.DataFrame({"x": x, "churn": churn}).to_csv(path, index=False)
    s = run_analysis(path)
    rate = next((f for f in s.findings if "rate is" in f.claim.lower()), None)
    assert rate is not None and "positive class" in rate.claim.lower()
    assert s.model_leaderboard is not None
    assert s.model_leaderboard.task_type == "binary_classification"


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{passed}/{len(tests)} eval checks passed")
    # Non-zero exit on any failure so CI (and shells) can detect it.
    sys.exit(0 if passed == len(tests) else 1)

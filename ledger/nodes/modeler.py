"""Modeler node — PART 1: custom AutoML loop (SPEC §8a).

Given a target, this trains its OWN panel of models, ensembles them, and selects
the winner BY THE RIGHT METRIC (PR-AUC for imbalanced classification, not raw
accuracy) using cross-validation + a held-out test set, then explains the winner
with model-agnostic permutation importance. The result is a ModelLeaderboard
artifact that grounds the Part-3 Q&A ("which model won and why?").

Deterministic (pure scikit-learn) — runs with no API key and is unit-tested.
xgboost/lightgbm are used automatically IF importable (need libomp on macOS);
otherwise scikit-learn's HistGradientBoosting fills the gradient-boosting slot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..dataio import load_csv
from ..state import ChartSpec, Finding, ModelLeaderboard, ModelResult
from ..targeting import resolve_target, to_binary01
from ..tools import charts

RS = 42
MAX_ROWS = 80_000          # subsample cap (prevalence-preserving) for tractability
PERM_SAMPLE = 5_000        # rows used for permutation importance
OUTPUT_DIR = "outputs/charts"


# --- problem framing --------------------------------------------------------
def _task_type(y: pd.Series) -> str:
    nun = y.nunique(dropna=True)
    if nun == 2:
        return "binary_classification"
    if not pd.api.types.is_numeric_dtype(y) or nun <= 10:
        return "multiclass_classification"
    return "regression"


def _feature_columns(df: pd.DataFrame, target: str, profile) -> list[str]:
    drop = {target}
    drop |= {c.name for c in profile.columns
             if "possible_id" in c.quality_flags or c.name.lower() == "id"
             or c.name.lower().endswith("_id")}
    if profile.time_column:
        drop.add(profile.time_column)  # raw dates -> leakage/complexity; drop for v1
    feats = []
    for c in df.columns:
        if c in drop or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        # skip very high-cardinality string columns (free text / hidden ids)
        if df[c].dtype == "object" and df[c].nunique() > 50:
            continue
        feats.append(c)
    return feats


# --- metric policy (the RIGHT metric, not accuracy) -------------------------
def _metric_plan(task: str, pos_rate: float) -> tuple[str, str, str]:
    """Return (display_name, sklearn_scoring, rationale)."""
    if task == "regression":
        return ("RMSE", "neg_root_mean_squared_error",
                "Regression task: selected by RMSE (lower is better); MAE and R2 also reported.")
    if pos_rate < 0.05 or pos_rate > 0.95:
        return ("PR-AUC", "average_precision",
                f"Imbalanced binary target ({pos_rate:.2%} positive): selected by PR-AUC "
                "(average precision), which focuses on the rare positive class. Raw accuracy "
                "and even ROC-AUC look deceptively high under this much imbalance.")
    return ("ROC-AUC", "roc_auc",
            "Balanced binary target: selected by ROC-AUC; PR-AUC and F1 also reported.")


def _candidates(task: str):
    if task == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
            "GradientBoosting": HistGradientBoostingRegressor(random_state=RS),
        }
    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                   class_weight="balanced", random_state=RS),
            "GradientBoosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                              random_state=RS),
        }
        # use xgboost/lightgbm automatically if the runtime can import them
        try:  # pragma: no cover
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(eval_metric="aucpr", random_state=RS)
        except Exception:
            pass
    return models


# --- user-requested technique (optional) ------------------------------------
SLOW_ROW_LIMIT = 20_000  # don't run O(n^2)-ish techniques on big data (would hang)
TECHNIQUE_CHOICES = [
    "Logistic Regression", "Linear Regression", "Ridge Regression", "Lasso Regression",
    "Random Forest", "Gradient Boosting", "Decision Tree", "K-Nearest Neighbors",
    "SVM", "Naive Bayes", "Neural Network (MLP)",
]
_ALIASES = {
    "logistic": "Logistic Regression", "logreg": "Logistic Regression",
    "linear": "Linear Regression", "ols": "Linear Regression",
    "ridge": "Ridge Regression", "lasso": "Lasso Regression",
    "rf": "Random Forest", "randomforest": "Random Forest",
    "gbm": "Gradient Boosting", "gradient boosting": "Gradient Boosting", "xgboost": "Gradient Boosting",
    "tree": "Decision Tree", "cart": "Decision Tree",
    "knn": "K-Nearest Neighbors", "svc": "SVM", "svr": "SVM",
    "support vector machine": "SVM", "naive bayes": "Naive Bayes", "nb": "Naive Bayes",
    "mlp": "Neural Network (MLP)", "neural network": "Neural Network (MLP)", "nn": "Neural Network (MLP)",
}


def _technique_registry() -> dict:
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    return {
        "Logistic Regression": dict(clf=lambda: LogisticRegression(max_iter=1000, class_weight="balanced"), reg=None, slow=False),
        "Linear Regression":   dict(clf=None, reg=lambda: LinearRegression(), slow=False),
        "Ridge Regression":    dict(clf=None, reg=lambda: Ridge(), slow=False),
        "Lasso Regression":    dict(clf=None, reg=lambda: Lasso(), slow=False),
        "Random Forest":       dict(clf=lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=RS), reg=lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RS), slow=False),
        "Gradient Boosting":   dict(clf=lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=RS), reg=lambda: HistGradientBoostingRegressor(random_state=RS), slow=False),
        "Decision Tree":       dict(clf=lambda: DecisionTreeClassifier(class_weight="balanced", random_state=RS), reg=lambda: DecisionTreeRegressor(random_state=RS), slow=False),
        "K-Nearest Neighbors": dict(clf=lambda: KNeighborsClassifier(), reg=lambda: KNeighborsRegressor(), slow=True),
        "SVM":                 dict(clf=lambda: SVC(probability=True, class_weight="balanced", random_state=RS), reg=lambda: SVR(), slow=True),
        "Naive Bayes":         dict(clf=lambda: GaussianNB(), reg=None, slow=False),
        "Neural Network (MLP)": dict(clf=lambda: MLPClassifier(max_iter=500, random_state=RS), reg=lambda: MLPRegressor(max_iter=500, random_state=RS), slow=False),
    }


def _resolve_user_technique(name: str, task: str, n_train: int):
    """Return (estimator_or_None, resolved_name_or_None, note_or_None)."""
    reg = _technique_registry()
    key = name.strip()
    resolved = next((k for k in reg if k.lower() == key.lower()), None) or _ALIASES.get(key.lower())
    if resolved is None:
        return None, None, f"Requested technique '{name}' isn't recognized — used the auto-selected panel instead."
    spec = reg[resolved]
    factory = spec["reg"] if task == "regression" else spec["clf"]
    if factory is None:
        kind = "regression" if task == "regression" else "classification"
        other = "classification" if task == "regression" else "regression"
        return None, resolved, (f"'{resolved}' is a {other} technique, but this is a {kind} "
                                "problem — it could not be trained; used the auto-selected panel instead.")
    if spec["slow"] and n_train > SLOW_ROW_LIMIT:
        return None, resolved, (f"'{resolved}' was skipped — too slow for {n_train:,} training rows; "
                                "used the auto-selected panel instead.")
    return factory(), resolved, None


def _preprocessor(df: pd.DataFrame, feats: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          # dense output: HistGradientBoosting rejects sparse, and
                          # ColumnTransformer would otherwise emit sparse for
                          # categorical-heavy data, breaking that model.
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    return pre, num, cat


# --- evaluation -------------------------------------------------------------
def _test_metrics(task, pipe, X_test, y_test) -> dict:
    if task == "regression":
        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        return {"RMSE": round(rmse, 4), "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
                "R2": round(float(r2_score(y_test, pred)), 4)}
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    return {
        "PR-AUC": round(float(average_precision_score(y_test, proba)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, proba)), 4),
        "F1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "Brier": round(float(brier_score_loss(y_test, proba)), 4),
    }


def _subsample(df, target, task):
    if len(df) <= MAX_ROWS:
        return df, False
    strat = df[target] if task != "regression" else None
    keep, _ = train_test_split(df, train_size=MAX_ROWS, stratify=strat, random_state=RS)
    return keep, True


def modeler(state) -> dict:
    df = load_csv(state.dataset_path)
    profile = state.profile
    target = resolve_target(state.target, profile.target_candidates, list(df.columns))
    if target is None:
        return {"log": state.log + ["modeler: no target -> skipped"]}

    df = df.dropna(subset=[target])
    y_all = df[target]
    task = _task_type(y_all)

    # need enough positives to model responsibly
    if task != "regression":
        if not pd.api.types.is_numeric_dtype(y_all) or set(y_all.unique()) != {0, 1}:
            y_all, _ = to_binary01(y_all)  # event class -> 1 (keeps PR-AUC meaningful)
            df = df.assign(**{target: y_all})
        pos = int(y_all.sum())
        if pos < 15:
            return {"log": state.log + [f"modeler: only {pos} positives -> skipped"]}
    pos_rate = float(y_all.mean()) if task != "regression" else 0.5

    df, sampled = _subsample(df, target, task)
    feats = _feature_columns(df, target, profile)
    X, y = df[feats], df[target]

    metric_name, scoring, rationale = _metric_plan(task, pos_rate)
    pre, num_cols, cat_cols = _preprocessor(df, feats)

    strat = y if task != "regression" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=strat)
    # StratifiedKFold needs discrete classes; regression must use plain KFold.
    n_splits = 3 if len(X_tr) > 20_000 else 5
    cv = (KFold(n_splits=n_splits, shuffle=True, random_state=RS) if task == "regression"
          else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RS))

    candidates = _candidates(task)
    auto_names = list(candidates)   # the reliable panel used to build the ensemble
    results: list[ModelResult] = []
    fitted: dict[str, Pipeline] = {}
    scored: dict[str, float] = {}  # internal higher-is-better CV score for selection

    # optional: fold in the user's requested ML technique alongside the auto panel
    user_note = None
    user_model_name = None
    if state.user_technique:
        est, resolved, note = _resolve_user_technique(state.user_technique, task, len(X_tr))
        if est is None:
            user_note = note
        elif resolved.lower().replace(" ", "") in {n.lower().replace(" ", "") for n in auto_names}:
            user_note = f"Your pick ({resolved}) is already part of the auto-selected panel."
        else:
            user_model_name = f"{resolved} (your pick)"
            candidates[user_model_name] = est

    for name, est in candidates.items():
        try:  # one bad model must not crash the pipeline (esp. on uploaded data)
            pipe = Pipeline([("pre", pre), ("model", est)])
            cv_internal = float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring)))
            pipe.fit(X_tr, y_tr)
            tm = _test_metrics(task, pipe, X_te, y_te)
        except Exception as exc:  # pragma: no cover
            import sys
            print(f"[ledger] model {name} failed, skipping: {exc}", file=sys.stderr)
            continue
        cv_disp = -cv_internal if metric_name == "RMSE" else cv_internal
        results.append(ModelResult(
            name=name, primary_metric=metric_name,
            cv_score=round(cv_disp, 4), test_score=tm[metric_name], all_metrics=tm,
            notes="class_weight=balanced" if task != "regression" else "",
        ))
        fitted[name] = pipe
        scored[name] = cv_internal

    if not scored:  # every model failed -> skip modeling rather than crash
        return {"log": state.log + ["modeler: all candidate models failed -> skipped"]}

    # finalize the note about the user's requested technique
    if user_model_name and user_note is None:
        clean = user_model_name.replace(" (your pick)", "")
        user_note = (f"Trained your requested technique ({clean}) and included it in the leaderboard."
                     if user_model_name in scored else
                     f"Your requested technique ({clean}) could not be trained (it errored during "
                     "fitting); the auto-selected panel was used.")

    # --- ensemble (soft-vote / average the AUTO panel only, for reliability) ---
    ens_result = None
    try:
        ests = [(n, candidates[n]) for n in auto_names if n in fitted]
        ens = (VotingClassifier(ests, voting="soft") if task != "regression"
               else VotingRegressor(ests))
        pipe = Pipeline([("pre", pre), ("model", ens)])
        cv_internal = float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring)))
        pipe.fit(X_tr, y_tr)
        tm = _test_metrics(task, pipe, X_te, y_te)
        cv_disp = -cv_internal if metric_name == "RMSE" else cv_internal
        ens_result = ModelResult(name="Ensemble", primary_metric=metric_name,
                                 cv_score=round(cv_disp, 4), test_score=tm[metric_name],
                                 all_metrics=tm, notes="soft-vote of the panel")
        fitted["Ensemble"] = pipe
        scored["Ensemble"] = cv_internal
    except Exception as exc:  # pragma: no cover
        import sys
        print(f"[ledger] ensemble failed: {exc}", file=sys.stderr)

    # --- selection by CV (higher internal score = better) ---
    best = max(scored, key=scored.get)
    best_pipe = fitted[best]
    best_res = next((r for r in results + ([ens_result] if ens_result else []) if r.name == best))
    overfit_gap = round(abs(best_res.cv_score - best_res.test_score), 4)

    # --- interpretation: model-agnostic permutation importance ---
    n_perm = min(PERM_SAMPLE, len(X_te))
    Xp, yp = X_te.iloc[:n_perm], y_te.iloc[:n_perm]
    drivers: list[tuple[str, float]] = []
    try:
        pi = permutation_importance(best_pipe, Xp, yp, scoring=scoring,
                                    n_repeats=5, random_state=RS, n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1]
        drivers = [(feats[i], round(float(pi.importances_mean[i]), 5))
                   for i in order[:8] if pi.importances_mean[i] > 0]
    except Exception as exc:  # pragma: no cover
        import sys
        print(f"[ledger] permutation importance failed: {exc}", file=sys.stderr)

    selection_reason = (
        f"Selected '{best}' — best cross-validated {metric_name} ({best_res.cv_score:.4f}); "
        f"held-out test {metric_name} {best_res.test_score:.4f} (CV-vs-test gap {overfit_gap:.4f}). "
        + (f"Top drivers: {', '.join(n for n, _ in drivers[:3])}." if drivers else "")
    )
    leaderboard = ModelLeaderboard(
        task_type=task, target=target, metric_rationale=rationale,
        candidates=results, ensemble=ens_result, selected=best,
        selection_reason=selection_reason, user_technique_note=user_note,
    )

    # --- a finding for the report + Q&A ---
    note_bits = []
    if sampled:
        note_bits.append(f"trained on a prevalence-preserving sample of {len(df):,} rows")
    acc_note = ""
    if task != "regression" and "Accuracy" in best_res.all_metrics:
        acc_note = (f" For contrast, raw accuracy is {best_res.all_metrics['Accuracy']:.3f} — "
                    "misleading under imbalance, which is why we select on " + metric_name + ".")
    finding = Finding(
        claim=(f"Best model is {best} with held-out {metric_name}="
               f"{best_res.test_score:.3f} (CV {best_res.cv_score:.3f}); "
               + (f"top drivers: {', '.join(n for n, _ in drivers[:3])}." if drivers else
                  "feature importance unavailable.") + acc_note
               + (f" Note on your requested technique: {user_note}" if user_note else "")),
        evidence={"leaderboard": {r.name: r.test_score for r in results}
                  | ({"Ensemble": ens_result.test_score} if ens_result else {}),
                  "metric": metric_name, "cv_test_gap": overfit_gap,
                  "drivers": dict(drivers)},
        confidence="high" if overfit_gap < 0.05 else "medium",
        method=f"custom model panel + ensemble, selected by CV {metric_name}, "
               "interpreted with permutation importance",
        limitations=[
            "Permutation importance reflects predictive association, not causation.",
            f"CV-vs-test gap {overfit_gap:.3f} indicates "
            + ("low" if overfit_gap < 0.05 else "some") + " overfitting risk.",
        ] + ([note_bits[0] + " for tractability."] if note_bits else []),
    )

    # --- charts: model comparison + feature importance ---
    new_charts: list[ChartSpec] = []
    figures = list(state.figures)
    all_res = results + ([ens_result] if ens_result else [])
    fig = charts.bar([r.name for r in all_res], [r.test_score for r in all_res],
                     title=f"Model comparison — held-out {metric_name}",
                     takeaway=f"{best} wins on {metric_name} ({best_res.test_score:.3f}).",
                     x_title="model", y_title=metric_name)
    p = charts.save_html(fig, OUTPUT_DIR, "model_comparison")
    figures.append(p)
    new_charts.append(ChartSpec(
        chart_id="model_comparison", title=f"Model comparison ({metric_name})",
        chart_type="bar", encoded_fields={"x": "model", "y": metric_name},
        underlying_numbers={r.name: r.test_score for r in all_res},
        takeaway=f"{best} is the best model by {metric_name}.", figure_path=p))

    if drivers:
        fig = charts.bar([n for n, _ in drivers], [v for _, v in drivers],
                         title=f"What drives {target} — permutation importance ({best})",
                         takeaway=f"{drivers[0][0]} is the most important predictor.",
                         x_title="feature", y_title=f"importance (drop in {metric_name})")
        p = charts.save_html(fig, OUTPUT_DIR, "feature_importance")
        figures.append(p)
        new_charts.append(ChartSpec(
            chart_id="feature_importance", title=f"Feature importance ({best})",
            chart_type="bar", encoded_fields={"x": "feature", "y": "importance"},
            underlying_numbers=dict(drivers),
            takeaway=f"{drivers[0][0]} most influences {target}.", figure_path=p))

    dash = state.dashboard_spec
    if dash is not None:
        dash.charts = dash.charts + new_charts

    return {
        "model_leaderboard": leaderboard,
        "findings": state.findings + [finding],
        "dashboard_spec": dash,
        "figures": figures,
        "log": state.log + [f"modeler: {task}, selected {best} "
                            f"({metric_name}={best_res.test_score:.3f})"],
    }

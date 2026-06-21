# Executive Data Analyst Agent — Design Spec

> Name: **Ledger**. An agentic AI that does the everyday job of a human data analyst
> and produces leadership-grade answers and dashboards.

Status: Draft v3 · Owner: Teja · Date: 2026-06-20 · Built: M1 ✅  M2 ✅  M3 ✅  M3b ✅  M4 ✅ (report)

---

## 0. Project structure — two parts

The project is delivered in two parts, both driven by the agent:

- **Part 1 — Automated modeling (custom AutoML).** On a (preferably financial) dataset, the
  agent detects the problem type, runs its **own** train/evaluate loop over multiple
  techniques (e.g. linear/logistic regression, random forest, gradient boosting), builds an
  **ensemble**, and selects the best model — using the *right* metric and proper validation
  (NOT raw accuracy; see §8a). Built as a custom model loop so Teja's own DS skill is on
  display, not hidden behind an AutoML library.
- **Part 2 — Visualization & insights.** The agent generates a **presentable, interactive
  dashboard in Plotly/Streamlit** (fully code-generated and agent-controlled) plus the
  executive report. **Tableau/Power BI authoring is explicitly out of scope** — no current
  API authors those dashboards from code cleanly; Plotly/Streamlit is the deliverable. (A
  clean-dataset + chart-spec handoff to Power BI may be added later as a stretch.)
- **Part 3 — Conversational Q&A (the "interrogate your analyst" layer).** Leadership can ask
  follow-up questions in plain English, and the agent answers grounded in the artifacts it
  produced (see §8b). It must answer questions about: (a) **the dataset**, (b) **the modeling
  run** — e.g. "which ML model gave the highest accuracy and why?", and (c) **the dashboard /
  visualizations** that were presented — e.g. "what does the second chart tell us?".

---

## 1. One-line pitch

Give it a dataset; it understands the data, builds an executive-presentable dashboard,
writes an elaborate explanation of what the data shows, and answers leadership questions
(projections, insights, root cause) — always stating its confidence and limitations.

## 2. Why this is a strong portfolio piece (fintech DS)

- **Agentic AI** + **real data-science rigor** in one project (rare combination).
- The differentiator is not "chat with CSV" — it is **executive judgment**: BLUF answers,
  projections with intervals, and an explicit *limitations & confidence* layer. Most
  autonomous-analyst projects confidently hallucinate; this one knows what it doesn't know.
- Flagship demo runs on **financial data**, so it reads as a *fintech* analyst, on-target
  for the roles Teja wants.

## 3. Goals / Non-goals

**Goals**
- Profile any tabular dataset and explain it in plain business language.
- Build dashboards presentable to leadership (Streamlit interactive + exportable HTML/PDF report).
- Answer arbitrary leadership questions grounded in computed results, e.g.
  "what does the data project?", "what are the key insights?", "what is the cause of the loss?".
- Produce projections with prediction intervals and **backtested** error.
- Always report limitations and a confidence level; refuse to over-claim on weak data.

**Non-goals (v1)**
- Not a BI replacement (no live DB connectors, no multi-user auth).
- Not investment/trading advice; analysis only.
- Not real-time streaming; batch datasets only.

## 4. Users & key scenarios

- **Analyst (operator)** uploads a dataset, reviews/curates agent output.
- **Leadership (consumer)** reads the report / asks questions in plain English.

Example leadership questions the agent must handle:
1. "Summarize this dataset for me." (descriptive)
2. "What are the top 3 insights?" (diagnostic + prioritization)
3. "What's causing the loss / the drop?" (root-cause)
4. "What do you project for next quarter?" (forecast + interval)
5. "How confident are you, and what are the caveats?" (uncertainty/limits)

## 5. Architecture (LangGraph)

Typed `AnalysisState` flows through a graph. The agentic loop is the
**Analyst ↔ Code Executor ↔ Validator** cycle: write analysis code → run in sandbox →
inspect result → refine → quantify confidence before reporting.

```
        dataset + (optional) question
                     │
                     ▼
              ┌──────────────┐
              │  Profiler    │  schema, dtypes, missingness, outliers,
              │              │  leakage scan, column-meaning inference
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  Planner     │  decompose the question into analysis steps
              └──────┬───────┘
                     ▼
        ┌─────────────────────────┐
        │  Analyst  ⇄  Code Exec  │  sandboxed pandas/sklearn/statsmodels
        │  (loop: run→inspect→fix)│  produces tables, figures, model outputs
        └───────────┬─────────────┘
                     ▼
              ┌──────────────┐
              │ Modeler      │  PART 1: custom AutoML loop — train panel,
              │ (if target)  │  ensemble, select by right metric, SHAP (§8a)
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Diagnostician│  root-cause: decomposition, driver analysis
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Forecaster   │  projections + intervals + walk-forward backtest
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Validator    │  recompute checks, error metrics, assumption
              │              │  checks → assigns confidence per claim
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Reporter     │  BLUF narrative + dashboard + limitations
              └──────┬───────┘
                     ▼
        Streamlit UI  +  exportable HTML/PDF executive report
```

Conditional edges:
- Profiler may route straight to Reporter for "just describe this" requests.
- Forecaster is skipped when no time dimension / projection isn't asked.
- Validator can send the Analyst back to redo a step if a check fails.

## 6. State & data contracts (Pydantic)

```python
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_pct: float
    n_unique: int
    inferred_meaning: str          # LLM-inferred business meaning
    quality_flags: list[str]       # e.g. ["high_missing", "possible_id", "outliers"]

class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    time_column: str | None
    target_candidates: list[str]
    quality_summary: str

class Finding(BaseModel):
    claim: str                     # executive-readable statement
    evidence: dict                 # numbers/tables backing it
    confidence: Literal["high", "medium", "low"]
    method: str                    # how it was computed
    limitations: list[str]

class Projection(BaseModel):
    metric: str
    horizon: str
    point: float
    lower: float
    upper: float                   # prediction interval
    backtest_error: dict           # e.g. {"MAPE": 0.07, "MAE": 1234}
    method: str
    caveats: list[str]

class ModelResult(BaseModel):
    name: str                      # e.g. "RandomForest", "LogisticRegression"
    primary_metric: str            # e.g. "ROC-AUC" (the RIGHT metric, not raw accuracy)
    cv_score: float
    test_score: float              # held-out; gap vs cv flags overfitting
    all_metrics: dict              # full metric set for transparency
    notes: str                     # imbalance handling, hyperparams, etc.

class ModelLeaderboard(BaseModel):          # Part 1 artifact, used by Part 3 Q&A
    task_type: str                 # "binary_classification" | "regression" | ...
    target: str
    metric_rationale: str          # WHY this metric was chosen
    candidates: list[ModelResult]
    ensemble: ModelResult | None
    selected: str                  # name of the winning model
    selection_reason: str

class ChartSpec(BaseModel):                 # one entry per dashboard chart
    chart_id: str
    title: str
    chart_type: str
    encoded_fields: dict           # x/y/color/etc.
    underlying_numbers: dict       # the data behind it, for Q&A grounding
    takeaway: str                  # written executive takeaway

class DashboardSpec(BaseModel):             # Part 2 artifact, used by Part 3 Q&A
    charts: list[ChartSpec]
    layout_rationale: str

class AnalysisState(BaseModel):
    dataset_path: str
    question: str | None
    profile: DatasetProfile | None
    plan: list[str]
    findings: list[Finding]
    projections: list[Projection]
    model_leaderboard: ModelLeaderboard | None   # Part 1 → grounds Part 3 Q&A
    dashboard_spec: DashboardSpec | None         # Part 2 → grounds Part 3 Q&A
    figures: list[str]             # paths to generated charts
    executive_summary: str | None
    limitations: list[str]
```

## 7. Tools (the analyst's "hands")

- **`run_python(code)`** — sandboxed execution with pandas/numpy/sklearn/statsmodels/plotly
  pre-imported; dataset pre-loaded as `df`. Returns stdout, result repr, and any figures.
  Sandbox via subprocess with restricted imports + timeout + memory cap (v1: subprocess;
  later: container). The agent never sees raw rows it shouldn't; only computed aggregates.
- **`make_chart(spec)`** — standardized, leadership-styled Plotly charts (titled, with a
  written takeaway annotation, consistent theme).
- **`describe_column(name)`** — quick stats helper to reduce token use.

## 8a. Part 1 — Automated modeling (custom model loop)

A dedicated **Modeler** node (sits between Analyst and Diagnostician when a target is present):

1. **Problem framing** — infer task type (regression vs binary/multiclass classification),
   identify the target, detect a time dimension. State assumptions explicitly.
2. **Leakage & data checks** — drop IDs/post-outcome columns, flag target leakage, handle
   missing values and categoricals in a leakage-safe pipeline (fit on train only).
3. **Candidate models** — train a panel: linear/logistic regression (baseline), random
   forest, gradient boosting (XGBoost/LightGBM), and 1–2 more as fits.
4. **Ensembling** — stack/vote the strongest candidates.
5. **Selection by the RIGHT metric** — NOT raw accuracy. Choose per task:
   - imbalanced classification → **ROC-AUC / PR-AUC / F1**, with class-imbalance handling
   - regression → **RMSE / MAE / R²**
   - where possible, a **business cost function** (e.g. cost of false negative in fraud)
   Selection uses **cross-validation + a held-out test set**; report the gap to catch
   overfitting.
6. **Interpretation** — model-agnostic **permutation importance** on the held-out set as the
   default (robust across estimators + ColumnTransformer; gives original-feature-level "what
   drives the prediction"). SHAP is installed and can be layered on for local explanations.
7. **Honest reporting** — report the chosen metric, CV vs test gap, calibration where
   relevant, and limitations. Never report a single inflated "accuracy" number.

> Design note: an AutoML library MAY be used as a breadth tool later, but v1 is a custom loop
> so the metric choice, leakage handling, validation, and interpretation are visibly Teja's.

## 8b. Part 3 — Conversational Q&A (grounded in artifacts)

A **Q&A node** (the "Briefer" in conversational mode) answers leadership follow-ups. The
rule: **never re-guess — answer from the structured artifacts the agent already produced**,
and run fresh `run_python` only when a genuinely new calculation is requested.

Three grounded answer paths:

1. **Dataset questions** ("how many rows? what's the date range? what does column X mean?")
   → answer from `DatasetProfile`; for new aggregates, call `run_python` on `df`.
2. **Modeling questions** ("which model gave the highest accuracy / why did it win / how does
   RF compare to logistic regression?") → answer from the **`ModelLeaderboard`** artifact the
   Modeler records: every candidate, its metric(s), CV-vs-test gap, ensembling result, the
   selected model, and the reason it was chosen. The agent quotes the actual numbers and the
   metric used (and corrects the framing if the user says "accuracy" but the right metric was
   AUC).
3. **Dashboard / visualization questions** ("what does chart 2 show? why did you pick this
   view? what's the takeaway?") → answer from the **`DashboardSpec`**: per-chart title, chart
   type, encoded fields, the underlying numbers, and the written takeaway.

Design notes:
- Every answer carries the same **confidence + limitations** discipline as the report.
- The Q&A node has read access to `profile`, `findings`, `projections`, `model_leaderboard`,
  and `dashboard_spec` in state — this is the agent's "memory" of its own analysis.
- Implemented as a loop back into the graph: a new question re-enters at Planner, which routes
  to a grounded answer or triggers fresh analysis if needed.

## 8. The rigor layer (what makes it data-science, not an app)

- **Projections:** statsmodels/Prophet with **prediction intervals** and **walk-forward
  backtesting**; the reported number always travels with its backtest error.
- **Root cause:** contribution/decomposition analysis (e.g. revenue = volume × price by
  segment), correlation-with-caveats, and explicit "correlation ≠ causation" framing.
- **Confidence:** Validator assigns each Finding a confidence from sample size, effect size,
  data quality, and whether a recompute matched. Low-confidence claims are surfaced as such.
- **Limitations:** dedicated section — sample-size limits, confounders, extrapolation risk,
  data-quality caveats, time coverage.

## 9. Dashboard & executive report

- **Streamlit app:** upload dataset → see profile → dashboard → chat box for live Q&A.
- **Executive report export:** one click → self-contained **HTML** (and **PDF** via
  print/`weasyprint`) with BLUF summary, KPI cards, titled charts w/ takeaways, projections,
  and a limitations section. This is the "presentable to leadership" deliverable.
- **Styling:** clean executive theme (KPI cards, takeaway-on-chart titles), not default gray.

## 10. Evaluation harness (credibility layer)

- **Known-answer datasets:** small datasets where the correct numbers/trends are known;
  assert the agent computes them correctly (not just "sounds right").
- **Forecast eval:** held-out periods; assert backtest error is reported and within bounds.
- **Confidence calibration:** check the agent says "low confidence" on deliberately weak data
  (tiny N, high missingness) — i.e. it correctly knows what it doesn't know.
- **Regression tests** on a fixed seed so results are reproducible.

## 11. Repo layout

```
lumen/
├── README.md                 # pitch, architecture diagram, demo gif
├── SPEC.md                   # this document
├── app.py                    # Streamlit entry point
├── graph.py                  # LangGraph wiring
├── state.py                  # Pydantic models (§6)
├── nodes/
│   ├── profiler.py
│   ├── planner.py
│   ├── analyst.py
│   ├── modeler.py            # Part 1: custom AutoML loop (§8a)
│   ├── diagnostician.py
│   ├── forecaster.py
│   ├── validator.py
│   └── reporter.py
├── tools/
│   ├── sandbox.py            # run_python
│   └── charts.py             # make_chart, executive theme
├── report/
│   └── render.py             # HTML/PDF executive report
├── data/
│   └── sample_finance.csv    # flagship demo dataset
├── evals/
│   ├── known_answers/
│   └── test_analyst.py
└── prompts/
```

## 12. Tech stack

- **Orchestration:** LangGraph (`StateGraph`)
- **LLMs:** Claude via `langchain-anthropic` — `claude-opus-4-8` for analyst/diagnostician/
  reporter reasoning, `claude-haiku-4-5` for profiling/routing/cheap steps
- **DS:** pandas, numpy, scikit-learn, statsmodels (and/or Prophet)
- **Viz/UI:** Plotly + Streamlit; HTML/PDF report export
- **Eval/test:** pytest

## 13. Flagship demo dataset (financial)

Recommended: a **lending / loan-portfolio** or **revenue-by-segment** dataset so leadership
questions land naturally — e.g. *"what's driving our loss rate?"* decomposed by segment and
time. Candidate public sources: Lending Club loan data, or a synthetic fintech P&L with
segments + monthly periods (lets us script a known "cause of loss" for the eval harness).
Final pick to be confirmed before build.

## 14. Milestones

1. **M1 — Skeleton:** repo, state models, LangGraph wiring, `run_python` sandbox, Streamlit
   shell. Runnable `profile → simple analysis → one chart` slice. ✅ DONE
2. **M2 — Analyst loop:** LLM writes pandas → runs in sandbox → self-corrects → grounded
   findings (`nodes/agentic_analyst.py`), layered on the deterministic baseline. ✅ DONE
3. **M3 — Part 1 modeler:** custom model panel (logistic/RF/HistGB + ensemble), leakage-safe
   pipelines, selection by the right metric (PR-AUC on imbalanced, ROC-AUC on balanced) via
   CV + held-out test, permutation-importance interpretation → `ModelLeaderboard`
   (`nodes/modeler.py`). ✅ DONE.
3b. **M3b — Rigor layer:** root-cause **diagnostician** (rate/mix decomposition),
   **forecaster** (OLS trend, 80% prediction intervals, walk-forward backtest vs naive),
   and **validator** (independent recompute + aggregate confidence + anti-overclaim
   guardrails the reporter must honor → fixed the "production-ready" overclaim). ✅ DONE.
4. **M4 — Executive report:** self-contained HTML export with BLUF + KPI cards + findings +
   leaderboard + projections + inline charts + limitations (`report/render.py`); PDF via
   browser Print→Save-as-PDF (renders Plotly JS; weasyprint/kaleido can't). ✅ DONE.
5. **M5 — Evals + README:** known-answer tests, confidence-calibration tests, architecture
   diagram, demo gif, writeup.

## 15. Risks & mitigations

- **Hallucinated insights** → ground every claim in `run_python` output; Validator recompute.
- **Unsafe code execution** → sandboxed subprocess, restricted imports, timeout/mem caps.
- **"Analyze anything" reads as shallow** → anchor demo in a financial dataset + strong evals.
- **Over-claiming accuracy** → confidence layer + mandatory limitations section.
- **Token cost / latency** → Haiku for cheap steps, aggregate-only context, caching.

## 16. Open questions

- Final flagship dataset choice (lending vs. synthetic P&L).
- Project name (Lumen / Clarity / Brief / Ledger / Insight — see §12).
- PDF rendering path → RESOLVED (M4): browser Print→Save-as-PDF, because it executes
  Plotly's JS and keeps the charts; weasyprint/kaleido drop them.
```

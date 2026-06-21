"""Typed state and data contracts for the Ledger graph (SPEC §6).

These Pydantic models ARE the architecture: the rigor (confidence, method,
limitations) is baked into the types, and the Part-1/Part-2 artifacts
(ModelLeaderboard, DashboardSpec) are what the Part-3 Q&A node reads from.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


# --- Profiling ---------------------------------------------------------------
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_pct: float
    n_unique: int
    inferred_meaning: str = ""
    quality_flags: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    time_column: Optional[str] = None
    target_candidates: list[str] = Field(default_factory=list)
    quality_summary: str = ""


# --- Findings & projections --------------------------------------------------
class Finding(BaseModel):
    claim: str
    evidence: dict = Field(default_factory=dict)
    confidence: Confidence = "medium"
    method: str = ""
    limitations: list[str] = Field(default_factory=list)


class Projection(BaseModel):
    metric: str
    horizon: str
    point: float
    lower: float
    upper: float
    backtest_error: dict = Field(default_factory=dict)
    method: str = ""
    caveats: list[str] = Field(default_factory=list)


# --- Part 1: modeling artifacts ---------------------------------------------
class ModelResult(BaseModel):
    name: str
    primary_metric: str
    cv_score: float
    test_score: float
    all_metrics: dict = Field(default_factory=dict)
    notes: str = ""


class ModelLeaderboard(BaseModel):
    task_type: str
    target: str
    metric_rationale: str
    candidates: list[ModelResult] = Field(default_factory=list)
    ensemble: Optional[ModelResult] = None
    selected: str = ""
    selection_reason: str = ""


# --- Part 2: dashboard artifacts --------------------------------------------
class ChartSpec(BaseModel):
    chart_id: str
    title: str
    chart_type: str
    encoded_fields: dict = Field(default_factory=dict)
    underlying_numbers: dict = Field(default_factory=dict)
    takeaway: str = ""
    figure_path: str = ""


class DashboardSpec(BaseModel):
    charts: list[ChartSpec] = Field(default_factory=list)
    layout_rationale: str = ""


# --- Graph state -------------------------------------------------------------
class AnalysisState(BaseModel):
    dataset_path: str
    question: Optional[str] = None

    profile: Optional[DatasetProfile] = None
    plan: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    projections: list[Projection] = Field(default_factory=list)

    model_leaderboard: Optional[ModelLeaderboard] = None   # Part 1 -> grounds Q&A
    dashboard_spec: Optional[DashboardSpec] = None         # Part 2 -> grounds Q&A

    figures: list[str] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    limitations: list[str] = Field(default_factory=list)

    # Validator output (M3b): guardrails the reporter must honor + an aggregate
    # confidence. This is the "knows what it doesn't know" layer.
    guardrails: list[str] = Field(default_factory=list)
    overall_confidence: Optional[str] = None

    # transient scratch (not a deliverable)
    log: list[str] = Field(default_factory=list)

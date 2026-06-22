"""LangGraph wiring for Ledger (SPEC §5).

M3b active path:
  profiler -> planner -> analyst (deterministic baseline) -> agentic_analyst
  (LLM writes+runs code) -> modeler (Part 1 AutoML) -> diagnostician (root-cause)
  -> forecaster (projections) -> validator (guardrails) -> reporter -> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes.agentic_analyst import agentic_analyst
from .nodes.analyst import analyst
from .nodes.diagnostician import diagnostician
from .nodes.forecaster import forecaster
from .nodes.modeler import modeler
from .nodes.planner import planner
from .nodes.profiler import profiler
from .nodes.reporter import reporter
from .nodes.validator import validator
from .state import AnalysisState


def build_graph():
    g = StateGraph(AnalysisState)
    g.add_node("profiler", profiler)
    g.add_node("planner", planner)
    g.add_node("analyst", analyst)
    g.add_node("agentic_analyst", agentic_analyst)
    g.add_node("modeler", modeler)
    g.add_node("diagnostician", diagnostician)
    g.add_node("forecaster", forecaster)
    g.add_node("validator", validator)
    g.add_node("reporter", reporter)

    g.add_edge(START, "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "analyst")
    g.add_edge("analyst", "agentic_analyst")
    g.add_edge("agentic_analyst", "modeler")
    g.add_edge("modeler", "diagnostician")
    g.add_edge("diagnostician", "forecaster")
    g.add_edge("forecaster", "validator")
    g.add_edge("validator", "reporter")
    g.add_edge("reporter", END)
    return g.compile()


def run_analysis(dataset_path: str, question: str | None = None,
                 target: str | None = None) -> AnalysisState:
    """Run the full pipeline and return the final state.

    `target` optionally forces which column is the modeling target (overrides
    auto-detection — the "pick your target column" feature)."""
    graph = build_graph()
    initial = AnalysisState(dataset_path=dataset_path, question=question, target=target)
    result = graph.invoke(initial)
    # langgraph returns a dict-like; coerce back to our typed model
    return AnalysisState.model_validate(result)

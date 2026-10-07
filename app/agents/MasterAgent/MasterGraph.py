"""The master agent's pipeline as a LangGraph graph: prepare workspace -> collect facts -> analyze -> save profile.

One router picks every next step from the state alone: the first stage without an output runs next. A resumed run
therefore skips what it already finished, and a project folder that is gone (deleted, or a restart on a machine
without it) is prepared again before a stage that reads it.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .MasterState import MasterState

if TYPE_CHECKING:
    # Type only: importing the nodes pulls in the sandbox and agent SDKs.
    from .MasterNodes import MasterNodes

PREPARE_WORKSPACE = "prepare_workspace"
COLLECT_FACTS = "collect_facts"
ANALYZE = "analyze"
SAVE_PROFILE = "save_profile"
NODE_NAMES = (PREPARE_WORKSPACE, COLLECT_FACTS, ANALYZE, SAVE_PROFILE)

# A run needs at most 4 steps. LangGraph's default limit is ~10000, which would turn a routing bug into
# thousands of project extractions; this turns it into a quick failure instead.
MAX_GRAPH_STEPS = 10


def route_next(state: MasterState) -> str:
    """The next node for this state, or END when every stage has its output."""
    if not state.get("fact_sheet"):
        next_stage = COLLECT_FACTS
    elif not state.get("findings") or not state.get("llm_model"):
        next_stage = ANALYZE
    elif not state.get("profile_id"):
        # Saving the profile only needs the saved outputs, not the project folder.
        return SAVE_PROFILE
    else:
        return END
    if not _project_dir_exists(state.get("project_dir")):
        return PREPARE_WORKSPACE
    return next_stage


def build_master_graph(nodes: MasterNodes) -> CompiledStateGraph:
    graph = StateGraph(MasterState)
    graph.add_node(PREPARE_WORKSPACE, nodes.prepare_workspace)
    graph.add_node(COLLECT_FACTS, nodes.collect_facts)
    graph.add_node(ANALYZE, nodes.analyze)
    graph.add_node(SAVE_PROFILE, nodes.save_profile)

    destinations = [*NODE_NAMES, END]
    for source in (START, *NODE_NAMES):
        graph.add_conditional_edges(source, route_next, destinations)
    return graph.compile().with_config(recursion_limit=MAX_GRAPH_STEPS)


def _project_dir_exists(project_dir: str | None) -> bool:
    return bool(project_dir) and Path(project_dir).is_dir()

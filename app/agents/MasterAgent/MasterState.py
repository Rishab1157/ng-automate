"""What the master agent knows about one run while it works.

Apart from the ids, every key is the output of one stage. The graph's router reads only this state to pick the
next stage, so a run that starts with the outputs it already saved skips the stages it finished.
"""

from typing import Any, TypedDict

from app.models.runModel import RunModel


class MasterState(TypedDict, total=False):
    run_id: str
    org_id: str
    project_id: str
    model_connection_id: str | None
    # Host folder with this run's copy of the project. Internal: never sent to the API, the events or the LLM.
    project_dir: str
    # FactSheetModel and AnalyzerFindingsModel as JSON dicts, the same shape the run document stores.
    fact_sheet: dict[str, Any]
    findings: dict[str, Any]
    # LiteLLM name of the model that produced `findings`.
    llm_model: str
    profile_id: str


def initial_state(run: RunModel) -> MasterState:
    """The run's ids plus every stage output it already saved (resume)."""
    state = MasterState(
        run_id=run.id,
        org_id=run.org_id,
        project_id=run.project_id,
        model_connection_id=run.model_connection_id,
    )
    saved = run.outputs.model_dump(mode="json")
    state.update({key: value for key, value in saved.items() if value is not None})  # type: ignore[typeddict-item]
    return state

"""What the master agent knows about one run while it works.

Apart from the ids and the run's settings, every key is the output of one stage. The graph's router reads only this
state to pick the next stage, so a run that starts with the outputs it already saved skips the stages it finished.
"""

from typing import Any, TypedDict

from app.models.runModel import RunMode, RunModel


class MasterState(TypedDict, total=False):
    run_id: str
    org_id: str
    project_id: str
    model_connection_id: str | None
    mode: str  # RunMode value
    test_selector: str | None
    test_data_id: str | None
    run_scope: str  # RunScope value
    # Host folder with this run's copy of the project. Internal: never sent to the API, the events or the LLM.
    project_dir: str
    # FactSheetModel and AnalyzerFindingsModel as JSON dicts, the same shape the run document stores.
    fact_sheet: dict[str, Any]
    findings: dict[str, Any]
    # LiteLLM name of the model that produced `findings`.
    llm_model: str
    profile_id: str
    # Generate mode: GenerationOutcomeModel as a JSON dict.
    generation: dict[str, Any] | None
    # Test and generate modes: TestCommandModel, TestAttemptModel list and TestReportModel as JSON dicts.
    test_command: dict[str, Any]
    test_attempts: list[dict[str, Any]] | None
    test_report: dict[str, Any]


def initial_state(run: RunModel) -> MasterState:
    """The run's ids and settings plus every stage output it already saved (resume).

    Work that edits the project copy and was interrupted starts over on a fresh copy: the agent may have been half-way
    through an edit, and the copy on disk no longer matches what was saved. That is a generate run whose tests are not
    saved yet, and a test phase without its report. A fresh copy also drops what the generator wrote, so it runs again.
    """
    state = MasterState(
        run_id=run.id,
        org_id=run.org_id,
        project_id=run.project_id,
        model_connection_id=run.model_connection_id,
        mode=run.mode.value,
        test_selector=run.test_selector,
        test_data_id=run.test_data_id,
        run_scope=run.run_scope.value,
    )
    saved = run.outputs.model_dump(mode="json")
    state.update({key: value for key, value in saved.items() if value is not None})  # type: ignore[typeddict-item]
    generating = run.mode == RunMode.GENERATE and not state.get("generation")
    testing = bool(state.get("test_attempts")) and not state.get("test_report")
    if generating or testing:
        state.pop("project_dir", None)
        state.pop("test_attempts", None)
    return state

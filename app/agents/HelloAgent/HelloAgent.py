"""Learning agent: looks at a repo and says what kind of project it is.

FileEditorTool can also edit files: only point this at a repo you can undo with git.
Run: python -m app.agents.HelloAgent.HelloAgent "<path to a repo>"
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from openhands.sdk import Agent, Conversation, Tool  # noqa: E402
from openhands.sdk.conversation.response_utils import get_agent_final_response  # noqa: E402
from openhands.tools.file_editor import FileEditorTool  # noqa: E402

from app.utils.LlmInstance import build_llm, get_default_llm_config  # noqa: E402

TASK = (
    "Do NOT create or edit any files. Only look. "
    "List the folders of this project, open the files that tell you how it is built, "
    "and tell me: language, framework, build tool, test framework, and what the project is for."
)


def run_hello_agent(repo_path: Path) -> str:
    """Run the agent and return its final answer."""
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repo folder not found: {repo_path}")

    agent = Agent(llm=build_llm(get_default_llm_config()), tools=[Tool(name=FileEditorTool.name)])
    conversation = Conversation(agent=agent, workspace=str(repo_path.resolve()))
    conversation.send_message(TASK)
    conversation.run()

    # Every step (LLM calls, tool calls, tool results) is an event in conversation.state.events.
    return get_agent_final_response(conversation.state.events)


if __name__ == "__main__":
    print("\n===== FINAL ANSWER =====\n" + run_hello_agent(Path(sys.argv[1])))

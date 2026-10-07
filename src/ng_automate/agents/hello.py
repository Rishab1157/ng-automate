"""First learning agent: looks at a repo and says what kind of project it is.

FileEditorTool can also edit files. The analyzer will get read-only tools;
until then, only point this at a repo you can undo with git.
"""

from pathlib import Path

from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.tools.file_editor import FileEditorTool

from ng_automate.llm.config import load_llm_config
from ng_automate.llm.factory import build_llm

TASK = (
    "Do NOT create or edit any files. Only look. "
    "List the folders of this project, open the files that tell you how it is built, "
    "and tell me: language, framework, build tool, test framework, and what the project is for."
)


def run_hello_agent(repo_path: Path) -> str:
    """Run the agent and return its final answer."""
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repo folder not found: {repo_path}")

    agent = Agent(
        llm=build_llm(load_llm_config()),
        tools=[Tool(name=FileEditorTool.name)],
    )
    conversation = Conversation(agent=agent, workspace=str(repo_path.resolve()))
    conversation.send_message(TASK)
    conversation.run()

    # Every step (LLM calls, tool calls, tool results) is an event in conversation.state.events.
    # Put a breakpoint here to inspect them.
    return get_agent_final_response(conversation.state.events)

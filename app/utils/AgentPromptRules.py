"""Rules that every agent's task message shares."""

from collections.abc import Sequence

# The user's own instructions for a run (given when it was created, or sent during it): they come before an agent's
# default rules, in the generator's and the healer's task messages alike.
USER_INSTRUCTIONS_BLOCK = (
    "\nThe user's instructions for this run (they come first: where they differ from the rules below, follow them; "
    "never delete or weaken an existing test):\n{instructions}\n"
)


def user_instructions_block(instructions: Sequence[str]) -> str:
    """The user's instructions for this run, one per line. Empty when there are none (blank ones are left out)."""
    if isinstance(instructions, str):  # one instruction, not a list of letters
        instructions = [instructions]
    texts = [text.strip() for text in instructions if text and text.strip()]
    if not texts:
        return ""
    return USER_INSTRUCTIONS_BLOCK.format(instructions="\n".join(f"- {text}" for text in texts))

# Models that write their tool calls as text (LLM_NATIVE_TOOL_CALLING=false) tend to wrap parameter values in quotes,
# and the quotes then become part of the value: a glob for "**/pages/**", quotes included, finds nothing. The example
# names no language: a small model copies examples, and one of the project's own would send it to the wrong files.
PLAIN_TOOL_VALUES_RULE = (
    'Write tool parameter values as plain text, without surrounding quotes: a glob pattern is **/pages/**, not '
    '"**/pages/**".'
)

# The sandbox runs browsers on a virtual screen that the user watches live (the live view): a headless browser shows
# nothing there. How a browser is made visible and maximized depends on the framework, so the agent finds out how
# this project starts its browser.
VISIBLE_BROWSER_RULE = (
    "Browser tests run on a virtual screen that the user watches live. Never make the browser headless, and give it a "
    "maximized window (or the full screen size) when the framework allows it, in the way this project starts its "
    "browser (its options, fixtures or config). If the project's own setup makes it headless or small, change only "
    "that setting. Never switch to headless to get around a problem."
)

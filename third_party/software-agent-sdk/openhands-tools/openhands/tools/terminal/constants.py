import re
from typing import Final


CMD_OUTPUT_PS1_BEGIN: Final[str] = "\n###PS1JSON###\n"
CMD_OUTPUT_PS1_END: Final[str] = "\n###PS1END###"
# Regex to match PS1 metadata blocks. Uses negative lookahead to handle corruption
# scenarios where concurrent output causes nested ###PS1JSON### markers. This ensures
# we match only the LAST ###PS1JSON### before each ###PS1END###.
CMD_OUTPUT_METADATA_PS1_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{CMD_OUTPUT_PS1_BEGIN.strip()}((?:(?!{CMD_OUTPUT_PS1_BEGIN.strip()}).)*?){CMD_OUTPUT_PS1_END.strip()}",
    re.DOTALL | re.MULTILINE,
)

# Default max size for command output content
# to prevent too large observations from being saved in the stream
# This matches the default max_message_chars in LLM class
MAX_CMD_OUTPUT_SIZE: Final[int] = 30000


# Common timeout message that can be used across different timeout scenarios: what
# the model can do about a command that is still running.
TIMEOUT_MESSAGE_TEMPLATE: Final[str] = (
    "Send an empty command '' to wait for more output, or send input or keys to it "
    "with is_input=true (`C-c` stops it); a new command runs only after it has "
    "ended. Run long commands with a larger `timeout` (in seconds)."
)

# NG Automate: the same for a pooled terminal (TerminalExecutor with tmux), which
# stops a command that is still running before it runs the next one there (see
# TerminalExecutor._prepare_pooled_session). Waiting with '' or sending keys there
# only failed with "No previous running command ...".
POOLED_TIMEOUT_MESSAGE: Final[str] = (
    "It cannot be waited for or answered: it is stopped (C-c) before its terminal "
    "runs another command. If it needs more time, run it again with a larger "
    "`timeout` (in seconds); if it waits for input, give the input in the command."
)

# NG Automate: what bash's own exit codes for a command it could not run mean.
EXIT_CODE_HINTS: Final[dict[int, str]] = {
    126: (
        "Exit code 126: a command was found but could not run (it is not "
        "executable, or it is a folder). Run a script through its interpreter (for "
        "example `sh ./script.sh`), or check the file with `ls -l`."
    ),
    127: (
        "Exit code 127: the shell found no command with that name (it is "
        "misspelled, or the program is not installed or not on PATH). Check the "
        "name in the output above; `command -v NAME` shows whether it is installed."
    ),
}

# How long to wait with no new output before considering it a no-change timeout
NO_CHANGE_TIMEOUT_SECONDS: Final[int] = 30

# How often to poll for new output in seconds
POLL_INTERVAL: Final[float] = 0.5
HISTORY_LIMIT: Final[int] = 10_000

TMUX_SOCKET_NAME: Final[str] = "openhands"

# Tmux session dimensions (columns x rows). Keep the viewport wide enough for
# common command output while leaving scrollback retention to HISTORY_LIMIT.
# Output wider than TMUX_SESSION_WIDTH columns will wrap; this is an accepted
# tradeoff to avoid the oversized 1000x1000 virtual terminal grid.
TMUX_SESSION_WIDTH: Final[int] = 256
TMUX_SESSION_HEIGHT: Final[int] = 200

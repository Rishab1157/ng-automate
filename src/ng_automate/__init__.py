import argparse
import os
from pathlib import Path

# Hide the OpenHands welcome banner printed on import.
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ng-automate")
    commands = parser.add_subparsers(dest="command", required=True)

    hello = commands.add_parser("hello", help="Run the first learning agent on a repo.")
    hello.add_argument("repo", type=Path, help="Path to the repo to look at.")

    args = parser.parse_args()

    if args.command == "hello":
        from ng_automate.agents.hello import run_hello_agent

        answer = run_hello_agent(args.repo)
        print("\n===== FINAL ANSWER =====\n" + answer)

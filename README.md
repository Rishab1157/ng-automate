# NG Automate

Agent system that analyzes a test-automation project, makes it build, generates tests, runs them and fixes technical failures.

## Setup

The OpenHands SDK source is included as a git submodule in `third_party/software-agent-sdk` (pinned to v1.53.0) and installed in editable mode, so you can read and debug it.

```
git clone --recurse-submodules <this repo>
cd ng-automate
python -m uv sync
cp .env.example .env    # then fill in the values
```

Already cloned without submodules? Run `git submodule update --init`.

## Run

```
python -m uv run ng-automate hello "<path to a repo>"
```

## Debug

Open the `ng-automate` folder in VS Code and press F5 (**Hello agent**). It can step into our code, the OpenHands SDK in `third_party/`, and libraries in `.venv`.

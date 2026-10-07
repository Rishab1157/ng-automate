# NG Automate

Agent system that analyzes a test-automation project, makes it build, generates tests, runs them and fixes technical failures.

## Setup

The OpenHands SDK is used from a local clone (editable install), so you can read and debug its source.

```
projects/
  ng-automate/          <- this repo
  software-agent-sdk/   <- git clone of OpenHands SDK, tag v1.53.0
```

```
cd projects
git clone --branch v1.53.0 https://github.com/OpenHands/software-agent-sdk.git
cd software-agent-sdk && git switch -c ng-automate-dev
cd ../ng-automate
python -m uv sync
cp .env.example .env    # then fill in the values
```

## Run

```
python -m uv run ng-automate hello "<path to a repo>"
```

## Debug

Open `ng-automate.code-workspace` in VS Code, then press F5:

- **Hello agent (pick a repo)** - steps through our code only.
- **Hello agent (step into OpenHands SDK too)** - also steps into the SDK source in `software-agent-sdk/`.

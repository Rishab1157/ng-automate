# NG Automate

Agent system that analyzes a test-automation project, makes it build, generates tests, runs them and fixes technical failures. Built to plug into QXcel: same token, same error format, same code structure.

## Setup

The OpenHands SDK source is a git submodule in `third_party/software-agent-sdk` (pinned to v1.53.0), installed in editable mode so you can read and debug it.

```
git clone --recurse-submodules <this repo>
cd ng-automate
python -m uv sync
copy .env.example .env    # then fill in the values
```

Already cloned without submodules? Run `git submodule update --init`.

## Run the API

With the `.venv` activated (`.venv\Scripts\activate`):

```
python -m uvicorn main:app --host 127.0.0.1 --port 4207 --reload
```

or `python main.py` (uses `HOST` / `PORT` from `.env`). Swagger: http://127.0.0.1:4207/docs, click **Authorize** and paste a QXcel token.

## Tests

```
python -m pytest
```

Tests use their own databases (`ng_automate_test`, `ng_automate_test_qxcel`) and a temporary data folder. Skip the ones that need internet with `-m "not network"`.

## Debug

Open the `ng-automate` folder in VS Code and press F5 (**NG Automate API**). It can step into our code, the OpenHands SDK in `third_party/`, and libraries in `.venv`.

## Structure (same as QXcel-API)

```
main.py                         starts the server
app/
  main.py                       FastAPI app: lifespan, CORS, error handlers, routers
  api/router.py                 includes every endpoint router
  api/endpoints/<Name>Endpoint/ routes (thin: validate, call service, return DTO)
  services/<name>Service/       business logic
  repositories/<name>Repository database access (BaseRepository + collection name)
  models/<name>Model/           DbModel (what is written), Model (what services use), Mapper
  dto/<name>Dto/                requestDto / responseDto (API bodies)
  projections/                  MongoDB projections
  permissions/                  permission codes from QXcel tokens
  core/exceptions/              ErrorCode, ErrorMessages, exception classes and handlers
  core/security/                token check (authentication) and access rules (authorization)
  config/settings.py            all settings, typed
  db/                           Mongo client, indexes
  utils/                        helpers (archives, LLM builder)
  agents/<Name>Agent/           OpenHands agents
tests/
```

QXcel's database is **read-only** for NG Automate: repositories over it (`use_qxcel_db = True`) refuse writes.

# Deploying NG Automate

This guide covers a fresh server and later updates. NG Automate is the API, its MongoDB and Qdrant, three Docker
images for the sandboxes, and a model server (Ollama). QXcel provides the users, the token, the git connections and
the model connections.

## 1. What you need

| Need | Why |
|---|---|
| Python 3.14 and `uv` (`python -m pip install uv`) | runs the API |
| Docker (Docker Desktop with WSL2 on Windows, or Docker Engine on Linux) | MongoDB, Qdrant and one sandbox container per agent session |
| Git, and the full repo including `third_party/` | the vendored OpenHands SDK and LangGraph; the sandbox images are built from them |
| Access to QXcel's MongoDB (read-only) | git connections, model connections, modules |
| QXcel's `SECRET_KEY` | NG Automate checks QXcel's tokens with it |
| An Ollama model server reachable from this machine | the agents' LLM and the embedding model, e.g. `http://192.168.3.90:11435` (on the office network or VPN) |
| Disk space: about 15 GB for images, plus projects and caches | sandbox images, per-org Maven/npm/pip caches, `DATA_DIR` |

The API runs **on the machine itself, not in a container**: it starts sandbox containers and mounts project folders
from this machine into them.

## 2. First deployment

### 2.1 Get the code and install

```
git clone <ng-automate repo>
cd ng-automate
python -m uv sync
copy .env.example .env        # Linux: cp .env.example .env
```

### 2.2 Fill in `.env`

Never commit `.env`. The values that matter:

| Variable | Set it to |
|---|---|
| `HOST`, `PORT` | `127.0.0.1` and `4207`. Use `0.0.0.0` only behind a reverse proxy or firewall. |
| `CORS_ORIGINS` | the QXcel UI's address, e.g. `["https://qxcel.example.com"]`. `["*"]` is for development only. |
| `SECRET_KEY` | exactly QXcel's `SECRET_KEY` |
| `NGAUTO_MONGO_PASSWORD` | a strong password; compose creates the database user with it |
| `MONGO_URI` | `mongodb://ngauto:<that password>@127.0.0.1:27018/?authSource=admin` |
| `QXCEL_MONGO_URI`, `QXCEL_DATABASE` | QXcel's server and database (`qxcel`). NG Automate only reads it. |
| `DATA_DIR` | where projects, test data and run copies live (default `data`). Back it up. |
| `USE_LOCAL_LLM` | `true`: every agent uses the local model below. `false`: every request must send a QXcel model connection (`model_connection_id`). |
| `LLM_MODEL`, `LLM_BASE_URL` | the local model, e.g. `ollama_chat/qwen2.5-coder:14b` and `http://192.168.3.90:11435` |
| `LLM_NUM_CTX` | `32768` (Ollama's default 4096 is too small for agents) |
| `LLM_NATIVE_TOOL_CALLING` | `false` for qwen2.5-coder (it writes tool calls as text); `true` for models with native tool calls |
| `LLM_THINKING` | `false` for models without a thinking mode |
| `HEAL_MEMORY_ENABLED`, `QDRANT_URL` | `true` and `http://127.0.0.1:6333` |
| `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` | the Ollama server with the embedding model, e.g. `http://192.168.3.90:11435` and `qwen3-embedding:0.6b` |
| `SANDBOX_ALLOWED_HOSTS` | only if sandboxes must reach something on a private address (a model connection's server, a package registry on the LAN), e.g. `["10.0.0.5:8081"]` |

Note: `.env.example` still shows Devstral on port 11434. Today's working setup is the qwen2.5-coder line above on
port 11435.

Check that the model server has both models:

```
curl http://192.168.3.90:11435/api/tags
```

### 2.3 Start MongoDB and Qdrant

```
docker compose up -d
```

This starts `ngauto-mongo` (on 127.0.0.1:27018) and `ngauto-qdrant` (127.0.0.1:6333). Both restart with Docker.

### 2.4 Build the sandbox images (once, in this order)

```
docker compose --profile build build agent-server
docker compose --profile build build sandbox-test
docker compose --profile build build sandbox-firewall
```

| Image | Used by |
|---|---|
| `ng-automate/agent-server:1.53.0-source` | the analyzer's read-only sandbox; the base of the next image |
| `ng-automate/sandbox-test:1.53.0-source` | runner, healer and generator: JDK, Maven, Node, Python, Chromium, the live view |
| `ng-automate/sandbox-firewall:1` | sets every sandbox's network rules. **Sandboxes do not start without it.** |

Rebuild `agent-server` and then `sandbox-test` after any change in `third_party/` or `docker/sandbox-test/`.

### 2.5 Set up QXcel

- QXcel's `modules` collection has the `NGAUTOMATE` module.
- Users who use NG Automate have the permission `NGAUTOMATE:ACCESS:OWN_ORG` (or `NGAUTOMATE:ACCESS:ALL_ORGS`) in their role.
- Git connections (repo URL, branch, token) and model connections are added in QXcel's Configurations and enabled for
  NG Automate. NG Automate reads them; the git token is only used for the clone and is never stored.
- The QXcel UI's `.env` points at the API: `VITE_NG_AUTOMATE_URL=http://<api-host>:4207`, then rebuild the UI.

### 2.6 Start the API

```
.venv\Scripts\activate        # Linux: source .venv/bin/activate
python main.py
```

Don't use `--reload` on a server. In development, use `--reload --reload-dir app` only: `data/` holds user projects.

On start the API resumes runs that were interrupted, and stops sandbox containers they left behind.

### 2.7 Check that it works

1. Open `http://<host>:4207/docs`, click **Authorize** and paste a QXcel token. `GET /tx-agents/ng-automate/project/`
   answers 200.
2. In the QXcel UI, open NG Automate, create a project (git connection or zip) and run **Analyze**. A sandbox
   container appears (`docker ps`) and the run completes in a few minutes.
3. The API log shows no "Healer memory is not working" warning after the first heal.

## 3. Updating a running deployment

1. Wait until no run is working (or accept that running runs resume after the restart).
2. Stop the API.
3. `git pull`, then `python -m uv sync`.
4. If `third_party/` or `docker/` changed, rebuild the images (2.4).
5. Add any new variables from `.env.example` to `.env`.
6. Start the API (2.6) and check it (2.7).

Never run two NG Automate servers on the same `MONGO_DATABASE`: both would pick up the same runs.

## 4. Security checklist

- **QXcel's MongoDB must require a login.** It holds git tokens and model API keys in plain text. Turn on
  `security.authorization` in `mongod.cfg`, create a read-only user for NG Automate, and put it in `QXCEL_MONGO_URI`.
- Keep `SANDBOX_EGRESS_FIREWALL=true` (the default). Sandboxes then reach the internet and the model server, but
  nothing else on a private address. Binding ports to 127.0.0.1 alone does not protect them on Docker Desktop.
- Keep MongoDB and Qdrant on 127.0.0.1. If Qdrant must be exposed, set `QDRANT__SERVICE__API_KEY` in compose and
  the same value as `QDRANT_API_KEY` in `.env`.
- Restrict `CORS_ORIGINS` to the QXcel UI and put the API behind HTTPS.
- Never commit `.env`. The project's `.gitignore` already leaves out `data/` and `workspace/`.

## 5. Backups

| What | Where |
|---|---|
| Projects, runs, events, test-data records | Docker volume `ng-automate_mongo-data` |
| Healer memory | Docker volume `ng-automate_qdrant-data` |
| Project files, test-data files | `DATA_DIR` |

The `ngauto-maven-<org>`, `ngauto-npm-<org>` and `ngauto-pip-<org>` volumes are download caches only.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Sandboxes fail to start: firewall image not found | build it: `docker compose --profile build build sandbox-firewall` |
| Runs fail at the start: the model server is not reachable | check `LLM_BASE_URL`, the network or VPN, and `curl <server>/api/tags` |
| "Healer memory is not working" in the log | `EMBEDDING_BASE_URL` unreachable or the model is missing on that server, or Qdrant is down |
| Every run waits, nothing happens after a reboot | Docker is not running: start it; MongoDB and Qdrant come back by themselves |
| A sandbox cannot download from a registry on the LAN | add `"host:port"` to `SANDBOX_ALLOWED_HOSTS` |
| Runs are slow or time out | the local model is slow on a shared GPU; raise the timeouts in `.env` (`RUN_TIMEOUT_SECONDS`, `GENERATOR_TIMEOUT_SECONDS`, `HEALER_TIME_BUDGET_SECONDS`) or use a stronger model |
| `USE_LOCAL_LLM=false` and requests fail with `MODEL_CONNECTION_REQUIRED` | the user has no model connection for NG Automate: add one in QXcel's Configurations |

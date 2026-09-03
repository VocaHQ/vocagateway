# AGENTS.md

VocaGateway is an optional self-hosted speech-to-text gateway for Voca clients
(vocaphone now; desktop apps planned). Clients send bounded recordings; FFmpeg
normalizes; a local engine transcribes; the gateway returns an idempotent
transcript. License: [AGPL-3.0](LICENSE). **Not on-device** — audio leaves the
client for the machine you run. No Voca account, no hosted Voca cloud.

Product and operator docs: [README.md](README.md) and the
[docs index](docs/README.md) ([configuration](docs/configuration.md),
[deployment](docs/deployment.md), [tailscale](docs/tailscale.md),
[troubleshooting](docs/troubleshooting.md), [models](docs/models.md)).
This file is for coding agents.

## Critical: git worktrees for every branch and PR

Never create a branch, commit, or open a pull request in the primary checkout. Always use a linked git worktree so the main working tree stays on `main` and stays clean. Do not `git switch` / `git checkout` a feature branch in the primary directory, and do not leave it dirty.

```bash
git fetch origin
git worktree add /tmp/vocagateway-<task> -b <type>/<short-name> origin/main

# All edits, commits, and `gh pr create` happen inside that worktree.

git worktree remove /tmp/vocagateway-<task>
git worktree prune
```

Rules:

- One worktree per branch, one branch per PR
- Place worktrees **outside** the primary working tree (`/tmp/vocagateway-<task>` or a sibling directory such as `../.worktrees/vocagateway-<task>`)
- Never run two tasks in the same worktree
- Never commit directly to `main`
- Clean up the worktree after the PR is pushed

## Layout

| Path | Role |
| --- | --- |
| `app/` | FastAPI package (`create_app` in `app/main.py`) |
| `app/routes/` | HTTP and `/v1/stream` WebSocket |
| `app/models/` | Engine adapters (sherpa-onnx, faster-whisper, Moonshine, whisper.cpp, MLX, WhisperKit, VocaMac, Handy) |
| `app/catalog.py` / `app/model_pins.json` | Downloadable models and SHA-256 pins |
| `app/fragments/` | HTMX HTML partials |
| `app/webui/` | Authenticated admin UI (static assets) |
| `tests/` | pytest (`asyncio_mode = auto`) |
| `docs/` | Operator docs; `models.md` is generated |
| `web/` | Public landing page, not the admin UI |
| `scripts/` | LaunchAgent/systemd installers, pin harvest, model-doc generator |
| `compose.yaml` | Documented container deploy; `.env` is the Compose token source |

## Commands

The justfile exports `UV_NO_SYNC=1`, so `uv run` in recipes does not auto-sync.
Run `just install` after clone or lockfile changes. FFmpeg is required (unit
audio tests shell out to it). `just doctor` reports missing `uv` / `ffmpeg` /
`docker`.

| Recipe | What it runs |
| --- | --- |
| `just install` | `uv sync --all-groups --all-extras` |
| `just lint` | ruff check (fail on would-fix) + ruff format `--check --diff` |
| `just format` | ruff format + ruff check `--fix` |
| `just type-check` | `uv run python -m mypy` (package `app`, strict) |
| `just unit` | `pytest -n auto` |
| `just package` | `uv sync --locked --check`, `uv pip check`, `uv audit` |
| `just compose` | `docker compose config` with a dummy ≥32-char token |
| `just test` | lint + type-check + package + unit + compose (needs Docker) |
| `just run` | `uv run vocagateway` (default `0.0.0.0:8765`) |
| `just site` | landing page at `http://127.0.0.1:4173/` |

Also: `just run-local` (loopback bind), `just token` / `just status` / `just diag`,
`just up` / `just down` / `just image`, `just site-check` (`node --test web/tests/site.test.mjs`).
Do not `docker compose down --volumes` unless deleting models, config, and sessions is intentional.

Primary CLIs: `vocagateway`, `vocagateway-token`, `vocagateway-status`,
`vocagateway-diagnostics`, `vocagateway-cleanup`. Deprecated aliases
(`vocaphone-server`, `vocaphone-token`, `vocaphone-status`,
`vocaphone-diagnostics`, `vocaphone-cleanup`) still resolve for one cycle.

## Python and lockfile

- `requires-python = ">=3.12"`; ruff/mypy target 3.12. Commit `uv.lock`.
- Dev group: ruff, mypy, pytest, pytest-asyncio, pytest-xdist, httpx.
- Optional extras: `engines` (faster-whisper, moonshine-voice, sherpa-onnx);
  `apple` (mlx-audio, Darwin arm64 only). Do not pass `--extra apple` on Linux.
- Native macOS (README): `uv sync --all-groups --extra engines --extra apple`.
  Native Linux: `uv sync --all-groups --extra engines`.
- CI quality: `uv sync --locked --all-groups` (no engine extras; tests mock engines).
- Images: `uv sync --frozen --no-dev --extra engines`.
- After `pyproject.toml` edits, regenerate and commit `uv.lock`. `just package`
  fails if the lock is stale.

## Pairing, tokens, network

Bearer token ≥ 32 characters. Native first run writes `~/.config/vocagateway/token`
(mode `600`). Override with `VOCAGATEWAY_TOKEN` or `VOCAGATEWAY_TOKEN_FILE`.
Compose reads `VOCAGATEWAY_TOKEN` from `.env` and mounts it as a secret at
`/run/secrets/vocagateway_token` — never as a container env var. Copy
`.env.example` → `.env`; never commit `.env`. Older `VOCAPHONE_*` names and
`~/.config/vocaphone/` are **unread**.

QR payload: `{"v":1,"url":"http://…:8765","token":"…"}`. Show with `just token`
or the WebUI **Pair & test** tab. Set `VOCAGATEWAY_PUBLIC_URL` (alias
`VOCAGATEWAY_PAIRING_URL`) when auto-discovery is wrong (Docker bridge, reverse
proxy). Bind/publish loopback for Tailscale Serve. **Never expose port `8765`
to the public internet.** HTTP is trusted LAN / VPN only.

## Deployment matrix

| Mode | Engines | Notes |
| --- | --- | --- |
| Native macOS | MLX Audio, WhisperKit, VocaMac, Handy, sherpa-onnx, faster-whisper, Moonshine, `whisper.cpp` | Best on Apple silicon |
| Native Linux | sherpa-onnx INT8, faster-whisper, Moonshine, optional `whisper.cpp` | No Apple extra |
| Docker Compose | sherpa-onnx INT8, faster-whisper INT8, Moonshine, `whisper.cpp` | Linux `amd64`/`arm64` images |

Docker Desktop cannot use macOS MLX / WhisperKit / Core ML. Host-only engines
(`vocamac`, `handy`, `mlx-audio`, `whisperkit`) are hidden in the WebUI on Linux
and in containers; API select is `422 invalid_engine`. Compose default publish is
`127.0.0.1:8765`. `VOCAGATEWAY_NETWORK_MODE=host` is Linux Docker Engine only
(not Docker Desktop) and ignores `VOCAGATEWAY_PUBLISH_HOST`/`PORT`. Profiles
`native` / `cuda` / `vulkan` share the same port and `vocagateway-data` volume —
run one service at a time.

## Consumers

| Project | How it uses this repo |
| --- | --- |
| [vocaphone](https://github.com/VocaHQ/vocaphone) | Git submodule at **`gateway/`** (not `server/`) |
| vocalinux / vocamac / vocawin | Planned: ship and start the headless server |

## WebUI vs `web/`

- **`app/webui/`** — authenticated HTMX admin at `/` (`/ui/…`, `/assets`). Token,
  models, pairing QR, mic benchmark, redacted diagnostics, device tokens.
- **`web/`** — static landing at [vocagateway.vocahq.com](https://vocagateway.vocahq.com/).
  No bundler. GitHub Pages from `web/` on `main` (`.github/workflows/deploy-pages.yml`).

Unauthenticated health: `GET /health/live`, `GET /health/ready` (`503` until a
model can transcribe), `GET /health`. Everything under `/v1/` needs
`Authorization: Bearer`. `/docs` and `/openapi.json` only when
`VOCAGATEWAY_DEBUG=true`. Upload ceiling is 25 MiB; max duration 120 s. Serve
behind a reverse proxy at a **domain root**, not a subpath.

## Code conventions

- `from __future__ import annotations`; ruff line-length 100; select
  `E,F,I,UP,B,SIM,ASYNC` (ignore `ASYNC240`).
- mypy strict on `app`. pytest files `tests/test_*.py`; prefer fakes over live
  engines (`tests/conftest.py`).
- Do not hand-edit `docs/models.md` — `uv run scripts/generate_model_docs.py`
  (`--check` is asserted in tests). Treat `app/model_pins.json` digest diffs as
  code; harvest with `uv run scripts/harvest-model-pins.py`.
- Do not weaken bearer auth, upload limits, retention, or default bind/publish
  without discussion.

## CI

| Workflow | When | What |
| --- | --- | --- |
| `quality.yml` | `app/`, `tests/`, `scripts/`, `pyproject.toml`, `uv.lock`, `compose.yaml` | ffmpeg, ruff, format `--check`, `mypy app`, pytest, compose config |
| `container.yml` | `Dockerfile*`, `.dockerignore`, `pyproject.toml`, `uv.lock` | `docker buildx` of `Dockerfile` only (`cacheonly`; not CUDA/Vulkan) |
| `verify-model-pins.yml` | pin/catalog/harvester paths + weekly | `scripts/verify-model-pins.py` |
| `deploy-pages.yml` | `web/**` on `main` | GitHub Pages |

Docs-only / `AGENTS.md` PRs skip quality and container jobs. That is expected.

## Privacy

Never commit secrets, recordings, transcripts, session ids, or private
hostnames. Diagnostics omit those. Keep `.env` and token files local.

## Git and pull requests

- Conventional Commits (`feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`).
- Never commit or push to `main`. Do not merge PRs; wait for review.
- PR body should match [`.github/pull_request_template.md`](.github/pull_request_template.md):
  **Summary** (what/why), **Verification** (`just test` or the equivalent ruff /
  mypy / pytest / compose checks; container build if Dockerfiles or the lockfile
  changed; docs if setup/network/config changed), **Privacy and security**
  checklist.
- One logical change per PR.

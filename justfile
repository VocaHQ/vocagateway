set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
# .env holds the Compose token. It belongs to the container, not to the
# recipes, so it is never loaded into this file's environment.
set dotenv-load := false

# Do not update the env, when running
export UV_NO_SYNC := '1'

# Commit stamped into every image these recipes build, surfaced by
# /v1/admin/status and the WebUI. Exported so `docker compose` interpolates them
# into the build args and `docker build --build-arg NAME` (no value) picks them
# up from the environment. Empty outside a checkout or without git — the gateway
# then reports `commit: null` instead of failing.
export VOCAGATEWAY_GIT_COMMIT := `git rev-parse HEAD 2>/dev/null || true`
export VOCAGATEWAY_GIT_COMMIT_SUBJECT := `git log -1 --format=%s 2>/dev/null || true`
export VOCAGATEWAY_GIT_COMMIT_DATE := `git log -1 --format=%cI 2>/dev/null || true`

# List all available recipes
_default:
    @just --list --unsorted --list-submodules

# Install dependencies
[group('dev')]
install:
    uv sync --all-groups --all-extras

# Serve the landing page at http://127.0.0.1:4173/
[group('dev')]
site:
    python3 -m http.server 4173 --directory web

# Check landing-page structure, assets, and product claims
[group('dev')]
site-check:
    node --test web/tests/site.test.mjs

# Format code with ruff
[group('dev')]
format:
    uv run python -m ruff format
    uv run python -m ruff check --fix

# Run all linters
[group('dev')]
lint:
    uv run python -m ruff check --exit-non-zero-on-fix
    uv run python -m ruff format --check --diff
    uv run flake8 --select=WPS,E999 app

# Run all checks
[group('dev')]
test: lint type-check package unit compose

# Run all type checkers
[group('type-check')]
type-check:
    uv run python -m mypy

# Run unit tests
[group('testing')]
unit *args='':
    uv run python -m pytest -n auto {{ args }}

# Validate package dependencies and run security audit
[group('testing')]
package:
    uv sync --all-groups --all-extras --locked --check
    uv pip check
    uv --preview-features audit audit

# The token only has to satisfy the length rule; it serves nothing. It sits up
# here because a comment inside a recipe body is echoed as a command.
#
# Validate the Compose deployment
[group('testing')]
compose:
    VOCAGATEWAY_TOKEN=test-token-with-at-least-thirty-two-characters \
      docker compose config --quiet

# Start the gateway on http://127.0.0.1:8765/
[group('run')]
run:
    uv run vocagateway

# Start the gateway bound to loopback only, ignoring any LAN or tailnet address
[group('run')]
run-local:
    VOCAGATEWAY_BIND_HOST=127.0.0.1 uv run vocagateway

# Print the bearer token; on a TTY also show a phone-scannable pairing QR
[group('run')]
token *args='':
    uv run vocagateway-token {{ args }}

# Ask a running gateway for its health
[group('run')]
status:
    uv run vocagateway-status

# Ask a running gateway for its diagnostics report
[group('run')]
diag:
    uv run vocagateway-diagnostics

# Build and start the container deployment in the background
[group('container')]
up:
    docker compose up --detach --build

# Stop the container deployment; `just down -v` also drops models, config and DB
[group('container')]
down *args='':
    docker compose down {{ args }}

# Follow the container's logs
[group('container')]
container-logs:
    docker compose logs --follow

# Build the gateway image without starting anything
[group('container')]
image:
    docker build --tag vocagateway:local \
      --build-arg VOCAGATEWAY_GIT_COMMIT \
      --build-arg VOCAGATEWAY_GIT_COMMIT_SUBJECT \
      --build-arg VOCAGATEWAY_GIT_COMMIT_DATE \
      .

# Remove tool caches; the virtualenv stays, rebuilding it is a long download
[group('build')]
clean:
    rm -rf .pytest_cache .ruff_cache .mypy_cache

# Report what this justfile needs and cannot find
[group('build')]
doctor:
    #!/usr/bin/env bash
    ok=0
    if command -v uv >/dev/null 2>&1; then
      echo "ok       $(uv --version)"
    else
      echo "MISSING  uv — curl -LsSf https://astral.sh/uv/install.sh | sh"
      ok=1
    fi
    # Every engine normalizes through FFmpeg, and tests/test_audio.py shells out
    # to it against real audio, so this is required, not optional.
    if command -v ffmpeg >/dev/null 2>&1; then
      echo "ok       ffmpeg"
    else
      echo "MISSING  ffmpeg — brew install ffmpeg (or apt install ffmpeg)"
      ok=1
    fi
    if command -v docker >/dev/null 2>&1; then
      echo "ok       docker"
    else
      echo "MISSING  docker — needed by just compose and the container recipes"
      ok=1
    fi
    if [ "$(uname -s)" = "Darwin" ]; then
      for tool in whisperkit-cli whisper-cli; do
        if command -v "${tool}" >/dev/null 2>&1; then
          echo "ok       ${tool}"
        else
          echo "note     ${tool} absent — optional, the gateway runs its own models"
        fi
      done
    fi
    if [ "${ok}" -ne 0 ]; then
      echo
      echo "Install what is marked MISSING above, then run just doctor again."
    fi
    exit "${ok}"

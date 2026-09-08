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
    uv run flake8 --select=WPS,E999 app tests

# `just copyright -check` only reports, which is what CI runs
# (.github/workflows/copyright-check.yml). HTML is skipped on purpose: a comment
# on top of an HTMX fragment would ship in every response body.
#
# Write the COPYRIGHT.txt header into every source file that is missing one
[group('dev')]
copyright *args='':
    addlicense {{ args }} -f COPYRIGHT.txt \
      -ignore '**/*.html' \
      -ignore 'app/webui/htmx.min.js' \
      -ignore 'app/webui/swagger/swagger-ui-bundle.js' \
      -ignore 'app/webui/swagger/swagger-ui.css' \
      -ignore 'web/sitemap.xml' \
      app tests scripts web

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
# Validate both Compose deployments: the source build and the published image
[group('testing')]
compose:
    VOCAGATEWAY_TOKEN=test-token-with-at-least-thirty-two-characters \
      docker compose config --quiet
    VOCAGATEWAY_TOKEN=test-token-with-at-least-thirty-two-characters \
      docker compose -f compose.prod.yaml config --quiet

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

# Run a published image instead of building. Needs a release that has pushed
# one; until then use `just up`. `just up-release 0.1.0` pins the version.
[group('container')]
up-release version='':
    #!/usr/bin/env bash
    set -euo pipefail
    # No version given means "whatever the deployment file and .env already
    # say", so an operator who pinned a version there keeps it. A version given
    # here wins for this one command.
    if [ -n "{{ version }}" ]; then
      export VOCAGATEWAY_IMAGE="docker.io/vocahq/vocagateway:{{ version }}"
    fi
    docker compose -f compose.prod.yaml up --detach --pull always

# Stop the container deployment; `just down -v` also drops models, config and DB
[group('container')]
down *args='':
    docker compose down {{ args }}

# Follow the container's logs
[group('container')]
container-logs:
    docker compose logs --follow

# Build a gateway image without starting anything; `just image cuda` or `vulkan`
[group('container')]
image accel='cpu':
    docker build --tag vocagateway:{{ if accel == "cpu" { "local" } else { accel } }} \
      --build-arg ACCEL={{ accel }} \
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
    # Transcript cleanup runs on this. The container builds its own, so this is
    # a note rather than a failure: only a native gateway needs one on the host.
    if command -v llama-server >/dev/null 2>&1; then
      echo "ok       llama-server"
    else
      echo "note     llama-server absent — transcript cleanup stays off until"
      echo "         one exists (brew install llama.cpp, or VOCAGATEWAY_CLEANUP_BINARY)"
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

# Gateway deployment

vocagateway uses the same HTTP API whether the gateway runs directly on macOS,
directly on Linux, or inside its Linux container. The meaningful differences are
the available speech engines, acceleration, isolation, and operational
portability.

## Contents

- [Which deployment should I choose?](#which-deployment-should-i-choose)
  - [Recommendation](#recommendation)
- [Host tool requirements](#host-tool-requirements)
- [Native macOS deployment](#native-macos-deployment) — [install](#install-and-run) · [run at login](#run-at-login)
- [Native Linux deployment](#native-linux-deployment) — [install](#install-and-run-1) · [systemd user service](#run-as-a-systemd-user-service)
- [Docker Compose deployment](#docker-compose-deployment) — [prerequisites](#prerequisites) · [from source](#building-from-the-checkout) · [published image](#running-a-published-image) · [first model](#first-model) · [transcript cleanup](#transcript-cleanup-in-the-container) · [routine operations](#routine-operations) · [backup](#persistent-data-and-backup) · [performance profiles](#performance-profiles) · [Vulkan GPU access](#giving-the-vulkan-container-access-to-the-gpu) · [build tuning](#tuning-the-compiled-runtimes)
- [Published images](#published-images) — [tags](#tags) · [upgrading](#upgrading-and-going-back) · [building your own](#building-and-pushing-your-own)
- [Gateway URL and network placement](#gateway-url-and-network-placement) — [trusted LAN](#trusted-local-network) · [Tailscale Serve](#tailscale-serve) · [VPS or public DNS](#vps-or-public-dns)
- [Configuration paths and env vars](#configuration-paths-and-env-vars)

Quick starts live in the [README](../README.md). This page is the longer form:
what to choose, how to keep it running, and how the phone reaches it.

## Which deployment should I choose?

| Consideration | Native macOS | Native Linux | Docker Compose |
| --- | --- | --- | --- |
| Recommended host | Apple silicon Mac | Linux desktop or home server | Linux `amd64`/`arm64` when you want an image |
| Engines | MLX Audio, WhisperKit, VocaMac, Handy, sherpa-onnx, faster-whisper, Moonshine, `whisper.cpp` | sherpa-onnx, faster-whisper, Moonshine, optional `whisper.cpp` | sherpa-onnx, faster-whisper, Moonshine, `whisper.cpp` |
| Acceleration | Apple-native MLX and WhisperKit/Core ML paths | Host CPU (Python wheels); CUDA via Docker profiles | Runtime-dispatched CPU backends; CUDA or Vulkan profiles |
| Performance | Recommended on Mac; no Linux VM | No container overhead on Linux | Slightly more isolation cost; strong for CUDA images |
| Portability | macOS LaunchAgent | systemd user unit | Reproducible across supported Linux architectures |
| Persistence | Files below `~/.local/share/vocagateway` | Same as native macOS | Named volume mounted at `/data` |
| Updates | Pull code, `uv sync`, restart LaunchAgent | Pull code, `uv sync`, restart systemd unit | Pull/build image and recreate the service |

### Recommendation

- On an Apple silicon Mac, run natively and compare MLX Whisper Turbo 4-bit,
  MLX Parakeet, and WhisperKit on the same recording. These avoid Docker
  Desktop's Linux VM and keep the chosen Apple-native model resident.
- On a Linux desktop or home server, run natively with
  `uv sync --all-groups --extra engines` when you already trust the host Python
  environment. Start with SenseVoice Small INT8 for its supported Asian languages
  plus English, or Parakeet TDT INT8 for 25 European languages. Both use
  sherpa-onnx wheels on `amd64` and `arm64`; faster-whisper remains the broad
  Whisper fallback.
- Use Docker on Linux when you want CUDA/Vulkan profiles, multi-arch images, or
  stronger isolation. Use Docker on a Mac only when reproducibility matters more
  than the lowest transcription latency.

There is no honest fixed speed multiplier: model size, audio length, thermals,
and host hardware all matter. For an apples-to-apples comparison, dictate the
same saved recording several times with equivalent model sizes and compare the
Pair & test tab's three-run benchmark. It treats run 1 as model warmup/load and
reports the warm average of runs 2 and 3. Compare inference time and real-time factor,
not only end-to-end time.

## Host tool requirements

Python packages in `pyproject.toml` / `uv.lock` are installed by `uv sync` or
`just install`. Native executables are separate host requirements:

| Tool | Required when | Installation |
| --- | --- | --- |
| Python 3.12+ and uv | Every native deployment | Native quick starts below |
| FFmpeg | Every engine; normalizes recordings | `brew install ffmpeg` or `sudo apt install ffmpeg` |
| `whisper-cli` | Selecting `whisper.cpp` | Homebrew `whisper-cpp`, or an upstream native build |
| `whisper-server` | Keeping `whisper.cpp` resident | Same whisper.cpp build as `whisper-cli`; optional fallback to CLI |
| `whisperkit-cli` | Standalone WhisperKit on macOS | `brew install whisperkit-cli` |
| `llama-server` | Transcript cleanup on a **native** install | `brew install llama.cpp`, or an upstream build named by `VOCAGATEWAY_CLEANUP_BINARY`. The container builds its own |

Run `just doctor` to check the host tools.
## Native macOS deployment

### Install and run

```sh
# ffmpeg (required), WhisperKit CLI, and the whisper.cpp CLI
brew install ffmpeg whisperkit-cli whisper-cpp
uv sync --all-groups --extra engines --extra apple
uv run vocagateway
```

The default listener is `0.0.0.0:8765`, while the local WebUI is
`http://127.0.0.1:8765/`. When Tailscale Serve is the only desired ingress,
override the listener:

```sh
VOCAGATEWAY_BIND_HOST=127.0.0.1 uv run vocagateway
```

The first run creates a mode-600 token file at
`~/.config/vocagateway/token`. Models default to
`~/.local/share/vocagateway/models`, the session database lives in the parent
data directory, and WebUI choices are stored in
`~/.config/vocagateway/config.json`.

### Run at login

```sh
./scripts/install-launch-agent.sh
launchctl print "gui/$(id -u)/com.vocahq.vocagateway"
```

Logs are written to `~/Library/Logs/VocaGateway/gateway.log` and
`gateway-error.log`. Re-run the installer after changing the checkout location
or gateway executable.

## Native Linux deployment

### Install and run

```sh
# Debian / Ubuntu example
sudo apt install ffmpeg
# https://docs.astral.sh/uv/ — curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-groups --extra engines
uv run vocagateway
```

Omit `--extra apple` on Linux. The default listener is `0.0.0.0:8765`. When
Tailscale Serve is the only desired ingress, override the listener:

```sh
VOCAGATEWAY_BIND_HOST=127.0.0.1 uv run vocagateway
```

Token, models, and config paths match the macOS native layout:

- token: `~/.config/vocagateway/token`
- models: `~/.local/share/vocagateway/models`
- config: `~/.config/vocagateway/config.json`

Phone clients need the bearer token (`cat ~/.config/vocagateway/token`) and a
reachable URL such as `http://192.168.1.20:8765` on a trusted LAN.

### Run as a systemd user service

```sh
./scripts/install-systemd-user.sh
systemctl --user status com.vocahq.vocagateway.service
journalctl --user -u com.vocahq.vocagateway.service -f
```

To keep the unit after logout:

```sh
loginctl enable-linger "$USER"
```

Re-run the installer after moving the checkout or recreating `.venv`.

## Docker Compose deployment

Two files, and the difference is where the image comes from:

| File | Image | Use it when |
| --- | --- | --- |
| [`compose.yaml`](../compose.yaml) | Built from this checkout | Cloning today, changing the code, or you need a `cuda` / `vulkan` image |
| [`compose.prod.yaml`](../compose.prod.yaml) | Pulled: `docker.io/vocahq/vocagateway` | After a GitHub release has published images. No checkout, no compiler, ~1 minute |

Everything else about the two is identical — the same environment, the same
volume, the same hardening — and a test asserts that, so a deployment never
gets a weaker container than a contributor's.

Until the first successful publish, Hub has nothing to pull. Build from
`compose.yaml`.

### Prerequisites

- Docker Engine with Compose v2, or Docker Desktop
- At least enough free memory and disk space for the selected model
- Tailscale on the host when the iPhone connects over the tailnet

### Building from the checkout

The path that works today, and the only one that produces a `cuda` or `vulkan`
image:

```sh
umask 077
cp .env.example .env
printf 'VOCAGATEWAY_TOKEN=%s\n' "$(openssl rand -hex 32)" >> .env
docker compose up --detach --build
```

Expect tens of minutes: the image compiles whisper.cpp and llama.cpp from
source for your accelerator.

[`.env.example`](../.env.example) is the annotated template for the same file,
organised in seven numbered sections: the token, the published host/port, the
pairing-QR address, the image tag, gateway behaviour (engine, retention, the
Swagger UI), the container listener, and the settings that look like they
belong in `.env` but are never forwarded to the container. It ships the
loopback publication defaults uncommented and everything else commented out
with an explanation. Start from it rather than writing `.env` by hand, so the
options are in front of you. Never commit the populated file.

The appended token overrides the empty `VOCAGATEWAY_TOKEN=` placeholder in the
template; Compose uses the last assignment when a key repeats in `.env`.

`VOCAGATEWAY_PUBLISH_HOST=127.0.0.1` is the safe default for Tailscale Serve. Set
it to `0.0.0.0` only when direct LAN access is intentional and protected by the
host firewall. Never forward the port from the public internet.

### Running a published image

Use this only after a GitHub release has successfully published images, and
after maintainers have set the Docker Hub secrets (`DOCKERHUB_USERNAME` /
`DOCKERHUB_TOKEN`). Until then `docker.io/vocahq/vocagateway` has nothing to
pull; stay on [Building from the checkout](#building-from-the-checkout).

Nothing to clone. Two files and a token:

```sh
umask 077
curl -O https://raw.githubusercontent.com/VocaHQ/vocagateway/main/compose.prod.yaml
curl -o .env https://raw.githubusercontent.com/VocaHQ/vocagateway/main/.env.example
printf 'VOCAGATEWAY_TOKEN=%s\n' "$(openssl rand -hex 32)" >> .env
docker compose -f compose.prod.yaml up --detach
```

That pulls one multi-architecture tag covering `linux/amd64` and `linux/arm64`;
Docker picks the right one. See [Published images](#published-images) for the
tags, the second registry, and how to upgrade or roll back.

### First model

The container starts before a model is installed. Confirm process liveness,
then open the WebUI and download/select a recommended sherpa-onnx, Moonshine,
or faster-whisper model:

```sh
docker compose ps
curl --fail http://127.0.0.1:8765/health/live
```

`/health/ready` returns HTTP `503` until the selected model is runnable. After
selection it should return HTTP `200` with `"status":"ready"`.

### Transcript cleanup in the container

Nothing to enable and nothing extra to run: the image compiles
`llama-server` for the same accelerator as the speech runtime, and the gateway
launches and owns it on loopback, on an unpublished ephemeral port, with a
credential of its own. `docker compose up` is the whole deployment.

What is left is the model, which is not baked into the image — it is downloaded
through the WebUI, verified against a pinned SHA-256, and stored in the same
`/data` volume as the speech models:

```sh
docker compose up --detach --build
# WebUI → Cleanup → download a model → Load model now
```

Until that download lands there is nothing to run cleanup on, and every
transcript takes exactly the path it would with the feature off. Once it lands
on a fresh install, a downloaded model needs no second "select" click while it
is the only one installed, and corrections start on the next dictation. A
config written before cleanup existed loads with the feature off, even if a
leftover model is already on disk; enable it in the Cleanup tab. **Overview →
Libraries & tools** reports the runtime beside FFmpeg and whisper.cpp, so a
missing one is visible without opening the Cleanup tab.

Worth knowing before you turn it loose:

- **Memory.** The cleanup model stays resident alongside whatever the speech
  engine is holding. On a CPU-only host under 16 GB RAM the managed worker uses
  the **compact** profile (4096 context, 8-bit KV). On a GPU or 16 GB+ host it
  uses **full** (8192 context, 16-bit KV). Force one with
  `VOCAGATEWAY_CLEANUP_PROFILE`. See [cleanup.md](cleanup.md). Measure both
  models together on the host you actually run on. *Unload after idle* is on
  for a fresh install and gives the memory back between bursts, at the cost of
  a reload on the next correction.
- **Build time.** Two ggml projects are compiled instead of one. See
  [Tuning the compiled runtimes](#tuning-the-compiled-runtimes) for narrowing
  the CUDA architecture spread and for capping concurrent compile jobs.
- **No new exposure.** The worker binds `127.0.0.1` inside the container, on a
  port that is never published, and the gateway authenticates to it with a key
  it generates per worker. A client's bearer token is never forwarded to it.
- **Turning it off.** Untick *Correct transcripts by default* in the Cleanup
  tab, or set `VOCAGATEWAY_CLEANUP_ENABLED=false` to take the decision away from
  the WebUI entirely. Deleting the model has the same practical effect.

#### Using a cleanup server you run yourself

For evaluating a runtime or a model this image did not build, point the gateway
at a server you run instead. Add it as a service of your own — Compose merges
`compose.override.yaml` automatically — and name it in `.env`:

```yaml
# compose.override.yaml
services:
  my-cleanup:
    image: ghcr.io/ggml-org/llama.cpp:server@sha256:...   # pin it by digest
    expose: ["8080"]                                      # never `ports:`
    volumes: [./models:/models:ro]
    command:
       [--model, /models/your.gguf, --host, 0.0.0.0, --port, "8080",
        --ctx-size, "4096", --batch-size, "512", --ubatch-size, "256",
        --cache-type-k, q8_0, --cache-type-v, q8_0,
        --parallel, "1", --jinja, --no-webui,
        --api-key, "${VOCAGATEWAY_CLEANUP_API_KEY}"]
```

```sh
VOCAGATEWAY_CLEANUP_ENDPOINT=my-cleanup:8080 \
VOCAGATEWAY_CLEANUP_API_KEY="$(openssl rand -hex 24)" \
docker compose up --detach
```

Do not reuse the old sidecar hostname. `VOCAGATEWAY_CLEANUP_ENDPOINT=cleanup:8080`
(or `cleanup` on any other port) is refused at startup: that Compose service is
gone. Unset the variable to use the in-image runtime. If you run a server of
your own, give it a different service name, as above.

Three things the gateway will hold you to. It refuses any address that is not
loopback, a private range, or a bare container service name on this project's
own network, because "runs on your gateway" has to stay true. It makes no
lifecycle promises for a process it did not start — no warm-up, no idle unload,
no restart. Match compact or full flags to the machine that runs that
server — see [cleanup.md](cleanup.md). And the window has to be
`--ctx-size 4096` or more: a window too
small does not fail loudly, it silently drops the system instruction, so the
gateway declines with `context_too_small` rather than trusting the result.

### Routine operations

Add `-f compose.prod.yaml` to each of these when you deploy a published image;
without it, Compose reads `compose.yaml` and would build.

```sh
# Follow gateway logs
docker compose logs --follow gateway

# Restart without deleting data
docker compose restart gateway

# Update to the newest published image
docker compose -f compose.prod.yaml pull
docker compose -f compose.prod.yaml up --detach

# Rebuild from an updated checkout
docker compose up --detach --build

# Stop the service while preserving the named volume
docker compose down
```

Do not add `--volumes` to `docker compose down` unless deleting every downloaded
model, stored configuration, and session record is intentional.

### Persistent data and backup

Compose mounts the `vocagateway_vocagateway-data` named volume at `/data`. Inspect it
with:

```sh
docker volume inspect vocagateway_vocagateway-data
```

Stop the gateway before taking a filesystem-level backup so the SQLite database
and model directory are captured consistently. A Docker or host-native backup
tool can then archive the volume shown by `docker volume inspect`. Keep backups
private because failed recordings may remain for the configured retry period.

WhisperKit model folders cannot run in a Linux container. Download a compatible
faster-whisper, Moonshine, or `whisper.cpp` model from the container WebUI
instead of copying the native macOS model directory blindly.

### Performance profiles

All three services come from the same `Dockerfile`; the `ACCEL` build argument
picks the accelerator, and Compose sets it per service. Stop the running one
before starting another, because all three publish the same port and share the
named volume.

```sh
docker compose down

# Portable CPU (default). No profile flag needed.
docker compose up --detach --build

# NVIDIA Container Toolkit and a supported NVIDIA GPU are required.
docker compose --profile cuda up --detach --build gateway-cuda

# A working host Vulkan driver and /dev/dri are required; see below.
docker compose --profile vulkan up --detach --build gateway-vulkan
```

There is no separate "native" CPU profile any more, and none is needed. The CPU
image is built with `GGML_CPU_ALL_VARIANTS`, which compiles one ggml CPU backend
per micro-architecture — `sse42` through `haswell`, `zen4`, `alderlake` and
`sapphirerapids` on x86, `armv8.0` through `armv8.2+dotprod`, `armv8.6+i8mm` and
`armv9.2+sme` on arm64 — and dlopens the best one the host reports at startup.
The image stays a portable registry artifact while still running AVX2/AVX-512
kernels on a modern x86 host and dotprod/i8mm kernels on a modern arm64 one,
which is what the old `native` build was for.

The CUDA and Vulkan images should be published only for architectures supported
by their base images and host drivers.

#### Giving the Vulkan container access to the GPU

Passing `/dev/dri` through is necessary but not sufficient. The render node is
owned by `root:render` on the host and the gateway runs as uid 10001, so without
a matching supplementary group the open fails with `EACCES`, Vulkan enumerates
no physical device, and whisper.cpp falls back to Mesa's software rasteriser —
slower than the plain CPU image, and silent about it.

Compose adds the groups for you, but their GIDs differ per distribution. Inspect
the render and card nodes, then put the numeric GIDs in `.env`:

```sh
stat -c '%G %g %n' /dev/dri/renderD128 /dev/dri/card0
echo 'VOCAGATEWAY_RENDER_GID=993' >> .env
echo 'VOCAGATEWAY_VIDEO_GID=44' >> .env
```

Use the GID reported for `renderD128` as `VOCAGATEWAY_RENDER_GID` and the GID
reported for `card0` as `VOCAGATEWAY_VIDEO_GID`. A missing `card0` is harmless;
the render node is the one compute requires.

The Vulkan profile ships the Mesa ICDs, so it covers AMD and Intel GPUs. NVIDIA
over Vulkan needs the host ICD injected by the Container Toolkit instead; on an
NVIDIA host use the CUDA profile.

#### Tuning the compiled runtimes

The image compiles two ggml projects: whisper.cpp for speech, and llama.cpp's
`llama-server` for transcript cleanup. Each has its own `.env` variable, which
is appended to that project's `cmake` line after the defaults, so a flag set
there overrides one the Dockerfile sets. They are separate because the builds
are separate — narrowing the CUDA architecture spread for one does nothing for
the other, and the two are roughly half the build each.

```sh
# Build CUDA kernels for one known GPU instead of the portable spread of
# virtual and real architectures ggml picks. Much faster nvcc, smaller binary,
# and the image no longer runs anywhere else.
VOCAGATEWAY_WHISPER_CMAKE_EXTRA=-DCMAKE_CUDA_ARCHITECTURES=89-real

# Build the CPU image without OpenBLAS.
VOCAGATEWAY_WHISPER_CMAKE_EXTRA=-DGGML_BLAS=OFF

# The same flags for the cleanup runtime. Set both when narrowing the CUDA
# spread, or the llama.cpp half still compiles every architecture.
VOCAGATEWAY_LLAMA_CMAKE_EXTRA=-DCMAKE_CUDA_ARCHITECTURES=89-real
```

`VOCAGATEWAY_BUILD_JOBS` caps how many compile jobs run at once; blank is
resolved to the builder's CPU count. The explicit default is important because
CMake otherwise lets Make interpret a bare `--parallel` as unlimited jobs. The
cuda image is the usual reason to set a lower value: nvcc instantiates a great
many templates and each job can want most of a gigabyte. Budget roughly one job
per 2 GB of builder memory — an 8 GB machine normally wants
`VOCAGATEWAY_BUILD_JOBS=3`.

OpenBLAS stays on by default because it measurably earns its place. On arm64
(Docker Desktop, `--cpus 4`, `ggml-tiny.en`, `samples/jfk.wav`, `-bs 2 -bo 2`)
the BLAS build ran the clip in ~1.95-2.06 s against ~3.15-3.18 s without it —
about 1.6x, consistently across five runs each. That is with the
per-micro-architecture CPU kernels already in play, so it is BLAS earning it
rather than BLAS compensating for a baseline build. The balance may differ on an
x86 host with AVX-512; time the same clip against both images before changing
it:

```sh
docker build --build-arg ACCEL=cpu --tag vocagateway:blas .
docker build --build-arg ACCEL=cpu \
  --build-arg WHISPER_CMAKE_EXTRA=-DGGML_BLAS=OFF --tag vocagateway:noblas .

for tag in blas noblas; do
  echo "== ${tag}"
  time docker run --rm \
    --volume vocagateway_vocagateway-data:/data:ro \
    --volume "$PWD/sample.wav:/tmp/sample.wav:ro" \
    --entrypoint whisper-cli "vocagateway:${tag}" \
      -m /data/models/<your-model>.bin -f /tmp/sample.wav -np -nt \
      -t "$(nproc)" -bs 2 -bo 2
done
```

The gateway passes `-bs 2 -bo 2` and a physical-core thread count of its own, so
matching them here keeps the measurement representative.

The image deliberately leaves `OPENBLAS_NUM_THREADS` unset. OpenBLAS-pthread
sizes its pool from the host CPU count and cannot see the cgroup quota, so a
container allowed two CPUs still reads `nproc` as whatever the host has — which
looks like it should oversubscribe badly. It does not: ggml's BLAS backend
issues a matmul from one thread at a time rather than from each of its workers,
so the two pools never nest. Pinned to 1 and unset were within run-to-run noise
for `ggml-tiny.en` and `ggml-base.en`, at `--cpus 4` and `--cpus 2` on a 10-CPU
host. Pinning it would only cap OpenBLAS's own parallelism for no gain.

## Published images

Once a GitHub release has published successfully, the CPU image is available
from two registries, with identical digests in both:

| Registry | Image |
| --- | --- |
| Docker Hub | `docker.io/vocahq/vocagateway` |
| GitHub Container Registry | `ghcr.io/vocahq/vocagateway` |

Until that first publish, build from [`compose.yaml`](../compose.yaml) instead.
`compose.prod.yaml` will 404 on pull.

GHCR is there for the day Docker Hub's anonymous pull limit gets in the way;
either serves the same bytes. Point `VOCAGATEWAY_IMAGE` at whichever you
prefer.

### Tags

| Tag | Moves | Use it for |
| --- | --- | --- |
| `0.1.0` | Never | A deployment you want to stay put. **Pin this in production** |
| `0.1` | Only when this patch is the newest final release of that minor series. Rebuilding an older patch does not rewrite it | Automatic patch updates, no minor jumps |
| `latest` | Only for a published GitHub release of the newest final version overall; never on a pre-release, and never when rebuilding an older tag | Trying it out, and home deployments that track the newest version |

`compose.prod.yaml` defaults to `latest`. Pin a version in `.env` when the
gateway matters to you:

```sh
VOCAGATEWAY_IMAGE=docker.io/vocahq/vocagateway:0.1.0
```

Each tag is a manifest list covering `linux/amd64` and `linux/arm64`, built
natively on a runner of each architecture — no emulation, and no per-platform
tag to choose between. Within an architecture the image is portable anyway:
`GGML_CPU_ALL_VARIANTS` compiles one ggml CPU backend per micro-architecture
and the best one for the host is loaded at startup.

### Upgrading, and going back

```sh
# Take the newest image the tag now points at
docker compose -f compose.prod.yaml pull
docker compose -f compose.prod.yaml up --detach

# Go back: pin the previous version and recreate
VOCAGATEWAY_IMAGE=docker.io/vocahq/vocagateway:0.1.0 \
  docker compose -f compose.prod.yaml up --detach
```

Both are safe for your data: models, config, the session database and device
tokens live in the `vocagateway_vocagateway-data` volume, which the container
only mounts. `docker compose -f compose.prod.yaml down` leaves it alone; only
`down -v` deletes it.

Check what you are actually running:

```sh
curl --fail --silent -H "Authorization: Bearer ${VOCAGATEWAY_TOKEN}" \
  http://127.0.0.1:8765/v1/admin/status | jq '{version, commit}'
```

`version` is always reported. `commit` — the source revision the image was
built from, which the release workflow stamps in — is `null` unless
`VOCAGATEWAY_DEBUG=true`, the same switch that adds the WebUI's **Build** row.
See [Stamping the build commit](../README.md#stamping-the-build-commit).

### What is not published

The `cuda` and `vulkan` images. Both are pinned to a driver stack and a GPU
generation, which is not a promise a public tag can keep, so they stay
build-it-yourself from `compose.yaml` and its profiles.

### Building and pushing your own

For a private registry, or a variant this project does not publish:

```sh
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag registry.example.com/vocagateway:latest \
  --push .
```

On an amd64 builder the arm64 half of that compiles whisper.cpp and llama.cpp
under QEMU emulation, which takes hours. A native arm64 builder — a remote
buildx node, or a CI runner of that architecture, which is what the release
workflow uses — is the fix; the build's ccache and uv cache mounts at least
make a repeat run cheap.

Conversely, Docker Desktop on an Apple silicon Mac can validate the Linux arm64
CPU image but not the Linux amd64 or NVIDIA CUDA paths. Pull requests that touch
container inputs run the Container GitHub Actions matrix for CPU, CUDA, and
Vulkan; use that result for cross-platform validation rather than treating one
local M1 build as coverage of all three variants. The CI CUDA build is a
compile-only check narrowed to one representative GPU architecture and omits
the CPU variants already exercised by the CPU job. It is deliberately faster
than the portable CUDA image produced from the unmodified Dockerfile defaults.

`VOCAGATEWAY_IMAGE` names the image for the `gateway` service in both files. In
`compose.prod.yaml` it selects what to pull. In `compose.yaml` it renames what
gets built rather than switching Compose to pulling, so `up --build` still
builds locally; the `gateway-cuda` and `gateway-vulkan` services carry fixed
tags and ignore it.

### How a release is published

[`.github/workflows/release.yml`](../.github/workflows/release.yml) runs when a
GitHub release is published. It builds `linux/amd64` on `ubuntu-24.04` and
`linux/arm64` on `ubuntu-24.04-arm`, pushes each by digest, smoke-tests the
amd64 digest under the same hardening Compose applies, and only then assembles
one manifest list per tag and copies those tags to GHCR. Digests may exist on
the registry while the smoke runs; public tags (`0.1.0`, `0.1`, `latest`) do
not move until it passes. A failed smoke leaves floating tags pointing at
whatever they already named.

The version tag always publishes. `major.minor` (for example `0.1`) is
emitted only when this tag is the newest final release of that line among git
tags. `latest` moves only for a published non-pre-release GitHub release of
the newest final tag overall.

Maintainers: the workflow needs two repository secrets, `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN` (a Docker Hub access token with **Read & Write** scope).
GHCR uses the workflow's own `GITHUB_TOKEN`. `workflow_dispatch` rebuilds a
given tag, with a `push` toggle so a dry run publishes nothing.

## Gateway URL and network placement

The iPhone app accepts any gateway URL with an explicit `http://` or `https://`
scheme and a valid hostname. Tailscale is one option rather than a requirement.

### Trusted local network

The native gateway already listens on all interfaces by default. For Docker,
set the Compose publication in `.env`:

```dotenv
VOCAGATEWAY_PUBLISH_HOST=0.0.0.0
VOCAGATEWAY_PUBLISH_PORT=8765
```

Protect port 8765 with the host firewall, ensure the hostname resolves from the
iPhone, and use a URL such as `http://homelabone:8765/`. Approve Local Network
access when iOS asks. Plain HTTP exposes recordings and the bearer token to
anyone who can inspect the network, so use it only on a trusted LAN or encrypted
VPN and never forward it from a router.

With the default bridge network, the container only ever sees its own private
bridge address (for example `172.19.0.2`), never the host's real Wi-Fi/Ethernet
interface — so the pairing card's auto-discovered candidate list won't include
a `192.168.x.x` address even after the change above.

The portable fix is to stop relying on discovery and name the address the phone
should use:

```dotenv
VOCAGATEWAY_PUBLIC_URL=http://192.168.1.20:8765
```

The gateway puts that URL first in the pairing card and encodes it in the QR.
It works on every Docker flavour, including Docker Desktop on macOS and
Windows, and is the only option there. `VOCAGATEWAY_PAIRING_URL` is an accepted
alias, checked second.

On Linux Docker Engine (not Docker Desktop) you can instead share the host's
network namespace so discovery sees the real LAN IP by itself:

```dotenv
VOCAGATEWAY_NETWORK_MODE=host
```

`VOCAGATEWAY_PUBLISH_HOST`/`VOCAGATEWAY_PUBLISH_PORT` are ignored in this mode —
Compose discards the `ports:` mapping and the container binds straight onto the
host per `VOCAGATEWAY_BIND_HOST` (`0.0.0.0` by default) and `VOCAGATEWAY_PORT`
(`8765` by default). That means the host firewall is now the only thing
standing between port 8765 and every interface on the box, including any
public one — lock it down before enabling this.

### Tailscale Serve

Both native and Compose deployments can remain on host loopback:

```sh
tailscale serve --bg 8765
tailscale serve status
```

Enter the reported private HTTPS URL in the iPhone app. Tailscale identity is an
additional network boundary; the vocaphone bearer token remains required. See
[Private Tailscale connectivity](tailscale.md) for the complete setup.

### VPS or public DNS

Keep the gateway published on `127.0.0.1`, place an HTTPS reverse proxy such as
Caddy or nginx in front of it, and use a trusted certificate for the public
hostname. Enter a URL such as `https://dictation.example.com/` in the app.

Keep bearer authentication enabled at the gateway even if the reverse proxy has
its own access control. Do not expose port 8765 directly or use unencrypted HTTP
over the public internet.

## Configuration paths and env vars

Native installs store the bootstrap token at `~/.config/vocagateway/token`,
engine settings at `~/.config/vocagateway/config.json`, and application data
under `~/.local/share/vocagateway/`. Docker Compose persists the same layout
under `/data` in the `vocagateway-data` named volume.

Under Compose those paths are not tunable from `.env`. `compose.yaml` forwards
only the variables it names, so `VOCAGATEWAY_DATA_DIR`,
`VOCAGATEWAY_MODELS_DIR`, `VOCAGATEWAY_CONFIG_FILE`, `VOCAGATEWAY_TOKEN_FILE`,
and the macOS engine paths are read out of `.env` and dropped —
`docker compose config` passes and nothing changes. Move container data by
remapping the `vocagateway-data` volume.

Environment variables use the `VOCAGATEWAY_*` prefix (for example
`VOCAGATEWAY_TOKEN`, `VOCAGATEWAY_BIND_HOST`, `VOCAGATEWAY_PUBLIC_URL`). The
Python package and primary CLI are `vocagateway` (`vocagateway`,
`vocagateway-token`, and related scripts). Deprecated `vocaphone-*` aliases
still resolve for one cycle. Older `VOCAPHONE_*` names and `~/.config/vocaphone/`
paths are not read. Full contract:
[configuration.md](configuration.md).

Downloaded models carry a `.vocagateway-model.json` marker file in each model
directory.

The WebUI, API routes (`/v1/*`, `/health/*`), pairing protocol, and engine IDs
are unchanged.

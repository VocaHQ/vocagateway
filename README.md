<div align="center">

<img src="https://raw.githubusercontent.com/VocaHQ/.github/main/brand/vocagateway/vocagateway-tower.svg" alt="VocaGateway" width="160" height="160">

# VocaGateway

[![Quality](https://github.com/VocaHQ/vocagateway/actions/workflows/quality.yml/badge.svg)](https://github.com/VocaHQ/vocagateway/actions/workflows/quality.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-Beta-0f6b57)](#vocagateway)
[![Release](https://img.shields.io/github/v/release/VocaHQ/vocagateway?label=Release)](https://github.com/VocaHQ/vocagateway/releases)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS + Linux + Docker](https://img.shields.io/badge/platform-macOS%20%2B%20Linux%20%2B%20Docker-lightgrey)](#deployment-summary)

[![Privacy: self-hosted, not on-device](https://img.shields.io/badge/privacy-self--hosted%20%7C%20not%20on--device-success)](#vocagateway)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/VocaHQ/vocagateway/pulls)
[![GitHub Issues](https://img.shields.io/github/issues/VocaHQ/vocagateway)](https://github.com/VocaHQ/vocagateway/issues)
[![Discord](https://img.shields.io/discord/1538633755877580810?logo=discord&logoColor=white&label=Discord)](https://discord.gg/t6muquAJbm)
[![Follow on X](https://img.shields.io/badge/Follow%20%40vocahq-000000?style=flat&logo=x&logoColor=white)](https://x.com/vocahq)
[![VocaHQ](https://img.shields.io/badge/VocaHQ-vocahq.com-1a7f4e)](https://vocahq.com)

</div>

**Beta** optional self-hosted transcription gateway for the
[Voca](https://github.com/VocaHQ) family. License:
[AGPL-3.0](LICENSE). Contact:
[hello@vocahq.com](mailto:hello@vocahq.com).

The public landing page is in [`web/`](web/) and deploys to
[vocagateway.vocahq.com](https://vocagateway.vocahq.com/).

Install it once on a machine you control, then pair phone clients to that host.
Desktop embed is Planned. A shipped VocaLinux can already point its `remote_api`
engine at `POST /v1/audio/transcriptions` on this host. You can self-host on
macOS or Linux, or use Docker Compose on Linux `amd64`/`arm64`. There is no Voca
account and no hosted Voca cloud.

The gateway takes bounded recordings from
[vocaphone](https://github.com/VocaHQ/vocaphone) (iOS/Android), and from
VocaLinux when that app is set to `remote_api`. FFmpeg normalizes the audio, a
local speech engine transcribes it, and the gateway returns an idempotent
transcript. The authenticated HTMX WebUI covers setup, model management, engine
selection, microphone testing, and operational status.

Gateway mode is not on-device processing. Audio leaves the client and travels to
the machine you chose. Prefer a trusted LAN, Tailscale, or HTTPS. Never expose
port `8765` to the public internet.

The CLI is `vocagateway` (package `vocagateway`). Deprecated aliases
(`vocaphone-server`, `vocaphone-token`, and related `vocaphone-*` scripts) still
resolve for one cycle so older justfiles keep working. Environment variables and
on-disk paths use the `vocagateway` prefix (`VOCAGATEWAY_*`,
`~/.config/vocagateway/`, `~/.local/share/vocagateway/`). The live pairing and env
contract is in [configuration.md](docs/configuration.md).

## Deploy in three steps

1. **Pick a host.** [Native macOS](#native-macos-quick-start) on Apple silicon,
   [native Linux](#native-linux-quick-start) on a desktop or home server, or
   [Docker Compose](#docker-compose-quick-start) for a reproducible Linux
   image. [Deployment summary](#deployment-summary) compares the three.
2. **Open the WebUI, enter the token, download one model.** The [fast model
   guide](#fast-model-guide) picks it for you. `GET /health/ready` answers
   `503` until that model can transcribe, and `200` once it can.
3. **Pair a phone** with the [QR in the Pair & test tab](#phone-pairing-qr),
   then decide how the phone reaches the host: [LAN, Tailscale, or an HTTPS
   reverse proxy](#listener-and-network-access).

## Contents

- [Deployment summary](#deployment-summary) — macOS vs Linux vs Docker
- [Native macOS quick start](#native-macos-quick-start)
- [Native Linux quick start](#native-linux-quick-start)
  - [Phone pairing QR](#phone-pairing-qr)
- [Docker Compose quick start](#docker-compose-quick-start)
  - [Stamping the build commit](#stamping-the-build-commit)
- [WebUI](#webui)
  - [Fast model guide](#fast-model-guide)
- [Model download integrity](#model-download-integrity)
- [Engine selection](#engine-selection)
- [Configuration](#configuration) — every `VOCAGATEWAY_*` variable and its default
- [Listener and network access](#listener-and-network-access)
  - [HTTPS reverse proxy (VPS)](#https-reverse-proxy-vps)
- [Health and readiness](#health-and-readiness)
- [Docker performance profiles](#docker-performance-profiles)
- [CLI and routine operations](#cli-and-routine-operations)
- [Development checks](#development-checks)
- [The Voca family](#the-voca-family) and [consumers](#consumers)

Longer form ([docs index](docs/)): [deployment.md](docs/deployment.md)
(operations, backup, Compose profiles) · [configuration.md](docs/configuration.md) (paths, environment
variables, pairing payload) · [tailscale.md](docs/tailscale.md) (private HTTPS)
· [troubleshooting.md](docs/troubleshooting.md) (what to check when it breaks)
· [models.md](docs/models.md) (all 58 models and 108 languages)

## Deployment summary

| Mode | Engines | Recommended use |
| --- | --- | --- |
| Native macOS | MLX Audio, WhisperKit, VocaMac, Handy, sherpa-onnx, faster-whisper, Moonshine, `whisper.cpp` | Best performance on Apple silicon |
| Native Linux | sherpa-onnx INT8, faster-whisper, Moonshine, optional `whisper.cpp` | Linux desktop or home server without Docker |
| Docker Compose | sherpa-onnx INT8, faster-whisper INT8, Moonshine, `whisper.cpp` | Reproducible Linux `amd64`/`arm64` images |

Native MLX Audio and WhisperKit are the accelerated choices on Apple silicon.
Docker Desktop runs the portable Linux image in a VM, so it cannot use the
macOS MLX/WhisperKit/Core ML paths. See [deployment.md](docs/deployment.md) for
performance notes, operational commands, and persistence.

## Native macOS quick start

Requires [Homebrew](https://brew.sh/), Python 3.12+, and
[uv](https://docs.astral.sh/uv/). Install the host dependencies first:

- `ffmpeg`: audio normalization (required by every engine)
- `whisperkit-cli`: standalone WhisperKit/Core ML engine on Apple silicon, and
  the compatibility path for VocaMac releases through 0.7.2
- `whisper-cpp`: provides `whisper-cli` for GGML `whisper.cpp` models,
  including the Handy model family, which runs without the Handy app

The [VocaMac](https://github.com/VocaHQ/vocamac) and
[Handy](https://handy.computer) desktop apps are **optional and Mac-only**.
VocaMac needs an Apple silicon Mac. Handy needs macOS. If you install neither,
the gateway downloads and runs its own models. VocaMac 0.8.0 and later expose
their selected WhisperKit, Parakeet, Apple Speech, or sherpa-onnx model through
a headless command; Handy similarly exposes its selected downloaded model. On
Linux and in containers both engines are hidden from the WebUI picker, and
selecting one through the API is rejected with `422 invalid_engine`.

```sh
brew install ffmpeg whisperkit-cli whisper-cpp
uv sync --all-groups --extra engines --extra apple
uv run vocagateway
```

The first run creates `~/.config/vocagateway/token` with mode `600`. Open
`http://127.0.0.1:8765/`, enter the token, download a recommended model, select
it, and confirm the Overview shows **Ready for dictation**.

To keep the gateway running after terminal sessions and restart it after login:

```sh
./scripts/install-launch-agent.sh
```

The LaunchAgent uses the checkout's `.venv`, adds standard Homebrew paths, and
writes logs to `~/Library/Logs/VocaGateway/`.

MLX Audio and WhisperKit are recommended on Apple silicon. The `apple` extra
installs MLX only on an arm64 Mac. It is left out of Linux and Docker on
purpose. The standalone `whisper.cpp` engine uses the `whisper-cli` binary
installed above (override its location with `VOCAGATEWAY_WHISPER_BINARY`). On a
native Linux host it is optional and can be built from source instead. When the
same build also ships `whisper-server` next to `whisper-cli`, the engine keeps
the model resident in that worker instead of reloading it for every clip; see
[Health and readiness](#health-and-readiness).

## Native Linux quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and FFmpeg on the host.

Run `just doctor` to check available host tools.

```sh
# Debian / Ubuntu
sudo apt install ffmpeg
# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --all-groups --extra engines
uv run vocagateway
```

Do not pass `--extra apple` on Linux. The first run creates
`~/.config/vocagateway/token` with mode `600`. The banner prints the WebUI URL and
token path. Show the secret (and a terminal pairing QR) with `just token` or
`uv run vocagateway-token`. Open `http://127.0.0.1:8765/`, enter the token,
download a recommended model (SenseVoice Small INT8 or Parakeet TDT INT8 on
CPU), select it, and confirm **Ready for dictation**.

To keep the gateway running after the terminal closes:

```sh
./scripts/install-systemd-user.sh
# optional: keep the user session (and unit) after logout
loginctl enable-linger "$USER"
```

```sh
systemctl --user status com.vocahq.vocagateway.service
journalctl --user -u com.vocahq.vocagateway.service -f
```

The unit uses the checkout's `.venv`. Re-run the installer after moving the
repository or recreating the virtualenv.

Phone clients on the same LAN can use `http://<host-lan-ip>:8765` while the
gateway binds `0.0.0.0` (the default). For Tailscale Serve only, bind loopback:

```sh
VOCAGATEWAY_BIND_HOST=127.0.0.1 uv run vocagateway
```

### Phone pairing QR

After you authenticate in the WebUI, the **Pair & test** tab shows a **Pair phone**
card with a QR. The iPhone and Android apps scan it to fill the gateway URL and
bearer token.
The code encodes a versioned JSON payload:

```json
{"v":1,"url":"http://192.168.1.20:8765","token":"..."}
```

Discovery prefers private Wi-Fi addresses (for example `192.168.x.x`). Override
with `VOCAGATEWAY_PUBLIC_URL` or `VOCAGATEWAY_PAIRING_URL` when automatic selection
is wrong. Discovery cannot see a public hostname, so this override is mandatory
behind a [reverse proxy](#https-reverse-proxy-vps). The same payload is available
without the WebUI: on a TTY,
`just token` (or `uv run vocagateway-token`) prints an ASCII QR for headless
setup. Use `just token --plain` when you only want the secret (pipes always get
plain output).

The same card can create a named per-device token and immediately show its own
QR instead of the shared bootstrap token. A **Token** dropdown switches which
one the QR currently encodes, along with the `/v1/admin/pairing` and
`/v1/admin/pairing/qr.svg` JSON/SVG endpoints via `?token_id=`. A device token's
plaintext is cached in memory only for the life of the gateway process, long
enough to regenerate its QR at a different address without creating a duplicate,
and is dropped immediately on revoke.

## Docker Compose quick start

[compose.yaml](compose.yaml) is the container deployment we document. It builds a
non-root Linux image containing FFmpeg, the gateway, and a pinned `whisper.cpp`
CLI. The same Dockerfile builds on Linux `amd64` and `arm64`.

```sh
umask 077
cp .env.example .env
printf 'VOCAGATEWAY_TOKEN=%s\n' "$(openssl rand -hex 32)" >> .env
docker compose up --detach --build
docker compose ps
curl --fail http://127.0.0.1:8765/health/live
```

[`.env.example`](.env.example) is the annotated template, in seven numbered
sections: the token, the published host/port, the pairing address, the image,
gateway behaviour, the container listener, and — the section that saves the
most time — the settings that look like they belong in `.env` but are never
passed to the container. It ships the loopback publication defaults
uncommented and everything else commented out, so starting from it is how you
find out what is tunable. Appending the token overrides the empty
`VOCAGATEWAY_TOKEN=` placeholder it ships with; Compose takes the last
assignment of a repeated key.

An empty `VOCAGATEWAY_TOKEN` is not a Compose error and not a startup error.
The gateway falls back to a secret it generates and never prints, so
`/health/live` looks healthy while every authenticated request returns `401`.
Fill it in before the first `up`.

The token is provided as a Compose secret rather than a container environment
variable. Models, configuration, and the SQLite database persist in the
`vocagateway_vocagateway-data` named volume mounted at `/data`.

Before pairing a phone, read the bridge-network note below. On the default
network the QR cannot auto-discover a reachable address, and
`VOCAGATEWAY_PUBLIC_URL` in `.env` is what fixes it.

The container is live before a model is installed, so `/health/ready` initially
returns `503`. Open the WebUI, enter the token from `.env`, download and select a
recommended sherpa-onnx, Moonshine, or faster-whisper model, and check again:

```sh
curl --fail http://127.0.0.1:8765/health/ready
```

The default Compose publication is host loopback only. That fits Tailscale
Serve. To allow direct LAN access, set `VOCAGATEWAY_PUBLISH_HOST=0.0.0.0` in
`.env` and protect the port with the host firewall. Never expose port 8765 to
the public internet.

The default bridge network also hides the host's real LAN address from the
gateway's own address auto-discovery (used for the pairing QR). The container
only ever sees its private bridge IP, not the host's Wi-Fi or Ethernet
interface. Two ways out, both set in `.env`:

- `VOCAGATEWAY_PUBLIC_URL=http://192.168.1.20:8765` names the address the phone
  should use and skips discovery entirely. This works everywhere, including
  Docker Desktop on macOS and Windows.
- `VOCAGATEWAY_NETWORK_MODE=host`, on Linux Docker Engine only, shares the
  host's network namespace so discovery finds the real `192.168.x.x` address by
  itself. This ignores `VOCAGATEWAY_PUBLISH_HOST`/`PORT`. The container binds
  directly on the host per `VOCAGATEWAY_BIND_HOST`/`VOCAGATEWAY_PORT`, so lock
  down port 8765 with the host firewall first.

### Stamping the build commit

A running container has no `.git` to read, so the commit it was built from is
baked in as a build argument. `just up` and `just image` do this for you. The
justfile exports `VOCAGATEWAY_GIT_COMMIT`, `VOCAGATEWAY_GIT_COMMIT_SUBJECT`, and
`VOCAGATEWAY_GIT_COMMIT_DATE` from `git`, Compose interpolates them into every
service's `build.args`, and `/v1/admin/status` then reports the revision.

Reporting is gated on `VOCAGATEWAY_DEBUG=true`, the same switch that mounts
`/docs`. A default deployment stamps the image but keeps the revision to itself:
`commit` is `null` and the WebUI drops its **Build** row. Stamp at build time
anyway. Turning debug on later then costs a restart rather than a rebuild.

Driving Compose or Docker directly works the same way once those variables are
in the environment:

```sh
export VOCAGATEWAY_GIT_COMMIT="$(git rev-parse HEAD)"
export VOCAGATEWAY_GIT_COMMIT_SUBJECT="$(git log -1 --format=%s)"
export VOCAGATEWAY_GIT_COMMIT_DATE="$(git log -1 --format=%cI)"
docker compose up --detach --build
```

Without them the build still succeeds and the gateway reports `commit: null`.
Stamping is informational, never a build requirement. Do not put them in `.env`.
Compose would pin every later build to whatever commit was current when you
wrote the file.

With [direnv](https://direnv.net), an `.envrc` keeps them current for every
command in the directory, so a plain `docker compose up --build` stamps the
running revision without the justfile in front of it:

```sh
watch_file .git/HEAD
watch_file .git/packed-refs
watch_dir .git/refs/heads

export VOCAGATEWAY_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
export VOCAGATEWAY_GIT_COMMIT_SUBJECT="$(git log -1 --format=%s 2>/dev/null || true)"
export VOCAGATEWAY_GIT_COMMIT_DATE="$(git log -1 --format=%cI 2>/dev/null || true)"
```

The three watches are what keep this from becoming the `.env` trap in a slower
form. direnv caches an evaluation until `.envrc` or something it watches
changes, so without them the exported sha would keep naming whatever commit was
current when the shell first entered the directory. Committing rewrites a loose
ref under `.git/refs/heads`, checking out a branch rewrites `.git/HEAD`, and
`git gc` folds loose refs into `.git/packed-refs`. Watching all three covers
every way HEAD moves. `.gitignore` already excludes `.envrc`, so it stays on
your machine.

The `ARG`/`ENV` pair sits after the last `COPY` in the Dockerfile, so a new
commit only invalidates that final metadata layer. Rebuilds stay cached, and the
`whisper.cpp` compile is never repeated for a commit change alone.

None of this is needed to run the gateway natively. `just run` inherits the
three variables from the justfile, which exports them for every recipe, and even
without them the gateway reads `git` directly whenever it is running from a
source checkout. Setting them by hand is only required where neither holds: an
installed wheel outside a checkout, or a container built without the build args
above. If a local `just run` reports no commit, check `VOCAGATEWAY_DEBUG` before
suspecting the variables. Reporting is gated on it, and it is off by default.

## WebUI

The authenticated WebUI provides:

- dependency, storage, model, and engine setup checks
- process uptime, active/queued work, outcomes, rejections, and stage-level latency
- detected CPU allocation/features, container state, and available accelerators
- hardware-aware model recommendations and disk-size/RAM guidance
- background downloads with scoped progress polling and cancellation
- model selection/deletion and persistent engine settings
- custom `.bin`/`.gguf` model downloads from HTTPS URLs, with an optional
  SHA-256 box that rejects the file unless it matches
- one-run or three-run microphone benchmarks with normalization, model-load,
  inference, real-time-factor, and peak-memory results
- selected engine/model and readiness/warmup status
- a redacted diagnostics export for bug reports (Settings tab or
  `uv run vocagateway-diagnostics`); never includes the token, audio, transcripts,
  or session identifiers
- named, independently revocable per-device tokens (Settings tab), so losing
  one phone means revoking that device's token instead of rotating everyone
  else's; the bootstrap `VOCAGATEWAY_TOKEN` always keeps working alongside them

Operational counters stay in process memory, contain no audio or transcript
content, and reset when the gateway process restarts.

The catalog has WhisperKit Core ML and MLX models for Apple silicon, portable
sherpa-onnx INT8 models, persistent CTranslate2 `faster-whisper` models,
Moonshine models for Arabic, English, Spanish, Japanese, Korean, Mandarin
Chinese, Ukrainian, and Vietnamese, and portable `whisper.cpp` models. It also
includes compact Whisper Medium, Whisper Large v3, and Breeze ASR builds from
[Handy's documented model family](https://handy.computer/docs/models) that run
directly through `whisper.cpp`. Handy does not need to be installed.
SenseVoice, Parakeet, GigaAM, Canary, Dolphin, and Qwen3-ASR all run through
sherpa-onnx without Handy. Parakeet, Qwen3-ASR, and Granite Speech also have
Apple-native MLX options.
GigaAM (Russian, CTC or RNNT) and Canary (English only in this build; the
underlying model also covers German, French, and Spanish, but source/target
language is fixed when the recognizer loads rather than per request) download
individual files directly from their Hugging Face model repos rather than a
packaged archive, since neither publishes one.

Qwen3-ASR reads a Hugging Face tokenizer
directory instead of a `tokens.txt`, so the gateway fetches `tokenizer/` and
passes the folder to the recognizer.

Parakeet ships in two generations, and newer is not automatically better. v3
covers 25 European languages. The English-only v2 spends all of its capacity on
English and transcribes it more accurately. Pick v2 if you dictate only in
English.

Full per-model language coverage (all 58 models, and a reverse index from each
of the 108 languages to the models that cover it) is in
[models.md](docs/models.md). The WebUI Models tab shows the same per card, with
a language filter.

### Fast model guide

| Model | Best host | Download | Languages | Choose it when |
| --- | --- | ---: | --- | --- |
| SenseVoice Small INT8 | Linux or macOS CPU | ~240 MB | Mandarin, Cantonese, English, Japanese, Korean | Lowest portable latency and small-server memory use matter most |
| Parakeet TDT 0.6B v3 INT8 | Linux or macOS CPU | ~672 MB | 25 European languages | You want stronger multilingual accuracy, punctuation, and capitalization |
| Parakeet TDT 0.6B v2 INT8 | Linux or macOS CPU | ~661 MB | English only | You dictate only in English and want the best accuracy at that speed |
| Dolphin Small CTC INT8 | Linux or macOS CPU | ~250 MB | 40 Eastern languages | You need Hindi, Bengali, Tamil, Urdu, Thai, or another South or Southeast Asian language |
| MLX Whisper Large v3 Turbo 4-bit | Apple silicon | ~469 MB | Multilingual | You want compact high accuracy through the M-series GPU |
| MLX Parakeet TDT 0.6B v3 | Apple silicon with at least 8 GB RAM | ~2.51 GB | 25 European languages | You want the full MLX Parakeet path and have enough unified memory |
| MLX Qwen3-ASR 0.6B 4-bit | Apple silicon with at least 8 GB RAM | ~713 MB | 11 languages | You want strong punctuation from an LLM decoder and can accept slower decoding |
| MLX Granite Speech 4.1 2B | Apple silicon with at least 12 GB RAM | ~2.38 GB | English only | You want top-ranked English accuracy on Apple silicon |

Every adapter keeps its loaded model in the gateway process. Benchmark three
runs in the Pair & test tab: the first includes model load, while runs two and
three show steady-state dictation speed. SenseVoice uses the FunASR Model License;
both Parakeet variants use CC BY 4.0; the quantized MLX Whisper model inherits
Whisper's MIT license; Dolphin, Qwen3-ASR, and Granite Speech are Apache 2.0.
Review the license shown on each model card before
redistributing weights.

## Model download integrity

Every catalog download is pinned and verified. Each entry in
[`app/model_pins.json`](app/model_pins.json) records the Hugging Face commit a
model is fetched from and the SHA-256 of its files. The gateway hashes each
file as it streams and discards anything that does not match, so a rejected
model is never left on disk for an engine to load.

Threats and how they are handled:

| Threat | Handled by |
| --- | --- |
| Network attacker swapping bytes in flight | TLS: every download is HTTPS with certificate verification, and non-HTTPS custom URLs are refused outright |
| Upstream repo or account compromise serving altered weights | **Pinned digests**, because the attacker is the origin and its certificate is perfectly valid |
| Silent re-upload changing a model under an existing catalog entry | **Pinned commits**, which stop downloads tracking `main` |
| Truncated or corrupted transfer | Digest verification, which also fixes the partially-downloaded-model failure mode |
| A repo listing naming a path outside the model directory | Listing paths are rejected the same way archive members already were |
| A paged listing steering the client to another host or scheme | Pagination follows `rel="next"` only within the original origin |

The pinned digest always wins over the digest Hugging Face reports at download
time. That ordering is the point: metadata fetched from a compromised host
would agree with the compromised file, so only a digest reviewed in git is
evidence of anything.

Model weights are executed by ONNX, GGUF, and Core ML runtimes. A swapped
model can execute code in those runtimes.

### Coverage

Every catalog source that publishes usable integrity metadata is pinned,
including all Hugging Face sources and the Moonshine asset manifests. The
current exceptions cannot be pinned from published metadata:

- 3 Handy-mirrored models on `blob.handy.computer` return a multipart S3 ETag,
  which is not a digest of the file content.
- 2 sherpa-onnx release tarballs on GitHub publish no checksum.

The last five can be pinned by hashing them locally, which transfers about
3.5 GB:

```sh
uv run scripts/harvest-model-pins.py --download-unpinnable
```

Unpinned models still download normally over HTTPS. They simply get no
digest check. A pinned model that fails verification fails the download. Nothing
is silently downgraded.

### Refreshing pins

The harvester is incremental by default: after adding a catalog model, it pins
only entries missing from `app/model_pins.json`. Existing records remain byte
for byte unchanged, so adding one model cannot silently refresh every model in
the catalog. Use `--refresh` only when intentionally reviewing every upstream
change; `--only` explicitly refreshes the matching model or family.

```sh
uv run scripts/harvest-model-pins.py                     # newly added models
uv run scripts/harvest-model-pins.py --only whisperkit:  # refresh one family
uv run scripts/harvest-model-pins.py --refresh           # refresh everything
```

Each revision and its digests are written as one snapshot. If the complete
snapshot cannot be collected, the command fails and preserves the previous
record rather than combining a new revision with stale hashes. Review the
resulting diff as carefully as code. A changed digest means the upstream bytes
changed, and the commit message should say why.

## Engine selection

The `auto` engine preference uses the first runnable option in this order:

1. VocaMac when the app is installed and its selected model is supported and
   downloaded
2. Handy when its macOS application binary is present
3. a downloaded WhisperKit model kept resident in a managed loopback service
4. a downloaded MLX Audio model on Apple silicon
5. a downloaded sherpa-onnx model
6. a downloaded `faster-whisper` model kept resident in the gateway process
7. a downloaded/configured `whisper.cpp` model

Steps 1 and 2 are skipped on any machine without those optional apps, which is
every Linux host and every container.

On a CPU-only Linux host, start with SenseVoice Small INT8 when its five
languages cover your use case, or Parakeet TDT INT8 for its 25 European
languages. Keep faster-whisper Base as the broad Whisper fallback. Compute
device and precision settings affect faster-whisper; sherpa models are already
INT8 CPU exports. Use the Pair & test tab's three-run benchmark after the first
warm run.

Moonshine's English Medium, Small, and Tiny Streaming tiers, and the sherpa-onnx
Streaming Zipformer English 20M INT8 model, accept float32 PCM over an
authenticated WebSocket while the iPhone records. That is real incremental
decoding with partial results, not a periodic re-transcription of the growing
buffer. Moonshine Medium favors accuracy, Small is the balanced Linux default,
and Tiny favors latency. The Zipformer model favors speed over accuracy at a
fraction of the download size. The ordinary WAV is still retained during the
request and automatically used by the batch API if streaming is unavailable or
interrupted. Streaming support is negotiated on that socket to avoid an extra
network round trip before every recording.

Every other model (the remaining Moonshine tiers, WhisperKit, faster-whisper,
and every other sherpa-onnx model above) uses its fast batch path after
recording. The server returns a structured unsupported response for those, so
the app immediately continues through the ordinary upload pipeline. In the
iPhone app or keyboard, **Automatic** uses the active gateway model. Choosing a
named language requires the active model to support that same language.

Moonshine's English code and weights use the MIT license. Its non-English weights
use the Moonshine Community License and are limited to non-commercial use. The
WebUI labels these models **personal use**. Review the current
[Moonshine licensing and model documentation](https://github.com/moonshine-ai/moonshine)
before deploying them outside a personal setup.

The WebUI can explicitly select an engine or installed model and persists that
choice in the runtime configuration file. `VOCAGATEWAY_ENGINE` outranks it:
anything other than `auto` pins the engine for the whole process, and the
WebUI's saved engine choice stops taking effect. Leave the variable at `auto`
unless you want that lock.

Four engines need a specific host, and the WebUI names the requirement next to
each one:

| Engine | Runs on |
| --- | --- |
| `vocamac` | Apple silicon Macs (the VocaMac app is Apple-silicon-only) |
| `mlx-audio` | Apple silicon Macs |
| `handy` | macOS |
| `whisperkit` | macOS |

The engine picker lists them only on a host that can run them, and both the
WebUI and `PUT /v1/admin/config` reject a selection the host cannot run with
`422 invalid_engine` rather than persisting a broken choice. `auto` skips them
on every other host.

That host check covers the WebUI and the API, not `VOCAGATEWAY_ENGINE`. Setting
`VOCAGATEWAY_ENGINE=vocamac` on Linux or in a container is accepted at startup;
the engine simply reports unavailable and `/health/ready` stays `503`. Check the
variable before hunting for a model problem.

On Apple silicon, current WhisperKit CLIs expose a local `serve` mode. The
gateway starts it on a random `127.0.0.1` port during warmup and reuses the
loaded Core ML model. If an older CLI does not support `serve`, transcription
falls back to the compatible one-shot command rather than becoming unavailable.

VocaMac 0.8.0 and later expose one-shot file transcription through the app
binary (`--transcribe-file`, shipped in
[vocamac#200](https://github.com/VocaHQ/vocamac/pull/200)). The adapter asks
that interface which model is selected and then invokes the same internal
router VocaMac uses for WhisperKit, FluidAudio Parakeet, Apple Speech, and
specialized sherpa-onnx models. Changing the selection in VocaMac affects the
next phone transcription without restarting the gateway, and the command does
not open, close, or disturb the VocaMac GUI.

Releases through 0.7.2 have no headless interface and keep the compatibility
path: the adapter reuses complete WhisperKit Core ML folders and tokenizers
through `whisperkit-cli`. That legacy path cannot run VocaMac's other embedded
engines. VocaMac does not need to be running in either mode.

To force VocaMac from the environment:

```sh
export VOCAGATEWAY_ENGINE=vocamac
export VOCAGATEWAY_VOCAMAC_MODEL='small'   # optional; otherwise VocaMac's own choice
uv run vocagateway
```

`VOCAGATEWAY_VOCAMAC_MODEL` accepts any VocaMac model ID, including `small`,
`parakeet-tdt-0.6b-v3`, `apple-speech`, or `canary-180m-flash`. WhisperKit folder
names such as `openai_whisper-small` remain accepted for compatibility. A
configured model is never substituted: if it is not downloaded or supported,
the engine reports unavailable rather than quietly using another.

To force Handy from the environment:

```sh
export VOCAGATEWAY_ENGINE=handy
export VOCAGATEWAY_HANDY_MODEL='owner/repository/model.gguf'
export VOCAGATEWAY_HANDY_FALLBACK_MODEL='owner/repository/fallback-model.gguf'
uv run vocagateway
```

To force standalone `whisper.cpp`:

```sh
export VOCAGATEWAY_ENGINE=whisper.cpp
export VOCAGATEWAY_WHISPER_BINARY=/absolute/path/to/whisper-cli
export VOCAGATEWAY_WHISPER_MODEL=/absolute/path/to/ggml-model.bin
uv run vocagateway
```

## Configuration

| Variable | Native default | Container default | Purpose |
| --- | --- | --- | --- |
| `VOCAGATEWAY_BIND_HOST` | `0.0.0.0` | `0.0.0.0` inside container | Gateway listener |
| `VOCAGATEWAY_PORT` | `8765` | `8765` | Gateway listener port |
| `VOCAGATEWAY_TOKEN` | unset | unset | Direct token override; at least 32 characters |
| `VOCAGATEWAY_TOKEN_FILE` | `~/.config/vocagateway/token` | `/run/secrets/vocagateway_token` | Bearer-token file |
| `VOCAGATEWAY_DATA_DIR` | `~/.local/share/vocagateway` | `/data` | Sessions and application data |
| `VOCAGATEWAY_MODELS_DIR` | `~/.local/share/vocagateway/models` | `/data/models` | Downloaded models |
| `VOCAGATEWAY_CONFIG_FILE` | `~/.config/vocagateway/config.json` | `/data/config/config.json` | WebUI engine/model choice |
| `VOCAGATEWAY_ENGINE` | `auto` | `auto` | `auto`, `vocamac`, `handy`, `mlx-audio`, `whisperkit`, `sherpa-onnx`, `faster-whisper`, `moonshine`, or `whisper.cpp` |
| `VOCAGATEWAY_WHISPER_BINARY` | `/opt/homebrew/bin/whisper-cli` | `/usr/local/bin/whisper-cli` | `whisper.cpp` executable |
| `VOCAGATEWAY_WHISPER_MODEL` | `~/.local/share/whisper.cpp/models/ggml-base.en.bin` | same, and normally absent | Fallback `whisper.cpp` model used only when no model is selected in the WebUI |
| `VOCAGATEWAY_WHISPER_SERVER_BINARY` | the `whisper-server` beside `whisper-cli`, else `PATH` | `/usr/local/bin/whisper-server` | Resident `whisper.cpp` worker; unset is normal, and a missing binary falls back to one `whisper-cli` run per request |
| `VOCAGATEWAY_WHISPER_DECODER_PRESET` | `quality` | `quality` | `quality` keeps the narrowed beam search; `fast` decodes greedily — cheaper on a CPU-only host, and worth a WER comparison on your own audio before you keep it |
| `VOCAGATEWAY_WHISPERKIT_BINARY` | `whisperkit-cli` | unavailable | Standalone WhisperKit executable and legacy VocaMac fallback |
| `VOCAGATEWAY_VOCAMAC_APP` | `/Applications/VocaMac.app` | unavailable | Optional VocaMac app bundle |
| `VOCAGATEWAY_VOCAMAC_MODEL` | unset | unset | Pin a VocaMac model instead of following the app's choice |
| `VOCAGATEWAY_HANDY_BINARY` | `/Applications/Handy.app/Contents/MacOS/handy` | unavailable | Optional Handy application binary |
| `VOCAGATEWAY_HANDY_MODEL` | unset | unset | Pin a Handy model (`owner/repository/model.gguf`) |
| `VOCAGATEWAY_HANDY_FALLBACK_MODEL` | `handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf` | unavailable | Model used when the pinned Handy model is missing |
| `VOCAGATEWAY_RETENTION_HOURS` | `24` | `24` | Failed-session retry retention |
| `VOCAGATEWAY_DELETE_SUCCESSFUL_AUDIO` | `true` | `true` | Delete source/normalized audio after success |
| `VOCAGATEWAY_PUBLIC_URL` | unset | unset | Address the pairing QR encodes, overriding auto-discovery |
| `VOCAGATEWAY_PAIRING_URL` | unset | unset | Alias for `VOCAGATEWAY_PUBLIC_URL`, checked second |
| `VOCAGATEWAY_DEBUG` | `false` | `false` | Serve the Swagger UI at `/docs` and the schema at `/openapi.json`, and report the build commit in `/v1/admin/status` |

Under Compose, `VOCAGATEWAY_BIND_HOST`, `PORT`, `ENGINE`, `RETENTION_HOURS`,
`DELETE_SUCCESSFUL_AUDIO`, `PUBLIC_URL`, `PAIRING_URL`, and `DEBUG` are read
from `.env` and passed into the container. `VOCAGATEWAY_TOKEN` becomes a
Compose secret at `/run/secrets/vocagateway_token` rather than an environment
variable. Every other variable in the table above is fixed by the image or
simply absent from a Linux container, and — this is the part that bites —
`compose.yaml` does not forward it, so writing `VOCAGATEWAY_DATA_DIR`,
`VOCAGATEWAY_MODELS_DIR`, `VOCAGATEWAY_CONFIG_FILE`, or any macOS engine path
into `.env` passes `docker compose config` and changes nothing. Relocate
container data by remapping the `vocagateway-data` volume instead.

Compose-only variables, which the gateway process itself never reads, also live
in `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCAGATEWAY_PUBLISH_HOST` | `127.0.0.1` | Host interface published by Docker |
| `VOCAGATEWAY_PUBLISH_PORT` | `8765` | Host port published by Docker |
| `VOCAGATEWAY_NETWORK_MODE` | `bridge` | Set to `host` on Linux Docker Engine to share the host's network namespace (ignores `VOCAGATEWAY_PUBLISH_HOST`/`PORT`); not supported by Docker Desktop |
| `VOCAGATEWAY_IMAGE` | `vocagateway:local` | Tag for the default CPU `gateway` service. It does not switch Compose from building to pulling — `up --build` still builds locally and applies the tag. To run a prebuilt image: `docker compose pull` then `up --no-build`. The `gateway-cuda` and `gateway-vulkan` services have fixed tags and ignore it |
| `VOCAGATEWAY_WHISPER_CMAKE_EXTRA` | unset | Extra CMake flags appended to the `whisper.cpp` build; use this for host-specific CUDA architectures or advanced CPU tuning |
| `VOCAGATEWAY_BUILD_JOBS` | builder CPU count | Maximum concurrent `whisper.cpp` compile jobs; lower it when a memory-constrained build is killed, especially for CUDA |
| `VOCAGATEWAY_RENDER_GID` | `993` | Host render-group GID added to the Vulkan container; set it from `/dev/dri/renderD128` |
| `VOCAGATEWAY_VIDEO_GID` | `44` | Host video-group GID added to the Vulkan container when required by the distribution |

Use [`.env.example`](.env.example) as a template and never commit the populated
`.env` file.

For the pairing QR payload (`{"v":1,"url":"...","token":"..."}`), native paths,
and the full `VOCAGATEWAY_*` contract (including the note that older
`VOCAPHONE_*` / `~/.config/vocaphone/` names are stale and unread), see
[configuration.md](docs/configuration.md).

## Listener and network access

The native default listener is `0.0.0.0:8765`. The startup banner and WebUI show
that listener separately from the local browser URL. An all-interface listener
is reachable from connected networks, so keep the host firewall enabled.

The iPhone and Android apps accept ordinary HTTP and HTTPS gateway URLs. A
Tailscale hostname is not mandatory. Supported arrangements include:

- a trusted LAN hostname such as `http://homelabone:8765/`; for Docker, set
  `VOCAGATEWAY_PUBLISH_HOST=0.0.0.0` and protect the port with the host firewall
- a loopback listener exposed privately through Tailscale Serve
- a VPS loopback listener behind an [HTTPS reverse proxy](#https-reverse-proxy-vps)
  and trusted certificate

HTTP does not encrypt the bearer token or recording. Use it only on a trusted
LAN or encrypted VPN, never over the public internet.

For the smallest private exposure, bind or publish on host loopback and use
Tailscale Serve:

```sh
tailscale serve --bg 8765
tailscale serve status
```

Use the reported private HTTPS URL in the iPhone or Android app. Do not use
Funnel. See [deployment.md](docs/deployment.md) for LAN/VPS alternatives and
[tailscale.md](docs/tailscale.md) for the private Serve setup.

### HTTPS reverse proxy (VPS)

Publish the gateway on host loopback, terminate TLS in nginx or Caddy, and keep
port 8765 closed at the firewall. Bearer authentication stays on even when the
proxy has its own access control.

Two things surprise most first deployments. The WebUI works over 443 while the
phone app cannot connect at all. Once pairing is fixed, recordings fail on
upload while short ones succeed.

The gateway does not know its own hostname. Pairing discovery inspects local
interfaces and builds candidates like `http://<vps-ip>:8765`, which is what the
QR encodes and what the phone then tries to reach, bypassing nginx entirely.
Set `VOCAGATEWAY_PUBLIC_URL` to the public address:

```dotenv
VOCAGATEWAY_PUBLIC_URL=https://vocagateway.example.com
```

Omit the port; 443 is implied. The override is placed ahead of discovery in the
pairing card and the QR.

The override only applies when the WebUI has no saved choice, and a saved public
address is never pruned as stale (only ambient LAN and tailnet IPs are). If the
pairing card was ever opened before the override was set, clear the old entry
with **Remove** under *Saved addresses* on the card, or remove `pairing_url` and
`pairing_urls` from `config.json` and restart.

The gateway accepts 25 MiB uploads. nginx defaults to 1 MiB and rejects real
recordings with a `413` before they reach the application, so raise
`client_max_body_size`. `/v1/stream` is a WebSocket and needs the upgrade
headers, and CPU transcription routinely outlives the 60-second proxy default:

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 443 ssl;
    http2 on;
    server_name vocagateway.example.com;

    ssl_certificate     /etc/letsencrypt/live/vocagateway.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vocagateway.example.com/privkey.pem;
    include             /etc/letsencrypt/options-ssl-nginx.conf;

    # 25 MiB upload ceiling plus headroom, so the gateway returns its own JSON
    # error envelope instead of an nginx HTML page.
    client_max_body_size 26m;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # /v1/stream
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        # transcription and model downloads
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;

        proxy_request_buffering off;
    }
}
```

Serve the gateway at a domain or subdomain root rather than a subpath. The WebUI
references `/assets/...` absolutely and sends `connect-src 'self'`, so a subpath
mount breaks the interface even though a path component is accepted in the
gateway URL itself.

Verify the proxy and the encoded address:

```sh
curl -s https://vocagateway.example.com/health/live
curl -s -H "Authorization: Bearer $TOKEN" \
  https://vocagateway.example.com/v1/admin/pairing
```

The pairing response should report the public hostname, not an interface address
with `:8765`. Re-scan the QR afterwards. A phone paired earlier still holds the
old URL.

## Health and readiness

- `GET /health/live` reports HTTP-process liveness and uptime without probing the
  selected engine.
- `GET /health/ready` returns `200` only when the engine/model can transcribe and
  returns `503` otherwise.
- `GET /health` is the backward-compatible iPhone health response and includes
  `streaming_supported` for status display and capability discovery. The iOS
  recording path negotiates on the socket itself to avoid a separate preflight.
- Authenticated `/v1/admin/status` exposes setup, metrics, and readiness details
  used by the WebUI.
- `/v1/admin/status` and the diagnostics bundle carry a `commit` object (`sha`,
  `short_sha`, `subject`, `committed_at`) naming the source revision the gateway
  runs, and the WebUI Overview shows it as `build <short sha>` plus a **Build**
  row under *Hardware details*. This requires `VOCAGATEWAY_DEBUG=true`; without
  it `commit` is `null` and the WebUI row is omitted. A source checkout reads
  the revision from git; containers need it stamped at build time. See
  [Stamping the build commit](#stamping-the-build-commit).

Engine probes are cached for five seconds. sherpa-onnx, MLX Audio,
`faster-whisper`, and Moonshine load their selected model once and keep it
resident. WhisperKit warmup starts its managed loopback service and keeps the
Core ML model resident there. `whisper.cpp` does the same when the build ships
`whisper-server`: warmup prefetches the model file and then starts a private
worker on an ephemeral `127.0.0.1` port, which holds the parsed model and the
CUDA/Vulkan/Metal context between requests. Nothing extra is published and no
host port is added. A build without that binary, or a worker that fails to
start, falls back to one `whisper-cli` run per transcription — the behavior
every earlier release had. VocaMac 0.8.0+ headless transcription is one-shot,
so its first-load cost is included in each request; older WhisperKit-only
VocaMac builds retain the persistent compatibility path. Handy retains the
filesystem-prefetch warmup behavior.

Choosing **Load** on a downloaded model waits for that warmup before the Models
view reports it active, so resident engines do not defer their model load to the
first transcription. In **Settings → Speech engine**, **Offload model when idle**
can release a resident model after 10, 15, 30, 60, or 120 minutes without a
transcription. It is off by default. The selected model stays selected and loads
again automatically on the next transcription. Saving a change to this setting
alone keeps whatever is already loaded — only a change that actually selects a
different engine, device, precision, thread count, or model rebuilds it. Active
batch and streaming jobs hold a model lease, so neither the idle monitor nor a
settings change unloads an engine while a transcription is still using it: the
replaced engine is closed once its last request finishes. The
setting also releases the `whisper.cpp` worker: the process is terminated and
the next transcription starts it again, paying one model load. It does not apply
to one-shot Handy or current VocaMac headless processes, because those engines
keep no model resident between requests.

## Docker performance profiles

Only run one gateway service at a time. Every profile publishes the same port
and shares the same model volume.

```sh
# Portable CPU (default; amd64 and arm64)
docker compose up --detach --build gateway

# NVIDIA host with Container Toolkit
docker compose --profile cuda up --detach --build gateway-cuda

# Intel/AMD Vulkan device exposed as /dev/dri
docker compose --profile vulkan up --detach --build gateway-vulkan
```

All three come from one `Dockerfile`, selected by the `ACCEL` build argument.
The CPU image needs no host-specific build: `GGML_CPU_ALL_VARIANTS` compiles a
ggml CPU backend per micro-architecture and the best one the host reports is
loaded at startup, so a portable image still runs AVX2/AVX-512 code on x86 and
dotprod/i8mm code on arm64.

An Apple silicon Docker Desktop build validates only the Linux arm64 CPU path;
it cannot validate Linux amd64 or an NVIDIA CUDA image. The Container GitHub
Actions workflow builds CPU, CUDA, and Vulkan separately and smoke-tests the CPU
image. Its compile-only CUDA check targets one representative GPU architecture
instead of producing the Dockerfile's portable architecture spread. Treat that
matrix as the cross-platform build result, not as a release image. If a local
build is killed for memory, set `VOCAGATEWAY_BUILD_JOBS` in `.env`; see
[Tuning the whisper.cpp build](docs/deployment.md#tuning-the-whispercpp-build).

The CUDA profile supports both faster-whisper CUDA and the CUDA `whisper.cpp`
binary. The Vulkan profile accelerates `whisper.cpp`; faster-whisper remains on
CPU there, and the container needs the host's render GID — see
[Vulkan GPU access](docs/deployment.md#giving-the-vulkan-container-access-to-the-gpu).
The dashboard reports what devices the container can actually see.

## CLI and routine operations

Primary console scripts are `vocagateway`, `vocagateway-token`,
`vocagateway-status`, `vocagateway-diagnostics`, and `vocagateway-cleanup`.
Deprecated aliases (`vocaphone-server`, `vocaphone-token`, `vocaphone-status`,
`vocaphone-diagnostics`, `vocaphone-cleanup`) call the same entry points for one
cycle.

```sh
# Query the local backward-compatible health response
uv run vocagateway-status

# Download a redacted diagnostics bundle for a bug report
uv run vocagateway-diagnostics

# Remove sessions older than the configured retention window
uv run vocagateway-cleanup

# Follow the native macOS LaunchAgent logs
tail -f ~/Library/Logs/VocaGateway/gateway.log

# Follow the native Linux systemd user unit logs
journalctl --user -u com.vocahq.vocagateway.service -f

# Follow container logs
docker compose logs --follow gateway

# Recreate a container from the current checkout
docker compose up --detach --build

# Stop containers but retain the named data volume
docker compose down
```

Do not run `docker compose down --volumes` unless deleting every downloaded
model, configuration file, and stored session is intentional.

## Development checks

```sh
# On macOS add --extra apple when you need MLX / WhisperKit in the dev environment.
uv sync --all-groups --extra engines
# Required for every change under app/; the WPS GitHub Actions workflow runs this too.
uv run flake8 --select=WPS,E999 app
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
VOCAGATEWAY_TOKEN=test-token-with-at-least-thirty-two-characters docker compose config --quiet
docker build --tag vocagateway:test .
```

Build and publish one tag for both supported Linux architectures from the
repository root:

```sh
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/your-user/vocagateway:latest \
  --push .
```

For backup, update, and native-vs-container guidance, continue with
[deployment.md](docs/deployment.md). For pairing, paths, and env vars, see
[configuration.md](docs/configuration.md). For failures, see
[troubleshooting.md](docs/troubleshooting.md). The [docs index](docs/) lists
all five pages.

## The Voca family

Directory: [vocahq.com](https://vocahq.com). [VocaPhone](https://vocaphone.vocahq.com)
is the live consumer. Embedding this gateway in a desktop app stays Planned.

| Product | Status | Website | Source |
| --- | --- | --- | --- |
| [VocaLinux](https://vocalinux.com/) | Available now (`v0.16.0`) | [vocalinux.com](https://vocalinux.com/) | [VocaHQ/vocalinux](https://github.com/VocaHQ/vocalinux) |
| [VocaMac](https://vocamac.com/) | Beta (`v0.9.0`) | [vocamac.com](https://vocamac.com/) | [VocaHQ/vocamac](https://github.com/VocaHQ/vocamac) |
| [VocaWin](https://vocawin.com/) | Unsigned beta (`v0.1.0-beta.1`) | [vocawin.com](https://vocawin.com/) | [VocaHQ/vocawin](https://github.com/VocaHQ/vocawin) |
| [VocaPhone](https://vocaphone.vocahq.com) | Android beta / iOS [TestFlight](https://testflight.apple.com/join/wd85wQ3W) (live consumer) | [vocaphone.vocahq.com](https://vocaphone.vocahq.com) | [VocaHQ/vocaphone](https://github.com/VocaHQ/vocaphone) |
| [VocaGateway](https://vocagateway.vocahq.com/) | Beta | [vocagateway.vocahq.com](https://vocagateway.vocahq.com/) | [VocaHQ/vocagateway](https://github.com/VocaHQ/vocagateway) |

## Consumers

| Project | How it uses this gateway |
| --- | --- |
| [vocaphone](https://github.com/VocaHQ/vocaphone) | Live consumer. Git submodule at `gateway/` for the iOS/Android clients |
| [vocalinux](https://github.com/VocaHQ/vocalinux) | `remote_api` can POST audio to `/v1/audio/transcriptions` on this host. Embedding the gateway in the app is still Planned. |
| [vocamac](https://github.com/VocaHQ/vocamac) / [vocawin](https://github.com/VocaHQ/vocawin) | Planned: ship and start the headless server from the desktop app |

### VocaLinux `remote_api`

A running VocaLinux can treat this host as an OpenAI transcription server. Set
the engine to `remote_api`. Server URL is the gateway origin, for example
`http://192.168.1.20:8765`. API Endpoint must be OpenAI
`/v1/audio/transcriptions`, not VocaLinux's default `/inference`. API Key is the
gateway bearer token. The Model field is ignored; the engine you loaded in the
WebUI is what runs.

```sh
curl -H "Authorization: Bearer $TOKEN" -F file=@sample.wav -F model=whisper-1 \
  http://127.0.0.1:8765/v1/audio/transcriptions
```

VocaLinux's Test Connection is `GET /` on that origin, which is the
unauthenticated WebUI, so a bad key can still look green. The first dictation is
the real check. The client times out after 30 seconds, and a cold model load can
miss that. Default concurrency is one in-flight transcription; a busy gateway
returns 503. The gateway speaks HTTP on the LAN by default. HTTPS needs a
certificate the desktop OS trusts.

This is still optional self-hosted compute. Audio leaves the desktop and travels
to the gateway host. It is not on-device transcription, and this endpoint does
not stream.

Clone with submodules when working from a consumer:

```sh
git clone --recurse-submodules https://github.com/VocaHQ/vocaphone.git
# or later: git submodule update --init --recursive
```

## License and contact

[AGPL-3.0](LICENSE). Questions and contributions:
[Discord](https://discord.gg/t6muquAJbm),
[@vocahq](https://x.com/vocahq) on X, or
[hello@vocahq.com](mailto:hello@vocahq.com).

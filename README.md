# VocaGateway

[![Discord](https://img.shields.io/discord/1538633755877580810?logo=discord&logoColor=white&label=Discord)](https://discord.gg/UMJduhcqn)
[![VocaHQ](https://img.shields.io/badge/VocaHQ-vocahq.com-1a7f4e)](https://vocahq.com)

**Early** optional self-hosted transcription gateway for the
[Voca](https://github.com/VocaHQ) family. License:
[AGPL-3.0](LICENSE). Contact:
[hello@vocahq.com](mailto:hello@vocahq.com).

The public landing page lives in [`web/`](web/) and deploys to
[vocagateway.vocahq.com](https://vocagateway.vocahq.com/).

Set it up once on hardware you control; pair phone clients (and, later, desktop
clients) to that host. Self-host on macOS or Linux, or use Docker Compose on
Linux `amd64`/`arm64`. There is no Voca account and no hosted Voca cloud.

The gateway accepts bounded recordings from
[vocaphone](https://github.com/VocaHQ/vocaphone) (iOS/Android) and, soon, the
Linux/macOS/Windows desktop apps. It normalizes audio with FFmpeg, invokes a
local speech engine, and returns an idempotent transcript. An authenticated
HTMX WebUI covers setup, model management, engine selection, microphone
testing, and operational status.

Gateway mode is **not** on-device processing: audio leaves the client and travels
to the machine you chose. Prefer a trusted LAN, Tailscale, or HTTPS. Never
expose port `8765` to the public internet.

CLI entry points still use the historical `vocaphone-server` script names from
the Python package (`vocaphone-gateway`). Environment variables and on-disk
paths use the `vocagateway` prefix (`VOCAGATEWAY_*`, `~/.config/vocagateway/`,
`~/.local/share/vocagateway/`). The live pairing and env contract is documented
in [configuration.md](docs/configuration.md).

## Consumers

| Project | How it uses this gateway |
| --- | --- |
| [vocaphone](https://github.com/VocaHQ/vocaphone) | Git submodule at `server/` for the iOS/Android clients |
| [vocalinux](https://github.com/VocaHQ/vocalinux) / [vocamac](https://github.com/VocaHQ/vocamac) / [vocawin](https://github.com/VocaHQ/vocawin) | Planned: ship and start the headless server from the desktop app |

Clone with submodules when working from a consumer:

```sh
git clone --recurse-submodules https://github.com/VocaHQ/vocaphone.git
# or later: git submodule update --init --recursive
```


## Deployment summary

| Mode | Engines | Recommended use |
| --- | --- | --- |
| Native macOS | MLX Audio, WhisperKit, VocaMac, Handy, sherpa-onnx, faster-whisper, Moonshine, `whisper.cpp` | Best performance on Apple silicon |
| Native Linux | sherpa-onnx INT8, faster-whisper, Moonshine, optional `whisper.cpp` | Linux desktop or home server without Docker |
| Docker Compose | sherpa-onnx INT8, faster-whisper INT8, Moonshine, `whisper.cpp` | Reproducible Linux `amd64`/`arm64` images |

Native MLX Audio and WhisperKit are the accelerated choices on Apple silicon.
Docker Desktop runs the portable Linux image in a VM, so it cannot use the
macOS MLX/WhisperKit/Core ML paths. See [deployment.md](docs/deployment.md) for the
performance explanation, operational commands, and persistence details.

## Native macOS quick start

Requires [Homebrew](https://brew.sh/), Python 3.12+, and
[uv](https://docs.astral.sh/uv/). Install the host dependencies first:

- `ffmpeg` — audio normalization (required by every engine)
- `whisperkit-cli` — WhisperKit/Core ML engine on Apple silicon, and the engine
  behind the VocaMac adapter
- `whisper-cpp` — provides `whisper-cli` for GGML `whisper.cpp` models,
  including the Handy model family, which runs without the Handy app

The [VocaMac](https://github.com/VocaHQ/vocamac) and
[Handy](https://handy.computer) desktop apps are **optional and Mac-only**:
VocaMac needs an Apple silicon Mac, Handy needs macOS. Install neither and the
gateway downloads and runs its own models; install either and the gateway can
reuse the models that app already downloaded instead of asking for a second
copy. On Linux and in containers both engines are hidden from the WebUI picker,
and selecting one through the API is rejected with `422 invalid_engine`.

```sh
brew install ffmpeg whisperkit-cli whisper-cpp
uv sync --all-groups --extra engines --extra apple
uv run vocaphone-server
```

The first run creates `~/.config/vocagateway/token` with mode `600`. Open
`http://127.0.0.1:8765/`, enter the token, download a recommended model, select
it, and confirm the Overview shows **Ready for dictation**.

To keep the gateway running after terminal sessions and restart it after login:

```sh
./scripts/install-launch-agent.sh
```

The LaunchAgent uses the checkout's `.venv`, adds standard Homebrew paths, and
writes logs to `~/Library/Logs/Vocaphone/`.

MLX Audio and WhisperKit are recommended on Apple silicon. The `apple` extra
installs MLX only on an arm64 Mac; it is deliberately absent from Linux and
Docker. The standalone `whisper.cpp` engine uses the `whisper-cli` binary
installed above (override its location with `VOCAGATEWAY_WHISPER_BINARY`); on a
native Linux host it is optional and can be built from source instead.

## Native Linux quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and FFmpeg on the host.

```sh
# Debian / Ubuntu
sudo apt install ffmpeg
# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --all-groups --extra engines
uv run vocaphone-server
```

Do not pass `--extra apple` on Linux. The first run creates
`~/.config/vocagateway/token` with mode `600`. The banner prints the WebUI URL and
token path; show the secret (and a terminal pairing QR) with `just token` or
`uv run vocaphone-token`. Open `http://127.0.0.1:8765/`, enter the token,
download a recommended model (SenseVoice Small INT8 or Parakeet TDT INT8 on
CPU), select it, and confirm **Ready for dictation**.

To keep the gateway running after the terminal closes:

```sh
./scripts/install-systemd-user.sh
# optional: keep the user session (and unit) after logout
loginctl enable-linger "$USER"
```

```sh
systemctl --user status com.vocahq.vocaphone.gateway.service
journalctl --user -u com.vocahq.vocaphone.gateway.service -f
```

The unit uses the checkout's `.venv`. Re-run the installer after moving the
repository or recreating the virtualenv.

Phone clients on the same LAN can use `http://<host-lan-ip>:8765` while the
gateway binds `0.0.0.0` (the default). For Tailscale Serve only, bind loopback:

```sh
VOCAGATEWAY_BIND_HOST=127.0.0.1 uv run vocaphone-server
```

### Phone pairing QR

After you authenticate in the WebUI, the **Pair & test** tab shows a **Pair phone**
card with a QR. The iPhone and Android apps scan it to fill the gateway URL and
bearer token.
The code encodes a versioned JSON payload:

```json
{"v":1,"url":"http://192.168.1.20:8765","token":"..."}
```

Discovery prefers private Wi‑Fi addresses (for example `192.168.x.x`). Override
with `VOCAGATEWAY_PUBLIC_URL` or `VOCAGATEWAY_PAIRING_URL` when automatic selection
is wrong — discovery cannot see a public hostname, so this override is mandatory
behind a [reverse proxy](#https-reverse-proxy-vps). The same payload is available
without the WebUI: on a TTY,
`just token` (or `uv run vocaphone-token`) prints an ASCII QR for headless
setup; use `just token --plain` when you only want the secret (pipes always get
plain output).

The same card can create a named per-device token and immediately show its own
QR instead of the shared bootstrap token, and a **Token** dropdown
switches which one the QR (and the `/v1/admin/pairing` and
`/v1/admin/pairing/qr.svg` JSON/SVG endpoints, via `?token_id=`) currently
encodes. A device token's plaintext is cached in memory only for the life of
the gateway process — long enough to regenerate its QR at a different address
without creating a duplicate — and is dropped immediately on revoke.

## Docker Compose quick start

[compose.yaml](compose.yaml) is the canonical container deployment. It builds a
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

[`.env.example`](.env.example) is the annotated template: it already sets the
loopback publication defaults and comments out every other supported setting,
so starting from it is how you find out what is tunable. Appending the token
overrides the empty `VOCAGATEWAY_TOKEN=` placeholder it ships with — Compose
takes the last assignment of a repeated key.

The token is provided as a Compose secret rather than a container environment
variable. Models, configuration, and the SQLite database persist in the
`vocagateway_vocagateway-data` named volume mounted at `/data`.

Before pairing a phone, read the bridge-network note below: on the default
network the QR cannot auto-discover a reachable address, and
`VOCAGATEWAY_PUBLIC_URL` in `.env` is what fixes it.

The container is live before a model is installed, so `/health/ready` initially
returns `503`. Open the WebUI, enter the token from `.env`, download/select a
recommended sherpa-onnx, Moonshine, or faster-whisper model, and check again:

```sh
curl --fail http://127.0.0.1:8765/health/ready
```

The default Compose publication is host loopback only. This is appropriate for
Tailscale Serve. To intentionally allow direct LAN access, set
`VOCAGATEWAY_PUBLISH_HOST=0.0.0.0` in `.env` and protect the port with the host
firewall. Never expose port 8765 to the public internet.

The default bridge network also hides the host's real LAN address from the
gateway's own address auto-discovery (used for the pairing QR): the container
only ever sees its private bridge IP, not the host's Wi-Fi/Ethernet interface.
Two ways out, both set in `.env`:

- `VOCAGATEWAY_PUBLIC_URL=http://192.168.1.20:8765` names the address the phone
  should use and skips discovery entirely. This works everywhere, including
  Docker Desktop on macOS and Windows.
- `VOCAGATEWAY_NETWORK_MODE=host`, on Linux Docker Engine only, shares the
  host's network namespace so discovery finds the real `192.168.x.x` address by
  itself. This ignores `VOCAGATEWAY_PUBLISH_HOST`/`PORT` — the container binds
  directly on the host per `VOCAGATEWAY_BIND_HOST`/`VOCAGATEWAY_PORT`, so lock
  down port 8765 with the host firewall first.

### Stamping the build commit

A running container has no `.git` to read, so the commit it was built from is
baked in as a build argument. `just up` and `just image` do this for you: the
justfile exports `VOCAGATEWAY_GIT_COMMIT`, `VOCAGATEWAY_GIT_COMMIT_SUBJECT`, and
`VOCAGATEWAY_GIT_COMMIT_DATE` from `git`, Compose interpolates them into every
service's `build.args`, and `/v1/admin/status` then reports the revision.

Reporting is gated on `VOCAGATEWAY_DEBUG=true`, the same switch that mounts
`/docs`. A default deployment stamps the image but keeps the revision to itself:
`commit` is `null` and the WebUI drops its **Build** row. Stamp at build time
regardless — turning debug on later then costs a restart rather than a rebuild.

Driving Compose or Docker directly works the same way once those variables are
in the environment:

```sh
export VOCAGATEWAY_GIT_COMMIT="$(git rev-parse HEAD)"
export VOCAGATEWAY_GIT_COMMIT_SUBJECT="$(git log -1 --format=%s)"
export VOCAGATEWAY_GIT_COMMIT_DATE="$(git log -1 --format=%cI)"
docker compose up --detach --build
```

Without them the build still succeeds and the gateway reports `commit: null` —
stamping is informational, never a build requirement. Do not put them in `.env`:
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
`git gc` folds loose refs into `.git/packed-refs` — watching all three covers
every way HEAD moves. `.gitignore` already excludes `.envrc`, so it stays on
your machine.

The `ARG`/`ENV` pair sits after the last `COPY` in each Dockerfile, so a new
commit only invalidates that final metadata layer. Rebuilds stay cached, and the
`whisper.cpp` compile is never repeated for a commit change alone.

None of this is needed to run the gateway natively. `just run` inherits the
three variables from the justfile, which exports them for every recipe, and even
without them the gateway reads `git` directly whenever it is running from a
source checkout. Setting them by hand is only required where neither holds — an
installed wheel outside a checkout, or a container built without the build args
above. If a local `just run` reports no commit, check `VOCAGATEWAY_DEBUG` before
suspecting the variables: reporting is gated on it, and it is off by default.

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
  `uv run vocaphone-diagnostics`); never includes the token, audio, transcripts,
  or session identifiers
- named, independently revocable per-device tokens (Settings tab), so losing
  one phone means revoking that device's token instead of rotating everyone
  else's; the bootstrap `VOCAGATEWAY_TOKEN` always keeps working alongside them

Operational counters stay in process memory, contain no audio or transcript
content, and reset when the gateway process restarts.

The catalog contains WhisperKit Core ML and MLX models for Apple silicon,
portable sherpa-onnx INT8 models, persistent CTranslate2 `faster-whisper`
models, Moonshine models for Arabic, English, Spanish, Japanese, Korean,
Mandarin Chinese, Ukrainian, and Vietnamese, and portable `whisper.cpp` models.
It also includes compact Whisper Medium,
Whisper Large v3, and Breeze ASR builds from
[Handy's documented model family](https://handy.computer/docs/models) that run
directly through `whisper.cpp`; Handy does not need to be installed.
SenseVoice, Parakeet, GigaAM, Canary, Dolphin, and Qwen3-ASR now all run independently of Handy through sherpa-onnx;
Parakeet, Qwen3-ASR, and Granite Speech also have Apple-native MLX options.
GigaAM (Russian, CTC or RNNT) and Canary (English only in this build; the
underlying model also covers German, French, and Spanish, but source/target
language is fixed when the recognizer loads rather than per request) download
individual files directly from their Hugging Face model repos rather than a
packaged archive, since neither publishes one.

Qwen3-ASR reads a Hugging Face tokenizer
directory instead of a `tokens.txt`, so the gateway fetches `tokenizer/` and
passes the folder to the recognizer.

Parakeet ships in two generations, and newer is not automatically better: v3
covers 25 European languages, while the English-only v2 spends all of its
capacity on English and transcribes it more accurately. Pick v2 if you dictate
only in English.

Full per-model language coverage — all 58 models, and a reverse index from each
of the 108 languages to the models that cover it — is in
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
model is fetched from and the SHA-256 of its files; the gateway hashes each
file as it streams and discards anything that does not match, so a rejected
model is never left on disk for an engine to load.

Be clear about what this does and does not buy you:

| Threat | Handled by |
| --- | --- |
| Network attacker swapping bytes in flight | TLS — every download is HTTPS with certificate verification, and non-HTTPS custom URLs are refused outright |
| Upstream repo or account compromise serving altered weights | **Pinned digests**, because the attacker is the origin and its certificate is perfectly valid |
| Silent re-upload changing a model under an existing catalog entry | **Pinned commits**, which stop downloads tracking `main` |
| Truncated or corrupted transfer | Digest verification, which also fixes the partially-downloaded-model failure mode |
| A repo listing naming a path outside the model directory | Listing paths are rejected the same way archive members already were |
| A paged listing steering the client to another host or scheme | Pagination follows `rel="next"` only within the original origin |

The pinned digest always wins over the digest Hugging Face reports at download
time. That ordering is the point: metadata fetched from a compromised host
would agree with the compromised file, so only a digest reviewed in git is
evidence of anything.

Model weights are executed by ONNX, GGUF, and Core ML runtimes, so a swapped
model is a code-execution concern rather than just a bad transcript.

### Coverage

40 of 58 catalog models are pinned, covering every Hugging Face source. The
remainder cannot be pinned from published metadata:

- **13 Moonshine models** are downloaded by the `moonshine_voice` library
  rather than by the gateway, so their transfer is outside our control.
- **3 Handy-mirrored models** on `blob.handy.computer` return a multipart S3
  ETag, which is not a digest of the file content.
- **2 sherpa-onnx release tarballs** on GitHub publish no checksum.

The last five can be pinned by hashing them locally, which transfers about
3.5 GB:

```sh
uv run scripts/harvest-model-pins.py --download-unpinnable
```

Unpinned models still download normally over HTTPS; they simply get no
digest check. Nothing is silently downgraded — a pinned model that fails
verification fails the download.

### Refreshing pins

When upstream legitimately re-uploads a model, its pinned download starts
failing until the pin is updated. That is intentional: the change becomes a
reviewable diff instead of a silent swap.

```sh
uv run scripts/harvest-model-pins.py                    # all free sources
uv run scripts/harvest-model-pins.py --only whisperkit:  # one family
```

Review the resulting diff as carefully as code. A changed digest means the
upstream bytes changed, and the commit message should say why.

## Engine selection

The `auto` engine preference uses the first runnable option in this order:

1. VocaMac when the app is installed and one of its downloaded Core ML models
   is complete
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
authenticated WebSocket while the iPhone records — real incremental decoding
with partial results, not a periodic re-transcription of the growing buffer.
Moonshine Medium favors accuracy, Small is the balanced Linux default, and Tiny
favors latency; the Zipformer model favors speed over accuracy at a fraction of
the download size. The ordinary WAV is still retained during the request and
automatically used by the batch API if streaming is unavailable or interrupted.
Streaming support is negotiated on that socket to avoid an extra network round
trip before every recording.

Every other model — the remaining Moonshine tiers, WhisperKit, faster-whisper,
and every other sherpa-onnx model above — uses its fast batch path after
recording. The server returns a structured unsupported response for those, so
the app immediately continues through the ordinary upload pipeline. In the
iPhone app or keyboard, **Automatic** uses the active gateway model. Choosing a
named language requires the active model to support that same language.

Moonshine's English code and weights use the MIT license. Its non-English weights
use the Moonshine Community License and are limited to non-commercial use; the
WebUI labels these models **personal use**. Review the current
[Moonshine licensing and model documentation](https://github.com/moonshine-ai/moonshine)
before deploying them outside a personal setup.

The WebUI can explicitly select an engine or installed model and persists that
choice in the runtime configuration file.

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

On Apple silicon, current WhisperKit CLIs expose a local `serve` mode. vocaphone
starts it on a random `127.0.0.1` port during warmup and reuses the loaded
Core ML model. If an older CLI does not support `serve`, transcription falls
back to the compatible one-shot command rather than becoming unavailable.

Tagged VocaMac releases through `0.7.2` have no headless transcription command.
VocaMac **main** now ships `--transcribe-file`
([vocamac#200](https://github.com/VocaHQ/vocamac/issues/200)), but this
gateway's `vocamac` adapter still runs through `whisperkit-cli`: it reuses the
app's Core ML model library and tokenizers, reads the model chosen in VocaMac's
Models tab, verifies downloads are complete, and skips partial downloads in
favour of another complete model. It does not call `--transcribe-file` yet.
VocaMac does not need to be running.

To force VocaMac from the environment:

```sh
export VOCAGATEWAY_ENGINE=vocamac
export VOCAGATEWAY_VOCAMAC_MODEL='small'   # optional; otherwise VocaMac's own choice
uv run vocaphone-server
```

`VOCAGATEWAY_VOCAMAC_MODEL` accepts either a VocaMac model size (`small`,
`large-v3-v20240930_turbo_632MB`) or a WhisperKit folder name
(`openai_whisper-small`). A configured model is never substituted: if it is not
downloaded, the engine reports unavailable rather than quietly using another.

To force Handy from the environment:

```sh
export VOCAGATEWAY_ENGINE=handy
export VOCAGATEWAY_HANDY_MODEL='owner/repository/model.gguf'
export VOCAGATEWAY_HANDY_FALLBACK_MODEL='owner/repository/fallback-model.gguf'
uv run vocaphone-server
```

To force standalone `whisper.cpp`:

```sh
export VOCAGATEWAY_ENGINE=whisper.cpp
export VOCAGATEWAY_WHISPER_BINARY=/absolute/path/to/whisper-cli
export VOCAGATEWAY_WHISPER_MODEL=/absolute/path/to/ggml-model.bin
uv run vocaphone-server
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
| `VOCAGATEWAY_WHISPERKIT_BINARY` | `whisperkit-cli` | unavailable | WhisperKit executable |
| `VOCAGATEWAY_VOCAMAC_APP` | `/Applications/VocaMac.app` | unavailable | Optional VocaMac app bundle |
| `VOCAGATEWAY_VOCAMAC_MODEL` | unset | unset | Pin a VocaMac model instead of following the app's choice |
| `VOCAGATEWAY_HANDY_BINARY` | `/Applications/Handy.app/Contents/MacOS/handy` | unavailable | Optional Handy application binary |
| `VOCAGATEWAY_HANDY_MODEL` | unset | unset | Pin a Handy model (`owner/repository/model.gguf`) |
| `VOCAGATEWAY_HANDY_FALLBACK_MODEL` | `handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf` | unavailable | Model used when the pinned Handy model is missing |
| `VOCAGATEWAY_RETENTION_HOURS` | `24` | `24` | Failed-session retry retention |
| `VOCAGATEWAY_DELETE_SUCCESSFUL_AUDIO` | `true` | `true` | Delete source/normalized audio after success |
| `VOCAGATEWAY_PUBLIC_URL` | unset | unset | Address the pairing QR encodes, overriding auto-discovery |
| `VOCAGATEWAY_PAIRING_URL` | unset | unset | Alias for `VOCAGATEWAY_PUBLIC_URL`, checked second |
| `VOCAGATEWAY_DEBUG` | `false` | `false` | Serve the Swagger UI at `/docs` and the schema at `/openapi.json` |

Under Compose, `VOCAGATEWAY_BIND_HOST`, `PORT`, `ENGINE`, `RETENTION_HOURS`,
`DELETE_SUCCESSFUL_AUDIO`, `PUBLIC_URL`, `PAIRING_URL`, and `DEBUG` are read
from `.env` and passed into the container. `VOCAGATEWAY_TOKEN` becomes a
Compose secret at `/run/secrets/vocagateway_token` rather than an environment
variable. The remaining paths and binaries are fixed by the image to their
container locations, and the macOS-only engine variables have no effect there.

Compose-only variables, which the gateway process itself never reads, also live
in `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCAGATEWAY_PUBLISH_HOST` | `127.0.0.1` | Host interface published by Docker |
| `VOCAGATEWAY_PUBLISH_PORT` | `8765` | Host port published by Docker |
| `VOCAGATEWAY_NETWORK_MODE` | `bridge` | Set to `host` on Linux Docker Engine to share the host's network namespace (ignores `VOCAGATEWAY_PUBLISH_HOST`/`PORT`); not supported by Docker Desktop |
| `VOCAGATEWAY_IMAGE` | `vocaphone-gateway:local` | Local or registry image tag |

Use [`.env.example`](.env.example) as a template and never commit the populated
`.env` file.

For the pairing QR payload (`{"v":1,"url":"...","token":"..."}`), native paths,
and the full `VOCAGATEWAY_*` contract — including the note that older
`VOCAPHONE_*` / `~/.config/vocaphone/` names are stale and unread — see
[configuration.md](docs/configuration.md).

## Listener and network access

The native default listener is `0.0.0.0:8765`; the startup banner and WebUI show
that listener separately from the local browser URL. An all-interface listener
is reachable from connected networks, so keep the host firewall enabled.

The iPhone and Android apps accept ordinary HTTP and HTTPS gateway URLs; a
Tailscale hostname is not mandatory. Supported arrangements include:

- a trusted LAN hostname such as `http://homelabone:8765/`; for Docker, set
  `VOCAGATEWAY_PUBLISH_HOST=0.0.0.0` and protect the port with the host firewall
- a loopback listener exposed privately through Tailscale Serve
- a VPS loopback listener behind an [HTTPS reverse proxy](#https-reverse-proxy-vps)
  and trusted certificate

HTTP does not encrypt the bearer token or recording. Use it only on a trusted
LAN or encrypted VPN, never over the public internet.

For the smallest private exposure, bind/publish on host loopback and use
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

Two things surprise most first deployments: the WebUI works over 443 while the
phone app cannot connect at all, and — once pairing is fixed — recordings fail
on upload while short ones succeed.

**Name the public URL.** The gateway does not know its own hostname. Pairing
discovery inspects local interfaces and builds candidates like
`http://<vps-ip>:8765`, which is what the QR encodes and what the phone then
tries to reach — bypassing nginx entirely. Set the address explicitly:

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

**Raise the nginx body limit.** The gateway accepts 25 MiB uploads; nginx
defaults to 1 MiB and rejects real recordings with a `413` before they reach the
application. `/v1/stream` is a WebSocket and needs the upgrade headers, and CPU
transcription routinely outlives the 60-second proxy default:

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
with `:8765`. Re-scan the QR afterwards — a phone paired earlier still holds the
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
  the revision from git; containers need it stamped at build time — see
  [Stamping the build commit](#stamping-the-build-commit).

Engine probes are cached for five seconds. sherpa-onnx, MLX Audio,
`faster-whisper`, and Moonshine load their selected model once and keep it
resident. WhisperKit warmup starts its managed loopback service and keeps the
Core ML model resident there, and the VocaMac adapter inherits that behavior for
VocaMac's own models. Handy and `whisper.cpp` retain the filesystem-prefetch
warmup behavior.

## Docker performance profiles

Only run one gateway service at a time; every profile publishes the same port
and shares the same model volume.

```sh
# Portable CPU + OpenBLAS (default; amd64 and arm64)
docker compose up --detach --build gateway

# Build CPU kernels for this exact host (fastest CPU image, not portable)
docker compose --profile native up --detach --build gateway-native

# NVIDIA host with Container Toolkit
docker compose --profile cuda up --detach --build gateway-cuda

# Intel/AMD Vulkan device exposed as /dev/dri
docker compose --profile vulkan up --detach --build gateway-vulkan
```

The CUDA profile supports both faster-whisper CUDA and the CUDA `whisper.cpp`
binary. The Vulkan profile accelerates `whisper.cpp`; faster-whisper remains on
CPU there. The dashboard reports what devices the container can actually see.

## CLI and routine operations

```sh
# Query the local backward-compatible health response
uv run vocaphone-status

# Download a redacted diagnostics bundle for a bug report
uv run vocaphone-diagnostics

# Remove sessions older than the configured retention window
uv run vocaphone-cleanup

# Follow the native macOS LaunchAgent logs
tail -f ~/Library/Logs/Vocaphone/gateway.log

# Follow the native Linux systemd user unit logs
journalctl --user -u com.vocahq.vocaphone.gateway.service -f

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
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
VOCAGATEWAY_TOKEN=test-token-with-at-least-thirty-two-characters docker compose config --quiet
docker build --tag vocaphone-gateway:test .
```

Build and publish one tag for both supported Linux architectures from the
repository root:

```sh
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/your-user/vocaphone-gateway:latest \
  --push .
```

For backup, update, and native-vs-container guidance, continue with
[deployment.md](docs/deployment.md). For pairing, paths, and env vars, see
[configuration.md](docs/configuration.md). For failures, see
[troubleshooting.md](docs/troubleshooting.md).

## License and contact

[AGPL-3.0](LICENSE). Questions and contributions:
[hello@vocahq.com](mailto:hello@vocahq.com).

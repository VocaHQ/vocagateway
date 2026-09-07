# Pairing, paths, and environment

Source of truth for the live VocaGateway contract. The Python package and CLI
are `vocagateway` (`vocagateway`, `vocagateway-token`, and related scripts).
Deprecated `vocaphone-*` console-script aliases still resolve for one cycle.
Environment variables and on-disk paths use the `vocagateway` prefix.

## Contents

- [Status and network boundary](#status-and-network-boundary)
- [Default port](#default-port)
- [On-disk paths (native)](#on-disk-paths-native)
- [QR pairing payload](#qr-pairing-payload)
- [Environment variables](#environment-variables) — [gateway process](#gateway-process) · [Compose-only](#compose-only-not-read-by-a-native-process)
- [Transcript cleanup](#transcript-cleanup)
- [Stale names (not read)](#stale-names-not-read)
- [VocaLinux remote_api](#vocalinux-remote_api)
- [Related docs](#related-docs)

## Status and network boundary

VocaGateway is **Beta** optional self-hosted infrastructure. There is no Voca
account and no hosted Voca cloud. When a client is configured to use the
gateway, audio travels to the machine you run it on — that is **not** on-device
processing. Prefer a trusted LAN, Tailscale, or HTTPS. Never expose port
`8765` to the public internet.

License: [AGPL-3.0](../LICENSE). Contact: [hello@vocahq.com](mailto:hello@vocahq.com).

## Default port

`8765` (`VOCAGATEWAY_PORT`).

## On-disk paths (native)

| Path | Contents |
| --- | --- |
| `~/.config/vocagateway/token` | Bootstrap bearer token (mode `600` on first run) |
| `~/.config/vocagateway/config.json` | WebUI engine/model choice and saved pairing URLs |
| `~/.local/share/vocagateway/` | Application data (sessions DB and related files) |
| `~/.local/share/vocagateway/models` | Downloaded models (`VOCAGATEWAY_MODELS_DIR` default) |

Docker Compose mounts the same layout under `/data` in the
`vocagateway_vocagateway-data` named volume (token via Compose secret).

## QR pairing payload

Version `1`. Fields are `url` (phone-reachable gateway base URL) and `token`
(bearer secret):

```json
{"v":1,"url":"http://192.168.1.20:8765","token":"..."}
```

Show the bootstrap token (and an ASCII QR on a TTY) with `just token` or
`uv run vocagateway-token`. Override the encoded address with
`VOCAGATEWAY_PUBLIC_URL` or `VOCAGATEWAY_PAIRING_URL` when auto-discovery is
wrong.

## Environment variables

Prefix: `VOCAGATEWAY_*`. Values below match `Settings.from_env()` and
[`.env.example`](../.env.example).

### Gateway process

The **In `.env`?** column is what `compose.yaml` actually does with the
variable. `compose.yaml` forwards only the keys it names, so a variable marked
*ignored* is read out of `.env`, interpolated into nothing, and dropped:
`docker compose config` passes and the container never sees it.

| Variable | Native default | In `.env`? | Purpose |
| --- | --- | --- | --- |
| `VOCAGATEWAY_BIND_HOST` | `0.0.0.0` | forwarded | Listener interface. Keep it wildcard on the default bridge network, or the published port cannot reach the process |
| `VOCAGATEWAY_PORT` | `8765` | forwarded | Listener port, and the container-side target of the published mapping |
| `VOCAGATEWAY_TOKEN` | unset (file or auto-create) | Compose **secret** | Bearer token override (≥ 32 characters). Mounted at `/run/secrets/vocagateway_token`, never as a container env var |
| `VOCAGATEWAY_TOKEN_FILE` | `~/.config/vocagateway/token` | ignored — image pins `/run/secrets/vocagateway_token` | Bearer-token file |
| `VOCAGATEWAY_DATA_DIR` | `~/.local/share/vocagateway` | ignored — image pins `/data` | Sessions and application data |
| `VOCAGATEWAY_MODELS_DIR` | `~/.local/share/vocagateway/models` | ignored — image pins `/data/models` | Downloaded models |
| `VOCAGATEWAY_CONFIG_FILE` | `~/.config/vocagateway/config.json` | ignored — image pins `/data/config/config.json` | Persisted WebUI settings |
| `VOCAGATEWAY_ENGINE` | `auto` | forwarded | Pin an engine id. Anything but `auto` overrides the WebUI's saved choice for the whole process |
| `VOCAGATEWAY_PUBLIC_URL` | unset | forwarded | Pairing QR URL override. A pairing address already saved in the WebUI card wins over it |
| `VOCAGATEWAY_PAIRING_URL` | unset | forwarded | Alias for `VOCAGATEWAY_PUBLIC_URL`, checked second |
| `VOCAGATEWAY_DEBUG` | `false` | forwarded | Serve `/docs` and `/openapi.json`, and report the build commit in `/v1/admin/status` |
| `VOCAGATEWAY_RETENTION_HOURS` | `24` | forwarded | Failed-session audio retention |
| `VOCAGATEWAY_DELETE_SUCCESSFUL_AUDIO` | `true` | forwarded | Delete audio after success |
| `VOCAGATEWAY_WHISPER_BINARY` | `/opt/homebrew/bin/whisper-cli` | ignored — image pins `/usr/local/bin/whisper-cli` | `whisper.cpp` CLI |
| `VOCAGATEWAY_WHISPER_MODEL` | `~/.local/share/whisper.cpp/models/ggml-base.en.bin` | ignored | Fallback `whisper.cpp` model, used only when no model is selected in the WebUI |
| `VOCAGATEWAY_WHISPER_SERVER_BINARY` | the `whisper-server` beside `whisper-cli`, else `PATH` | ignored — image ships `/usr/local/bin/whisper-server` | Resident `whisper.cpp` worker; a missing binary falls back to one `whisper-cli` run per request |
| `VOCAGATEWAY_WHISPER_DECODER_PRESET` | `quality` | forwarded | `quality` keeps the narrowed beam search; `fast` decodes greedily — cheaper on a CPU-only host, and worth a WER comparison on your own audio before you keep it |
| `VOCAGATEWAY_WHISPERKIT_BINARY` | `whisperkit-cli` | ignored — macOS only | WhisperKit CLI (macOS); also the 0.7.2 VocaMac fallback |
| `VOCAGATEWAY_VOCAMAC_APP` | `/Applications/VocaMac.app` | ignored — macOS only | Optional VocaMac bundle |
| `VOCAGATEWAY_VOCAMAC_MODEL` | unset | ignored — macOS only | Pin a VocaMac model ID instead of following the app |
| `VOCAGATEWAY_HANDY_BINARY` | `/Applications/Handy.app/Contents/MacOS/handy` | ignored — macOS only | Optional Handy binary |
| `VOCAGATEWAY_HANDY_MODEL` | unset | ignored — macOS only | Pin a Handy model id |
| `VOCAGATEWAY_HANDY_FALLBACK_MODEL` | `handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf` | ignored — macOS only | Model used when the pinned Handy model is missing |
| `VOCAGATEWAY_CLEANUP_ENABLED` | unset (shipped default: on for a fresh install) | forwarded | Force transcript cleanup on or off. **Unset is not "off"** — it leaves the WebUI's saved choice in charge. A brand-new install starts on (inert until a model is downloaded). A config written before cleanup existed loads as off until the operator enables it. Setting the variable locks the toggle in the UI |
| `VOCAGATEWAY_CLEANUP_MODE` | unset (`conservative`) | forwarded | Gateway default mode: `off` or `conservative`. Only reached when cleanup is enabled, and only acted on once a model is installed |
| `VOCAGATEWAY_CLEANUP_MODEL` | unset | forwarded | Pin a cleanup model id, e.g. `cleanup:qwen3-0.6b`. Unset, the gateway uses the installed one while exactly one is installed |
| `VOCAGATEWAY_CLEANUP_TIMEOUT_SECONDS` | unset (`5`) | forwarded | Total deadline for one correction, 1–30 s. Past it the plain transcript is returned |
| `VOCAGATEWAY_CLEANUP_LANGUAGES` | unset (the model's list) | forwarded | Comma-separated allowlist of languages cleanup may run for |
| `VOCAGATEWAY_CLEANUP_AUTO_LANGUAGE` | unset (do not guess) | forwarded | Which language a transcript left on `auto` is corrected as. Unset means `auto` falls back uncorrected unless its writing system names a language. Ignored if not on the allowlist |
| `VOCAGATEWAY_CLEANUP_BINARY` | `llama-server` on `PATH` | `/opt/llama/bin/llama-server`, built into the image | Explicit `llama-server` for the gateway to launch and own |
| `VOCAGATEWAY_CLEANUP_ENDPOINT` | unset | forwarded | `host:port` of a cleanup server the **operator** runs, instead of the one the gateway would launch. Setting it gives up gateway-controlled warm-up and idle unloading, because the gateway then does not own the process. Only loopback, private addresses, and bare container service names are accepted; anything routable is refused at startup. The old Compose sidecar hostname `cleanup` is refused too (including `cleanup:8080`); unset the variable to use the in-image runtime, or use a different service name such as `my-cleanup` |
| `VOCAGATEWAY_CLEANUP_API_KEY` | unset | forwarded | Credential the gateway presents to that operator-run server. Unused by the gateway-managed worker, which generates a key of its own per launch. A client's bearer token is never forwarded |

`VOCAGATEWAY_ENGINE` accepts `auto`, `sherpa-onnx`, `faster-whisper`,
`moonshine`, `whisper.cpp`, `mlx-audio`, `whisperkit`, `vocamac`, or `handy`.
Only the first five run in the Linux container, and only the value's spelling
is checked at startup — the host check that answers `422 invalid_engine` in the
WebUI and on `PUT /v1/admin/config` does not apply here. A macOS-only engine
pinned through the variable on Linux starts fine and leaves `/health/ready` at
`503`.

Optional build/status stamps (when set): `VOCAGATEWAY_GIT_COMMIT`,
`VOCAGATEWAY_GIT_COMMIT_SUBJECT`, `VOCAGATEWAY_GIT_COMMIT_DATE`. These are
build arguments, not settings — keep them out of `.env`, where they would pin
every later build to one commit. `just up` and `just image` export them from
git; see [Stamping the build commit](../README.md#stamping-the-build-commit).

### Compose-only (not read by a native process)

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCAGATEWAY_PUBLISH_HOST` | `127.0.0.1` | Host interface Docker publishes |
| `VOCAGATEWAY_PUBLISH_PORT` | `8765` | Host port Docker publishes |
| `VOCAGATEWAY_NETWORK_MODE` | `bridge` | Set `host` on Linux Docker Engine only |
| `VOCAGATEWAY_IMAGE` | `docker.io/vocahq/vocagateway:latest` in `compose.prod.yaml`, `vocagateway:local` in `compose.yaml` | The image the `gateway` service runs. In the deployment file it selects what to pull — pin a version for production. In the build file it only renames what gets built. `ghcr.io/vocahq/vocagateway` serves the same digests. The `gateway-cuda` and `gateway-vulkan` services ignore it |
| `VOCAGATEWAY_WHISPER_CMAKE_EXTRA` | unset | Extra CMake flags appended to the image's `whisper.cpp` build |
| `VOCAGATEWAY_LLAMA_CMAKE_EXTRA` | unset | The same for the image's `llama.cpp` cleanup-runtime build. Separate builds, so narrowing one does nothing for the other |
| `VOCAGATEWAY_BUILD_JOBS` | builder CPU count | Maximum concurrent compile jobs, for both builds; lower it when a build is memory constrained |
| `VOCAGATEWAY_RENDER_GID` | `993` | Host render-group GID added to the Vulkan container |
| `VOCAGATEWAY_VIDEO_GID` | `44` | Host video-group GID added to the Vulkan container |

Container defaults for data paths are under `/data` (and the token secret under
`/run/secrets/vocagateway_token`). The five build and Vulkan values above are
Compose interpolation inputs, not gateway-process environment variables. See
[Tuning the compiled runtimes](deployment.md#tuning-the-compiled-runtimes) and
[Giving the Vulkan container access to the GPU](deployment.md#giving-the-vulkan-container-access-to-the-gpu).

## Transcript cleanup

On by default on a brand-new install, and inert until a model is installed. It
runs **after** speech recognition, on the recognised text only: audio never
reaches it, and nothing it does can turn a successful transcription into a
failed one. Where it cannot finish safely — no model, wrong language, text too
long, busy, timed out, or an edit the checks refuse — the gateway returns
exactly the transcript it would have returned with the feature off.

"On by default" is a setting, not a behaviour, and it applies to a fresh
install (no config file yet). With no cleanup model installed there is nothing
to run, and a request that leaves `cleanup` at `inherit` resolves to `off`
rather than to a correction that falls back — so a deployment that never
downloads one returns byte-identical responses to a gateway built before the
feature existed. A config written before the cleanup block existed loads with
the feature off, even if a leftover model is already on disk; enable it in the
Cleanup tab. Downloading a model in the WebUI is what starts corrections on a
fresh install; unticking *Correct transcripts by default*, or deleting the
model, stops them.

Two deployment shapes, and they are not interchangeable:

- **Managed** (the default, native and in the container). The gateway launches
  and owns a `llama-server` on loopback with an ephemeral, unpublished port and
  a credential of its own. It can be warmed from the settings page and unloaded
  when idle. The container image builds this runtime; a native install supplies
  one (`brew install llama.cpp`, or `VOCAGATEWAY_CLEANUP_BINARY`).
- **External** (`VOCAGATEWAY_CLEANUP_ENDPOINT`). The operator runs the server,
  typically to evaluate a runtime or model the image did not build. The gateway
  will use it but promises nothing about its lifecycle, because it does not own
  the process. Give it `--ctx-size 8192` or more, or it is declined with
  `context_too_small`.

Which model runs is the operator's choice, but only once there is a choice to
make: with exactly one cleanup model installed and nothing selected, that one is
used. A second installed model, or an explicit `VOCAGATEWAY_CLEANUP_MODEL`, ends
the fallback.

Requests choose per call. Sessions and `/v1/stream` take
`cleanup: "off" | "conservative" | "inherit"` (default `inherit`);
`POST /v1/audio/transcriptions` takes a multipart `cleanup=off|conservative`
field that defaults to `off`. `GET /v1/capabilities` reports what this gateway
actually supports, so a client can stop offering a mode it cannot get. An older
gateway answers `404` there — omit the new fields when that happens, because
the session schema rejects unknown ones rather than ignoring them.

Two things it deliberately does not do. **Raw is never corrected**, whatever a
request asks for. And a transcript whose language is left on `auto` is only
corrected when its writing system names one supported language on its own —
Latin script does not, so an `auto` English dictation falls back with
`unsupported_language` rather than being sent to an English-tuned corrector on
the strength of its alphabet. Ask for `en` explicitly to have it corrected.

The `original_transcript` field carries the recognised text before correction,
for sessions that opted in — including ones where cleanup fell back. It lives
under the same retention and deletion rules as the transcript, and is absent
(`null`) for legacy and cleanup-off sessions rather than reconstructed.

## Stale names (not read)

Older `VOCAPHONE_*` environment variables and `~/.config/vocaphone/` paths are
**not** read by the current gateway. Use `VOCAGATEWAY_*` and
`~/.config/vocagateway/` / `~/.local/share/vocagateway/` only.

Deprecated CLI console-script aliases (`vocaphone-server`, `vocaphone-token`,
`vocaphone-status`, `vocaphone-diagnostics`, `vocaphone-cleanup`) still resolve to
the same entry points as `vocagateway*` for one cycle; prefer the new names.

## VocaLinux remote_api

A shipped VocaLinux can POST dictation to this gateway over the OpenAI
transcription path. Set the engine to `remote_api`, the server URL to the
gateway origin, the API endpoint to `/v1/audio/transcriptions` (not
`/inference`), and the API key to the gateway bearer token. The model field is
ignored; the WebUI's loaded engine runs.

The phone pairing contract is unchanged. This path does not create a session and
does not stream. Audio still travels to the gateway host, so it is not on-device
processing.

VocaLinux Test Connection is unauthenticated `GET /`, so it can look green with
a bad key. First dictation is the real check. The client timeout is 30 seconds.
Default concurrency is 1 (busy returns 503). LAN HTTP is the gateway default;
HTTPS needs a certificate the desktop OS trusts.

## Related docs

- [README](../README.md) — quick starts and full configuration table
- [deployment.md](deployment.md) — native vs Compose operations
- [troubleshooting.md](troubleshooting.md) — 401 and readiness failures

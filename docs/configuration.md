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
| `~/.local/share/vocagateway/` | Application data (sessions DB, device tokens, and gateway-owned data/logs under `VOCAGATEWAY_DATA_DIR`) |
| `~/.local/share/vocagateway/models` | Downloaded models (`VOCAGATEWAY_MODELS_DIR` default) |

Docker Compose mounts the same layout under `/data` in the
`vocagateway_vocagateway-data` named volume (token via Compose secret).

Desktop embedders must keep gateway data/logs under `data_dir` (default
`~/.local/share/vocagateway`). Never relocate that tree into a host app's
Application Support directory. See [desktop-embed.md](desktop-embed.md).

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
| `VOCAGATEWAY_WHISPERKIT_BINARY` | `whisperkit-cli` | ignored — macOS only | WhisperKit CLI (macOS); also the 0.7.2 VocaMac fallback |
| `VOCAGATEWAY_VOCAMAC_APP` | `/Applications/VocaMac.app` | ignored — macOS only | Optional VocaMac bundle |
| `VOCAGATEWAY_VOCAMAC_MODEL` | unset | ignored — macOS only | Pin a VocaMac model ID instead of following the app |
| `VOCAGATEWAY_HANDY_BINARY` | `/Applications/Handy.app/Contents/MacOS/handy` | ignored — macOS only | Optional Handy binary |
| `VOCAGATEWAY_HANDY_MODEL` | unset | ignored — macOS only | Pin a Handy model id |
| `VOCAGATEWAY_HANDY_FALLBACK_MODEL` | `handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf` | ignored — macOS only | Model used when the pinned Handy model is missing |

`VOCAGATEWAY_ENGINE` accepts `auto`, `sherpa-onnx`, `faster-whisper`,
`moonshine`, `whisper.cpp`, `mlx-audio`, `whisperkit`, `vocamac`, or `handy`.
Only the first five run in the Linux container, and only the value's spelling is
checked at startup — the host check that answers `422 invalid_engine` in the
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
| `VOCAGATEWAY_IMAGE` | `vocagateway:local` | Tag for the default CPU `gateway` service. It renames what gets built rather than switching Compose to pulling; use `docker compose pull` then `up --no-build` for a registry image. The `gateway-cuda` and `gateway-vulkan` services ignore it |
| `VOCAGATEWAY_WHISPER_CMAKE_EXTRA` | unset | Extra CMake flags appended to the image's `whisper.cpp` build |
| `VOCAGATEWAY_BUILD_JOBS` | builder CPU count | Maximum concurrent `whisper.cpp` compile jobs; lower it when a build is memory constrained |
| `VOCAGATEWAY_RENDER_GID` | `993` | Host render-group GID added to the Vulkan container |
| `VOCAGATEWAY_VIDEO_GID` | `44` | Host video-group GID added to the Vulkan container |

Container defaults for data paths are under `/data` (and the token secret under
`/run/secrets/vocagateway_token`). The four build and Vulkan values above are
Compose interpolation inputs, not gateway-process environment variables. See
[Tuning the whisper.cpp build](deployment.md#tuning-the-whispercpp-build) and
[Giving the Vulkan container access to the GPU](deployment.md#giving-the-vulkan-container-access-to-the-gpu).

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
- [desktop-embed.md](desktop-embed.md): Planned desktop embed contract (Pairable vs Ready, Compose pin)
- [troubleshooting.md](troubleshooting.md) — 401 and readiness failures

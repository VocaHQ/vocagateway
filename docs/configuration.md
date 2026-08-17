# Pairing, paths, and environment

Source of truth for the live VocaGateway contract. The Python package and CLI
are `vocagateway` (`vocagateway`, `vocagateway-token`, and related scripts).
Deprecated `vocaphone-*` console-script aliases still resolve for one cycle.
Environment variables and on-disk paths use the `vocagateway` prefix.

## Status and network boundary

VocaGateway is **Early** optional self-hosted infrastructure. There is no Voca
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

| Variable | Native default | Purpose |
| --- | --- | --- |
| `VOCAGATEWAY_BIND_HOST` | `0.0.0.0` | Listener interface |
| `VOCAGATEWAY_PORT` | `8765` | Listener port |
| `VOCAGATEWAY_TOKEN` | unset (file or auto-create) | Bearer token override (≥ 32 characters) |
| `VOCAGATEWAY_TOKEN_FILE` | `~/.config/vocagateway/token` | Bearer-token file |
| `VOCAGATEWAY_DATA_DIR` | `~/.local/share/vocagateway` | Sessions and application data |
| `VOCAGATEWAY_MODELS_DIR` | `~/.local/share/vocagateway/models` | Downloaded models |
| `VOCAGATEWAY_CONFIG_FILE` | `~/.config/vocagateway/config.json` | Persisted WebUI settings |
| `VOCAGATEWAY_ENGINE` | `auto` | Force an engine id |
| `VOCAGATEWAY_PUBLIC_URL` | unset | Pairing QR URL override |
| `VOCAGATEWAY_PAIRING_URL` | unset | Alias for `VOCAGATEWAY_PUBLIC_URL` |
| `VOCAGATEWAY_DEBUG` | `false` | Serve `/docs` and `/openapi.json` |
| `VOCAGATEWAY_RETENTION_HOURS` | `24` | Failed-session audio retention |
| `VOCAGATEWAY_DELETE_SUCCESSFUL_AUDIO` | `true` | Delete audio after success |
| `VOCAGATEWAY_WHISPER_BINARY` | `/opt/homebrew/bin/whisper-cli` | `whisper.cpp` CLI |
| `VOCAGATEWAY_WHISPER_MODEL` | `~/.local/share/whisper.cpp/models/ggml-base.en.bin` | Fallback `whisper.cpp` model |
| `VOCAGATEWAY_WHISPERKIT_BINARY` | `whisperkit-cli` | WhisperKit CLI (macOS) |
| `VOCAGATEWAY_VOCAMAC_APP` | `/Applications/VocaMac.app` | Optional VocaMac bundle |
| `VOCAGATEWAY_VOCAMAC_MODEL` | unset | Pin a VocaMac model size/folder |
| `VOCAGATEWAY_HANDY_BINARY` | `/Applications/Handy.app/Contents/MacOS/handy` | Optional Handy binary |
| `VOCAGATEWAY_HANDY_MODEL` | unset | Pin a Handy model id |
| `VOCAGATEWAY_HANDY_FALLBACK_MODEL` | `handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf` | Handy fallback model |

Optional build/status stamps (when set): `VOCAGATEWAY_GIT_COMMIT`,
`VOCAGATEWAY_GIT_COMMIT_SUBJECT`, `VOCAGATEWAY_GIT_COMMIT_DATE`.

### Compose-only (not read by a native process)

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCAGATEWAY_PUBLISH_HOST` | `127.0.0.1` | Host interface Docker publishes |
| `VOCAGATEWAY_PUBLISH_PORT` | `8765` | Host port Docker publishes |
| `VOCAGATEWAY_NETWORK_MODE` | `bridge` | Set `host` on Linux Docker Engine only |
| `VOCAGATEWAY_IMAGE` | `vocagateway:local` | Local or registry image tag |

Container defaults for data paths are under `/data` (and the token secret under
`/run/secrets/vocagateway_token`). See the [README configuration table](../README.md#configuration).

## Stale names (not read)

Older `VOCAPHONE_*` environment variables and `~/.config/vocaphone/` paths are
**not** read by the current gateway. Use `VOCAGATEWAY_*` and
`~/.config/vocagateway/` / `~/.local/share/vocagateway/` only.

Deprecated CLI console-script aliases (`vocaphone-server`, `vocaphone-token`,
`vocaphone-status`, `vocaphone-diagnostics`, `vocaphone-cleanup`) still resolve to
the same entry points as `vocagateway*` for one cycle; prefer the new names.

## Related docs

- [README](../README.md) — quick starts and full configuration table
- [deployment.md](deployment.md) — native vs Compose operations
- [troubleshooting.md](troubleshooting.md) — 401 and readiness failures

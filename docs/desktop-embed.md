# Desktop embed contract

Frozen contract for shipping VocaGateway inside a desktop app (VocaMac,
VocaWin, VocaLinux). Embedding remains Planned; this page is the spike
contract so host apps and the gateway stay aligned.

VocaGateway is Beta optional self-hosted speech-to-text. It is never
on-device: audio leaves the client for the machine that runs the gateway.
Product site: [vocagateway.vocahq.com](https://vocagateway.vocahq.com/).

## Platform matrix

**macOS.** Prefer a native gateway process so MLX Audio, WhisperKit, VocaMac,
and Handy stay available. Compose on Docker Desktop is a fallback only: the
container runs Linux in a VM and cannot use Apple engines.

**Windows.** Compose-primary: Docker Desktop with WSL2. If Docker is missing,
disable the self-hosted option and point the user at the Docker install docs.
Microsoft Store / MSIX packaging is out of v1.

**Linux.** Prefer Podman when present, then Docker. Flatpak and AppImage hosts
often cannot see the host container socket without extra permissions; document
that path before assuming Compose will start.

## Shared pairing contract

Pairing uses the compact JSON document phones already scan
(`vocaphone-pair-v1` is the scheme hint):

```json
{"v":1,"url":"http://192.168.1.20:8765","token":"..."}
```

`GET /v1/admin/pairing` returns `{version,url,payload,candidates}`. It does
**not** put a raw bearer token or `token_id` at the top level. Clients MUST
decode `payload` to recover `{v,url,token}`. Optional
`GET /v1/admin/pairing/qr.svg` renders the same payload as SVG.

`VOCAGATEWAY_PUBLIC_URL` (alias `VOCAGATEWAY_PAIRING_URL`) must be a
phone-reachable, non-loopback base URL. The gateway drops loopback and
link-local overrides from discovery, and the QR never encodes loopback. Set
the override whenever auto-discovery would otherwise advertise an address the
phone cannot open (Docker bridge, reverse proxy, Tailscale Serve hostname).

## Pairable vs Ready

| Probe | Meaning for the host app |
| --- | --- |
| `GET /health/live` | Process is up. Treat this as the Pairable candidate signal. |
| `GET /health/ready` | Engine can transcribe. Treat this as Ready-for-dictation. |

Pairable-before-Ready is intentional. When liveness is OK and a
non-loopback pairing URL exists, show Pair / QR even if readiness is still
`503` (no model selected yet). Dictation waits for Ready; pairing does not.

`GET /v1/admin/status` exposes the same distinction without the secret:
`pairable` (bool), `pairing_url` (phone-reachable base URL or null), and
`ready_for_dictation` (bool). It never includes the bearer token.

## Endpoints host apps use

| Endpoint | Auth | Role |
| --- | --- | --- |
| `GET /health/live` | none | Process up (Pairable candidate) |
| `GET /health/ready` | none | Engine ready (Ready-for-dictation); `503` until a model can transcribe |
| `GET /v1/admin/status` | Bearer | `pairable`, `pairing_url`, `ready_for_dictation`, paths, setup |
| `GET /v1/admin/pairing` | Bearer | `{version,url,payload,candidates}`; decode `payload` for the token |
| `GET /v1/admin/pairing/qr.svg` | Bearer | Same payload as SVG |
| `POST /v1/audio/transcriptions` | Bearer | Dictation after Ready |

## Compose image pin

Until a `v0.2.0` release exists, pin the Compose image to the published GitHub
Release tag `v0.1.0` (or that image's digest). Do not invent tags.

```sh
# .env for a desktop-launched Compose stack
# Build or pull from the v0.1.0 release, then point Compose at that tag:
VOCAGATEWAY_IMAGE=vocagateway:v0.1.0
VOCAGATEWAY_PUBLIC_URL=http://192.168.1.20:8765
```

Use `docker compose up --no-build` (or the Podman equivalent) once the pinned
image is present locally or in your registry. Local `vocagateway:local` builds
remain fine for developers; the pin is for shipped desktop embeds.

## Data and logs

The gateway owns its data and logs under `VOCAGATEWAY_DATA_DIR` (default
`~/.local/share/vocagateway`). Sessions, device tokens, models, and
gateway-owned on-disk logs live there. Never place that tree under a desktop
app's Application Support (or Windows/Linux equivalent). The host app may tee
gateway stdout/stderr into its own logger; it must not relocate the gateway's
on-disk state.

## Launch notes by platform

**Mac (native primary).** Start the `vocagateway` process with the usual env
and token file under `~/.config/vocagateway/`. Prefer Apple engines when the
user wants lowest latency. If you fall back to Compose, set
`VOCAGATEWAY_PUBLIC_URL` to a LAN or Tailscale address the phone can reach;
loopback inside the Linux VM is useless for pairing.

**Windows (Compose-primary).** Require Docker Desktop + WSL2. Generate or
read the bearer token, write `.env` with a pinned `VOCAGATEWAY_IMAGE` and a
phone-reachable `VOCAGATEWAY_PUBLIC_URL`, then start Compose. Without Docker,
keep the feature disabled with an install link. Store/MSIX is out of scope for
v1.

**Linux (podman-first).** Try Podman Compose, then Docker Compose. Bind or
publish so the phone reaches the host, and set `VOCAGATEWAY_PUBLIC_URL` when
bridge networking would otherwise advertise an unreachable address. Flatpak
and AppImage builds need explicit socket access to talk to the container
engine.

## Related docs

- [configuration.md](configuration.md): paths, env vars, QR payload
- [deployment.md](deployment.md): native vs Compose operations
- [README](../README.md): quick starts and consumers

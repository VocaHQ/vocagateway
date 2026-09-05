# VocaGateway documentation

Start at the [README](../README.md) — it carries the three quick starts and the
full `VOCAGATEWAY_*` table. These pages go deeper on one topic each.

| Page | Read it when |
| --- | --- |
| [deployment.md](deployment.md) | Choosing between native macOS, native Linux, and Docker; running at login; backups; the portable CPU service and `cuda`/`vulkan` Compose profiles; where the phone reaches the host |
| [configuration.md](configuration.md) | You need the exact value of a path, an environment variable, or the pairing QR payload |
| [tailscale.md](tailscale.md) | You want private HTTPS to the gateway without opening a port |
| [troubleshooting.md](troubleshooting.md) | Something is failing and you want the symptom, not the theory |
| [models.md](models.md) | Picking a model: all 69 in the catalog, what each speaks, and a reverse index from 109 languages back to the models that cover them |
| [additional-models.md](additional-models.md) | Judging the newer multilingual and English additions — Cohere Transcribe, Parakeet Unified, Granite Speech |

## I just want it running

1. **Docker on Linux** — [Compose quick start](../README.md#docker-compose-quick-start),
   then [`.env.example`](../.env.example) for every knob. On the default bridge
   network, set `VOCAGATEWAY_PUBLIC_URL` before you scan the pairing QR.
2. **Apple silicon Mac** — [native macOS quick start](../README.md#native-macos-quick-start).
   MLX Audio and WhisperKit are the fast paths; Docker Desktop cannot reach them.
3. **Linux desktop or home server, no container** —
   [native Linux quick start](../README.md#native-linux-quick-start).

In all three, the gateway is live before any model exists. `GET /health/ready`
answers `503` until you download and select one in the WebUI.

## Before you expose it

Gateway mode is not on-device processing: audio leaves the phone for the machine
you chose. Keep port `8765` off the public internet. Use a trusted LAN,
[Tailscale Serve](tailscale.md), or an
[HTTPS reverse proxy](../README.md#https-reverse-proxy-vps) with a real
certificate. HTTP protects neither the bearer token nor the recording.

## Contributing to the docs

`models.md` is generated — run `uv run scripts/generate_model_docs.py` rather
than editing it; a test asserts `--check`. Everything else on this page is
hand-written. [AGENTS.md](../AGENTS.md) has the repository conventions.

# Threat model: VocaGateway

## What this project does and where untrusted input enters
VocaGateway is a self-hosted, headless speech-to-text server (FastAPI + Uvicorn, Python 3.12) for the Voca
client apps (VocaPhone on iOS/Android, desktop apps). A user runs it on their own machine or home server,
usually on a LAN, behind Tailscale, or behind a reverse proxy. It listens on `0.0.0.0:8765` by default, over
plain HTTP; TLS is expected to come from a proxy or the tunnel.

It is single-operator. There are no user accounts and no tenants. Every request except `/health*`, `/`, and
the static `/assets` must carry `Authorization: Bearer <token>`. Valid tokens are:
- the bootstrap token from `VOCAGATEWAY_TOKEN` / the token file (compared with `hmac.compare_digest`);
- named device tokens issued through `/v1/admin/tokens`, persisted only as SHA-256 digests in
  `<data_dir>/device_tokens.json`.

All valid tokens currently have the same privileges, admin endpoints included (see "Anything to leave alone").

Attackers, in priority order:
1. **Unauthenticated network attacker** who can reach the port (same LAN or Wi-Fi, a misconfigured
   port-forward, a malicious web page in the operator's browser doing CSRF or DNS rebinding against the WebUI).
   This is the most important boundary.
2. **Holder of a revoked or rotated device token** (a lost phone). Revocation and rotation must take effect
   immediately on HTTP and on the WebSocket.
3. **Authenticated client sending hostile content**: audio uploads, streaming frames, JSON bodies, model IDs,
   custom model URLs, and Hugging Face repo listings or archives the gateway downloads.

Untrusted input enters through:
- `/v1/sessions*`: create a session, `PUT /v1/sessions/{id}/audio` (file upload, size-capped by
  `VOCAGATEWAY_MAXIMUM_UPLOAD_BYTES`, 25 MiB by default), finish, retry, delete.
- `/v1/audio/transcriptions`: OpenAI-compatible one-shot upload (multipart `file`, plus `model`, `language`, `response_format`, `stream`, `cleanup` form fields).
- `/v1/stream`: a WebSocket that streams PCM/audio frames. The token comes from the `Authorization` header
  on the upgrade request.
- `/v1/admin/*`: config, tokens, pairing (returns token plaintext and a QR SVG), models (download from
  pinned Hugging Face sources or a custom HTTPS `.bin`/`.gguf` URL), cleanup (starts a local `llama-server`),
  diagnostics.
- `/ui/partials/*`: HTML fragments for the htmx WebUI, rendered with Jinja2 templates.
- Audio decoding: uploaded bytes are written to the data dir and decoded by `ffmpeg`
  (`app/audio.py`, `asyncio.create_subprocess_exec`, no shell). The result goes to whisper.cpp
  (`whisper-cli` or a resident `whisper-server` on a private loopback port) or to a Python engine
  (faster-whisper, sherpa-onnx, moonshine).
- Model downloads: Hugging Face tree listings, archives (`_safe_extract_archive`), and file paths joined onto
  the models dir (`is_safe_relative_path`).

## Components that matter most / least
Most important:
- Authentication on every router (`app/context.py:require_token`, `token_is_valid`, the WebSocket checks in
  `app/routes/streaming.py`), and the token store (`app/tokens.py`): bypasses, timing leaks, revoked tokens
  still working, plaintext leaking into logs, diagnostics bundles, or `/v1/admin/status`.
- The WebUI against a browser-based attacker: XSS in Jinja2 templates or htmx partials (model names,
  token labels, transcripts and error strings are all attacker-influenced), CSRF, DNS rebinding.
- File system safety: session IDs, model IDs, custom model filenames, archive members, and HF listing paths
  must never escape `VOCAGATEWAY_DATA_DIR` / `VOCAGATEWAY_MODELS_DIR`.
- Subprocess construction for `ffmpeg`, `whisper-cli`, `whisper-server`, and `llama-server`: argument
  injection through filenames, model paths, language codes, or prompts.
- Custom model URL download (`_validate_custom_url`): SSRF and redirect handling.
- Resource exhaustion that one unauthenticated request can cause: unbounded reads before auth, upload cap
  bypass (chunked encoding, multipart), WebSocket frames before auth.

Less important, still in scope:
- DoS by an authenticated client (long audio, many sessions). The queue limits
  (`VOCAGATEWAY_MAX_CONCURRENT_TRANSCRIPTIONS`, `..._MAX_QUEUED_...`, queue timeout) are meant to bound it.
- Correctness of transcript post-processing (`app/text_styles.py`, `app/scripts.py`).

Out of scope:
- `app/webui/swagger/` (vendored Swagger UI bundle) and `web/` (the static marketing site).
- Bugs inside third-party engines (whisper.cpp, ffmpeg, llama.cpp, CTranslate2, onnxruntime), unless the
  gateway passes them input it should have rejected.
- The production `Dockerfile` / compose files' build reproducibility.

## How to exercise it
The image has the dev dependencies, whisper.cpp built with debug info at `/usr/local/bin/whisper-{cli,server}`,
the `tiny.en` model at `/opt/models/ggml-tiny.en.bin`, and a sample clip at `/opt/samples/jfk.wav`.
Everything works offline.

- Tests: `cd /src && python -m pytest -n auto -q` (about 1,000 tests; they use fake engines. Two pairing tests in `tests/test_pairing_api.py` expect a reachable LAN address and fail with networking off).
- Start the server (the env is preset; the bootstrap token is `$VOCAGATEWAY_TOKEN`, a fixed test value):
  `vocagateway &` then `curl -s localhost:8765/health/ready`.
- Transcribe:
  `curl -s -H "Authorization: Bearer $VOCAGATEWAY_TOKEN" -F "file=@/opt/samples/jfk.wav;type=audio/wav" localhost:8765/v1/audio/transcriptions`
  (the session flow is in `app/routes/sessions.py`).
- API schema: start with `VOCAGATEWAY_DEBUG=1` and open `/docs`, or read `app/schemas.py`.
- Model downloads and the cleanup LLM need the network or a `llama-server` binary, and neither is in the
  image. Exercise those paths with unit tests or a stub HTTP server on localhost.

## How you rate severity
- **Critical**: unauthenticated remote code execution; unauthenticated auth bypass of any `/v1/*` or
  `/v1/admin/*` route; unauthenticated disclosure of a token.
- **High**: authenticated RCE (command or argument injection into ffmpeg/whisper/llama, malicious model
  archives writing outside the models dir); arbitrary file read or write; a revoked or rotated token still
  accepted; stored XSS in the WebUI that an unauthenticated party can plant (e.g. through a value a
  network attacker controls) or that leaks a token; CSRF or DNS rebinding that performs admin actions.
- **Medium**: SSRF from the custom-model URL to internal hosts; stored XSS that needs a valid token to
  plant; an unauthenticated request that exhausts memory or disk or wedges the server.
- **Low**: DoS by an authenticated client within the configured limits; information leaks of versions,
  paths, or hardware details through admin endpoints; missing security headers with no demonstrated exploit.

## Anything to leave alone
- Every valid token is admin. Device tokens can call `/v1/admin/*` by design. Do not report this as privilege
  escalation, but do report anything that lets a token act after revocation.
- Plain HTTP and `0.0.0.0` as the default bind are deliberate. TLS is the deployment's job, and the WebUI
  shows an exposure banner.
- `/health`, `/health/live`, `/health/ready`, `/`, and `/assets/*` are intentionally unauthenticated. Report
  them only if they leak secrets or more than coarse status.
- The bootstrap token cannot be revoked through the API. Whoever controls the token file or env var owns
  the gateway.
- `VOCAGATEWAY_TOKEN` in this image is a public test value, not a leaked secret.

## Reports and patches
Patches should be minimal, should come with a regression test in `tests/` (pytest, fake engines as in
`tests/conftest.py`), and must keep `ruff`, `flake8 --select=WPS`, and `mypy` clean.

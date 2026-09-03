# Troubleshooting

Find the symptom, not the subsystem.

**The gateway**

- [Gateway reachable, model not ready](#gateway-reachable-model-not-ready) — liveness `200`, readiness `503`
- [Gateway unavailable](#gateway-unavailable)
- [401 unauthorized](#401-unauthorized)
- [413, 415, or 422](#413-415-or-422) — upload rejected
- [A model download fails SHA-256 verification](#a-model-download-fails-sha-256-verification)
- [Reporting a gateway bug](#reporting-a-gateway-bug) — attach the redacted diagnostics bundle

**Network and Docker**

- [Docker service does not start](#docker-service-does-not-start)
- [A LAN hostname such as homelabone does not connect](#a-lan-hostname-such-as-homelabone-does-not-connect) — also covers a pairing QR that only offers a `172.x` address

**Speed and accuracy**

- [Native Apple silicon transcription is slow](#native-apple-silicon-transcription-is-slow)
- [Linux transcription is still slow](#linux-transcription-is-still-slow)
- [The transcript came back in the wrong language](#the-transcript-came-back-in-the-wrong-language)

**The phone app and keyboard**

- [Keyboard is missing](#keyboard-is-missing)
- [Start does not open vocaphone](#start-does-not-open-vocaphone)
- [Keyboard never shows recording](#keyboard-never-shows-recording)
- [Microphone is denied](#microphone-is-denied)
- [AirPods are connected but the wrong microphone is used](#airpods-are-connected-but-the-wrong-microphone-is-used)
- [Media plays through the receiver after dictation](#media-plays-through-the-receiver-after-dictation)
- [Transcript did not insert](#transcript-did-not-insert)
- [Finish appears unresponsive](#finish-appears-unresponsive)

## Keyboard is missing

Confirm the extension is signed with the containing app, then add vocaphone in
iOS Settings → General → Keyboard → Keyboards. Some secure or specialized fields
intentionally reject third-party keyboards.

## Start does not open vocaphone

Launch vocaphone once directly, confirm the `vocaphone` URL scheme is present,
and retry. If iOS does not open it from the keyboard, open vocaphone manually
within two minutes; it recovers the waiting keyboard request and starts recording.
Automatic return to the original app is not available, so swipe back manually
after recording starts.

## Keyboard never shows recording

Confirm that Settings → General → Keyboard → Keyboards → vocaphone has Allow
Full Access enabled. The app and extension must also use exactly the same
registered App Group. Without Full Access, the keyboard now displays an explicit
warning instead of silently failing to create shared state.

## Microphone is denied

Open iOS Settings → Privacy & Security → Microphone and enable vocaphone. The
keyboard extension itself cannot receive microphone permission.

## AirPods are connected but the wrong microphone is used

Open vocaphone and check **Microphone → Input in use**. Use **Automatic** to let
iOS select the combined input/output route, or **iPhone Microphone** to request
the built-in input. Bluetooth input and output routes are linked by iOS, so
changing the microphone can also change where playback is heard while recording.

If the displayed input does not change, stop the current dictation and Quick
Dictation standby, reconnect the accessory, choose the preference again, and
start a new recording.

## Media plays through the receiver after dictation

Update to a build containing the current audio-session handling, then stop any
active recording and disable/re-enable Quick Dictation. With no external audio
route, vocaphone requests the built-in speaker and deactivates its audio session
when standby ends so other apps can restore their normal playback session.

If the orange microphone indicator remains after the ready window should have
expired, force-quit vocaphone once and reopen it. Include the selected input,
connected accessories, and whether Quick Dictation was Ready in a bug report.

## Gateway reachable, model not ready

Liveness and readiness distinguish these states:

```sh
curl --fail http://127.0.0.1:8765/health/live
curl --include http://127.0.0.1:8765/health/ready
```

If liveness is `200` but readiness is `503`, inspect the selected model in the
WebUI. For a native Handy setup, also check:

```sh
test -x /Applications/Handy.app/Contents/MacOS/handy
/Applications/Handy.app/Contents/MacOS/handy --list-models --json
uv run vocagateway-status
```

For a native VocaMac setup, check that the app is installed and its selected
model is reported as downloaded and supported:

```sh
test -x /Applications/VocaMac.app/Contents/MacOS/VocaMac
defaults read com.vocamac.app vocamac.selectedModelSize
/Applications/VocaMac.app/Contents/MacOS/VocaMac --list-models --json
```

The selected entry must have `"selected":true`, `"downloaded":true`, and
`"supported":true`. VocaMac 0.8.0 and later can run WhisperKit, Parakeet, Apple
Speech, and specialized ONNX selections through the headless interface. A
release through 0.7.2 can use only complete VocaMac WhisperKit downloads and
also requires `whisperkit-cli` on `PATH`; update VocaMac to follow other model
families.

With `VOCAGATEWAY_ENGINE=whisper.cpp`, also check
`$VOCAGATEWAY_WHISPER_BINARY` and `$VOCAGATEWAY_WHISPER_MODEL`.

Check `VOCAGATEWAY_ENGINE` itself before going further. Any value other than
`auto` pins the engine for the whole process and overrides the WebUI's saved
choice, and the host check that returns `422 invalid_engine` in the WebUI does
not apply to the variable. `VOCAGATEWAY_ENGINE=vocamac` (or `handy`,
`whisperkit`, `mlx-audio`) on Linux or in a container is accepted at startup and
then reports unavailable forever, which looks exactly like a missing model.

For Docker, open Models and download/select SenseVoice Small INT8, Parakeet TDT
INT8, or a faster-whisper Base model. CPU + INT8 applies to faster-whisper;
sherpa entries are already quantized. The container cannot run MLX Audio,
WhisperKit folders, VocaMac, or Handy itself.

## Native Apple silicon transcription is slow

For an MLX model, the active engine should start with `mlx-audio:` and the
dependency card should show MLX Audio available. For WhisperKit, Overview should
report `Metal/Core ML` and the active model should start with `whisperkit:`.
Current WhisperKit builds stay resident behind a random
loopback-only service; after gateway startup, `ps` should show a
`whisperkit-cli serve` child process. Restart the gateway after upgrading
WhisperKit so warmup can start the service. If `serve` is unavailable, vocaphone
deliberately falls back to the slower compatible one-shot CLI.

Use the Pair & test tab's three-run benchmark. It reports warm runs 2 and 3
separately from the first model-load run. If normalization is small but inference is slow,
try MLX Whisper Turbo 4-bit or a smaller WhisperKit model before changing
network or iPhone settings. MLX requires a native arm64 macOS gateway installed
with `uv sync --extra engines --extra apple`; it is unavailable inside Docker.

## Linux transcription is still slow

Use the Pair & test tab's three-run benchmark, which reports the warm
second/third run. Model load should be zero after the first persistent-engine
request. Check:

- the active engine is `sherpa-onnx`, Moonshine, or `faster-whisper`, not the
  per-request `whisper.cpp` CLI
- Precision is INT8 and CPU threads is 0 or no higher than the effective CPU
  allocation shown on Overview. Leave it at 0 unless you are deliberately
  capping the gateway: 0 means the engine counts the host's physical cores,
  clamped by the container's CPU quota, so it neither oversubscribes a cgroup
  nor spreads a batch across hyperthread siblings that then hold it back
- the container has not been assigned a fractional CPU quota
- Tiny/Base is used before Small/Medium on low-power servers
- for capable hardware, the `native`, `cuda`, or `vulkan` Compose profile is
  running instead of the portable default

Do not run multiple profile services together: they share port 8765 and the
model volume. SenseVoice Small INT8 is the smallest portable multilingual
choice, while Parakeet TDT INT8 covers 25 European languages with punctuation.

If you need Whisper itself rather than one of the faster architectures, the
choice is between two tiers, and it is a speed/accuracy trade rather than a
free win. On the Open ASR Leaderboard's English average, full Whisper Large v3
scores 5.78 WER and Large v3 Turbo 6.36; on German the gap is wider, 4.00
against 6.12. Turbo keeps Large v3's encoder and replaces its 32-layer decoder
with four, so it is roughly 1.7x faster and half the download. Pick Turbo when
latency matters, full Large v3 when accuracy does — as a faster-whisper model
either way, so the weights stay resident between requests.

Distil-Whisper Large v3.5 beats both on English (5.40) at the same size, if you
do not need other languages. The whisper.cpp Medium builds and the 3 GB
whisper.cpp Large v3 are retired; an installed copy keeps working and the WebUI
points at its replacement.
Moonshine English Tiny/Small Streaming are fast Linux options;
Tiny prioritizes latency and Small balances speed with accuracy. Other Moonshine
languages use the batch upload path after recording. The app automatically
falls back to that batch path whenever live streaming is unavailable.

## Docker service does not start

Run commands from the directory containing the canonical Compose file:

```sh
docker compose config
docker compose ps
docker compose logs gateway
```

Confirm `.env` contains a `VOCAGATEWAY_TOKEN` of at least 32 characters and
is not a copy with the placeholder unchanged. Nothing enforces this for you:
`docker compose config` accepts the empty placeholder, and a container started
without a token falls back to a secret it generates and never prints, so
`/health/live` stays green while every authenticated request returns `401`. A
healthy container can also be not ready until a model is selected; the Docker
healthcheck measures liveness.

If port 8765 is already in use, change `VOCAGATEWAY_PUBLISH_PORT` in `.env` and
recreate the service. Tailscale Serve must then point to that same host port.

## Gateway unavailable

Check that the gateway host is awake, reachable, and running. For a Tailscale
deployment, also confirm Tailscale is connected and Serve is active. The
recording should remain on the iPhone for Retry.

For a container deployment, also check `docker compose ps` from the repository
root and confirm the `vocagateway_vocagateway-data` volume is still mounted.

## A LAN hostname such as homelabone does not connect

Confirm the app URL includes the scheme and port, for example
`http://homelabone:8765/`, and approve Local Network access in iOS Settings.
Then verify the hostname from another LAN device and check that the gateway is
actually listening beyond loopback.

For Docker, `VOCAGATEWAY_PUBLISH_HOST` must be `0.0.0.0` rather than the secure
loopback default. Recreate the service after changing `.env`:

```sh
docker compose up --detach
```

Keep the host firewall enabled. Do not use this LAN configuration to expose port
8765 directly to the internet; use an HTTPS reverse proxy for a VPS.

If the pairing QR itself shows no LAN address to pick from (or only shows a
`172.x`/bridge address), that's the same root cause: the container's default
bridge network only exposes its own private interface to address
auto-discovery, never the host's real LAN NIC. Set
`VOCAGATEWAY_PUBLIC_URL=http://192.168.1.20:8765` in `.env` to name the address
the phone should use — this works on every Docker flavour, including Docker
Desktop. On Linux Docker Engine only, `VOCAGATEWAY_NETWORK_MODE=host` is the
alternative: the container shares the host's network namespace and discovery
finds the `192.168.x.x` address directly. Recreate the service after either
change. See [deployment.md](deployment.md#trusted-local-network).

## 401 unauthorized

If this device was paired with its own token, open the WebUI Settings tab and
confirm it is still listed under **Paired device tokens** — revoking a token
there immediately rejects it. Otherwise confirm the bootstrap token with
`just token` or `uv run vocagateway-token` (reads `VOCAGATEWAY_TOKEN` or
`~/.config/vocagateway/token`), copy the exact value into vocaphone, and
save/test again. Never put the token in a URL or screenshot. See
[configuration.md](configuration.md).

## 413, 415, or 422

- `413 audio_too_large`: keep recording below two minutes / 25 MB.
- `415 unsupported_audio_type`: use M4A, CAF, or WAV.
- `422 audio_empty`, `invalid_audio`, or `silent_audio`: record again and inspect
  the phone's input route.
- `422 language_unsupported`: the model loaded on your gateway cannot transcribe
  the language selected in the app. Either set the language to Automatic, pick a
  language the model covers, or download a model that covers it — the Models tab
  lists each model's languages, and [models.md](models.md#language-index) maps
  every language to the models covering it. This failure is deliberately not retryable,
  because retrying sends the same language to the same model. For Hindi and other
  South Asian languages, pin the language and use a multilingual Whisper model.

## The transcript came back in the wrong language

Some models decide the language themselves and cannot be pinned to one. Dolphin,
SenseVoice, and Qwen3-ASR all predict the language as part of
decoding, so the language chosen in the app does not constrain them — their model
cards carry an **auto language** badge. On short recordings they can confuse
closely related languages, most often Hindi with Urdu, Marathi, or Nepali.

If you need a guaranteed language, use a Whisper model. `whisper.cpp`,
faster-whisper, WhisperKit, and MLX Whisper are all passed the language
explicitly, so selecting Hindi transcribes Hindi.

Speaking for longer also helps the auto-detecting models: a two-second clip
carries much less evidence of which language it is than a full sentence.

## Transcript did not insert

Return to the same target field and tap Insert. If the keyboard context changed,
vocaphone intentionally refuses automatic insertion to avoid putting private
text in the wrong app. vocaphone uses iOS's document identifier so returning to
the same field still works if the keyboard extension was recreated while the
containing app was open.

If the transcript is visible but Insert appears inactive, tap once in the target
field, switch back to vocaphone keyboard, and wait for the current session card.
Do not start another dictation for the same text: session revisions deliberately
prevent duplicate insertion.

## Finish appears unresponsive

Finish first writes a finalizing revision that the containing app observes. Keep
vocaphone's Quick Dictation session alive, verify the orange microphone
indicator was present, and wait for the Transcribing state. If the gateway is
offline, the keyboard should surface Retry rather than discarding the recording.

Repeated Finish taps are safe, but they do not create a second server session.
When reporting a failure, include the keyboard state shown before and after the
tap, whether vocaphone was open in the background, and the gateway readiness
response—never include the token or a private transcript.

## Reporting a gateway bug

Attach the redacted diagnostics bundle instead of manually describing gateway
state: open the WebUI **Settings** tab and click **Download diagnostics**, or run
`uv run vocagateway-diagnostics` on the gateway host. It contains version, engine
and dependency status, hardware detection, and operational counters, and never
includes the bearer token, recordings, transcripts, or session identifiers.

## A model download fails SHA-256 verification

The gateway pins the expected digest of every catalog model it can
(`app/model_pins.json`) and discards a download that does not match. It retries
short responses and checksum mismatches three times against that same pin; it
never changes or bypasses the expected digest. A reported failure therefore
means the mismatch persisted. The partial file is already deleted; nothing
unverified is kept.

Two very different causes look identical here, so check which one it is before
retrying:

1. **Upstream re-uploaded the model.** Common and usually benign. Confirm the
   repo has a newer commit than the one pinned for that model, then refresh the
   pins and review the diff:

   ```sh
   uv run scripts/harvest-model-pins.py --only sherpa-onnx:
   git diff app/model_pins.json
   ```

2. **The bytes were altered in transit or at the source.** A corrupted proxy or
   mirror, or a compromised upstream. Because the gateway already retried the
   transfer, do not keep clicking Retry. Do not "fix" a reproducible mismatch
   by refreshing the pin unless the upstream commit genuinely changed.

Never work around this by deleting the pin. The check is the only thing
standing between a swapped model file and an ONNX/GGUF/Core ML runtime that
will execute it.

For a custom `.bin`/`.gguf` URL, the digest is whatever you pasted into the
SHA-256 box. Confirm it against the model card; leaving the box empty skips
verification for that download.

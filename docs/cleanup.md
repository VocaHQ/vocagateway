# Transcript cleanup: compact vs full

The gateway launches a local `llama-server` for grammar cleanup. One set of
flags cannot fit both a CPU-only 8 GB box and a GPU machine with 32 GB. It
therefore picks a **profile** from this host, and you can force one.

You do not need this page if you only use the WebUI **Cleanup** tab: download a
model, optionally press **Load model now**. The status card already says which
profile is running.

Read this when:

- cleanup is slow or the machine is swapping, and you want the low-end flags
- you have a GPU or 16 GB+ RAM and want the model run at full quality
- you run `llama-server` yourself (`VOCAGATEWAY_CLEANUP_ENDPOINT`) and need
  the matching command line

## What the two profiles do

| | **Compact** | **Full** |
| --- | --- | --- |
| Who it is for | CPU-only host with under 16 GB RAM | GPU (Metal, CUDA, or AMD) **or** 16 GB RAM or more |
| `--ctx-size` | 4096 | 8192 |
| KV cache | 8-bit (`q8_0`) | 16-bit (`f16`) |
| `--batch-size` / `--ubatch-size` | 512 / 256 | 2048 / 512 |
| Long dictations | Split on sentences, then stitched | Usually one pass; still split if the text is huge |
| RAM next to a speech model | A few hundred MB of cache | Roughly 2× that cache, higher quality |

Shared on both: `--parallel 1`, `--jinja`, `--no-webui`, loopback only, a
private API key. Threads follow the same cap as speech (physical performance
cores, at most 8, unless you set CPU threads in the WebUI).

## How the gateway chooses

Default is **auto**:

1. If the host is CPU-only **and** RAM is under 16 GB → **compact**
2. Otherwise → **full** (Apple silicon counts as having a GPU via Metal)

Override with an environment variable. A set value wins and is mentioned on
the Cleanup tab:

```sh
# Force the low-end flags on a strong machine that is also running a large ASR model
VOCAGATEWAY_CLEANUP_PROFILE=compact

# Force the high-end flags on a CPU-only host that has plenty of RAM
VOCAGATEWAY_CLEANUP_PROFILE=full
```

Unset, or `auto`, leaves the automatic choice in charge. Invalid values are
ignored and auto is used.

In Docker Compose, add the same line to `.env` (copy from `.env.example`).
Blank means unset.

After changing the variable, restart the gateway so the worker is launched
again with the new flags.

## Managed worker (the default)

Do nothing extra. The gateway passes the profile flags when it starts
`llama-server`. The Cleanup tab status line names the profile and whether it
was automatic or forced.

Pick the compact Qwen3 0.6B Q4 model on a small CPU host; the Q8 0.6B model is
the quality default. The 1.7B model is only worth it on a full-profile host.

## External `llama-server`

If you set `VOCAGATEWAY_CLEANUP_ENDPOINT`, the gateway does **not** apply these
flags — it did not start that process. Give the server the compact or full
command below so it matches the machine it runs on.

The window must still be **4096 or more**. Smaller is declined with
`context_too_small`. Long transcripts are packed using the window the server
reports, so a full-profile external server automatically gets larger pieces.

Always keep `--host` on loopback or the Compose network, `--parallel 1`,
`--jinja`, `--no-webui`, and an API key. Never `ports:` that server to the
public internet.

### Compact (CPU-only, under 16 GB)

```sh
llama-server \
  --model /path/to/Qwen3-0.6B-Q4_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 256 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --threads 4 \
  --parallel 1 \
  --jinja \
  --no-webui \
  --api-key "$VOCAGATEWAY_CLEANUP_API_KEY"
```

Set `--threads` to the number of **performance** cores, at most 8. Then:

```sh
VOCAGATEWAY_CLEANUP_ENDPOINT=127.0.0.1:8080
VOCAGATEWAY_CLEANUP_API_KEY=...   # same key as above
```

Compose equivalent (never publish the port):

```yaml
# compose.override.yaml
services:
  my-cleanup:
    image: ghcr.io/ggml-org/llama.cpp:server@sha256:...
    expose: ["8080"]
    volumes: [./models:/models:ro]
    command:
      [
        --model, /models/Qwen3-0.6B-Q4_0.gguf,
        --host, 0.0.0.0, --port, "8080",
        --ctx-size, "4096", --batch-size, "512", --ubatch-size, "256",
        --cache-type-k, q8_0, --cache-type-v, q8_0,
        --parallel, "1", --jinja, --no-webui,
        --api-key, "${VOCAGATEWAY_CLEANUP_API_KEY}",
      ]
```

### Full (GPU or 16 GB+)

```sh
llama-server \
  --model /path/to/Qwen3-0.6B-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --parallel 1 \
  --jinja \
  --no-webui \
  --api-key "$VOCAGATEWAY_CLEANUP_API_KEY"
```

llama.cpp offloads to Metal, CUDA, or Vulkan when those backends are built in.
You do not need `--n-gpu-layers` on current builds; it defaults to auto. Use
the compact command instead if this host is swapping.

Compose: same as above, with `--ctx-size 8192`, batch 2048 / ubatch 512, and
`f16` caches.

## If something is wrong

| Symptom | What to try |
| --- | --- |
| Machine swapping, cleanup slow, ASR also slow | `VOCAGATEWAY_CLEANUP_PROFILE=compact`, restart, turn on **Free memory when idle** |
| Strong GPU host still using compact | `VOCAGATEWAY_CLEANUP_PROFILE=full`, restart. Confirm Overview shows a GPU |
| External server `context_too_small` | Restart it with `--ctx-size 4096` (compact) or `8192` (full) |
| Long dictation uncorrected (`input_too_long`) | That is only the 16 KiB ceiling. Shorter long dictations are split automatically |

Cleanup still cannot turn a successful transcription into a failure: every
timeout, busy slot, or unsafe edit returns the plain styled transcript.

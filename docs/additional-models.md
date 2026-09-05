# Additional speech-to-text models

These are optional, self-hosted models. Audio runs on the gateway host; adding
models does not make transcription run on the phone. Hardware, recording length,
noise, language, and model quantization all affect speed and accuracy.

## Hindi spoken, Roman text written

Choose **Hinglish — Roman** for Hindi written using Latin letters, for example
`mujhe ghar jaana hai`. This is transcription, not translation into English.

- **Apex Q5** remains the compact, balanced cross-platform option (574 MB).
- **Prime Q5** adds a larger alternative (1.18 GB) through whisper.cpp. Its
  publisher reports slightly better Common Voice/FLEURS results than Apex but
  worse Indic-Voices results. It is not an across-the-board accuracy upgrade.
- **MLX Swift** is a much smaller Whisper Base fine-tune (296 MB), for Apple
  silicon. It has higher published error rates than Apex and trades accuracy for
  size and speed. The gateway forces the Hindi decoder token and validates Roman
  output, including when the requested language is Automatic.

The Prime GGML file is a community conversion; Swift uses the publisher's weights.
Both are experimental and are not automatically recommended over existing models.
Sources: [Prime conversion](https://huggingface.co/curiophile/whisper-hindi2hinglish-ggml),
[Oriserve comparison](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex),
[Swift](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Swift).

## Hindi and mixed-script speech

**MLX Whisper Small Hindi** writes Devanagari. Its publisher evaluates read Hindi;
conversational speech, noise, and code-switching are not established strengths.
A short gateway smoke test produced repeated extra words, so treat this entry as
an evaluation candidate, not an accuracy upgrade or a recommended default.
**MLX Srota** and **Srota Conversational** write Hindi in Devanagari and English in
Latin. These are separate from the Roman Hindi selector. Srota's conversational
benchmark shares speakers between training and test; the scores do not establish
accuracy on unseen speakers. The gateway preserves its literal `language None`
decoding prefix rather than substituting a Hindi-only prefix.

Sources: [Small Hindi](https://huggingface.co/zindagi-technologies/whisper-small-hindi),
[Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish).

## Multilingual and English additions

- **Cohere Transcribe INT8** supports 14 languages on CPU through sherpa-onnx.
  Select the spoken language explicitly: this model does not detect it
  automatically. Hindi is not supported. Its external ONNX encoder data file is
  downloaded and verified alongside the encoder and decoder.
- **Parakeet Unified English INT8** has separate batch and streaming exports.
  The streaming entry uses 560 ms model context. That is not a guarantee of
  560 ms end-to-end latency. These are English-only models released under the
  NVIDIA Open Model License; consult the linked upstream terms before use.
- **Granite Speech 4.1 Multilingual** supports English, French, German, Spanish,
  Portuguese, and Japanese. The MLX BF16 model is separate from the existing
  English NAR quantization. A smaller native Q5 entry is available through the
  optional transcribe.cpp runtime. The MLX adapter supplies a transcription prompt
  so an explicit language selection does not turn into a translation request.
- **Canary-Qwen 2.5B Q5** is English-only and uses transcribe.cpp. It is a different
  architecture from Canary 180M and cannot be loaded using that sherpa adapter.

Sources: [Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026),
[sherpa Cohere](https://k2-fsa.github.io/sherpa/onnx/cohere_transcribe/index.html),
[Parakeet Unified](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
[Granite](https://huggingface.co/ibm-granite/granite-speech-4.1-2b),
[Canary GGUF](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf).

## Install the optional transcribe.cpp runtime

The gateway's Python extras and default container image do not bundle this
binary. Install it on the gateway host before choosing its models. The model
weights are downloaded by the gateway with revision and SHA-256 pins.

The adapter was checked with transcribe.cpp 0.2.3, commit
`e2f82cb6702315a1194f3bf1a6fee67cd2678447`. It requires the `-o` text-file output
option. Older binaries without that option are not compatible.
Longer recordings also use its `--batch` and `--batch-jsonl` options.

Follow the [pinned build and installation instructions](deployment.md#install-transcribecpp)
for macOS or Linux. If installing elsewhere, set `VOCAGATEWAY_TRANSCRIBE_BINARY` to the absolute
path of `build/bin/transcribe-cli` and restart the gateway. If `transcribe-cli` is
already on the service's PATH, no override is needed. Service PATH can differ
from the interactive shell's PATH.

The WebUI Overview page lists **transcribe.cpp CLI** under Libraries & tools. It
reads Missing until the gateway resolves the binary, and the tile carries the
install hint; it shows the resolved path once the override or PATH lookup works.
Every transcribe.cpp model card carries the same warning, so a download that
cannot run yet is flagged before it starts. The weights are still correct
without the binary, so the download stays available and the warning clears on
the next page load once the gateway resolves `transcribe-cli`.

Use the upstream CPU build or enable its Metal, CUDA, Vulkan, or HIP backend for
the host. A GPU build is not guaranteed to run on another machine. A container
needs a compatible Linux build and its runtime libraries inside the image;
macOS binaries cannot run in Docker Desktop's Linux container.

This gateway adapter runs a bounded subprocess for each recording, cleans up its
transcript file, and kills the process on timeout or cancellation. It does not
keep weights resident between recordings. Recordings over 20 seconds are split
near quiet boundaries without dropping samples, then decoded in one process so
the weights are loaded once. Each chunk is checked for errors or truncation before
returning a combined transcript. Chunk boundaries can still affect recognition;
measure accuracy on your recordings. Select an explicit language for
multilingual Granite; Automatic resolves to English only for English-only Canary.

## Evaluation and availability

Published WER is specific to a dataset, normalization, decoder, and weight
precision. It is not a measured ranking of these gateway integrations. Compare
models on the same held-out recordings and script convention, and measure memory
and latency on the actual host before choosing a default.

Orato Hindi v1 is not in the downloadable catalog: its publisher currently states
that access is private, and anonymous weight access is unavailable. It cannot
provide the gateway's unauthenticated public model-download experience.

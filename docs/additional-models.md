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
  English NAR quantization. The MLX adapter supplies a transcription prompt so an
  explicit language selection does not turn into a translation request.

Sources: [Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026),
[sherpa Cohere](https://k2-fsa.github.io/sherpa/onnx/cohere_transcribe/index.html),
[Parakeet Unified](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
[Granite](https://huggingface.co/ibm-granite/granite-speech-4.1-2b).

## Evaluation and availability

Published WER is specific to a dataset, normalization, decoder, and weight
precision. It is not a measured ranking of these gateway integrations. Compare
models on the same held-out recordings and script convention, and measure memory
and latency on the actual host before choosing a default.

Orato Hindi v1 is not in the downloadable catalog: its publisher currently states
that access is private, and anonymous weight access is unavailable. It cannot
provide the gateway's unauthenticated public model-download experience.

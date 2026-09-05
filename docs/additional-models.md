# Additional speech-to-text models

These are optional, self-hosted models. Audio runs on the gateway host; adding
models does not make transcription run on the phone. Hardware, recording length,
noise, language, and model quantization all affect speed and accuracy.

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

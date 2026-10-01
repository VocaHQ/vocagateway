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

## Indian languages

- **IndicConformer INT8** is AI4Bharat's Conformer for the 22 scheduled
  languages of India: Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada,
  Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia,
  Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu and Urdu. Each language is
  a separate download of about 198 MB that runs on CPU through sherpa-onnx.
  Install the one you speak and select it; it transcribes only that language,
  whatever the client asks for, and English words come out transliterated.
- The output has no punctuation or capitalization. Four languages come back in
  a script you might not expect: Kashmiri in Perso-Arabic, Sindhi in
  Devanagari, Manipuri in Meetei Mayek and Santali in Ol Chiki. The last two
  need fonts that not every device ships.
- These are the 120M-parameter per-language checkpoints, decoded through their
  CTC head. AI4Bharat's own repositories, including the larger 600M
  multilingual model, are gated behind a Hugging Face login, which the
  gateway's anonymous downloader cannot pass. The catalog therefore pins a
  community INT8 export of the same MIT-licensed weights by commit and SHA-256.
- Dolphin still covers many of these languages in one download and adds
  punctuation, but it guesses the language and is markedly less accurate on
  each of them.

Sources: [AI4Bharat IndicConformer](https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large),
[sherpa-onnx export](https://huggingface.co/parismitaglobalsolutions/indicconformer-sherpa-onnx).

## Evaluation and availability

Published WER is specific to a dataset, normalization, decoder, and weight
precision. It is not a measured ranking of these gateway integrations. Compare
models on the same held-out recordings and script convention, and measure memory
and latency on the actual host before choosing a default.

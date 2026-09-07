# English cleanup model evaluation

Date: 2026-09-07. This is a preliminary engineering smoke test, not the full
600-case, multi-host release evaluation. The catalog therefore reports no model
as having passed the final English quality gates yet.

## Result

The gateway adds `cleanup:qwen3-0.6b-q4` as an English-only compact candidate.
Its pinned `Qwen3-0.6B-Q4_0.gguf` artifact is 428,970,080 bytes, 33% smaller than
the existing 639,446,688-byte Q8_0 artifact. On an Apple M1 Pro with 16 GB RAM,
the compact artifact had a 192 ms warm p50 and 224 ms p95 over 45 requests. The
Q8_0 baseline had a 224 ms p50 in the smaller baseline run. These measurements
show a modest local improvement, not the 2× latency reduction required for a
"much faster" recommendation.

The model uses the existing managed `llama-server`, prompt, strict JSON response,
and preservation checks. It is offered only for explicit English (`en`). The
artifact comes from the llama.cpp project's GGUF repository and remains attributed
to the upstream Qwen model separately in catalog metadata.

The catalog also offers two unevaluated English-only Gemma 3 candidates from
the same ggml-org conversion path: `cleanup:gemma-3-270m` (Q8_0, 291,545,600
bytes) for a smaller download than Qwen3 0.6B Q4, and `cleanup:gemma-3-1b`
(Q4_K_M, 806,058,240 bytes) as a 1B English instruction-following option.
Neither has `evaluated_languages` set. Sub-0.5B chat models in the table below
mostly copied errors or invented text, so the 270M entry is a footprint option
only until it clears the smoke corpus.

## Candidates not promoted

All candidates used the 15 synthetic cases in
`tests/fixtures/cleanup/english_core.jsonl`. The corpus covers already-correct
text, punctuation, grammar, protected spans, negation, deliberate repetition,
and dictated instructions. Model files and raw benchmark output remain outside
git.

| Candidate | Artifact size | Warm p50 | Outcome |
| --- | ---: | ---: | --- |
| SmolLM2 360M Q8_0 | 386 MB | 196 ms | Mostly copied incorrect input unchanged |
| SmolLM2 135M Q8_0 | 145 MB | 202 ms | Invented text and timed out on some cases |
| Granite 4 H 350M Q4_K_M | 223 MB | 330 ms | Mostly copied incorrect input unchanged |
| LFM2.5 350M Q4_K_M | 229 MB | 169 ms | Changed meaning and removed content; license also needs distribution review |
| Qwen2.5 0.5B Q4_K_M | 491 MB | 214 ms | Inconsistent grammar and occasional wrapper output |
| Unbabel GEC T5-small FP32 | about 231 MB | 219 ms | Corrected grammar but changed meaning and damaged URLs, email, and paths |

The dedicated T5 model is English grammatical-error correction rather than a
chat model, but the current `llama-server` path cannot serve encoder-decoder T5
models correctly. Adding a second runtime would not be justified by these quality
and latency results. The source model card documents the `gec:` input contract:
<https://huggingface.co/Unbabel/gec-t5_small>. The current llama.cpp limitation
is tracked at <https://github.com/ggml-org/llama.cpp/issues/26565>.

## Reproduction

Start a candidate with the same `llama-server` flags the gateway uses, then run:

```sh
uv run scripts/benchmark_cleanup.py \
  --endpoint 127.0.0.1:8080 \
  --api-key local-key \
  --model-id cleanup:qwen3-0.6b-q4 \
  --corpus tests/fixtures/cleanup/english_core.jsonl \
  --warmups 5 --repeats 3 \
  --artifact-revision b5f37287796e5be0ea3dab2e7430873fb3f73e49 \
  --artifact-sha256 da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4 \
  --runtime-version "llama.cpp c1d0e7a00" \
  --host-note "Apple M1 Pro, 16 GB"
```

Record the exact artifact revision, SHA-256, runtime version, host, and launch
flags beside any output used for a decision. Do not use private transcripts as
benchmark fixtures. Promotion to `evaluated_languages=("en",)` still requires
the full held-out corpus and human review described in the implementation plan.

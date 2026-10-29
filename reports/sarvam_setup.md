# Sarvam-30B: how the model file was made and tested

Recorded 2026-10-20 on an Apple M4 Pro, 24 GB (Metal GPU budget 19.07 GB).

## Why we quantized it ourselves

- llama.cpp loads Sarvam-30B as the existing `bailingmoe2` architecture with a Sarvam-specific
  tokenizer (`sarvam-moe`), added in [ggml-org/llama.cpp#20275](https://github.com/ggml-org/llama.cpp/pull/20275) (merged 2026-05-09).
- The official 4-bit file (`sarvamai/sarvam-30b-gguf`, Q4_K_M, 19.57 GB) is too large for the GPU budget (see `model_fit.md`).
- A community 3-bit file (`DevQuasar/sarvamai.sarvam-30b-GGUF`, Q3_K_M) was converted before #20275 and is labelled
  `sarvam_moe`; llama.cpp b11669 rejects it with `unknown model architecture: 'sarvam_moe'`. **Do not use it.**

## Provenance

| Step | Detail |
|---|---|
| Source | `Sumitc13/sarvam-30b-GGUF` → `sarvam-30B-full-BF16.gguf` (uploaded 2026-05-09, header: `bailingmoe2`, tokenizer pre `sarvam-moe`) |
| Source SHA-256 | `b5dbeb1083a93694a76b5e244e60867239dbc38407e866bdabf6043464bf62f8` (matches Hugging Face LFS record) |
| Quantize | `llama-quantize sarvam-30B-full-BF16.gguf sarvam-30b-Q3_K_M.gguf Q3_K_M` (llama.cpp b11669, Homebrew) |
| Result | `models/sarvam-30b/sarvam-30b-Q3_K_M.gguf`, 15.68 GB, 3.90 bits per weight, 123 s |

## Serving

```bash
llama-server -m models/sarvam-30b/sarvam-30b-Q3_K_M.gguf --alias sarvam-30b-q3 \
  -ngl 99 -c 4096 --jinja --reasoning-budget 0 --host 127.0.0.1 --port 8080
```

`--reasoning-budget 0` is required. Sarvam is a reasoning model: without it, it spends the whole
answer budget "thinking", even when the chat template adds its `<|nothink|>` token
(`enable_thinking: false`). With it, Sarvam answers directly, like the other models (thinking is switched off for every model that allows it).

## Smoke test result

`python -m evaluation.smoke --model sarvam-30b-q3 --configs eng_Latn hin_Deva hin_Latn --n 10` → `smoke_sarvam-30b-q3.md`.

- Loads in about 15 s, fully on the GPU (`-ngl 99`); about 0.7 s per Belebele item.
- 30/30 answers were valid JSON letters.
- English 8/10, Hindi (Devanagari) 8/10, romanized Hindi 6/10. **10 items cannot support any conclusion**;
  this only shows the pipeline works.
- System memory free while loaded: 13%. Close other heavy apps during runs.

## Known caveats (for the limitations section)

- llama.cpp warns that `</s>` is not marked as a control token and that the EOS id is not in the end-of-generation set;
  related open issue: [ggml-org/llama.cpp#30879](https://github.com/ggml-org/llama.cpp/issues/30879) (byte-fallback tokens mislabelled).
  Rare characters in Indian scripts may tokenize slightly differently than in Sarvam's reference implementation.
- Sarvam runs at Q3_K_M while the other models run at ~Q4 (see the Llama 3.1 8B Q3_K_M control in `paper/paper.md`, section 3.3).

# Does each model fit on this Mac?

GPU memory budget: **19.07 GB** (Metal recommendedMaxWorkingSetSize). Context assumed: 4096 tokens.
Only one model is loaded at a time, so each row is checked on its own.

| Model | Runtime / quantization | Weights (GB) | KV + buffers (GB) | Total (GB) | Headroom (GB) | Fits on GPU? |
|---|---|---:|---:|---:|---:|---|
| Gemma 3 12B | Ollama `gemma3:12b` | 7.31 | 1.80 | 9.11 | +9.96 | yes |
| Qwen3 8B | Ollama `qwen3:8b` | 5.23 | 1.80 | 7.03 | +12.04 | yes |
| Llama 3.1 8B | Ollama `llama3.1:8b` | 4.92 | 1.80 | 6.72 | +12.35 | yes |
| gpt-oss 20B | Ollama `gpt-oss:20b` | 13.79 | 1.80 | 15.59 | +3.48 | yes |
| Sarvam-30B | llama.cpp, Q4_K_M (official) | 19.57 | 1.08 | 20.65 | -1.58 | **no** |
| Sarvam-30B | llama.cpp, Q3_K_M (size reference) | 15.68 | 1.08 | 16.76 | +2.31 | yes |

Sarvam-30B KV cache is small (80 MB at 4096 tokens) because it has 19 layers and 4 KV heads; the weights are what decide the fit.

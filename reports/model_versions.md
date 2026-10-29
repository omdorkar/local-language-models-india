# Exact model versions

Runtimes: Ollama 0.32.5; llama.cpp version: 0.7.0 (build 11669, commit 620d2225c).
The paper (section 3.3, Table 2) lists the first 12 characters of each digest.

| Model | Tag / runtime | Digest | Size (GB) | Arch | Params | Quant | Capabilities | Licence |
|---|---|---|---:|---|---|---|---|---|
| Gemma 3 12B | `gemma3:12b` | `b92b218582c521ba287…` | 8.16 | gemma3 | 11.8B | Q4_K_M | completion, vision | Gemma Terms of Use  |
| Qwen3 8B | `qwen3:8b` | `500a1f067a9f782620b…` | 5.23 | qwen3 | 8.2B | Q4_K_M | completion, tools, thinking | Apache License |
| Llama 3.1 8B | `llama3.1:8b` | `46e0c10c039e0191193…` | 4.92 | llama | 8.0B | Q4_K_M | completion, tools | LLAMA 3.1 COMMUNITY LICENSE AGREEMENT |
| gpt-oss 20B | `gpt-oss:20b` | `f38aa0c53da5f8c49d0…` | 13.79 | gpt-oss | 20.9B | MXFP4 | completion, tools, thinking | Apache License |
| Sarvam-30B | `llama.cpp local GGUF` | `sha256:4cf4375650e3…` | 15.68 | bailingmoe2 | 32B (2.4B active) | Q3_K_M | completion, thinking | Apache-2.0 |

Full digests are in `model_versions.json`.

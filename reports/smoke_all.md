# Smoke test: all five models

Recorded 2026-10-21. The same 10 Belebele items (seed 0) in English, Hindi (Devanagari) and romanized Hindi;
prompt template `d3ee3d7bd6e7`; temperature 0. Exact model versions: `model_versions.md`. Per-model detail: `smoke_<model>.md`.

**10 items cannot support any conclusion.** This shows only that every model loads, answers in the required JSON format,
and how long a run will take.

| Model | English | Hindi | Romanized Hindi | Hindi token ratio | Romanized token ratio | s / item | Invalid (of 30) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Gemma 3 12B | 9 | 9 | 9 | 1.19 | 1.38 | 2.44 | 0 |
| gpt-oss 20B | 10 | 7 | 8 | 2.01 | 1.44 | 3.95 | 0 |
| Llama 3.1 8B | 10 | 5 | 2 | 2.05 | 1.56 | 1.59 | 0 |
| Qwen3 8B | 9 | 7 | 5 | 3.30 | 1.55 | 1.99 | 0 |
| Sarvam-30B (Q3_K_M) | 8 | 8 | 6 | 1.19 | 1.38 | 0.69 | 0 |

Token ratio = mean prompt tokens for the Hindi version ÷ English version, from each runtime's own count. The English
instructions (~80 tokens) are identical in every prompt, so the ratio for the passage alone is higher than shown.
Gemma 3 and Sarvam gave identical counts on these Hindi items (both have 262,144-token vocabularies); in the main run their counts differ for several other scripts (`results/token_ratio.csv`).

## Settings that make each model answer directly

| Model | Runtime | Setting | Why |
|---|---|---|---|
| Gemma 3 12B, Llama 3.1 8B | Ollama | none needed | no thinking mode |
| Qwen3 8B | Ollama | `reasoning_effort: "none"` | otherwise it thinks and uses up the 32-token limit |
| gpt-oss 20B | Ollama | `reasoning_effort: "low"`, `max_tokens: 512` | reasoning cannot be switched off: `"none"` returns an empty answer |
| Sarvam-30B | llama-server | `--reasoning-budget 0` | `enable_thinking: false` alone does not stop it (see `sarvam_setup.md`) |

gpt-oss is therefore the only model that reasons briefly before answering; this goes in the limitations.
Ollama's reported output tokens for gpt-oss (6) do not include its reasoning text.

## Time budget for the main run (estimate)

Average across the five models: about 2.1 s per item per condition (≈10.7 s for one item on all five models).
The Belebele plan of 300 items × 18 conditions × 5 models would take about **16 hours**. Two nights, or 200 items
(≈11 hours), both fit; decide before the main run. Longer native-script prompts in other languages may
add time for Qwen3 and Llama, whose Hindi prompts were 2–3× longer than English.

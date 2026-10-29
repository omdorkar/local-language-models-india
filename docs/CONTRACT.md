# Internal contract: data flow between modules

Every module reads and writes these formats. Change this file first if a format must change.

## Python environment

- Main env: `.venv` (Python 3.11, `uv pip install -e ".[dev]"`). Run code as `.venv/bin/python -m evaluation.<module>`.
- Heavy or fragile extras (torch, fairseq, Multi-VALUE, …) go in a **separate** venv (`.venv-bridge`, `.venv-extra`)
  with its own `requirements-*.txt`. Never install them into `.venv`.
- Seed everywhere: **1909** (`evaluation.config.SEED`).

## Item samples — `data/derived/samples/<benchmark>.json`

```json
{"benchmark": "belebele", "seed": 1909, "items": ["<item_id>", ...]}
```
- Belebele `item_id` = `"<link>#<question_number>"` (unique across the 900 items).
- Global-MMLU `item_id` = `sample_id` (e.g. `"astronomy/test/12"`).
- Benchmarks: `belebele` (300), `gmmlu_lite` (all 400), `gmmlu` (300, full Global-MMLU).

## Conditions

A condition is one way of presenting the same item. Condition id format:

| Benchmark | Condition id | Meaning |
|---|---|---|
| belebele | `eng_Latn`, `hin_Deva`, `hin_Latn`, … | Belebele config, human-made |
| belebele | `hin_Latn+xlit` | romanized input converted to native script by IndicXlit (fix, H4) |
| belebele | `hin_Deva+mt_en` | native input machine-translated to English (fix, H5) |
| belebele | `eng_Latn+inde` | English rewritten as Indian English by Multi-VALUE (exploratory) |
| belebele | `hin_Latn+cm` | machine-made Hinglish (exploratory) |
| gmmlu_lite | `en`, `hi`, `bn`, `or` | Global-MMLU-Lite language config |
| gmmlu | `en`, `hi`, `bn`, `te`, `ne` | Global-MMLU language config |

Derived (machine-made) item text lives in `data/derived/conditions/<benchmark>/<condition_id>.jsonl`, one row per item,
with **the same fields as the source rows** (for Belebele: `flores_passage`, `question`, `mc_answer1..4`, `correct_answer_num`)
plus `item_id`, `source_condition`, `method`, `method_version`, and any validation scores. The harness treats a derived
condition exactly like a human one.

## Run output — `runs/<benchmark>/<model_key>.jsonl` (append-only, one row per item × condition)

| Field | Type | Notes |
|---|---|---|
| `benchmark` | str | `belebele` / `gmmlu_lite` / `gmmlu` |
| `model_key` | str | from `config.MODELS` (`gemma3-12b`, …) or `llama3.1-8b-q3` for the sensitivity run |
| `model_tag` | str | Ollama tag or llama-server alias |
| `condition` | str | condition id above |
| `lang` | str | ISO 639-1 (`en`, `hi`, …) |
| `script` | str | ISO 15924 (`Latn`, `Deva`, …) |
| `source` | str | `human` or `machine` |
| `item_id` | str | see samples |
| `gold` | str | `A`–`D` |
| `answer` | str \| null | parsed letter; null if invalid |
| `correct` | bool | `answer == gold` (invalid counts as wrong) |
| `raw` | str | raw model content |
| `prompt_tokens`, `completion_tokens` | int \| null | from the runtime's usage |
| `seconds` | float | wall time |
| `template_sha` | str | `prompts.TEMPLATE_SHA` or the Global-MMLU template's hash |
| `seed` | int | 1909 |
| `timestamp` | str | ISO 8601 UTC |

Resumability key: (`model_key`, `condition`, `item_id`). A row already present is never re-run.

Template calibration (for token tax): one row per (benchmark, model_key) with `condition = "__template__"`,
prompt built from the template with every field empty; `item_id = "__template__"`, `gold = ""`, `correct = false`.
Analysis must exclude these rows from accuracy.

## Analysis outputs

- `reports/results/*.csv` and `*.md` — tables; `reports/figures/*.png` (+ `.svg`) — figures.
- Primary hypothesis tests (H1-H5) as described in `paper/paper.md`, section 3.7.

## File ownership while agents work in parallel

Only the owner edits a file. Shared files (`config.py`, `pyproject.toml`,
`docs/CONTRACT.md`) are edited by the orchestrator only — report needed changes instead.

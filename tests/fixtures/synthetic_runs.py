"""Synthetic runs/*.jsonl generator following docs/CONTRACT.md, with planted effects.

Used only for tests and for smoke-testing the analysis layer; never writes into the real runs/ directory unless
the caller points it there (do not).

Planted truth (see PLANTED below):
  * English accuracy 0.85 for every model.
  * Native-script gap = coef[model] * (token_ratio[lang] - 1), except languages in NO_GAP (gap exactly 0).
    gpt-oss-20b has coef 0 (no planted token-ratio relation; small random gaps instead).
  * Romanized gaps (vs native): hin 0.06, ben 0.10, urd 0.15, npi 0 (none).
  * Fix xlit: +0.05 for hin and ben, 0 for urd, npi.   Fix mt_en: acc = English acc - 0.02 (all languages).
  * llama3.1-8b-q3 = llama3.1-8b - 0.03 everywhere.
  * Some invalid answers (answer null).
  * Global-MMLU-Lite: non-English accuracy lower, extra 0.08 loss on culturally sensitive (CS) items.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.config import LANGUAGES, MODELS

N_ITEMS = 300
EN_ACC = 0.85
TOKEN_RATIO = {"hi": 1.5, "bn": 2.5, "ur": 1.8, "ne": 1.6, "mr": 1.7, "te": 3.5, "ta": 3.8, "gu": 3.0,
               "kn": 3.6, "ml": 4.0, "pa": 2.8, "or": 4.5, "as": 2.6}
COEF = {"gemma3-12b": 0.045, "qwen3-8b": 0.05, "llama3.1-8b": 0.055, "gpt-oss-20b": 0.0, "sarvam-30b": 0.02}
NO_GAP = {"mr", "gu"}  # planted: no gap in any model
ROMAN_GAP = {"hi": 0.06, "bn": 0.10, "ur": 0.15, "ne": 0.0}
XLIT_GAIN = {"hi": 0.05, "bn": 0.05, "ur": 0.0, "ne": 0.0}
MT_EN_COST = 0.02
INVALID_RATE = {"gpt-oss-20b": 0.05}
DEFAULT_INVALID = 0.02
BASE_TOKENS = 250
TEMPLATE_TOKENS = {"gemma3-12b": 60, "qwen3-8b": 55, "llama3.1-8b": 58, "gpt-oss-20b": 70, "sarvam-30b": 52,
                   "llama3.1-8b-q3": 58}
Q3_KEY = "llama3.1-8b-q3"
GMMLU_LITE_LANGS = ["en", "hi", "bn", "or"]
GMMLU_LANGS = ["en", "hi", "bn", "te", "ne"]


def planted_native_acc(model: str, iso: str) -> float:
    if iso in NO_GAP:
        return EN_ACC
    return EN_ACC - COEF.get(model, 0.03) * (TOKEN_RATIO[iso] - 1)


def planted_accuracy(model: str, condition: str) -> float:
    """True population accuracy of a Belebele condition for a model (before the q3 shift)."""
    if condition == "eng_Latn":
        return EN_ACC
    base, _, suffix = condition.partition("+")
    iso = next(i for i, l in LANGUAGES.items() if base in (l.belebele_native, l.belebele_latn))
    native = planted_native_acc(model, iso)
    is_latn = base.endswith("_Latn")
    acc = native - ROMAN_GAP[iso] if is_latn else native
    if suffix == "xlit":
        acc += XLIT_GAIN[iso]
    elif suffix == "mt_en":
        acc = EN_ACC - MT_EN_COST
    return acc


def belebele_conditions(with_fixes: bool = True) -> list[tuple[str, str, str, str]]:
    """[(condition, iso, script, source)]"""
    out = [("eng_Latn", "en", "Latn", "human")]
    for iso, l in LANGUAGES.items():
        if iso == "en":
            continue
        script = l.belebele_native.split("_")[1]
        out.append((l.belebele_native, iso, script, "human"))
        if l.belebele_latn:
            out.append((l.belebele_latn, iso, "Latn", "human"))
    if with_fixes:
        for iso, l in LANGUAGES.items():
            if l.belebele_latn:
                out.append((f"{l.belebele_latn}+xlit", iso, l.belebele_native.split("_")[1], "machine"))
        for iso, l in LANGUAGES.items():
            if iso != "en":
                out.append((f"{l.belebele_native}+mt_en", iso, "Latn", "machine"))
    return out


def _correct_vector(rng, v_shared, acc, rho=0.7):
    """Correctness with marginal P = acc and paired correlation via a shared uniform."""
    fresh = rng.random(len(v_shared))
    v = np.where(rng.random(len(v_shared)) < rho, v_shared, fresh)
    return v < acc


def _row(bench, model, cond, iso, script, source, item, gold, correct, invalid_rate, rng, tokens, seconds=1.0):
    letters = "ABCD"
    if rng.random() < invalid_rate:
        ans, corr = None, False
    elif correct:
        ans, corr = gold, True
    else:
        ans = rng.choice([x for x in letters if x != gold])
        corr = False
    return {"benchmark": bench, "model_key": model, "model_tag": f"tag:{model}", "condition": cond, "lang": iso,
            "script": script, "source": source, "item_id": item, "gold": gold, "answer": ans, "correct": corr,
            "raw": json.dumps({"answer": ans}), "prompt_tokens": int(tokens), "completion_tokens": 7,
            "seconds": seconds, "template_sha": "d3ee3d7bd6e7", "seed": 1909,
            "timestamp": "2026-10-21T00:00:00Z"}


def write_synthetic_runs(root: Path, n_items: int = N_ITEMS, seed: int = 7, models: list[str] | None = None,
                         with_fixes: bool = True, fix_models: list[str] | None = None, with_templates: bool = True,
                         with_q3: bool = True, with_gmmlu: bool = True, drop_template_for: list[str] | None = None) -> dict:
    """Write root/runs/<benchmark>/<model>.jsonl (+ root/raw/global_mmlu_lite/<lang>/test-...parquet).

    Returns {"runs": Path, "raw": Path, "items": [...]}.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    runs, raw = root / "runs", root / "raw"
    models = list(models or [m.key for m in MODELS])
    fix_models = models if fix_models is None else fix_models
    all_models = models + ([Q3_KEY] if with_q3 and "llama3.1-8b" in models else [])
    items = [f"https://example.org/passage{i // 4}#{i % 4 + 1}" for i in range(n_items)]
    golds = rng.choice(list("ABCD"), size=n_items)
    base_tok = rng.integers(BASE_TOKENS - 60, BASE_TOKENS + 60, size=n_items)

    v_by_model: dict = {}
    for model in all_models:
        is_q3 = model == Q3_KEY
        src_model = "llama3.1-8b" if is_q3 else model
        shift = -0.03 if is_q3 else 0.0
        inv = INVALID_RATE.get(model, DEFAULT_INVALID)
        # per model, shared across conditions -> paired correlation; q3 shares the Q4 model's draws
        v_shared = v_by_model["llama3.1-8b"] if is_q3 else rng.random(n_items)
        v_by_model[model] = v_shared
        rows = []
        tmpl = TEMPLATE_TOKENS[model]
        for cond, iso, script, source in belebele_conditions(with_fixes):
            if ("+xlit" in cond or "+mt_en" in cond) and src_model not in fix_models:
                continue
            acc = planted_accuracy(src_model, cond) + shift
            if model == "gpt-oss-20b" and cond not in ("eng_Latn",):
                acc += rng.normal(0, 0.02)  # unstructured per-language noise instead of a token-ratio relation
            corr = _correct_vector(rng, v_shared, acc)
            ratio = 1.0 if cond == "eng_Latn" else (TOKEN_RATIO[iso] if "+mt_en" not in cond else 1.05)
            if cond.endswith("_Latn") or "_Latn+" in cond:
                ratio = ratio * 0.8 if iso != "en" else ratio
            for k, item in enumerate(items):
                tokens = tmpl + base_tok[k] * ratio
                rows.append(_row("belebele", model, cond, iso, script, source, item, golds[k], corr[k], inv, rng, tokens))
        if with_templates and model not in (drop_template_for or []):
            rows.append({"benchmark": "belebele", "model_key": model, "model_tag": f"tag:{model}", "condition": "__template__",
                         "lang": "en", "script": "Latn", "source": "human", "item_id": "__template__", "gold": "",
                         "answer": None, "correct": False, "raw": "", "prompt_tokens": tmpl, "completion_tokens": 0,
                         "seconds": 0.1, "template_sha": "d3ee3d7bd6e7", "seed": 1909, "timestamp": "2026-10-21T00:00:00Z"})
        _dump(runs / "belebele" / f"{model}.jsonl", rows)

    # ---- Global-MMLU(-Lite)
    if with_gmmlu:
        n_lite = 400
        lite_ids = [f"subj{i % 20}/test/{i}" for i in range(n_lite)]
        cs = np.array(["CS" if i % 2 == 0 else "CA" for i in range(n_lite)])
        lite_gold = rng.choice(list("ABCD"), size=n_lite)
        for lang in GMMLU_LITE_LANGS:
            d = pd.DataFrame({"sample_id": lite_ids, "cultural_sensitivity_label": cs, "answer": lite_gold})
            (raw / "global_mmlu_lite" / lang).mkdir(parents=True, exist_ok=True)
            d.to_parquet(raw / "global_mmlu_lite" / lang / "test-00000-of-00001.parquet")
        gm_ids = [f"subj{i % 20}/test/{i}" for i in range(300)]
        gm_gold = rng.choice(list("ABCD"), size=300)
        for model in models:
            v_lite, v_gm = rng.random(n_lite), rng.random(300)
            rows_lite, rows_gm = [], []
            for lang in GMMLU_LITE_LANGS:
                for k, item in enumerate(lite_ids):
                    acc = 0.7 if lang == "en" else 0.62 - (0.08 if cs[k] == "CS" else 0.0)
                    c = (np.where(rng.random() < 0.7, v_lite[k], rng.random()) < acc)
                    rows_lite.append(_row("gmmlu_lite", model, lang, lang, "Latn" if lang == "en" else "Deva", "human", item,
                                          lite_gold[k], bool(c), DEFAULT_INVALID, rng, 200))
            for lang in GMMLU_LANGS:
                for k, item in enumerate(gm_ids):
                    acc = 0.65 if lang == "en" else 0.55
                    c = (np.where(rng.random() < 0.7, v_gm[k], rng.random()) < acc)
                    rows_gm.append(_row("gmmlu", model, lang, lang, "Latn" if lang == "en" else "Deva", "machine", item,
                                        gm_gold[k], bool(c), DEFAULT_INVALID, rng, 200))
            _dump(runs / "gmmlu_lite" / f"{model}.jsonl", rows_lite)
            _dump(runs / "gmmlu" / f"{model}.jsonl", rows_gm)
    return {"runs": runs, "raw": raw, "items": items}


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            r = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":  # python tests/fixtures/synthetic_runs.py <outdir>
    import sys

    out = Path(sys.argv[1])
    write_synthetic_runs(out)
    print("wrote", out)

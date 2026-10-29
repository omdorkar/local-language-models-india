"""Belebele loading helpers shared by the bridge modules (paths follow docs/CONTRACT.md)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/belebele/data"
SAMPLE = ROOT / "data/derived/samples/belebele.json"
COND_DIR = ROOT / "data/derived/conditions/belebele"
CACHE_DIR = ROOT / "data/derived/cache"
REPORT_DIR = ROOT / "reports/bridge"

TEXT_FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]

# Belebele config -> (ISO 639-1 used by IndicXlit, English name). Romanized configs for H4.
LATN_TO_XLIT = {"hin_Latn": "hi", "ben_Latn": "bn", "urd_Latn": "ur", "npi_Latn": "ne"}
LATN_TO_NATIVE = {"hin_Latn": "hin_Deva", "ben_Latn": "ben_Beng", "urd_Latn": "urd_Arab", "npi_Latn": "npi_Deva"}
# Config codes are NLLB language codes (checked against the NLLB tokenizer).
NATIVE_CONFIGS = [
    "hin_Deva", "ben_Beng", "urd_Arab", "npi_Deva", "mar_Deva", "tel_Telu", "tam_Taml",
    "guj_Gujr", "kan_Knda", "mal_Mlym", "pan_Guru", "ory_Orya", "asm_Beng",
]


def item_id(row: dict) -> str:
    return f"{row['link']}#{row['question_number']}"


def sample_ids() -> list[str]:
    return json.loads(SAMPLE.read_text())["items"]


def load_config(config: str, ids: list[str] | None = None) -> list[dict]:
    """Rows of one Belebele config for the sampled items, in sample order, with `item_id` added."""
    ids = ids if ids is not None else sample_ids()
    by_id = {}
    for line in (RAW / f"{config}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["item_id"] = item_id(r)
            by_id[r["item_id"]] = r
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{config}: {len(missing)} sampled items missing, e.g. {missing[0]}")
    return [by_id[i] for i in ids]


def write_condition(condition: str, rows: list[dict]) -> Path:
    COND_DIR.mkdir(parents=True, exist_ok=True)
    p = COND_DIR / f"{condition}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p

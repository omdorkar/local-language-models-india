"""Draw the item samples (sampling rule in paper/paper.md section 3.1) and write data/derived/samples/<benchmark>.json.

    python -m evaluation.samples
"""

import json

import numpy as np
import pandas as pd

from . import DATA_RAW, ROOT
from .config import SEED

SAMPLES = ROOT / "data" / "derived" / "samples"
N_ITEMS = 300


def belebele_item_id(link: str, question_number: int) -> str:
    return f"{link}#{int(question_number)}"


def sample_belebele() -> list[str]:
    """Shuffle passages; add whole passages while the total stays <= 300, skipping any that would overshoot."""
    eng = pd.read_json(DATA_RAW / "belebele" / "data" / "eng_Latn.jsonl", lines=True)
    passages = list(dict.fromkeys(eng.link))  # order of first appearance in the file
    questions = eng.groupby("link", sort=False).question_number.apply(list)
    chosen: list[str] = []
    for i in np.random.default_rng(SEED).permutation(len(passages)):
        qs = questions[passages[i]]
        if len(chosen) + len(qs) <= N_ITEMS:
            chosen += [belebele_item_id(passages[i], q) for q in qs]
        if len(chosen) == N_ITEMS:
            break
    assert len(chosen) == N_ITEMS
    return chosen


def _gmmlu_ids(root: str, langs: list[str]) -> list[str]:
    sets = [set(pd.read_parquet(DATA_RAW / root / l / "test-00000-of-00001.parquet").sample_id) for l in langs]
    return sorted(set.intersection(*sets))


def sample_gmmlu_lite() -> list[str]:
    return _gmmlu_ids("global_mmlu_lite", ["en", "hi", "bn", "or"])


def sample_gmmlu() -> list[str]:
    ids = _gmmlu_ids("global_mmlu", ["en", "hi", "bn", "te", "ne"])
    picked = np.random.default_rng(SEED).choice(len(ids), size=N_ITEMS, replace=False)
    return [ids[i] for i in sorted(picked)]


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, fn in [("belebele", sample_belebele), ("gmmlu_lite", sample_gmmlu_lite), ("gmmlu", sample_gmmlu)]:
        items = fn()
        (SAMPLES / f"{name}.json").write_text(json.dumps({"benchmark": name, "seed": SEED, "items": items}, indent=1))
        print(f"{name}: {len(items)} items")


if __name__ == "__main__":
    main()

"""Shared helpers for the exploratory conditions (Indian English, machine Hinglish)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import _env

ROOT = _env.ROOT
SEED = _env.SEED
RAW = ROOT / "data" / "raw" / "belebele" / "data"
SAMPLE = ROOT / "data" / "derived" / "samples" / "belebele.json"
OUT_DIR = ROOT / "data" / "derived" / "conditions" / "belebele"
REPORT_DIR = ROOT / "reports" / "extra"
TEXT_FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
LOW_SIM = 0.85


def item_id(row: dict) -> str:
    return f"{row['link']}#{row['question_number']}"


def sample_ids() -> list[str]:
    return json.loads(SAMPLE.read_text())["items"]


def load_rows(cond: str, ids: list[str] | None = None) -> dict[str, dict]:
    ids = ids if ids is not None else sample_ids()
    want = set(ids)
    out = {}
    with open(RAW / f"{cond}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            iid = item_id(r)
            if iid in want:
                out[iid] = r
    missing = want - set(out)
    assert not missing, f"{cond}: {len(missing)} sampled ids missing"
    return out


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def edit_ratio(a: str, b: str) -> float:
    """Character-level Levenshtein distance / max(len(a), len(b)); 0 = identical, 1 = completely different."""
    from rapidfuzz.distance import Levenshtein

    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


_labse = None


def labse():
    global _labse
    if _labse is None:
        from sentence_transformers import SentenceTransformer

        _labse = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    return _labse


def labse_cos(a: list[str], b: list[str]) -> list[float]:
    m = labse()
    ea = m.encode(a, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    eb = m.encode(b, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return [float(x) for x in (ea * eb).sum(axis=1)]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

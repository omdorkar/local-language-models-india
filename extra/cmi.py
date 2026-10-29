"""Code-Mixing Index (Das & Gambäck 2014) and the COMI-LINGUA reference distribution.

CMI(utterance) = 100 * (1 - max_i(w_i) / (n - u))  if n > u, else 0,
n = tokens, u = language-independent tokens, w_i = tokens tagged with language i.
(Das & Gambäck, "Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text", ICON 2014, eq. 2;
 https://aclanthology.org/W14-5152/)
"""
from __future__ import annotations

import ast
import re
from collections import Counter

import numpy as np
import pandas as pd

from .common import ROOT

LID_CSV = ROOT / "data" / "raw" / "comi_lingua" / "LID_test.csv"
ANN_COLS = ["Annotated by: Annotator 1", "Annotated by: Annotator 2", "Annotated by: Annotator 3"]
LANGS = ("hi", "en")  # 'ot' (other: punctuation, numbers, symbols, ...) is language-independent


def cmi(tags) -> float:
    n = len(tags)
    u = sum(t not in LANGS for t in tags)
    if n <= u:
        return 0.0
    c = Counter(t for t in tags if t in LANGS)
    return 100.0 * (1.0 - max(c.values()) / (n - u))


def _parse(x):
    return [(d["key"], d["value"]) for d in ast.literal_eval(x)]


def _script_frac_latin(tokens) -> float:
    lat = sum(bool(re.search("[A-Za-z]", t)) and not re.search("[ऀ-ॿ]", t) for t in tokens)
    dev = sum(bool(re.search("[ऀ-ॿ]", t)) for t in tokens)
    return lat / (lat + dev) if lat + dev else float("nan")


def load_comi() -> pd.DataFrame:
    """One row per COMI-LINGUA LID test sentence with majority-vote token tags (>=2 of 3 annotators; 3-way split -> annotator 1).

    Columns: sentence, n_tokens, cmi (majority), cmi_a1..a3 (each annotator), latin_frac (share of alphabetic tokens in Latin script).
    Sentences where the three annotators' token counts disagree (5 of 5000) are dropped.
    """
    df = pd.read_csv(LID_CSV)
    out = []
    for _, r in df.iterrows():
        ann = [_parse(r[c]) for c in ANN_COLS]
        if not (len(ann[0]) == len(ann[1]) == len(ann[2])):
            continue
        toks = [k for k, _ in ann[0]]
        maj = []
        for i in range(len(toks)):
            votes = Counter(a[i][1] for a in ann)
            v, c = votes.most_common(1)[0]
            maj.append(v if c >= 2 else ann[0][i][1])
        out.append({
            "sentence": r["Sentences"], "n_tokens": len(toks), "cmi": cmi(maj),
            "cmi_a1": cmi([t for _, t in ann[0]]), "cmi_a2": cmi([t for _, t in ann[1]]), "cmi_a3": cmi([t for _, t in ann[2]]),
            "frac_en": float(np.mean([t == "en" for t in maj])), "frac_ot": float(np.mean([t == "ot" for t in maj])),
            "latin_frac": _script_frac_latin(toks),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = load_comi()
    print(len(d), d.cmi.describe())

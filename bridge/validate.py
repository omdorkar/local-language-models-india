"""Validation of the bridge outputs against Belebele's own human-made parallel text.

* xlit: character error rate (CER) of IndicXlit output vs the native-script Belebele config (e.g. hin_Latn+xlit vs hin_Deva).
* mt:   chrF and BLEU (sacrebleu) of NLLB English output vs Belebele's eng_Latn text of the same item.
"""
from __future__ import annotations

import csv
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

from . import data as D

FIELD_GROUPS = {
    "passage": ["flores_passage"],
    "question": ["question"],
    "options": ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"],
}
_DROP = re.compile("[​‌‍‎‏ً-ٰٟـ]")  # ZW chars, Arabic diacritics, tatweel
_ARABIC_VARIANTS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ہ", "ھ": "ھ"})


def norm(text: str, strip_punct: bool = True) -> str:
    """NFC, drop zero-width joiners / Arabic diacritics, unify Arabic/Urdu letter variants, optionally drop punctuation+symbols."""
    t = unicodedata.normalize("NFC", text)
    t = _DROP.sub("", t).translate(_ARABIC_VARIANTS)
    t = "".join(str(unicodedata.decimal(c)) if unicodedata.category(c) == "Nd" else c for c in t)  # native digits -> ASCII
    if strip_punct:
        t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def edit_stats(hyp: str, ref: str, strip_punct: bool = True) -> tuple[int, int, int, int]:
    """(char edits, ref chars, word edits, ref words)"""
    h, r = norm(hyp, strip_punct), norm(ref, strip_punct)
    hw, rw = h.split(), r.split()
    return Levenshtein.distance(h, r), len(r), Levenshtein.distance(hw, rw), len(rw)


def _write(name: str, header: list[str], rows: list[dict], title: str, notes: list[str]) -> None:
    D.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (D.REPORT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {title}", ""] + notes + ["", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in header) + " |")
    (D.REPORT_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def xlit_validation(ids: list[str] | None = None) -> list[dict]:
    import json

    ids = ids or D.sample_ids()
    out, row_scores = [], {}
    for latn, native in D.LATN_TO_NATIVE.items():
        hyp_rows = [json.loads(l) for l in (D.COND_DIR / f"{latn}+xlit.jsonl").read_text(encoding="utf-8").splitlines()]
        ref = {r["item_id"]: r for r in D.load_config(native, ids)}
        acc = {}  # group -> [ce, rc, we, rw, ce_raw, rc_raw, n]
        for row in hyp_rows:
            tot = [0, 0]
            for g, fields in {**FIELD_GROUPS, "all": D.TEXT_FIELDS}.items():
                for f in fields:
                    ce, rc, we, rw = edit_stats(row[f], ref[row["item_id"]][f])
                    ce_raw, rc_raw, _, _ = edit_stats(row[f], ref[row["item_id"]][f], strip_punct=False)
                    a = acc.setdefault(g, [0] * 7)
                    for i, v in enumerate((ce, rc, we, rw, ce_raw, rc_raw, 1)):
                        a[i] += v
                    if g == "all":
                        tot[0] += ce
                        tot[1] += rc
            row_scores[(latn, row["item_id"])] = tot[0] / max(tot[1], 1)
        for g in ["all", "passage", "question", "options"]:
            ce, rc, we, rw, ce_raw, rc_raw, n = acc[g]
            out.append({
                "config": f"{latn}+xlit", "reference": native, "field": g, "n_texts": n,
                "CER": round(100 * ce / rc, 2), "CER_with_punct": round(100 * ce_raw / rc_raw, 2),
                "WER": round(100 * we / rw, 2),
            })
    header = ["config", "reference", "field", "n_texts", "CER", "CER_with_punct", "WER"]
    _write(
        "xlit_validation", header, out,
        "IndicXlit validation: CER vs Belebele native-script text",
        [
            "Hypothesis: IndicXlit (Roman -> native) applied to the human-romanized Belebele item. Reference: Belebele's own native-script text for the same item.",
            "CER / WER in percent (micro-averaged edit distance / reference length) on 300 sampled items. `CER` ignores punctuation/symbols, zero-width characters and Arabic diacritics and "
            "unifies Arabic/Urdu letter variants and native digits (২০ = 20); `CER_with_punct` only applies NFC. `all` pools the six text fields; `options` pools mc_answer1..4. "
            "`n_texts` counts field texts (passages repeat across questions of the same passage).",
            "IMPORTANT: in Belebele the romanized *passage* is a faithful romanization of the native passage, but the romanized *question and options* are an independent "
            "(Hinglish-style) translation, not a transliteration of the native question/options (e.g. hin_Latn \"Passage ke hisab se\" vs hin_Deva \"परिच्छेद के अनुसार\"). "
            "So CER for `question`/`options` mostly measures lexical divergence between two human translations, not transliteration error; the `passage` CER is the clean transliteration-quality number.",
            "Caveat: the Latin Belebele text is itself a human romanization (spelling is free), and the native reference is one of several valid spellings, so CER is an upper bound on the transliteration error of the pipeline.",
        ],
    )
    return out, row_scores


def mt_validation(ids: list[str] | None = None) -> tuple[list[dict], dict]:
    import json

    import sacrebleu

    ids = ids or D.sample_ids()
    ref_rows = {r["item_id"]: r for r in D.load_config("eng_Latn", ids)}
    out, row_scores = [], {}
    chrf_m, bleu_m = sacrebleu.metrics.CHRF(), sacrebleu.metrics.BLEU()
    for cfg in D.NATIVE_CONFIGS:
        hyp_rows = [json.loads(l) for l in (D.COND_DIR / f"{cfg}+mt_en.jsonl").read_text(encoding="utf-8").splitlines()]
        groups = {"passage": ([], []), "question": ([], []), "options": ([], []), "all": ([], [])}
        seen_links = set()
        for row in hyp_rows:
            ref = ref_rows[row["item_id"]]
            hyp_all, ref_all = [], []
            for g, fields in FIELD_GROUPS.items():
                if g == "passage" and row["link"] in seen_links:
                    continue  # unique passages only (several questions share a passage)
                for f in fields:
                    groups[g][0].append(row[f])
                    groups[g][1].append(ref[f])
                    groups["all"][0].append(row[f])
                    groups["all"][1].append(ref[f])
            seen_links.add(row["link"])
            row_scores[(cfg, row["item_id"])] = chrf_m.sentence_score(
                " ".join(row[f] for f in D.TEXT_FIELDS), [" ".join(ref[f] for f in D.TEXT_FIELDS)]).score
        for g in ["all", "passage", "question", "options"]:
            hyp, ref = groups[g]
            out.append({
                "config": f"{cfg}+mt_en", "reference": "eng_Latn", "field": g, "n_texts": len(hyp),
                "chrF": round(chrf_m.corpus_score(hyp, [ref]).score, 2), "BLEU": round(bleu_m.corpus_score(hyp, [ref]).score, 2),
            })
    header = ["config", "reference", "field", "n_texts", "chrF", "BLEU"]
    _write(
        "mt_validation", header, out,
        "NLLB-200-distilled-1.3B validation: chrF / BLEU vs Belebele English",
        [
            "Hypothesis: NLLB translation (native config -> eng_Latn). Reference: Belebele's eng_Latn text for the same item (the FLORES-200 English source).",
            f"sacrebleu {sacrebleu.__version__}: chrF = `CHRF()` default (char 6-gram, no word n-grams, beta=2); BLEU = `BLEU()` default (13a tokenizer, exp smoothing). Corpus-level over the 300 sampled items; "
            "`passage` uses each unique passage once (165), `options` pools mc_answer1..4, `all` pools the three groups.",
        ],
    )
    return out, row_scores

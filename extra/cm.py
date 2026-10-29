"""Condition `hin_Latn+cm`: machine-made romanized Hindi-English code-mixed (Hinglish) Belebele items.

Method (simplified lexical-substitution variant of the alignment + Equivalence-Constraint pipeline behind Microsoft's GCM
toolkit, Rizvi et al., EACL 2021 Demos; Pratapa et al., ACL 2018):
  1. Word-align Belebele's parallel `hin_Latn` (matrix language, human romanization) and `eng_Latn` fields with SimAlign
     (Jalili Sabet et al., Findings of EMNLP 2020) using MuRIL (Khanuja et al. 2021), a multilingual encoder pre-trained on
     transliterated Indic text, so romanized Hindi gets usable embeddings (mBERT / XLM-R did not; see README).
  2. Keep only one-to-one aligned pairs (Argmax-in-both-directions pairs, or one-to-one pairs of the IterMax matching).
  3. Candidate = Hindi common-noun / adjective token whose English partner is a content NOUN/ADJ (spaCy), or a Hindi
     nominal followed by a light verb (kar-/ho-) whose partner is an English VERB (-> "<verb lemma> karna" pattern).
     Replacing a single token by its aligned partner in the same slot respects the Equivalence Constraint trivially.
  4. Mixing level control: for each unit (sentence of a passage / question / option) a target CMI is drawn from the empirical
     distribution of COMI-LINGUA (expert LID tags, majority of 3 annotators); randomly ordered candidates are substituted until
     the unit's CMI (Das & Gambäck 2014) is closest to the target.
Run from repo root:  .venv-extra/bin/python -m extra.cm
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from importlib.metadata import version

import numpy as np
import pandas as pd

from . import _env
from .cmi import cmi, load_comi
from .common import (LOW_SIM, OUT_DIR, REPORT_DIR, SEED, TEXT_FIELDS, edit_ratio, labse_cos, load_rows, sample_ids,
                     stable_seed, write_jsonl)

COND = "hin_Latn+cm"
ALIGN_CACHE = REPORT_DIR / "cm_alignment_cache.json"
ALIGN_MODEL = "google/muril-base-cased"
ALIGN_LAYER = 8
TOK_RE = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]")
SENT_END = {".", "?", "!", "।"}

# Hindi (romanized) function words / auxiliaries: never replaced and never counted as English
HI_STOP = set("""ke ki ka ko se me mein main par pe per ne hai hain tha the thi aur ya jo is us ye yeh wo woh kya kaun kis kisi kab kahan
kyun kyunki kaise nahi nahin na bhi to hi toh sabse bahut ek do teen char aur koi kuch sab kar kiya kiye kare karna karne karta karte
karti karein karen ho hua hue hui hoga hogi honge hona hone hota hote hoti gaya gayi gaye jata jate jati jaata jaate jaati ja jaa
raha rahe rahi diya di diye liya li liye sakta sakte sakti chahiye tak liye liya baad pehle pahle saath sath dwara dvara bhi
unka unke unki inka inke inki apna apne apni uska uske uski iska iske iski mera tera hamara tumhara unhe unhen inhe inhen
usne unhone isne inhone aap hum tum main mujhe tujhe hum apne jab tab agar lekin magar par parantu kyonki isliye isi usi""".split())
# English function-word homographs of Hindi tokens (do not treat as English when identical to the aligned English word)
HOMOGRAPH = set("the to so is us in me he hi na ne se ko ka ki ke par per or no am a i as at on an of do go pe ho den".split())
KAR_FORMS = set("karna karne karta karte karti karein karen kare karo karenge karegi karega kiya kiye kiyaa kijiye karke kar".split())
HI_VERB_FORMS = set("""gaya gayi gaye diya liya aaya aayi aaye mila mili mile bana bani bane rakha rakhi rakhe hua hue hui
kiya kiye gayi jata jaata jaate jate sakta sakte raha rahe rahi chala chali chale""".split())

_sp = None


def spacy_nlp():
    global _sp
    if _sp is None:
        import spacy

        _sp = spacy.load("en_core_web_sm")
    return _sp


_aligner = None


def aligner():
    global _aligner
    if _aligner is None:
        from simalign import SentenceAligner

        _aligner = SentenceAligner(model=ALIGN_MODEL, token_type="bpe", matching_methods="mai", device="cpu", layer=ALIGN_LAYER)
    return _aligner


def hi_tokens(text: str):
    return [(m.group(), m.start(), m.end()) for m in TOK_RE.finditer(text)]


def en_tokens(text: str):
    doc = spacy_nlp()(text)
    return [t for t in doc if not t.is_space]


def _word_embeddings(words: list[str]):
    """Mean-pooled sub-word embeddings (layer ALIGN_LAYER) per word; words truncated beyond 510 sub-words get NaN rows."""
    import torch

    L = aligner().embed_loader
    tok = L.tokenizer
    pieces = [tok.tokenize(w) or [tok.unk_token] for w in words]
    ids_ = []
    spans = []
    for p in pieces:
        spans.append((len(ids_), len(ids_) + len(p)))
        ids_ += tok.convert_tokens_to_ids(p)
    ids_ = ids_[:510]
    inp = torch.tensor([[tok.cls_token_id] + ids_ + [tok.sep_token_id]], device=L.device)
    with torch.no_grad():
        hid = L.emb_model(input_ids=inp, attention_mask=torch.ones_like(inp))["hidden_states"][L.layer][0, 1:-1].cpu().numpy()
    out = np.full((len(words), hid.shape[1]), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        if b <= len(ids_):
            out[i] = hid[a:b].mean(0)
    return out


def align_pair(hi_text: str, en_text: str, cache: dict):
    """SimAlign word-level 'Inter' (argmax-in-both-directions) alignment, MuRIL layer 8, mean-pooled sub-words.

    Returns {"pairs": [[hindi_idx, english_idx, sim, margin], ...]} where sim in [0,1] is (cos+1)/2 and margin is the pair's
    similarity minus the best competing similarity in its row/column.
    """
    key = hi_text + "\t" + en_text
    if key in cache:
        return cache[key]
    h = [t for t, _, _ in hi_tokens(hi_text)]
    e = [t.text for t in en_tokens(en_text)]
    X, Y = _word_embeddings(h), _word_embeddings(e)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    sim = (Xn @ Yn.T + 1.0) / 2.0
    sim = np.nan_to_num(sim, nan=-1.0)
    fwd, bwd = sim.argmax(1), sim.argmax(0)
    pairs = []
    for i, j in enumerate(fwd):
        if bwd[j] == i and sim[i, j] > 0:
            row = np.delete(sim[i], j)
            col = np.delete(sim[:, j], i)
            comp = max(row.max() if row.size else 0.0, col.max() if col.size else 0.0)
            pairs.append([int(i), int(j), round(float(sim[i, j]), 4), round(float(sim[i, j] - comp), 4)])
    res = {"pairs": pairs}
    cache[key] = res
    return res


def tag_units(htoks, etoks, al):
    """Language tag per Hindi-side token: 'en' | 'hi' | 'ot'. Used identically for the plain hin_Latn text (before) and the output.

    ot: no letters (punctuation, digits), or the aligned English token is a proper noun / named entity (language-independent in
    Das & Gambäck's sense). en: token identical (case-insensitive) to its aligned English token, or to any token / lemma of the parallel
    English field, length >= 3, not a Hindi/English homograph. hi: everything else.
    """
    en_set = {t.text.lower() for t in etoks if t.is_alpha} | {t.lemma_.lower() for t in etoks if t.is_alpha}
    partner = {h: e for h, e, _, _ in al["pairs"]}
    tags = []
    for i, (tok, _, _) in enumerate(htoks):
        if not re.search(r"[^\W\d_]", tok):
            tags.append("ot")
            continue
        low = tok.lower()
        p = partner.get(i)
        if p is not None and low == etoks[p].text.lower() and (etoks[p].pos_ == "PROPN" or etoks[p].ent_type_):
            tags.append("ot")
        elif len(low) >= 3 and low not in HOMOGRAPH and low not in HI_STOP and ((p is not None and low == etoks[p].text.lower()) or low in en_set):
            tags.append("en")
        else:
            tags.append("hi")
    return tags


SWITCH_WORDS = set("""and or but so because if when while although though however also very only often almost now then really
maybe instead whereas unless all some every any both several""".split())
SIM_MIN = 0.70     # alignment-confidence filter on the SimAlign similarity of a pair (0-1 scale), see README
MARGIN_MIN = 0.01  # ... and on its margin over the best competing similarity in its row/column


def candidates(htoks, etoks, al, tags):
    """List of dicts {hi_idx, replacement, rule, ...} for one aligned field."""
    from rapidfuzz import fuzz

    out = []
    for h, e, sim, margin in al["pairs"]:
        tok, s, t = htoks[h]
        et = etoks[e]
        low = tok.lower()
        if sim < SIM_MIN or margin < MARGIN_MIN:
            continue
        if tags[h] != "hi" or not tok.isalpha() or not tok.isascii():
            continue
        el0 = et.text.lower()
        if et.pos_ in ("CCONJ", "SCONJ", "ADV", "NUM", "DET") and (el0 in SWITCH_WORDS or (et.pos_ == "NUM" and el0.isalpha())):
            # frequent single-word switches of Hinglish: conjunctions, discourse adverbs, number words
            if low in HOMOGRAPH or low == el0 or len(low) < 2 or fuzz.ratio(low, el0) >= 75:
                continue
            rep = el0
            if tok[0].isupper():
                rep = rep[0].upper() + rep[1:]
            out.append({"hi_idx": h, "replacement": rep, "rule": "function_switch", "en_text": et.text, "pos": et.pos_,
                        "hindi": tok, "sim": sim, "margin": margin})
            continue
        if len(low) < 3 or low in HI_STOP:
            continue
        if et.pos_ == "PROPN" or et.ent_type_ or not et.is_alpha or len(et.text) < 3 or et.is_stop:
            continue
        el = et.text.lower()
        # already English (spelling variant / agglutinated suffix such as "reportsse"): nothing to substitute
        if low == el or low == et.lemma_.lower() or fuzz.ratio(low, el) >= 75 or el in low or low in el:
            continue
        if et.pos_ in ("NOUN", "ADJ"):
            rep, rule = (et.text if et.text.isupper() else el), "noun_adj"
        elif et.pos_ == "VERB" and h + 1 < len(htoks) and low not in HI_VERB_FORMS:
            nxt = htoks[h + 1][0].lower()
            if nxt in KAR_FORMS:
                rep, rule = et.lemma_.lower(), "light_verb_kar"          # "<verb> karna"
            else:
                continue
            if not rep.isalpha():
                continue
        else:
            continue
        if tok[0].isupper():
            rep = rep[0].upper() + rep[1:]
        out.append({"hi_idx": h, "replacement": rep, "rule": rule, "en_text": et.text, "pos": et.pos_, "hindi": tok,
                    "sim": sim, "margin": margin})
    return out


def units_of(htoks, field):
    """Index ranges (start, end) of CMI units: sentences for the passage, the whole field otherwise."""
    if field != "flores_passage":
        return [(0, len(htoks))]
    units, s = [], 0
    for i, (t, _, _) in enumerate(htoks):
        if t in SENT_END:
            units.append((s, i + 1))
            s = i + 1
    if s < len(htoks):
        units.append((s, len(htoks)))
    return units


def mix_field(hi_text, en_text, field, comi_cmis, cache):
    """Returns dict(out, replacements, unit stats)."""
    htoks = hi_tokens(hi_text)
    etoks = en_tokens(en_text)
    al = align_pair(hi_text, en_text, cache)
    tags = tag_units(htoks, etoks, al)
    cands = candidates(htoks, etoks, al, tags)
    by_pos = {}
    for c in cands:
        by_pos.setdefault(c["hi_idx"], c)
    chosen, ustats = [], []
    for (a, b) in units_of(htoks, field):
        utags = tags[a:b]
        if not utags:
            continue
        unit_text = " ".join(t for t, _, _ in htoks[a:b])
        rng = random.Random(stable_seed(SEED, "cm", unit_text))
        target = float(rng.choice(comi_cmis))
        pool = [by_pos[i] for i in range(a, b) if i in by_pos]
        rng.shuffle(pool)
        base = cmi(utags)
        allin = list(utags)
        for c in pool:
            allin[c["hi_idx"] - a] = "en"
        ceiling = cmi(allin)
        best_k, best_gap = 0, abs(base - target)
        cur = list(utags)
        for k, c in enumerate(pool, 1):
            cur[c["hi_idx"] - a] = "en"
            gap = abs(cmi(cur) - target)
            if gap < best_gap - 1e-9:
                best_k, best_gap = k, gap
        picked = pool[:best_k]
        final = list(utags)
        for c in picked:
            final[c["hi_idx"] - a] = "en"
        chosen += picked
        ustats.append({"field": field, "n_tokens": b - a, "n_candidates": len(pool), "target_cmi": target, "base_cmi": base,
                       "out_cmi": cmi(final), "n_replaced": len(picked), "max_cmi": ceiling})
    out, last = [], 0
    for c in sorted(chosen, key=lambda c: c["hi_idx"]):
        _, s, t = htoks[c["hi_idx"]]
        out.append(hi_text[last:s]); out.append(c["replacement"]); last = t
    out.append(hi_text[last:])
    out_tags = list(tags)
    for c in chosen:
        out_tags[c["hi_idx"]] = "en"
    return {"out": "".join(out), "field_cmi_base": cmi(tags), "field_cmi_out": cmi(out_tags),
            "replacements": [{k: c[k] for k in ("hindi", "replacement", "en_text", "pos", "rule", "hi_idx")} for c in sorted(chosen, key=lambda c: c["hi_idx"])],
            "units": ustats, "n_candidates_field": len(by_pos)}


def build():
    ids = sample_ids()
    en, hl = load_rows("eng_Latn", ids), load_rows("hin_Latn", ids)
    comi = load_comi()
    comi_cmis = comi.cmi.tolist()
    cache = json.loads(ALIGN_CACHE.read_text()) if ALIGN_CACHE.exists() else {}
    res = {}
    for n, iid in enumerate(ids):
        for f in TEXT_FIELDS:
            res[(iid, f)] = mix_field(hl[iid][f], en[iid][f], f, comi_cmis, cache)
        if n % 25 == 0:
            print(n, flush=True)
            ALIGN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            ALIGN_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    ALIGN_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return ids, en, hl, res, comi


def main():
    global SIM_MIN, MARGIN_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-min", type=float, default=SIM_MIN, help="min SimAlign similarity of a substituted pair (default %(default)s)")
    ap.add_argument("--margin-min", type=float, default=MARGIN_MIN, help="min similarity margin over the best competing pair (default %(default)s)")
    a = ap.parse_args()
    SIM_MIN, MARGIN_MIN = a.sim_min, a.margin_min
    from .cm_validate import finish

    ids, en, hl, res, comi = build()
    finish(ids, en, hl, res, comi)


if __name__ == "__main__":
    main()

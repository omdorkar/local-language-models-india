"""Condition `eng_Latn+inde`: Belebele English rewritten into Indian English with Multi-VALUE.

Ziems et al. (2023), Multi-VALUE, ACL 2023, https://aclanthology.org/2023.acl-long.44 ;
code: https://github.com/SALT-NLP/multi-value (PyPI: value-nlp). Dialect class: `Dialects.IndianDialect`
(eWAVE variety "Indian English", abbr IndE). Rules are applied per field with a field-specific fixed seed
(sha256 of "1909|item_id|field"), so output does not depend on processing order.

Run from repo root:  .venv-extra/bin/python -m extra.inde            (transform + validate)
                     .venv-extra/bin/python -m extra.inde --validate-only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter
from importlib.metadata import version

from . import _env
from .common import (LOW_SIM, OUT_DIR, REPORT_DIR, SEED, TEXT_FIELDS, edit_ratio, labse_cos, load_rows, sample_ids,
                     stable_seed, write_jsonl)

CACHE = REPORT_DIR / "inde_field_cache_v2.json"
COND = "eng_Latn+inde"


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save_cache(c: dict):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, ensure_ascii=False))


def install_guard(d):
    """One local guard on Multi-VALUE's `progressives` rule (feature 88).

    The stock rule rewrites every base-form / present-tense verb that has no auxiliary into "is/are/am + V-ing". spaCy tags
    imperatives ("Make sure ...") and the noun-phrase fragments that Belebele options often are ("Meat", "The spiky leaves") as
    such verbs, which yields "Are Making sure ..." / "Are Meating". The guard skips verbs that are the sentence root (or a
    base-form conjunct) and have no grammatical subject. All other rules are untouched.
    """
    orig = d.progressives

    def guarded(swap_do_aux=True):
        keep = []
        for t in d.tokens:
            has_subj = any(c.dep_ in ("nsubj", "nsubjpass", "csubj", "expl") for c in t.children)
            subjectless = (t.dep_ == "ROOT" or (t.dep_ == "conj" and t.tag_ == "VB")) and not has_subj
            if not (subjectless and t.tag_ in ("VB", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS")):
                keep.append(t)
        all_tokens = d.tokens
        d.tokens = keep
        try:
            orig(swap_do_aux=swap_do_aux)
        finally:
            d.tokens = all_tokens

    d.progressives = guarded
    return d


def _run(d, text: str, seed: int):
    """Transform one string; returns (out, rules list, status)."""
    random.seed(seed)
    out = d.transform(text)
    rules = [{"type": v["type"], "start": k[0], "end": k[1], "value": v["value"]} for k, v in d.executed_rules.items()]
    rules.sort(key=lambda r: (r["start"], r["end"]))
    return out, rules


def transform_field(d, text: str, seed: int):
    """Whole-field transform; on a Stanza/spaCy alignment error fall back to sentence-wise, then to identity."""
    try:
        out, rules = _run(d, text, seed)
        return {"out": out, "rules": rules, "status": "ok"}
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:200]
    # sentence-wise fallback (offsets are relative to each sentence; we store sentence index)
    sents = [s.text_with_ws for s in d.nlp(text).sents]
    outs, allrules, bad = [], [], 0
    for i, s in enumerate(sents):
        try:
            o, r = _run(d, s.strip(), seed + i)
            lead = s[: len(s) - len(s.lstrip())]
            trail = s[len(s.rstrip()):]
            outs.append(lead + o + trail)
            for x in r:
                x["sent"] = i
            allrules += r
        except Exception:  # noqa: BLE001
            outs.append(s)
            bad += 1
    return {"out": "".join(outs), "rules": allrules, "status": f"sentence_fallback({bad}/{len(sents)} sentences unchanged; whole-field error: {err})"}


def build(args):
    _env.patch_for_multivalue()
    from multivalue import Dialects

    ids = sample_ids()
    src = load_rows("eng_Latn", ids)
    d = install_guard(Dialects.IndianDialect(seed=SEED))
    print(d, flush=True)
    cache = _load_cache()
    t0 = time.time()
    for n, iid in enumerate(ids):
        for f in TEXT_FIELDS:
            key = f"{iid}|{f}"
            text = src[iid][f]
            if key in cache and cache[key]["src"] == text:
                continue
            res = transform_field(d, text, stable_seed(SEED, iid, f))
            res["src"] = text
            cache[key] = res
        if n % 10 == 0:
            _save_cache(cache)
            print(f"{n}/{len(ids)} items, {time.time() - t0:.0f}s", flush=True)
    _save_cache(cache)


def assemble():
    ids = sample_ids()
    src = load_rows("eng_Latn", ids)
    cache = _load_cache()
    rows = []
    for iid in ids:
        s = src[iid]
        r = dict(s)
        rules_by_field, status = {}, {}
        for f in TEXT_FIELDS:
            c = cache[f"{iid}|{f}"]
            r[f] = c["out"]
            rules_by_field[f] = c["rules"]
            if c["status"] != "ok":
                status[f] = c["status"]
        # options must stay distinguishable: if a transform made two options identical, revert the later one
        reverted = []
        opts = ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
        for i, f in enumerate(opts):
            if any(r[f].strip().lower() == r[g].strip().lower() for g in opts[:i]):
                r[f] = s[f]
                rules_by_field[f] = []
                reverted.append(f)
        distinct = len({r[f].strip().lower() for f in opts}) == 4
        n_rules = Counter()
        for f, rl in rules_by_field.items():
            for x in rl:
                n_rules[x["type"]] += 1
        r.update({
            "item_id": iid, "dialect": COND, "source_condition": "eng_Latn",
            "method": "multivalue.Dialects.IndianDialect (eWAVE Indian English; Ziems et al. 2023) + local subjectless-verb guard on the progressives rule + LaBSE>=0.85 field gate",
            "method_version": f"value-nlp=={version('value-nlp')}; spacy=={version('spacy')} en_core_web_sm=={version('en-core-web-sm')}; stanza=={version('stanza')}; seed={SEED} (per-field sha256 seed)",
            "rules_fired": dict(n_rules), "rules_detail": rules_by_field,
            "n_rules_fired": sum(n_rules.values()),
            "fallback_status": status, "reverted_option_fields": reverted, "options_distinct": distinct,
        })
        rows.append((s, r))
    return rows


def validate(pairs):
    """Score raw Multi-VALUE output, apply the LaBSE gate (field reverted to the original if cos < 0.85), write outputs."""
    import pandas as pd

    flat = [(r["item_id"], f, s[f], r[f]) for s, r in pairs for f in TEXT_FIELDS]
    cos = labse_cos([x[2] for x in flat], [x[3] for x in flat])
    recs = []
    for (iid, f, a, b), c in zip(flat, cos):
        recs.append({"item_id": iid, "field": f, "changed": a != b, "edit_ratio": edit_ratio(a, b), "labse_cos": c,
                     "low_sim": c < LOW_SIM, "gated_reverted": bool(a != b and c < LOW_SIM),
                     "final_changed": bool(a != b and c >= LOW_SIM), "final_edit_ratio": edit_ratio(a, b) if c >= LOW_SIM else 0.0,
                     "len_src": len(a), "len_out": len(b)})
    df = pd.DataFrame(recs)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "inde_validation_fields.csv", index=False)

    rows = []
    for s, r in pairs:
        sub = df[df.item_id == r["item_id"]].set_index("field")
        r["mv_raw_fields"] = {f: r[f] for f in TEXT_FIELDS}  # unfiltered Multi-VALUE output
        gated = []
        for f in TEXT_FIELDS:
            if bool(sub.loc[f, "gated_reverted"]):
                r[f] = s[f]
                gated.append(f)
        r["gate_reverted_fields"] = gated
        opts = ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
        r["options_distinct"] = len({r[f].strip().lower() for f in opts}) == 4
        r["validation"] = {
            "labse_cos_raw": {f: round(float(sub.loc[f, "labse_cos"]), 4) for f in TEXT_FIELDS},
            "edit_ratio_raw": {f: round(float(sub.loc[f, "edit_ratio"]), 4) for f in TEXT_FIELDS},
            "changed_raw": {f: bool(sub.loc[f, "changed"]) for f in TEXT_FIELDS},
            "changed_final": {f: bool(sub.loc[f, "final_changed"]) for f in TEXT_FIELDS},
            "any_changed_final": bool(sub.final_changed.any()),
            "min_labse_cos_raw": round(float(sub.labse_cos.min()), 4),
            "flag_low_sim": bool(sub.low_sim.any()),
            "gate": f"field reverted to original if LaBSE cos(original, Multi-VALUE output) < {LOW_SIM}",
        }
        rows.append(r)
    write_jsonl(OUT_DIR / f"{COND}.jsonl", rows)

    item_df = pd.DataFrame([{
        "item_id": r["item_id"], "any_changed_final": r["validation"]["any_changed_final"], "min_labse_cos": r["validation"]["min_labse_cos_raw"],
        "flag_low_sim": r["validation"]["flag_low_sim"], "n_rules_fired": r["n_rules_fired"], "n_fields_reverted": len(r["gate_reverted_fields"]),
        "options_distinct": r["options_distinct"], "reverted_collision": ",".join(r["reverted_option_fields"]),
        "fallback": ";".join(r["fallback_status"]),
    } for r in rows])
    item_df.to_csv(REPORT_DIR / "inde_validation_items.csv", index=False)
    rule_counts, rule_items = Counter(), Counter()
    for r in rows:
        for k, v in r["rules_fired"].items():
            rule_counts[k] += v
            rule_items[k] += 1
    rc = pd.DataFrame([{"rule": k, "n_fired": rule_counts[k], "n_items": rule_items[k]} for k in rule_counts]).sort_values("n_fired", ascending=False)
    rc.to_csv(REPORT_DIR / "inde_rule_counts.csv", index=False)
    return rows, df, item_df, rc


def report(rows, df, item_df, rc):
    L = []
    n = len(rows)
    n_fb = int((item_df.fallback.astype(str).str.len() > 0).sum())
    L.append("# Indian English (`eng_Latn+inde`) validation\n")
    L.append("*Exploratory condition.* Method: Multi-VALUE (Ziems et al., 2023, ACL; https://aclanthology.org/2023.acl-long.44), "
             "`Dialects.IndianDialect` = eWAVE variety \"Indian English\" (IndE), applied to the 300 sampled Belebele English items "
             f"(passage, question, 4 options). Package `{rows[0]['method_version']}`. No LLM is involved.\n")
    L.append("**Quality gate.** Multi-VALUE is rule-based over spaCy/Stanza parses and misfires on the short fragments that Belebele options are "
             "(for example \"Meat\" -> \"Are Meating\", \"Police\" -> \"Polouses\"). Every field whose LaBSE cosine to the original is below "
             f"{LOW_SIM} is therefore reverted to the original English in the condition file; the unfiltered Multi-VALUE output of every field is kept in "
             "`mv_raw_fields` of each row. Statistics below are given for the raw output first, then for the final (gated) file.\n")
    L.append("## Raw Multi-VALUE output\n")
    L.append(f"- Rows: **{n}**; items with at least one changed field: **{(df.groupby('item_id').changed.any()).mean():.1%}**")
    L.append(f"- Items flagged (any field LaBSE cosine < {LOW_SIM}): **{int(item_df.flag_low_sim.sum())}** ({item_df.flag_low_sim.mean():.1%}); fields below {LOW_SIM}: {int(df.low_sim.sum())} of {len(df)}")
    L.append(f"- Items where Stanza/spaCy sentence alignment failed on the whole field and Multi-VALUE was run sentence by sentence: {n_fb}")
    L.append(f"- Option collisions reverted: {int((item_df.reverted_collision.astype(str).str.len() > 0).sum())}\n")
    L.append("| field | fraction changed | mean edit ratio (all) | mean edit ratio (changed only) | mean LaBSE cos | min LaBSE cos | n < 0.85 |")
    L.append("|---|---|---|---|---|---|---|")
    for f in TEXT_FIELDS + ["ALL"]:
        s = df if f == "ALL" else df[df.field == f]
        ch = s[s.changed]
        L.append(f"| {f if f != 'ALL' else '**all fields**'} | {s.changed.mean():.3f} | {s.edit_ratio.mean():.4f} | {ch.edit_ratio.mean():.4f} | "
                 f"{s.labse_cos.mean():.4f} | {s.labse_cos.min():.4f} | {int(s.low_sim.sum())} |")
    L.append("\nLaBSE = `sentence-transformers/LaBSE`, cosine between original and transformed field. Edit ratio = character Levenshtein distance / max length.\n")
    L.append("## Final file (after the gate)\n")
    L.append(f"- Items with at least one changed field: **{item_df.any_changed_final.mean():.1%}** ({int(item_df.any_changed_final.sum())}/{n}); "
             f"fields changed: **{df.final_changed.mean():.1%}**; fields reverted by the gate: {int(df.gated_reverted.sum())}; "
             f"mean character edit ratio over all fields: {df.final_edit_ratio.mean():.4f}; all four options distinct in every row: {bool(item_df.options_distinct.all())}\n")
    L.append("| field | fraction changed (final) | fields reverted by gate |")
    L.append("|---|---|---|")
    for f in TEXT_FIELDS:
        s = df[df.field == f]
        L.append(f"| {f} | {s.final_changed.mean():.3f} | {int(s.gated_reverted.sum())} |")
    imp = 0
    for r in rows:
        for f in TEXT_FIELDS:
            for x in r["rules_detail"][f]:
                if x["type"] == "progressives" and x["value"][:3] in ("Are", "Is ", "Am "):
                    imp += 1
    L.append(f"\n**Known tool limitation.** The `progressives` rule also fires on sentence-initial imperatives (\"Make sure ...\" -> \"Are Making sure ...\"); "
             f"{imp} sentence-initial firings of this kind occur in the raw output (they are kept where the field passes the gate).\n")
    L.append("## Rules fired (top 25; counts over all fields of all 300 items, raw output)\n")
    L.append("| rule (Multi-VALUE feature function) | times fired | items |")
    L.append("|---|---|---|")
    for _, x in rc.head(25).iterrows():
        L.append(f"| {x.rule} | {x.n_fired} | {x.n_items} |")
    L.append(f"\n({len(rc)} distinct rules fired in total; full table in `inde_rule_counts.csv`; per-field rule spans in `rules_detail` of the condition file.)\n")
    low = df[df.low_sim].sort_values("labse_cos").head(10)
    L.append("## Lowest-similarity fields (reverted by the gate)\n")
    for _, x in low.iterrows():
        L.append(f"- `{x.item_id}` field `{x.field}` cos={x.labse_cos:.3f}")
    cand = sorted([r for r in rows if r["validation"]["any_changed_final"]], key=lambda r: r["n_rules_fired"])
    picks = [cand[len(cand) // 5], cand[len(cand) // 2], cand[(4 * len(cand)) // 5]]
    src = load_rows("eng_Latn")
    L.append("\n## Three examples (original -> final file; raw output shown when the gate reverted a field)\n")
    for r in picks:
        s = src[r["item_id"]]
        L.append(f"### `{r['item_id']}`  (rules fired: {r['n_rules_fired']})\n")
        for f in TEXT_FIELDS:
            extra = f"\n  - raw Multi-VALUE (reverted by gate): {r['mv_raw_fields'][f]}" if f in r["gate_reverted_fields"] else ""
            L.append(f"- **{f}** (cos {r['validation']['labse_cos_raw'][f]}, edit {r['validation']['edit_ratio_raw'][f]})\n  - original: {s[f]}\n  - Indian English: {r[f]}{extra}")
        L.append(f"- rules: {json.dumps(r['rules_fired'])}\n")
    (REPORT_DIR / "inde_validation.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    if not a.validate_only:
        build(a)
    pairs = assemble()
    rows, df, item_df, rc = validate(pairs)
    report(rows, df, item_df, rc)
    print("done", len(rows))


if __name__ == "__main__":
    main()

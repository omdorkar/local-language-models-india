"""Assemble `hin_Latn+cm` rows, validate against COMI-LINGUA and the English/Hindi originals, write the report."""
from __future__ import annotations

import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import stats

from .cm import ALIGN_LAYER, ALIGN_MODEL, COND
from .common import (LOW_SIM, OUT_DIR, REPORT_DIR, SEED, TEXT_FIELDS, edit_ratio, labse_cos, write_jsonl)

QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
OPTS = ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]


def q_row(name, x):
    x = np.asarray(x, float)
    return {"set": name, "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), **{f"q{int(q * 100):02d}": np.quantile(x, q) for q in QS},
            "share_zero": float(np.mean(x == 0))}


def ks_row(name, x, ref):
    r = stats.ks_2samp(x, ref)
    return {"set": name, "KS_D": r.statistic, "p_value": r.pvalue, "n": len(x)}


def finish(ids, en, hl, res, comi):
    # ---- assemble rows (revert an option to its source text if it collides with another option) ----
    rows, unit_recs = [], []
    for iid in ids:
        r = dict(hl[iid])
        reverted = []
        outs = {f: res[(iid, f)]["out"] for f in TEXT_FIELDS}
        for i, f in enumerate(OPTS):
            if any(outs[f].strip().lower() == outs[g].strip().lower() for g in OPTS[:i]):
                outs[f] = hl[iid][f]
                reverted.append(f)
        r.update(outs)
        r["dialect"] = COND
        r["item_id"] = iid
        r["source_condition"] = "hin_Latn"
        r["aligned_with"] = "eng_Latn"
        r["method"] = (f"SimAlign({ALIGN_MODEL}, layer {ALIGN_LAYER}, itermax/argmax) word alignment hin_Latn<->eng_Latn + EC-style "
                       "single-token noun/adjective/light-verb substitution, CMI-calibrated to COMI-LINGUA (GCM-inspired; Rizvi et al. 2021)")
        r["method_version"] = (f"extra.cm v1; simalign=={version('simalign')}; transformers=={version('transformers')}; torch=={version('torch')}; "
                               f"spacy=={version('spacy')} en_core_web_sm=={version('en-core-web-sm')}; seed={SEED} (per-unit sha256 seed)")
        r["replacements"] = {f: ([] if f in reverted else res[(iid, f)]["replacements"]) for f in TEXT_FIELDS}
        r["n_replaced"] = int(sum(len(v) for v in r["replacements"].values()))
        r["reverted_option_fields"] = reverted
        r["options_distinct"] = len({r[f].strip().lower() for f in OPTS}) == 4
        rows.append(r)
        for f in TEXT_FIELDS:
            for k, u in enumerate(res[(iid, f)]["units"]):
                u = dict(u)
                if f in reverted:
                    u["out_cmi"], u["n_replaced"] = u["base_cmi"], 0
                u.update(item_id=iid, unit=k)
                unit_recs.append(u)
    units = pd.DataFrame(unit_recs)
    units.to_csv(REPORT_DIR / "cm_units.csv", index=False)

    # ---- per-field text similarity ----
    flat = [(r["item_id"], f, en[r["item_id"]][f], hl[r["item_id"]][f], r[f]) for r in rows for f in TEXT_FIELDS]
    c_out_en = labse_cos([x[4] for x in flat], [x[2] for x in flat])
    c_base_en = labse_cos([x[3] for x in flat], [x[2] for x in flat])
    c_out_base = labse_cos([x[4] for x in flat], [x[3] for x in flat])
    recs = []
    for (iid, f, e, b, o), a1, a2, a3 in zip(flat, c_out_en, c_base_en, c_out_base):
        recs.append({"item_id": iid, "field": f, "changed": o != b, "edit_ratio_vs_hin_Latn": edit_ratio(b, o),
                     "labse_out_vs_eng": a1, "labse_hinLatn_vs_eng": a2, "labse_out_vs_hinLatn": a3, "low_sim": a3 < LOW_SIM,
                     "drift_vs_eng": (a1 < a2 - 0.05)})
    fdf = pd.DataFrame(recs)
    fdf.to_csv(REPORT_DIR / "cm_validation_fields.csv", index=False)
    for r in rows:
        sub = fdf[fdf.item_id == r["item_id"]].set_index("field")
        r["validation"] = {
            "changed": {f: bool(sub.loc[f, "changed"]) for f in TEXT_FIELDS},
            "edit_ratio_vs_hin_Latn": {f: round(float(sub.loc[f, "edit_ratio_vs_hin_Latn"]), 4) for f in TEXT_FIELDS},
            "labse_out_vs_eng": {f: round(float(sub.loc[f, "labse_out_vs_eng"]), 4) for f in TEXT_FIELDS},
            "labse_hinLatn_vs_eng": {f: round(float(sub.loc[f, "labse_hinLatn_vs_eng"]), 4) for f in TEXT_FIELDS},
            "labse_out_vs_hinLatn": {f: round(float(sub.loc[f, "labse_out_vs_hinLatn"]), 4) for f in TEXT_FIELDS},
            "any_changed": bool(sub.changed.any()), "min_labse_out_vs_hinLatn": round(float(sub.labse_out_vs_hinLatn.min()), 4),
            "flag_low_sim": bool(sub.low_sim.any()),
            "flag_drift_vs_eng": bool(sub.drift_vs_eng.any()),
            "cmi_by_unit": {f: [round(u["out_cmi"], 2) for u in units[(units.item_id == r["item_id"]) & (units.field == f)].to_dict("records")] for f in TEXT_FIELDS},
        }
    write_jsonl(OUT_DIR / f"{COND}.jsonl", rows)

    # ---- CMI distribution vs COMI-LINGUA ----
    sent = units[(units.field.isin(["flores_passage", "question"])) & (units.n_tokens >= 8)]
    lat = comi[comi.latin_frac > 0.9]
    ref = comi.cmi.values
    qtab = pd.DataFrame([
        q_row("COMI-LINGUA test, majority LID (all)", ref),
        q_row("COMI-LINGUA, annotator 1", comi.cmi_a1), q_row("COMI-LINGUA, annotator 2", comi.cmi_a2), q_row("COMI-LINGUA, annotator 3", comi.cmi_a3),
        q_row("COMI-LINGUA, Latin-script-dominant subset", lat.cmi),
        q_row("Ours: passage sentences + questions (>=8 tokens), after", sent.out_cmi),
        q_row("Ours: same units, BEFORE (plain hin_Latn, same tagger)", sent.base_cmi),
        q_row("Ours: all units incl. options, after", units.out_cmi),
        q_row("Ours: all units incl. options, BEFORE", units.base_cmi),
        q_row("Ours: target CMI drawn for the units (all)", units.target_cmi),
        q_row("Ours: ceiling = CMI if every candidate were substituted (sentences+questions)", sent.max_cmi),
    ])
    qtab.to_csv(REPORT_DIR / "cm_cmi_quantiles.csv", index=False)
    ktab = pd.DataFrame([
        ks_row("after: sentences+questions vs COMI (all)", sent.out_cmi, ref),
        ks_row("before: sentences+questions vs COMI (all)", sent.base_cmi, ref),
        ks_row("after: sentences+questions vs COMI Latin-dominant", sent.out_cmi, lat.cmi),
        ks_row("after: all units vs COMI (all)", units.out_cmi, ref),
        ks_row("before: all units vs COMI (all)", units.base_cmi, ref),
        ks_row("after: units with mixing reachable (n_candidates>=1), >=8 tokens, vs COMI", units[(units.n_candidates >= 1) & (units.n_tokens >= 8)].out_cmi, ref),
        ks_row("target draws vs COMI (sanity)", units.target_cmi, ref),
    ])
    ktab.to_csv(REPORT_DIR / "cm_cmi_ks.csv", index=False)
    plot_hist(sent, units, comi)
    write_report(rows, fdf, units, qtab, ktab, en, hl)
    print("wrote", len(rows), "rows")


def plot_hist(sent, units, comi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bins = np.arange(0, 56, 2.5)
    ax.hist(comi.cmi, bins=bins, density=True, alpha=0.45, label="COMI-LINGUA test (expert LID)", color="#444")
    ax.hist(sent.out_cmi, bins=bins, density=True, histtype="step", lw=2, label="hin_Latn+cm (passage sentences + questions)", color="#d62728")
    ax.hist(sent.base_cmi, bins=bins, density=True, histtype="step", lw=2, ls="--", label="plain hin_Latn (same tagger)", color="#1f77b4")
    ax.set_xlabel("Code-Mixing Index (Das & Gambäck 2014)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cm_cmi_hist.png", dpi=150)
    plt.close(fig)


def fmt(df, cols):
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating)) else str(r[c])) for c in cols) + " |")
    return "\n".join(L)


def write_report(rows, fdf, units, qtab, ktab, en, hl):
    n = len(rows)
    L = ["# Machine-made Hinglish (`hin_Latn+cm`) validation\n",
         "*Exploratory condition.* Romanized Hindi-English code-mixed versions of the 300 sampled Belebele items, made by word-aligning "
         "the human romanized Hindi (`hin_Latn`) with the English original (`eng_Latn`) and substituting aligned English nouns, adjectives and "
         "light-verb complements for their Hindi counterparts; the amount of mixing per unit is drawn from the CMI distribution of "
         "COMI-LINGUA (expert-annotated real Hinglish). See `extra/README.md` for the full method and citations.\n",
         f"- Rows: **{n}**; items with at least one changed field: **{np.mean([r['validation']['any_changed'] for r in rows]):.1%}**; "
         f"fields changed: **{fdf.changed.mean():.1%}**; tokens substituted per item (mean): {np.mean([r['n_replaced'] for r in rows]):.1f}",
         f"- Options distinct in every row: {all(r['options_distinct'] for r in rows)}; option fields reverted to hin_Latn because of a collision: "
         f"{sum(len(r['reverted_option_fields']) for r in rows)}",
         f"- Items flagged (LaBSE(cm, hin_Latn) < {LOW_SIM} in any field): **{sum(r['validation']['flag_low_sim'] for r in rows)}**. "
         "Caution: LaBSE is weak on romanized Hindi (mean cos(hin_Latn, English) is only about 0.48), and swapping words for English ones moves "
         "the embedding toward English, so this flag is noisy and over-triggers; the more informative check is the next line.",
         f"- Fields where cm is *closer* to the English original than plain hin_Latn is: {(fdf.labse_out_vs_eng > fdf.labse_hinLatn_vs_eng)[fdf.changed].mean():.1%} of changed fields; "
         f"fields where cm is more than 0.05 *farther* from English than hin_Latn (possible meaning drift): {int(fdf.drift_vs_eng.sum())} in "
         f"{sum(r['validation']['flag_drift_vs_eng'] for r in rows)} items (flag `flag_drift_vs_eng`)\n",
         "## CMI distribution vs COMI-LINGUA\n",
         "CMI = 100 x (1 - max_i w_i / (n - u)) per unit (Das & Gambäck 2014). Units: a sentence of a passage, the question, or one option. "
         "Reference: 4,995 COMI-LINGUA LID test sentences (token tags = majority of 3 annotators; per-annotator rows shown for spread). "
         "Our tags come from alignment provenance, not annotators (see README), so absolute values are comparable only approximately.\n",
         fmt(qtab, ["set", "n", "mean", "sd", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "share_zero"]), "",
         "Two-sample Kolmogorov-Smirnov statistic D (0 = identical distributions):\n",
         fmt(ktab, ["set", "KS_D", "p_value", "n"]), "",
         "![CMI histogram](cm_cmi_hist.png)\n",
         "Note: COMI-LINGUA's test split consists of sentences that are code-mixed (no sentence has CMI = 0; minimum about 7.7), so the target is a "
         "*mixed-only* distribution. Units for which the aligned English counterparts offer too few safe candidates stay below their target "
         "(see `n_candidates` in `cm_units.csv`), and option-length units have coarse CMI values (e.g. 1 of 4 words English = 25).\n",
         "Sensitivity (`python -m extra.cm_sensitivity` -> `cm_sensitivity.csv`): dropping the alignment-confidence filter entirely raises the mean "
         "realised CMI of sentences+questions only from about 19.3 to 20.6 (ceiling 22.9 -> 25.2; COMI-LINGUA 27.8), so the binding constraint is the "
         "number of safe single-token substitution slots, not the filter. The method cannot reach COMI-LINGUA's level of mixing without also switching "
         "function words or multi-word phrases, which this simplified version deliberately does not do.\n",
         "## Text similarity (LaBSE, `sentence-transformers/LaBSE`)\n",
         "| field | fraction changed | mean edit ratio vs hin_Latn | mean cos(cm, eng) | mean cos(hin_Latn, eng) | mean cos(cm, hin_Latn) | min cos(cm, hin_Latn) | n < 0.85 |",
         "|---|---|---|---|---|---|---|---|"]
    for f in TEXT_FIELDS + ["ALL"]:
        s = fdf if f == "ALL" else fdf[fdf.field == f]
        L.append(f"| {f} | {s.changed.mean():.3f} | {s.edit_ratio_vs_hin_Latn.mean():.4f} | {s.labse_out_vs_eng.mean():.4f} | {s.labse_hinLatn_vs_eng.mean():.4f} | "
                 f"{s.labse_out_vs_hinLatn.mean():.4f} | {s.labse_out_vs_hinLatn.min():.4f} | {int(s.low_sim.sum())} |")
    rc = {}
    for r in rows:
        for f in TEXT_FIELDS:
            for x in r["replacements"][f]:
                rc[x["rule"]] = rc.get(x["rule"], 0) + 1
    L.append(f"\nSubstitutions by rule: {json.dumps(rc)}\n")
    # examples
    rng = np.random.default_rng(SEED)
    cand = [r for r in rows if r["n_replaced"] > 0]
    picks = [cand[i] for i in rng.choice(len(cand), 3, replace=False)]
    L.append("## Three random examples (seeded pick)\n")
    for r in picks:
        L.append(f"### `{r['item_id']}` ({r['n_replaced']} substitutions)\n")
        for f in TEXT_FIELDS:
            L.append(f"- **{f}**\n  - English: {en[r['item_id']][f]}\n  - hin_Latn (human): {hl[r['item_id']][f]}\n  - hin_Latn+cm: {r[f]}\n"
                     f"  - substitutions: {', '.join(x['hindi'] + '->' + x['replacement'] for x in r['replacements'][f]) or '(none)'}")
        L.append("")
    (REPORT_DIR / "cm_validation.md").write_text("\n".join(L))

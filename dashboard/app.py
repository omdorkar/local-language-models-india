"""Language and Script Gaps in Local Language Models for Indian Users: local results dashboard.

    .venv/bin/streamlit run dashboard/app.py
    .venv/bin/streamlit run dashboard/app.py -- --runs /path/to/runs     # other folder
    EVALUATION_RUNS=/path/to/runs .venv/bin/streamlit run dashboard/app.py
    http://localhost:8501/?runs=/path/to/runs&view=...                   # query parameter

Read-only: it opens runs/*/*.jsonl for reading and writes nothing. It works while the experiment is still running
(partial cells are shown with their n). All statistics come from ``evaluation.stats``; the dashboard only displays them.
Folder settings and defaults are documented in ``dashboard/dash_data.py``.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import dash_charts as C  # noqa: E402
import dash_data as D  # noqa: E402
from evaluation import plots, stats  # noqa: E402

LANG_NAME = plots.LANG_NAME
NATIVE_ORDER = plots.NATIVE_ORDER
MODEL_NAME = plots.MODEL_NAME

VIEWS = [
    "Run progress",
    "1. Overview: all languages vs English",
    "2. Native script vs romanized",
    "3. Token ratio vs gap",
    "4. Item browser",
    "5. Fixes and exploratory conditions",
    "6. Global-MMLU",
]
CONF = ":blue-badge[PRIMARY: main hypothesis test]"
EXPL = ":orange-badge[EXPLORATORY: not one of H1-H5]"
FONT_CSS = """
<style>
.ek-text{font-family:"Noto Sans","Noto Sans Devanagari","Noto Sans Bengali","Noto Naskh Arabic","Noto Nastaliq Urdu",
"Noto Sans Telugu","Noto Sans Tamil","Noto Sans Gujarati","Noto Sans Kannada","Noto Sans Malayalam","Noto Sans Gurmukhi",
"Noto Sans Oriya","Kohinoor Devanagari","Kohinoor Bangla","Kohinoor Telugu","Geeza Pro","Arial Unicode MS",system-ui,sans-serif;
line-height:1.65;font-size:0.95rem}
.ek-card{border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:.6rem .8rem;margin-bottom:.4rem}
.ek-opt{padding:.1rem .3rem;margin:.1rem 0;border-left:4px solid transparent}
.ek-gold{border-left:4px solid #0072B2;font-weight:600;background:rgba(0,114,178,.10)}
.ek-lab{font-size:.8rem;opacity:.7;text-transform:uppercase;letter-spacing:.04em}
</style>
"""


def show(df: pd.DataFrame, **kw) -> None:
    """Table without the index; works on old and new Streamlit versions."""
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:  # noqa: BLE001
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def fig_out(fig, caption: str) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(caption)


def mname(m: str) -> str:
    return MODEL_NAME.get(m, m)


# ------------------------------------------------------------------------------------------ context


def build_context() -> dict:
    st.sidebar.title("Language and Script Gaps")
    st.sidebar.caption("Results dashboard. Local and read-only.")
    qp = st.query_params.get("runs")
    runs_in = st.sidebar.text_input("Runs folder", value=qp or D.default_runs_dir(), key="runs_dir",
                                    help="Folder that contains belebele/, gmmlu_lite/, gmmlu/ with one .jsonl per model. Opened read-only.")
    choices = sorted(set(D.NBOOT_CHOICES) | {D.default_nboot()})
    n_boot = st.sidebar.selectbox("Bootstrap resamples", choices, index=choices.index(D.default_nboot()), key="n_boot",
                                  help="Confidence intervals and p-values use this many resamples. The paper's analysis uses 10,000; "
                                       "fewer is faster and changes intervals only slightly.")
    if st.sidebar.button("Reload data now"):
        st.cache_data.clear()
        st.cache_resource.clear()
    want = st.query_params.get("view")
    idx = next((i for i, v in enumerate(VIEWS) if want and v.lower().startswith(want.lower())), 1)
    view = st.sidebar.radio("View", VIEWS, index=idx, key="view")

    paths = D.resolve_paths(runs_in)
    runs_dir = str(paths.runs)
    sig = D.runs_signature(runs_dir)
    data = D.load_all(runs_dir, sig)
    items = data["items"]
    counts = D.row_counts(items)
    models = D.ordered_models(items["model_key"].unique()) if len(items) else []
    main_models = [m for m in models if m in stats.MAIN_MODELS]
    comp = D.completeness(counts, paths.derived, stats.MAIN_MODELS)
    ctx = dict(paths=paths, runs_dir=runs_dir, sig=sig, data=data, items=items, counts=counts, models=models,
               main_models=main_models, comp=comp, n_boot=int(n_boot), view=view)

    st.sidebar.divider()
    if not paths.runs.is_dir():
        st.sidebar.error("Runs folder not found.")
    elif not sig:
        st.sidebar.warning("No run files yet.")
    else:
        st.sidebar.markdown(f"**{len(items):,}** answer rows  \n**{len(sig)}** run files  \n**{len(models)}** models")
        for b in D.BENCHMARKS:
            cb = comp[b]
            if cb["cells_total"] and cb["rows"]:
                st.sidebar.progress(min(1.0, cb["rows"] / max(1, cb["rows_total"])), text=f"{b}: {cb['cells_done']}/{cb['cells_total']} cells complete")
    if data["n_bad"]:
        st.sidebar.caption(f"{data['n_bad']} unreadable line(s) skipped (probably a row being written right now).")
    st.sidebar.divider()
    st.sidebar.caption("Statistics come from `evaluation.stats`: percentile bootstrap over items (seed 1909), exact McNemar, Holm correction "
                       "inside each hypothesis family. Invalid answers count as wrong.")
    return ctx


def provisional(ctx: dict, benchmark: str = "belebele") -> bool:
    cb = ctx["comp"].get(benchmark, {})
    return not cb or cb.get("cells_done", 0) < cb.get("cells_total", 1)


def provisional_note(ctx: dict, benchmark: str = "belebele") -> None:
    if provisional(ctx, benchmark):
        cb = ctx["comp"].get(benchmark, {})
        st.warning(f"**The run is not finished** ({cb.get('cells_done', 0)} of {cb.get('cells_total', 0)} planned model x condition cells "
                   f"have all {cb.get('expected_n', 0)} items). Numbers below are provisional and rest on different numbers of items per cell "
                   "(shown as n). Primary verdicts count a test that cannot be run yet as p = 1, so they are conservative "
                   "but should not be quoted until the run is complete.")


def need_data(ctx: dict, benchmark: str | None = None) -> bool:
    """Show a friendly message and return True if there is nothing to draw."""
    items = ctx["items"]
    if items.empty:
        st.info("No answer rows found yet. Start the experiment (`python -m evaluation.run ...`) or point the **Runs folder** in the sidebar "
                "at a folder with `belebele/<model>.jsonl` files.")
        return True
    if benchmark and not (items["benchmark"] == benchmark).any():
        st.info(f"No `{benchmark}` rows yet.")
        return True
    return False


def acc_table(ctx: dict) -> pd.DataFrame:
    return D.accuracy(ctx["runs_dir"], ctx["sig"], ctx["n_boot"])


def conf_tables(ctx: dict):
    return D.confirmatory(ctx["runs_dir"], ctx["sig"], ctx["n_boot"])


def conf_display(df: pd.DataFrame, cond_name: str, ref_name: str, flip: bool, gap_name: str) -> pd.DataFrame:
    """Readable table of primary (H1-H5) rows. ``flip`` shows reference minus condition (positive = condition is worse)."""
    rows = []
    for _, r in df.iterrows():
        row = {"Model": mname(r["model"]), "Language": LANG_NAME.get(r["language"], r["language"])}
        if r["status"] != "ok":
            row |= {"n": "", ref_name: "", cond_name: "", gap_name: "", "95% CI": "", "p (Holm)": "", "Verdict": r["status"]}
        else:
            gap, lo, hi = (-r["gap"], -r["ci_high"], -r["ci_low"]) if flip else (r["gap"], r["ci_low"], r["ci_high"])
            row |= {"n": str(int(r["n"])), ref_name: D.fmt_pp(r["acc_reference"], False), cond_name: D.fmt_pp(r["acc_condition"], False),
                    gap_name: D.fmt_pp(gap), "95% CI": D.fmt_ci(lo, hi), "p (Holm)": D.fmt_p(r["p_holm"]), "Verdict": D.verdict(r)}
        rows.append(row)
    return pd.DataFrame(rows)


def family_summary(df: pd.DataFrame, label: str) -> str:
    ran = int((df["status"] == "ok").sum())
    conf = int(df["confirmed"].fillna(False).astype(bool).sum())
    return (f"**{label}: confirmed in {conf} of {len(df)} planned model-language tests** "
            f"({ran} could be run so far, {len(df) - ran} not run). A test that is not confirmed means the data do not show the predicted "
            "difference at this sample size; it is not evidence that there is none.")


# ------------------------------------------------------------------------------------------ views


def view_progress(ctx: dict) -> None:
    st.header("Run progress")
    st.caption("How many answers exist for every model x condition. The experiment appends to the run files while it works, "
               "so partial cells are normal. 'Complete' means all sampled items are present.")
    if need_data(ctx):
        return
    counts, items = ctx["counts"], ctx["items"]
    acc = acc_table(ctx)
    for b in D.BENCHMARKS:
        sub = counts[counts["benchmark"] == b]
        if sub.empty:
            continue
        exp_n, from_file = D.sample_size(ctx["paths"].derived, b, int(sub["n"].max()))
        st.subheader(D.BENCHMARK_NAME[b])
        cb = ctx["comp"][b]
        c1, c2, c3 = st.columns(3)
        c1.metric("Items per condition (sample)", exp_n)
        c2.metric("Planned cells complete", f"{cb['cells_done']} / {cb['cells_total']}")
        c3.metric("Rows in the planned grid", f"{cb['rows']:,} / {cb['rows_total']:,}")
        if not from_file:
            st.caption("Sample file not found; the expected n is the largest count seen.")
        models = D.ordered_models(sub["model_key"].unique())
        conds = [c for c in D.ordered_conditions(b, sub["condition"].unique()) if c in set(sub["condition"]) or "+" not in c]
        piv = sub.pivot(index="condition", columns="model_key", values="n").reindex(index=conds, columns=models)

        def cell(v):
            if pd.isna(v):
                return "not started"
            return f"{int(v)} (complete)" if v >= exp_n else f"{int(v)} / {exp_n}"

        tbl = piv.map(cell) if hasattr(piv, "map") else piv.applymap(cell)
        tbl.columns = [mname(m) for m in tbl.columns]
        tbl.insert(0, "Condition", [D.cond_label(c, b) for c in tbl.index])
        tbl.insert(1, "Id", list(tbl.index))
        show(tbl.reset_index(drop=True))
        if not acc.empty:
            with st.expander("Invalid-answer rate (%) per model x condition (invalid answers count as wrong)"):
                a = acc[acc["benchmark"] == b]
                inv = a.pivot(index="condition", columns="model", values="invalid_rate").reindex(index=conds, columns=models) * 100
                inv = inv.map(lambda v: "" if pd.isna(v) else f"{v:.1f}") if hasattr(inv, "map") else inv
                inv.columns = [mname(m) for m in inv.columns]
                inv.insert(0, "Condition", [D.cond_label(c, b) for c in inv.index])
                show(inv.reset_index(drop=True))
    st.divider()
    st.subheader("Stored (frozen) analysis outputs")
    res, figs = ctx["paths"].results, ctx["paths"].figures
    if res.is_dir() and any(res.iterdir()):
        files = sorted(p for p in res.iterdir() if p.is_file())
        newest_run = max((s[2] for s in ctx["sig"]), default=0)
        show(pd.DataFrame([{"File": p.name, "KB": round(p.stat().st_size / 1024, 1),
                            "Modified": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")} for p in files]))
        summary = res / "summary.md"
        if summary.exists():
            if summary.stat().st_mtime_ns < newest_run:
                st.caption("The stored summary is older than the newest run file: re-run `python -m evaluation.stats` to refresh it. "
                           "The views in this dashboard are always computed live from the run files.")
            with st.expander("Stored summary (reports/results/summary.md)"):
                st.markdown(summary.read_text(encoding="utf-8"))
    else:
        st.caption(f"No stored results in `{res}` yet. They appear after `python -m evaluation.stats`.")
    pngs = sorted(figs.glob("*.png")) if figs.is_dir() else []
    if pngs:
        with st.expander(f"Stored figures ({len(pngs)})"):
            for p in pngs:
                st.image(str(p), caption=p.stem)


def view_overview(ctx: dict) -> None:
    st.header("Overview: accuracy by model and condition, and the gap to English")
    if need_data(ctx, "belebele"):
        return
    provisional_note(ctx)
    acc = acc_table(ctx)
    bel = acc[acc["benchmark"] == "belebele"]
    exp_n, _ = D.sample_size(ctx["paths"].derived, "belebele", int(bel["n"].max()))
    show_q3 = False
    if "llama3.1-8b-q3" in set(bel["model"]):
        show_q3 = st.checkbox("Include the Llama 3.1 8B Q3_K_M sensitivity run", value=False, key="show_q3")
    models = [m for m in D.ordered_models(bel["model"].unique()) if show_q3 or m != "llama3.1-8b-q3"]

    st.markdown(f"#### Accuracy (%) by condition  {EXPL}")
    conds = [c for c in D.belebele_human_conditions() if c in set(bel["condition"])]
    piv = bel.pivot(index="condition", columns="model", values="accuracy").reindex(index=conds, columns=models) * 100
    npv = bel.pivot(index="condition", columns="model", values="n").reindex(index=conds, columns=models)
    texts = [[("" if pd.isna(piv.iloc[i, j]) else f"{piv.iloc[i, j]:.1f}" + (f"\nn={int(npv.iloc[i, j])}" if npv.iloc[i, j] < exp_n else ""))
              for j in range(len(models))] for i in range(len(conds))]
    fig = C.annotated_heatmap(piv.to_numpy(float), texts, [D.cond_label(c) for c in conds], [mname(m) for m in models],
                              "Accuracy (%)", diverging=False, vmin=0, vmax=100)
    fig_out(fig, f"Share of Belebele questions answered correctly, per model (columns) and condition (rows). Each cell uses up to {exp_n} questions; "
                 "cells with fewer show their n. Chance is 25%. Human-made conditions only: English, 13 native-script languages and 4 romanized versions. "
                 "This descriptive overview is not itself a hypothesis test; the tests are below.")
    with st.expander("Accuracy with 95% bootstrap confidence intervals, n and invalid-answer rate"):
        t = bel[bel["model"].isin(models) & bel["condition"].isin(conds)].copy()
        t["Model"] = t["model"].map(mname)
        t["Condition"] = t["condition"].map(D.cond_label)
        t["Accuracy % [95% CI]"] = [f"{a * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]" for a, lo, hi in zip(t["accuracy"], t["ci_low"], t["ci_high"])]
        t["Invalid %"] = (t["invalid_rate"] * 100).round(1)
        show(t[["Model", "Condition", "condition", "n", "Accuracy % [95% CI]", "Invalid %"]].rename(columns={"condition": "Id"}))

    st.markdown(f"#### English minus native-script accuracy (H1)  {CONF}")
    conf, _ = conf_tables(ctx)
    h1 = conf[(conf["hypothesis"] == "H1")]
    h1ok = h1[h1["status"] == "ok"]
    if h1ok.empty:
        st.info("H1 needs English and native-script rows for the same model; none yet.")
        return
    h1m = [m for m in models if m in set(h1ok["model"])]
    langs = [l for l in NATIVE_ORDER if l in set(h1ok["language"])]
    g = np.full((len(langs), len(h1m)), np.nan)
    tx = [["" for _ in h1m] for _ in langs]
    for _, r in h1ok.iterrows():
        if r["model"] in h1m:
            i, j = langs.index(r["language"]), h1m.index(r["model"])
            g[i, j] = -r["gap"] * 100
            tx[i][j] = f"{g[i, j] + 0.0:+.1f}{'*' if r['confirmed'] else ''}" + (f"\nn={int(r['n'])}" if r["n"] < exp_n else "")
    fig = C.annotated_heatmap(g, tx, [LANG_NAME[l] for l in langs], [mname(m) for m in h1m],
                              "English minus native accuracy (percentage points)\npositive = worse in the Indian language", diverging=True)
    fig_out(fig, "Paired gap on the same questions: English accuracy minus accuracy in each native-script language, in percentage points. "
                 "Positive (red) means the model does worse in the Indian language. * = H1 confirmed for that model and language "
                 "(Holm-adjusted p < 0.05 and the gap in the predicted direction). Confidence intervals and exact p-values are in the table below.")
    st.markdown(family_summary(h1, "H1"))
    show(conf_display(h1[h1["model"].isin(models)], "Native acc %", "English acc %", True, "English minus native (pp)"))
    st.caption("95% CI: percentile bootstrap over questions (paired). p (Holm): exact McNemar p-value, Holm-adjusted within the planned 65-test family.")


def view_native_vs_roman(ctx: dict) -> None:
    st.header("Native script vs romanized (Latin-letter) script")
    st.markdown(f"Hindi, Bengali, Urdu and Nepali, same questions written in the native script and in romanized form (human-made, from Belebele).  {CONF} (H2)")
    if need_data(ctx, "belebele"):
        return
    provisional_note(ctx)
    acc = acc_table(ctx)
    bel = acc[acc["benchmark"] == "belebele"]
    models = [m for m in D.ordered_models(bel["model"].unique()) if m != "llama3.1-8b-q3"]
    panels = [(i, n, r) for i, n, r in stats.romanized_languages() if ((bel["condition"] == n).any() and (bel["condition"] == r).any())]
    if not panels:
        st.info("No language has both a native-script and a romanized condition yet.")
        return
    fig = C.dumbbell_panels(bel, panels, models, LANG_NAME)
    fig_out(fig, "Accuracy of each model in the native script (filled circle) and in romanized script (open diamond), with 95% bootstrap confidence "
                 "intervals. The line joining them is the script gap: if the diamond lies to the left of the circle, the model does worse on the romanized text.")
    conf, _ = conf_tables(ctx)
    h2 = conf[conf["hypothesis"] == "H2"]
    st.markdown(family_summary(h2, "H2"))
    show(conf_display(h2, "Romanized acc %", "Native acc %", True, "Native minus romanized (pp)"))
    st.caption("Positive gap = romanized text is answered worse than the native-script text of the same questions. "
               "p (Holm): exact McNemar, Holm-adjusted inside the planned 20-test family.")
    with st.expander("What is in the romanized versions?"):
        st.markdown("Belebele's romanized passages were transliterated by annotators with IndicXlit support. The romanized *questions and options* "
                    "are an independent informal translation, not a letter-by-letter transliteration of the native ones "
                    "(see `reports/bridge/xlit_validation.md`). A script gap therefore mixes script with some wording differences.")


def view_tokens(ctx: dict) -> None:
    st.header("Token ratio vs accuracy gap")
    st.markdown(f"Do languages that need more tokens than English also lose more accuracy?  {CONF} (H3)")
    if need_data(ctx, "belebele"):
        return
    provisional_note(ctx)
    conf, pts = conf_tables(ctx)
    h3 = conf[conf["hypothesis"] == "H3"]
    models = [m for m in ctx["main_models"]]
    if pts.empty:
        st.info("No points yet. H3 needs, for each model, English and several native-script conditions plus the empty-template calibration row "
                "(`__template__`).")
    else:
        fig = C.token_scatter(pts, h3, [m for m in models if m in set(pts["model"])])
        fig_out(fig, "Each point is one native-script language. Horizontal: how many more prompt tokens that language needs than English "
                     "(the model's own count, with the fixed instruction text removed). Vertical: English accuracy minus accuracy in that language, in "
                     "percentage points. The rank correlation (Spearman rho) is over the 13 languages. A positive rho is an association, not proof that tokens cause the gap.")
    rows = []
    for _, r in h3.iterrows():
        ok = r["status"] == "ok"
        rows.append({"Model": mname(r["model"]), "Languages with data": int(r["n"]) if pd.notna(r["n"]) else 0,
                     "Spearman rho": f"{r['rho']:+.2f}" if ok else "", "p (one-sided)": D.fmt_p(r["p"]) if ok else "",
                     "p (Holm)": D.fmt_p(r["p_holm"]) if ok else "", "Verdict": D.verdict(r)})
    st.markdown(family_summary(h3, "H3"))
    show(pd.DataFrame(rows))
    tok = D.token_table(ctx["runs_dir"], ctx["sig"])
    if not tok.empty:
        with st.expander("Token ratios per model and condition (net = template removed, as used by H3; raw = no subtraction)"):
            t = tok[tok["benchmark"] == "belebele"]
            t = t[t["model"].isin(models)]
            piv = t.pivot(index="condition", columns="model", values="token_ratio_net")
            conds = [c for c in D.ordered_conditions("belebele", piv.index) if c in piv.index]
            piv = piv.reindex(conds)
            piv.columns = [mname(m) for m in piv.columns]
            piv = piv.map(lambda v: "" if pd.isna(v) else f"{v:.2f}") if hasattr(piv, "map") else piv
            piv.insert(0, "Condition", [D.cond_label(c) for c in piv.index])
            show(piv.reset_index(drop=True))
            st.caption("Ratio of mean net prompt tokens per item to English, from each runtime's own token count. Gemma 3 and Sarvam share a tokenizer, "
                       "so their counts coincide.")


def item_card_html(txt: dict, rtl: bool) -> str:
    d = ' dir="rtl"' if rtl else ""
    opts = "".join(
        f'<div class="ek-opt{" ek-gold" if "ABCD"[i] == txt["gold"] else ""}"{d}><b>{"ABCD"[i]}.</b> {html.escape(o)}'
        f'{" &nbsp;[gold answer]" if "ABCD"[i] == txt["gold"] else ""}</div>' for i, o in enumerate(txt["options"]))
    return (f'<div class="ek-card ek-text"{d}><div class="ek-lab">Passage</div>{html.escape(txt["passage"])}'
            f'<div class="ek-lab" style="margin-top:.6rem">Question</div>{html.escape(txt["question"])}'
            f'<div class="ek-lab" style="margin-top:.6rem">Options</div>{opts}</div>')


def view_items(ctx: dict) -> None:
    st.header("Item browser")
    st.caption("Pick one Belebele question and see it in several conditions side by side, with what each model answered against the gold answer. "
               "Item text comes from `data/raw/belebele/data` (human-made conditions) and `data/derived/conditions/belebele` (machine-made).")
    st.markdown(CONF + " conditions are the human-made ones and the +xlit / +mt_en fixes; " + EXPL + " conditions are +inde and +cm.")
    st.markdown(FONT_CSS, unsafe_allow_html=True)
    if need_data(ctx, "belebele"):
        return
    paths, items = ctx["paths"], ctx["items"]
    bel = items[items["benchmark"] == "belebele"]
    avail = D.available_item_conditions(paths.raw, paths.derived, bel["condition"].unique())
    default = [c for c in ("eng_Latn", "hin_Deva", "hin_Latn") if c in avail]
    chosen = st.multiselect("Conditions to compare", avail, default=default, key="item_conds",
                            format_func=lambda c: f"{D.cond_label(c)}  [{c}]")
    if not chosen:
        st.info("Choose at least one condition.")
        return
    sample = D.sample_items(paths.derived, "belebele", bel["item_id"].unique())
    flt = st.selectbox("Which items?", ["All sampled items", "Some model is right in English but wrong in a chosen condition",
                                        "Models disagree in a chosen condition"], key="item_filter")
    pool = sample
    if flt != "All sampled items":
        sub = bel[bel["condition"].isin(set(chosen) | {"eng_Latn"})]
        piv = sub.pivot_table(index=["item_id", "model_key"], columns="condition", values="correct", aggfunc="first")
        if flt.startswith("Some model"):
            others = [c for c in chosen if c != "eng_Latn" and c in piv.columns]
            if "eng_Latn" in piv.columns and others:
                hit = piv[(piv["eng_Latn"] == True) & (piv[others] == False).any(axis=1)]  # noqa: E712
                keep = set(hit.index.get_level_values(0))
                pool = [i for i in sample if i in keep]
            else:
                pool = []
        else:
            cols = [c for c in chosen if c in piv.columns]
            keep = set()
            for c in cols:
                nun = piv[c].dropna().groupby(level=0).nunique()
                keep |= set(nun[nun > 1].index)
            pool = [i for i in sample if i in keep]
        st.caption(f"{len(pool)} of {len(sample)} items match.")
    if not pool:
        st.info("No item matches this filter yet.")
        return
    item = st.selectbox("Item", pool, format_func=D.item_title, key="item_id")
    models = D.ordered_models(bel["model_key"].unique())
    per = bel[bel["item_id"] == item].set_index(["condition", "model_key"])

    summary_rows = {}
    ncols = min(3, len(chosen))
    for start in range(0, len(chosen), ncols):
        cols = st.columns(ncols)
        for col, cond in zip(cols, chosen[start:start + ncols]):
            with col:
                st.markdown(f"**{D.cond_label(cond)}**  \n`{cond}` ({'machine-made' if '+' in cond else 'human-made'})")
                p = D.condition_path(paths.raw, paths.derived, cond)
                txt = D.belebele_texts(str(paths.raw), str(paths.derived), cond, D.path_signature(p)).get(item)
                if txt is None:
                    st.warning(f"Text for this item not found in `{p.name}`.")
                else:
                    st.markdown(item_card_html(txt, D.condition_script(cond) in D.RTL_SCRIPTS), unsafe_allow_html=True)
                lines, mismatch = [], False
                for m in models:
                    if (cond, m) in per.index:
                        r = per.loc[(cond, m)]
                        ans = r["answer"] if isinstance(r["answer"], str) else "invalid"
                        mark = "correct" if r["correct"] else "wrong"
                        lines.append(f"- {mname(m)}: **{ans}** ({mark}; gold {r['gold']})")
                        summary_rows.setdefault(m, {})[D.cond_label(cond)] = f"{ans} {'✓' if r['correct'] else '✗'}"
                        if txt and txt["gold"] and txt["gold"] != r["gold"]:
                            mismatch = True
                    else:
                        lines.append(f"- {mname(m)}: not run yet")
                st.markdown("\n".join(lines))
                if mismatch:
                    st.warning("The gold letter in the run file differs from this item file's gold answer. Check that the data folder matches the runs.")
    if summary_rows:
        st.markdown("##### Answers at a glance")
        tbl = pd.DataFrame(summary_rows).T.reindex([m for m in models if m in summary_rows])
        tbl.index = [mname(m) for m in tbl.index]
        tbl = tbl.reset_index().rename(columns={"index": "Model"}).fillna("not run yet")
        show(tbl)
        st.caption("Letter chosen by the model, followed by ✓ if it matches the gold answer and ✗ if not (an invalid reply counts as wrong).")


def view_fixes(ctx: dict) -> None:
    st.header("On-device fixes and exploratory conditions")
    if need_data(ctx, "belebele"):
        return
    provisional_note(ctx)
    conf, _ = conf_tables(ctx)
    acc = acc_table(ctx)
    models = [m for m in D.ordered_models(acc["model"].unique()) if m != "llama3.1-8b-q3"]
    t1, t2, t3, t4 = st.tabs(["H4: IndicXlit (+xlit)", "H5: translate to English (+mt_en)", "Indian English (+inde) and Hinglish (+cm)",
                              "Q3 vs Q4 sensitivity"])

    def fix_tab(hyp: str, title: str, unfixed: str, fixed: str, ref_name: str, cond_name: str, caption: str) -> None:
        st.markdown(f"{CONF}  \n{title}")
        d = conf[conf["hypothesis"] == hyp]
        ok = d[d["status"] == "ok"]
        if ok.empty:
            built = ctx["paths"].derived / "conditions" / "belebele"
            n_files = len(list(built.glob(f"*+{fixed}.jsonl"))) if built.is_dir() else 0
            st.info(f"No {hyp} results yet: the model has not been run on the +{fixed} conditions. "
                    f"({n_files} input file(s) for +{fixed} exist in `data/derived/conditions/belebele`.)")
            return
        langs = [l for l in NATIVE_ORDER if l in set(ok["language"])]
        ms = [m for m in models if m in set(ok["model"])]
        df = pd.DataFrame({"model": ok["model"], "cat": ok["language"], "value": ok["gap"] * 100, "lo": ok["ci_low"] * 100,
                           "hi": ok["ci_high"] * 100, "star": ok["confirmed"].astype(bool)})
        fig = C.grouped_bars(df, langs, LANG_NAME, ms, "Change in accuracy (pp)\nfixed minus unfixed")
        fig_out(fig, caption)
        st.markdown(family_summary(d, hyp))
        show(conf_display(d, cond_name, ref_name, False, "Fixed minus unfixed (pp)"))

    with t1:
        fix_tab("H4", "Romanized input converted to the native script by IndicXlit, before the model sees it (Hindi, Bengali, Urdu, Nepali).",
                "romanized", "xlit", "Romanized acc %", "Converted acc %",
                "Bars: change in accuracy when the romanized text is first converted to native script by IndicXlit (positive = the fix helps), "
                "same questions, with 95% bootstrap CI. * = confirmed (Holm-adjusted p < 0.05). The conversion is machine-made and imperfect; "
                "see reports/bridge/xlit_validation.md.")
    with t2:
        fix_tab("H5", "Native-script input machine-translated to English by NLLB-200-distilled-1.3B before the model sees it (13 languages).",
                "native", "mt_en", "Native acc %", "Translated acc %",
                "Bars: change in accuracy when the question is first machine-translated to English (positive = the fix helps), same questions, "
                "with 95% bootstrap CI. * = confirmed (Holm-adjusted p < 0.05). Translation errors count against the fix.")
    with t3:
        st.markdown(f"{EXPL}  \nSynthetic conditions. Indian English is made by rule-based Multi-VALUE; Hinglish by word alignment plus single-word English "
                    "substitution. Both approximate, not depict, how people write. No multiple-testing correction; p-values are unadjusted.")
        eg = D.exploratory_gaps(ctx["runs_dir"], ctx["sig"], ctx["n_boot"])
        if eg.empty:
            st.info("No rows for `eng_Latn+inde` or `hin_Latn+cm` (with their reference conditions) yet.")
        else:
            ms = [m for m in models if m in set(eg["model"])]
            df = pd.DataFrame({"model": eg["model"], "cat": eg["condition"], "value": eg["gap"] * 100, "lo": eg["ci_low"] * 100, "hi": eg["ci_high"] * 100})
            cats = [c for c in ("eng_Latn+inde", "hin_Latn+cm") if c in set(eg["condition"])]
            fig = C.grouped_bars(df, cats, {c: D.cond_label(c) for c in cats}, ms, "Accuracy change vs unmodified (pp)", rotate=0)
            fig_out(fig, "Change in accuracy relative to the unmodified text of the same questions (Indian English vs English; Hinglish vs plain romanized Hindi), "
                         "with 95% bootstrap CI. Negative = the synthetic variant is harder for the model. Exploratory.")
            t = pd.DataFrame([{"Model": mname(r.model), "Condition": D.cond_label(r.condition), "Reference": D.cond_label(r.reference), "n": r.n,
                               "Reference acc %": D.fmt_pp(r.acc_reference, False), "Condition acc %": D.fmt_pp(r.acc_condition, False),
                               "Gap (pp)": D.fmt_pp(r.gap), "95% CI": D.fmt_ci(r.ci_low, r.ci_high), "p (unadjusted)": D.fmt_p(r.p)}
                              for r in eg.itertuples()])
            show(t)
    with t4:
        st.markdown(f"{EXPL}  \nSarvam-30B runs at Q3_K_M while the other models run near Q4. This check runs Llama 3.1 8B at both precisions to estimate "
                    "how much Q4 to Q3 alone changes accuracy.")
        q = D.quantization(ctx["runs_dir"], ctx["sig"], ctx["n_boot"])
        if q.empty:
            st.info("Needs both `llama3.1-8b` and `llama3.1-8b-q3` runs on the same items.")
        else:
            q = q[q["benchmark"] == "belebele"]
            t = pd.DataFrame([{"Condition": D.cond_label(str(r.condition).split(" ")[0]) if not str(r.condition).startswith("ALL") else "All conditions pooled",
                               "n": int(r.n), "Q3 minus Q4 (pp)": D.fmt_pp(r.gap), "95% CI": D.fmt_ci(r.ci_low, r.ci_high), "p (unadjusted)": D.fmt_p(r.p)}
                              for r in q.itertuples()])
            show(t)
            st.caption("Paired by question. The pooled row's interval resamples questions (all conditions of a question stay together).")


def view_gmmlu(ctx: dict) -> None:
    st.header("Global-MMLU")
    st.markdown(f"{EXPL}  \nKnowledge questions in English and in Indian languages. Not one of the five main hypotheses (H1-H5).")
    if need_data(ctx):
        return
    if not ((ctx["items"]["benchmark"] == "gmmlu_lite") | (ctx["items"]["benchmark"] == "gmmlu")).any():
        st.info("No Global-MMLU rows yet. These benchmarks run after Belebele.")
        return
    paths = ctx["paths"]
    gm = D.gmmlu_bundle(ctx["runs_dir"], ctx["sig"], str(paths.raw), D.path_signature(paths.raw / "global_mmlu_lite"), ctx["n_boot"])
    acc = acc_table(ctx)
    tabs = st.tabs(["Global-MMLU-Lite", "Global-MMLU (full)", "Culturally sensitive vs agnostic"])

    def bench_tab(bench: str, expect: list[str], note: str) -> None:
        a = acc[acc["benchmark"] == bench]
        if a.empty:
            st.info(f"No `{bench}` rows yet.")
            return
        provisional_note(ctx, bench)
        st.caption(note)
        models = [m for m in D.ordered_models(a["model"].unique()) if m != "llama3.1-8b-q3"]
        langs = [l for l in expect if l in set(a["condition"])]
        df = pd.DataFrame({"model": a["model"], "cat": a["condition"], "value": a["accuracy"] * 100, "lo": a["ci_low"] * 100, "hi": a["ci_high"] * 100})
        fig = C.grouped_bars(df, langs, LANG_NAME, models, "Accuracy (%)", zero_line=False, ylim=(0, 100), rotate=0)
        fig_out(fig, "Accuracy by language and model with 95% bootstrap confidence intervals. English is the reference; chance is 25%.")
        g = gm["gaps"]
        g = g[g["benchmark"] == bench] if len(g) else g
        if len(g):
            gl = [l for l in expect if l != "en" and l in set(g["condition"])]
            df2 = pd.DataFrame({"model": g["model"], "cat": g["condition"], "value": g["gap"] * 100, "lo": g["ci_low"] * 100, "hi": g["ci_high"] * 100})
            fig = C.grouped_bars(df2, gl, LANG_NAME, [m for m in models if m in set(g["model"])], "Accuracy minus English (pp)", rotate=0)
            fig_out(fig, "Paired gap to English on the same questions (negative = lower than English), 95% bootstrap CI.")
            show(pd.DataFrame([{"Model": mname(r.model), "Language": LANG_NAME.get(r.condition, r.condition), "n": int(r.n),
                                "English acc %": D.fmt_pp(r.acc_reference, False), "Language acc %": D.fmt_pp(r.acc_condition, False),
                                "Gap (pp)": D.fmt_pp(r.gap), "95% CI": D.fmt_ci(r.ci_low, r.ci_high),
                                "p (Holm within benchmark)": D.fmt_p(r.p_holm_within_benchmark)} for r in g.itertuples()]))
        with st.expander("Accuracy table with n, CI and invalid-answer rate"):
            show(pd.DataFrame([{"Model": mname(r.model), "Language": LANG_NAME.get(r.condition, r.condition), "n": int(r.n),
                                "Accuracy % [95% CI]": f"{r.accuracy * 100:.1f} [{r.ci_low * 100:.1f}, {r.ci_high * 100:.1f}]",
                                "Invalid %": round(r.invalid_rate * 100, 1)} for r in a.itertuples()]))

    with tabs[0]:
        bench_tab("gmmlu_lite", ["en", "hi", "bn", "or"],
                  "Global-MMLU-Lite: 400 questions, fully human-translated into Hindi, Bengali and Odia.")
    with tabs[1]:
        bench_tab("gmmlu", ["en", "hi", "bn", "te", "ne"],
                  "Global-MMLU (full): 300 questions. The Telugu and Nepali translations mix machine translation and post-editing, "
                  "so read them separately from the human-translated Lite languages.")
    with tabs[2]:
        cs = gm["cs_ca"]
        if not gm["have_labels"]:
            st.info("The Global-MMLU-Lite files (with the `cultural_sensitivity_label` column) were not found in the raw-data folder.")
        elif cs.empty:
            st.info("No Global-MMLU-Lite rows yet.")
        else:
            provisional_note(ctx, "gmmlu_lite")
            c = cs[cs["label"].isin(["CS", "CA"])]
            models = [m for m in D.ordered_models(c["model"].unique()) if m != "llama3.1-8b-q3"]
            langs = [l for l in ("en", "hi", "bn", "or") if l in set(c["condition"])]
            fig = C.cs_ca_panels(c, models, langs, LANG_NAME)
            fig_out(fig, "Global-MMLU-Lite accuracy on culturally sensitive questions (CS, solid) and culturally agnostic questions (CA, hatched), "
                         "by language, with 95% bootstrap CI. Each question carries a CS/CA label from the dataset's authors; about half the questions are CS.")
            d = gm["cs_diff"]
            if len(d):
                st.markdown("##### Do culturally sensitive questions lose more accuracy than agnostic ones?")
                show(pd.DataFrame([{"Model": mname(r.model), "Language": LANG_NAME.get(r.condition, r.condition), "n CS": int(r.n_cs), "n CA": int(r.n_ca),
                                    "CS gap vs English (pp)": D.fmt_pp(r.gap_cs), "CA gap vs English (pp)": D.fmt_pp(r.gap_ca),
                                    "CS minus CA (pp)": D.fmt_pp(r.diff_cs_minus_ca), "95% CI": D.fmt_ci(r.ci_low, r.ci_high)} for r in d.itertuples()]))
                st.caption("Gap = language accuracy minus English accuracy on the same questions within each label. A negative 'CS minus CA' means "
                           "culturally sensitive questions lose more. If the interval includes 0 the data do not show a difference. No multiple-testing correction.")


VIEW_FUNCS = {VIEWS[0]: view_progress, VIEWS[1]: view_overview, VIEWS[2]: view_native_vs_roman, VIEWS[3]: view_tokens,
              VIEWS[4]: view_items, VIEWS[5]: view_fixes, VIEWS[6]: view_gmmlu}


def main() -> None:
    st.set_page_config(page_title="Language and Script Gaps in Local Language Models for Indian Users: results", page_icon=None, layout="wide")
    ctx = build_context()
    st.title("Language and Script Gaps in Local Language Models for Indian Users: results dashboard")
    st.caption("Does a local model answer the same multiple-choice question worse in Indian languages, in romanized script, in Indian English or in Hinglish? "
               "Recomputed live from the run files. Labels mark each section as PRIMARY (one of the five main hypotheses, H1-H5) or EXPLORATORY.")
    VIEW_FUNCS[ctx["view"]](ctx)
    st.divider()
    st.caption("Indian-script text appears only in the item browser and uses the fonts installed on your computer (macOS has Devanagari, Bengali, Telugu, Tamil and "
               "other Indian scripts built in; on other systems install Noto Sans fonts for the script). Charts use English labels only.")


main()

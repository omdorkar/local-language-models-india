"""Analysis layer: paired gaps, bootstrap CIs, exact McNemar, Holm-corrected hypothesis families.

    .venv/bin/python -m evaluation.stats [--runs runs] [--out reports/results]

Implements the five main hypothesis tests (H1-H5, primary analyses) as described in the paper's Methods
(paper/paper.md, section 3.7); everything else is exploratory and labelled as such in every output file name / table heading.

Conventions
-----------
* Input: runs/<benchmark>/<model_key>.jsonl (docs/CONTRACT.md). Rows with condition "__template__" are used only
  to calibrate the token count of the empty template; they never enter an accuracy.
* Invalid answers (answer null / not A-D) count as wrong; the invalid rate is reported separately.
* ``gap`` in every CSV is a *proportion* (0.10 = 10 percentage points) and is always
  accuracy(condition) - accuracy(reference), over items present in both, EXCEPT where a column says
  "English minus native" (H3 and the exploratory gap-vs-resource files), where positive = native is worse.
* CI: percentile bootstrap over items, 10,000 resamples, numpy default_rng(1909) (a fresh generator per test,
  so results do not depend on the order in which tests are run).
* McNemar: exact (statsmodels), two-sided. b = condition right & reference wrong, c = condition wrong & reference right.
* Holm: statsmodels multipletests('holm'), separately inside each hypothesis family. By default the family is the
  fixed-size family (all models in config.MODELS x all languages); a test that could not be run enters with p = 1,
  which keeps the correction conservative when data are incomplete (``--holm-run-only`` changes this).
* "confirmed" iff Holm-adjusted p < 0.05 AND the sign of the gap equals the predicted direction.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .config import LANGUAGES, MODELS, SEED

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 10_000
ALPHA = 0.05
TEMPLATE_COND = "__template__"
LETTERS = set("ABCD")
ENG_REF = {"belebele": "eng_Latn", "gmmlu_lite": "en", "gmmlu": "en"}
MAIN_MODELS = [m.key for m in MODELS]
Q4_KEY, Q3_KEY = "llama3.1-8b", "llama3.1-8b-q3"
SARVAM_KEY = "sarvam-30b"

CONF_COLUMNS = [
    "hypothesis", "model", "language", "condition", "reference", "n", "acc_condition", "acc_reference",
    "gap", "ci_low", "ci_high", "b", "c", "rho", "p", "p_holm", "predicted_direction", "confirmed", "status",
]


# ----------------------------------------------------------------------------------------------- loading


def load_runs(runs_dir: Path, benchmarks: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Read runs/<benchmark>/*.jsonl.

    Returns (items, templates): ``items`` has one row per (benchmark, model_key, condition, item_id) with
    columns correct (bool, invalid counted wrong), invalid (bool), prompt_tokens; ``templates`` maps
    (benchmark, model_key) -> prompt_tokens of the empty template.
    """
    runs_dir = Path(runs_dir)
    recs, templates = [], {}
    if not runs_dir.exists():
        return pd.DataFrame(), templates
    for path in sorted(runs_dir.glob("*/*.jsonl")):
        bench_dir = path.parent.name
        if benchmarks and bench_dir not in benchmarks:
            continue
        with path.open() as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    warnings.warn(f"{path}:{ln}: unreadable JSON line skipped")
                    continue
                bench = r.get("benchmark") or bench_dir
                model = r.get("model_key") or path.stem
                if r.get("condition") == TEMPLATE_COND:
                    if r.get("prompt_tokens") is not None:
                        templates[(bench, model)] = int(r["prompt_tokens"])
                    continue
                ans = r.get("answer")
                valid = isinstance(ans, str) and ans in LETTERS
                recs.append({
                    "benchmark": bench, "model_key": model, "condition": r.get("condition"),
                    "lang": r.get("lang"), "script": r.get("script"), "source": r.get("source"),
                    "item_id": r.get("item_id"), "gold": r.get("gold"), "answer": ans if valid else None,
                    "invalid": not valid, "correct": bool(valid and ans == r.get("gold")),
                    "prompt_tokens": r.get("prompt_tokens"),
                })
    if not recs:
        return pd.DataFrame(), templates
    df = pd.DataFrame(recs)
    df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    # Resumability key is unique by contract; if a duplicate sneaks in, the last row wins.
    df = df.drop_duplicates(["benchmark", "model_key", "condition", "item_id"], keep="last").reset_index(drop=True)
    return df, templates


def build_tables(items: pd.DataFrame) -> dict[str, dict[tuple[str, str], pd.DataFrame]]:
    """benchmark -> (model, condition) -> DataFrame indexed by item_id with columns correct (0/1), invalid (0/1), tokens."""
    out: dict[str, dict] = {}
    if items.empty:
        return out
    for (b, m, c), g in items.groupby(["benchmark", "model_key", "condition"], sort=False):
        t = pd.DataFrame({
            "correct": g["correct"].astype(int).to_numpy(),
            "invalid": g["invalid"].astype(int).to_numpy(),
            "tokens": g["prompt_tokens"].to_numpy(),
        }, index=g["item_id"].to_numpy())
        out.setdefault(b, {})[(m, c)] = t
    return out


# ------------------------------------------------------------------------------------------ core statistics


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling items with replacement."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // n)
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, n, size=(e - s, n))
        means[s:e] = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)


def paired_gap(cond: pd.Series, ref: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """accuracy(cond) - accuracy(ref) over items present in both, with bootstrap CI and exact McNemar.

    ``cond`` and ``ref`` are 0/1 Series indexed by item_id.
    """
    common = cond.index.intersection(ref.index)
    n = len(common)
    if n == 0:
        return dict(n=0, acc_condition=np.nan, acc_reference=np.nan, gap=np.nan, ci_low=np.nan, ci_high=np.nan,
                    b=0, c=0, p=np.nan)
    x = cond.loc[common].to_numpy(dtype=int)
    y = ref.loc[common].to_numpy(dtype=int)
    d = x - y
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return dict(n=n, acc_condition=float(x.mean()), acc_reference=float(y.mean()), gap=float(d.mean()),
                ci_low=lo, ci_high=hi, b=b, c=c, p=mcnemar_exact(b, c))


def holm(pvals: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if len(p) == 0:
        return p
    return multipletests(p, alpha=ALPHA, method="holm")[1]


def direction_ok(gap: float, predicted: str) -> bool:
    """predicted is 'lower' (gap < 0) or 'higher' (gap > 0). A gap of exactly 0 satisfies neither."""
    if gap is None or not np.isfinite(gap):
        return False
    return gap < 0 if predicted == "lower" else gap > 0 if predicted == "higher" else False


# ------------------------------------------------------------------------------------------ language helpers


def native_languages() -> list[tuple[str, str]]:
    """[(iso, Belebele native-script config)] for the 13 Indian languages (English excluded)."""
    return [(iso, l.belebele_native) for iso, l in LANGUAGES.items() if iso != "en" and l.belebele_native]


def romanized_languages() -> list[tuple[str, str, str]]:
    """[(iso, native config, romanized config)] for Hindi, Bengali, Urdu, Nepali."""
    return [(iso, l.belebele_native, l.belebele_latn) for iso, l in LANGUAGES.items() if l.belebele_latn]


def condition_lang(cond: str) -> str | None:
    base = cond.split("+")[0]
    for iso, l in LANGUAGES.items():
        if base in (l.belebele_native, l.belebele_latn) or base == iso:
            return iso
    return None


# -------------------------------------------------------------------------------------------- token ratios


def token_ratios(tables: dict, templates: dict) -> pd.DataFrame:
    """Per (benchmark, model, condition): mean net prompt tokens and ratio to the English condition.

    net tokens = prompt_tokens - template_tokens (template_tokens from the model's ``__template__`` row).
    Ratios use only items present (with tokens) in both the condition and English. ``token_ratio_raw`` ignores the
    template (tokens per item / English tokens per item, as defined in the paper, section 3.7) and is
    reported for comparison; H3 uses ``token_ratio_net``.
    """
    rows = []
    for bench, tb in tables.items():
        ref_c = ENG_REF.get(bench)
        for (model, cond), t in tb.items():
            ref = tb.get((model, ref_c))
            tmpl = templates.get((bench, model))
            row = dict(benchmark=bench, model=model, condition=cond, language=condition_lang(cond) if bench == "belebele" else cond,
                       template_tokens=tmpl, n_items=0, mean_tokens_raw=np.nan, mean_tokens_net=np.nan,
                       eng_mean_tokens_raw=np.nan, eng_mean_tokens_net=np.nan,
                       token_ratio_raw=np.nan, token_ratio_net=np.nan, status="ok")
            if ref is None:
                row["status"] = "no English reference condition"
            else:
                common = t.index.intersection(ref.index)
                tc = t.loc[common, "tokens"].astype(float)
                tr = ref.loc[common, "tokens"].astype(float)
                ok = tc.notna() & tr.notna()
                tc, tr = tc[ok], tr[ok]
                row["n_items"] = int(len(tc))
                if len(tc) == 0:
                    row["status"] = "no token counts"
                else:
                    row["mean_tokens_raw"], row["eng_mean_tokens_raw"] = float(tc.mean()), float(tr.mean())
                    row["token_ratio_raw"] = float(tc.mean() / tr.mean())
                    if tmpl is None:
                        row["status"] = "no __template__ row for this model"
                    else:
                        nc, nr = tc - tmpl, tr - tmpl
                        row["mean_tokens_net"], row["eng_mean_tokens_net"] = float(nc.mean()), float(nr.mean())
                        row["token_ratio_net"] = float(nc.mean() / nr.mean()) if nr.mean() > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ accuracy table


def accuracy_table(tables: dict, n_boot: int = N_BOOT) -> pd.DataFrame:
    meta = {}
    rows = []
    for bench, tb in tables.items():
        for (model, cond), t in tb.items():
            lo, hi = bootstrap_ci(t["correct"].to_numpy(), n_boot)
            rows.append(dict(benchmark=bench, model=model, condition=cond, n=len(t), n_correct=int(t["correct"].sum()),
                             accuracy=float(t["correct"].mean()), ci_low=lo, ci_high=hi,
                             invalid_rate=float(t["invalid"].mean())))
    df = pd.DataFrame(rows)
    return df.sort_values(["benchmark", "model", "condition"]).reset_index(drop=True) if len(df) else df


# ----------------------------------------------------------------------------------- hypothesis families (H1-H5)


def _row(hyp, model, lang, cond, ref, predicted):
    return dict(hypothesis=hyp, model=model, language=lang, condition=cond, reference=ref, n=0,
                acc_condition=np.nan, acc_reference=np.nan, gap=np.nan, ci_low=np.nan, ci_high=np.nan,
                b=np.nan, c=np.nan, rho=np.nan, p=np.nan, p_holm=np.nan, predicted_direction=predicted,
                confirmed=pd.NA, status="not run")


def _paired_family(hyp: str, specs: list[tuple[str, str, str, str]], predicted: str, bel: dict, models: list[str],
                   n_boot: int, holm_planned: bool) -> pd.DataFrame:
    """specs: (iso, condition, reference_condition); one test per model x spec."""
    rows = []
    for model in models:
        for iso, cond, ref in [(s[0], s[1], s[2]) for s in specs]:
            r = _row(hyp, model, iso, cond, ref, predicted)
            tc, tr = bel.get((model, cond)), bel.get((model, ref))
            if tc is None:
                r["status"] = f"not run: no rows for condition {cond}"
            elif tr is None:
                r["status"] = f"not run: no rows for reference {ref}"
            else:
                res = paired_gap(tc["correct"], tr["correct"], n_boot)
                if res["n"] == 0:
                    r["status"] = "not run: no items in common"
                else:
                    r.update(res)
                    r["status"] = "ok"
            rows.append(r)
    df = pd.DataFrame(rows, columns=CONF_COLUMNS)
    return _finish_family(df, predicted, holm_planned)


def _finish_family(df: pd.DataFrame, predicted: str, holm_planned: bool) -> pd.DataFrame:
    ok = df["status"] == "ok"
    p = df["p"].astype(float).copy()
    if holm_planned:
        padj = holm(p.fillna(1.0).to_numpy())
        df["p_holm"] = np.where(ok, padj, np.nan)
    else:
        padj = holm(p[ok].to_numpy())
        df.loc[ok, "p_holm"] = padj
    stat = df["rho"] if predicted == "rho>0" else df["gap"]
    dir_ok = [direction_ok(s, "higher" if predicted in ("higher", "rho>0") else "lower") if o else False
              for s, o in zip(stat, ok)]
    conf = [bool(o and ph < ALPHA and d) for o, ph, d in zip(ok, df["p_holm"].fillna(1.0), dir_ok)]
    df["confirmed"] = pd.array([c if o else pd.NA for c, o in zip(conf, ok)], dtype="boolean")
    return df


def h3_family(tables: dict, templates: dict, models: list[str], holm_planned: bool, n_boot: int = N_BOOT):
    """H3: per model, Spearman rho (token ratio vs English-minus-native gap) across the 13 languages.

    Returns (rows DataFrame, points DataFrame of per-language token ratio and gap).
    """
    bel = tables.get("belebele", {})
    tr_df = token_ratios({"belebele": bel}, templates) if bel else pd.DataFrame()
    rows, points = [], []
    for model in models:
        r = _row("H3", model, "all 13 native-script languages", "native vs eng_Latn", "eng_Latn", "rho>0")
        eng = bel.get((model, "eng_Latn"))
        pts = []
        if eng is None:
            r["status"] = "not run: no rows for reference eng_Latn"
        elif templates.get(("belebele", model)) is None:
            r["status"] = f"not run: no {TEMPLATE_COND} row for {model}"
        else:
            for iso, cond in native_languages():
                t = bel.get((model, cond))
                if t is None:
                    continue
                tr = tr_df[(tr_df["model"] == model) & (tr_df["condition"] == cond)]
                ratio = float(tr["token_ratio_net"].iloc[0]) if len(tr) else np.nan
                g = paired_gap(t["correct"], eng["correct"], n_boot=10)  # CI not needed here
                if g["n"] and np.isfinite(ratio):
                    pts.append(dict(model=model, language=iso, condition=cond, n=g["n"], token_ratio=ratio,
                                    gap_eng_minus_native=-g["gap"]))
            r["n"] = len(pts)
            if len(pts) < 5:
                r["status"] = f"not run: only {len(pts)} languages with data (need >= 5)"
            else:
                x = [q["token_ratio"] for q in pts]
                y = [q["gap_eng_minus_native"] for q in pts]
                if np.ptp(x) == 0 or np.ptp(y) == 0:
                    r["status"] = "not run: token ratio or gap constant across languages"
                else:
                    res = sps.spearmanr(x, y, alternative="greater")
                    r["rho"], r["p"], r["status"] = float(res.statistic), float(res.pvalue), "ok"
        rows.append(r)
        points.extend(pts)
    df = _finish_family(pd.DataFrame(rows, columns=CONF_COLUMNS), "rho>0", holm_planned)
    return df, pd.DataFrame(points)


def confirmatory(tables: dict, templates: dict, models: list[str] | None = None, n_boot: int = N_BOOT,
                 holm_planned: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = models or MAIN_MODELS
    bel = tables.get("belebele", {})
    nat = native_languages()
    lat = romanized_languages()
    h1 = _paired_family("H1", [(i, c, "eng_Latn") for i, c in nat], "lower", bel, models, n_boot, holm_planned)
    h2 = _paired_family("H2", [(i, lt, n) for i, n, lt in lat], "lower", bel, models, n_boot, holm_planned)
    h3, points = h3_family(tables, templates, models, holm_planned, n_boot)
    h4 = _paired_family("H4", [(i, f"{lt}+xlit", lt) for i, n, lt in lat], "higher", bel, models, n_boot, holm_planned)
    h5 = _paired_family("H5", [(i, f"{n}+mt_en", n) for i, n in nat], "higher", bel, models, n_boot, holm_planned)
    out = pd.concat([h1, h2, h3, h4, h5], ignore_index=True)
    for col in ("n", "b", "c"):
        out[col] = out[col].astype("Int64")
    out["predicted_direction"] = out["predicted_direction"].replace(
        {"lower": "gap < 0 (condition lower than reference)", "higher": "gap > 0 (condition higher than reference)",
         "rho>0": "rho > 0 (one-sided)"})
    return out, points


# --------------------------------------------------------------------------------------------- exploratory


def gmmlu_gaps(tables: dict, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for bench in ("gmmlu_lite", "gmmlu"):
        tb = tables.get(bench, {})
        models = sorted({m for m, _ in tb})
        sub = []
        for model in models:
            ref = tb.get((model, "en"))
            if ref is None:
                continue
            for (m, cond), t in sorted(tb.items()):
                if m != model or cond == "en":
                    continue
                res = paired_gap(t["correct"], ref["correct"], n_boot)
                sub.append(dict(benchmark=bench, model=model, condition=cond, reference="en", **res))
        if sub:
            s = pd.DataFrame(sub)
            s["p_holm_within_benchmark"] = holm(s["p"].fillna(1.0).to_numpy())
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_cs_labels(raw_dir: Path) -> pd.Series | None:
    """sample_id -> 'CS' / 'CA' from Global-MMLU-Lite (label is a property of the item, identical across languages)."""
    base = Path(raw_dir) / "global_mmlu_lite"
    files = sorted(base.glob("en/test-*.parquet")) + sorted(base.glob("*/test-*.parquet"))
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["sample_id", "cultural_sensitivity_label"])
        except Exception:  # noqa: BLE001
            continue
        return d.drop_duplicates("sample_id").set_index("sample_id")["cultural_sensitivity_label"]
    return None


def gmmlu_lite_cs_ca(tables: dict, labels: pd.Series | None, n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    tb = tables.get("gmmlu_lite", {})
    if labels is None or not tb:
        return pd.DataFrame(), pd.DataFrame()
    acc_rows, diff_rows = [], []
    for model in sorted({m for m, _ in tb}):
        ref = tb.get((model, "en"))
        for (m, cond), t in sorted(tb.items()):
            if m != model:
                continue
            lab = labels.reindex(t.index)
            for name, mask in (("all", lab.notna()), ("CS", lab == "CS"), ("CA", lab == "CA")):
                sub = t[mask.to_numpy()]
                if sub.empty:
                    continue
                lo, hi = bootstrap_ci(sub["correct"].to_numpy(), n_boot)
                row = dict(model=model, condition=cond, label=name, n=len(sub), accuracy=float(sub["correct"].mean()),
                           ci_low=lo, ci_high=hi, gap_vs_en=np.nan, gap_ci_low=np.nan, gap_ci_high=np.nan,
                           b=np.nan, c=np.nan, p=np.nan)
                if ref is not None and cond != "en":
                    rsub = ref[ref.index.isin(sub.index)]
                    g = paired_gap(sub["correct"], rsub["correct"], n_boot)
                    row.update(gap_vs_en=g["gap"], gap_ci_low=g["ci_low"], gap_ci_high=g["ci_high"], b=g["b"], c=g["c"], p=g["p"])
                acc_rows.append(row)
            if ref is not None and cond != "en":
                parts = {}
                for name in ("CS", "CA"):
                    ids = labels.index[labels == name]
                    ids = t.index.intersection(ref.index).intersection(ids)
                    parts[name] = (t.loc[ids, "correct"] - ref.loc[ids, "correct"]).to_numpy()
                if all(len(v) for v in parts.values()):
                    rng = np.random.default_rng(SEED)
                    bs = []
                    for v in parts.values():
                        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
                        bs.append(v[idx].mean(axis=1))
                    diff = bs[0] - bs[1]
                    diff_rows.append(dict(
                        model=model, condition=cond, n_cs=len(parts["CS"]), n_ca=len(parts["CA"]),
                        gap_cs=float(parts["CS"].mean()), gap_ca=float(parts["CA"].mean()),
                        diff_cs_minus_ca=float(parts["CS"].mean() - parts["CA"].mean()),
                        ci_low=float(np.percentile(diff, 2.5)), ci_high=float(np.percentile(diff, 97.5))))
    return pd.DataFrame(acc_rows), pd.DataFrame(diff_rows)


def quantization(tables: dict, n_boot: int = N_BOOT) -> pd.DataFrame:
    """llama3.1-8b-q3 minus llama3.1-8b, paired by item, per condition and pooled over conditions."""
    rows = []
    for bench, tb in tables.items():
        conds = sorted({c for (m, c) in tb if m == Q3_KEY and (Q4_KEY, c) in tb})
        pooled_d, pooled_ids = [], []
        for cond in conds:
            res = paired_gap(tb[(Q3_KEY, cond)]["correct"], tb[(Q4_KEY, cond)]["correct"], n_boot)
            rows.append(dict(benchmark=bench, condition=cond, **res))
            common = tb[(Q3_KEY, cond)].index.intersection(tb[(Q4_KEY, cond)].index)
            d = tb[(Q3_KEY, cond)].loc[common, "correct"] - tb[(Q4_KEY, cond)].loc[common, "correct"]
            pooled_d.append(d)
        if pooled_d:
            allv = pd.concat(pooled_d, keys=range(len(pooled_d)))
            per_item = allv.groupby(level=1).agg(["sum", "count"])
            s, k = per_item["sum"].to_numpy(float), per_item["count"].to_numpy(float)
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(s), size=(n_boot, len(s)))
            est = s[idx].sum(axis=1) / k[idx].sum(axis=1)
            tot_b = int((allv == 1).sum())
            tot_c = int((allv == -1).sum())
            rows.append(dict(benchmark=bench, condition="ALL (pooled over conditions; CI clusters on item; p not computed)",
                             n=int(k.sum()), acc_condition=np.nan, acc_reference=np.nan, gap=float(s.sum() / k.sum()),
                             ci_low=float(np.percentile(est, 2.5)), ci_high=float(np.percentile(est, 97.5)),
                             b=tot_b, c=tot_c, p=np.nan))
    return pd.DataFrame(rows)


def gap_table(tables: dict, models: list[str], n_boot: int = 10) -> pd.DataFrame:
    """English-minus-native Belebele accuracy gap per model x language (point estimates; used by several exploratory views)."""
    bel = tables.get("belebele", {})
    rows = []
    for model in models:
        eng = bel.get((model, "eng_Latn"))
        if eng is None:
            continue
        for iso, cond in native_languages():
            t = bel.get((model, cond))
            if t is None:
                continue
            g = paired_gap(t["correct"], eng["correct"], n_boot)
            if g["n"]:
                rows.append(dict(model=model, language=iso, condition=cond, n=g["n"],
                                 acc_native=g["acc_condition"], acc_english=g["acc_reference"],
                                 gap_eng_minus_native=-g["gap"]))
    return pd.DataFrame(rows)


def sarvam_vs_others(gaps: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: no test, no CI. Sarvam's English-minus-native gap vs the mean of the other main models."""
    if gaps.empty or SARVAM_KEY not in set(gaps["model"]):
        return pd.DataFrame()
    others = [m for m in MAIN_MODELS if m != SARVAM_KEY]
    piv = gaps.pivot(index="language", columns="model", values="gap_eng_minus_native")
    cols = [m for m in others if m in piv.columns]
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame({
        "sarvam_gap": piv[SARVAM_KEY], "others_mean_gap": piv[cols].mean(axis=1),
        "others_min_gap": piv[cols].min(axis=1), "others_max_gap": piv[cols].max(axis=1)})
    out["sarvam_minus_others_mean"] = out["sarvam_gap"] - out["others_mean_gap"]
    out["n_other_models"] = len(cols)
    out.loc["MEAN over languages"] = [out["sarvam_gap"].mean(), out["others_mean_gap"].mean(),
                                      out["others_min_gap"].mean(), out["others_max_gap"].mean(),
                                      out["sarvam_minus_others_mean"].mean(), len(cols)]
    out = out.reset_index().rename(columns={"index": "language"})
    return out


def resource_tables(gaps: pd.DataFrame, langstats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gap vs language resource level (speakers, Wikipedia articles). Exploratory; Spearman, two-sided."""
    if gaps.empty or langstats.empty:
        return pd.DataFrame(), pd.DataFrame()
    ls = langstats.drop(columns=["language"], errors="ignore").rename(columns={"iso": "language", "census2011_first_language_speakers": "speakers"})
    pts = gaps.merge(ls[["language", "speakers", "wikipedia_articles"]], on="language", how="left")
    rows = []
    groups = [(m, g) for m, g in pts.groupby("model")]
    mean_g = pts[pts["model"].isin(MAIN_MODELS)].groupby("language", as_index=False).agg(
        gap_eng_minus_native=("gap_eng_minus_native", "mean"), speakers=("speakers", "first"),
        wikipedia_articles=("wikipedia_articles", "first"))
    groups.append(("MEAN over the 5 main models", mean_g))
    for name, g in groups:
        for res in ("speakers", "wikipedia_articles"):
            gg = g.dropna(subset=[res, "gap_eng_minus_native"])
            gg = gg[gg[res] > 0]
            if len(gg) >= 5 and gg["gap_eng_minus_native"].nunique() > 1:
                r = sps.spearmanr(np.log10(gg[res].astype(float)), gg["gap_eng_minus_native"])
                rows.append(dict(model=name, resource=res, n_languages=len(gg), rho=float(r.statistic), p_two_sided=float(r.pvalue)))
            else:
                rows.append(dict(model=name, resource=res, n_languages=len(gg), rho=np.nan, p_two_sided=np.nan))
    return pts, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ summary


def _pp(x, signed=True):
    return "n/a" if x is None or not np.isfinite(x) else (f"{x * 100:+.1f}" if signed else f"{x * 100:.1f}")


def _pf(p) -> str:
    return "" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _md(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _mark(row) -> str:
    if row["status"] != "ok":
        return "not run"
    sig = row["p_holm"] < ALPHA
    pred = row["predicted_direction"]
    if row["confirmed"]:
        return "✓"
    if sig:
        return "⚠ opposite direction"
    return "✗"


def _family_pivot(df: pd.DataFrame, models: list[str]) -> str:
    langs = list(dict.fromkeys(df["language"]))
    rows = []
    for lg in langs:
        row = [lg]
        for m in models:
            r = df[(df["model"] == m) & (df["language"] == lg)]
            if r.empty:
                row.append("n/a")
                continue
            r = r.iloc[0]
            row.append(f"{_pp(r['gap'])} {_mark(r)}" if r["status"] == "ok" else "not run")
        rows.append(row)
    conf_row = ["**confirmed**"]
    for m in models:
        sub = df[df["model"] == m]
        ran = int((sub["status"] == "ok").sum())
        conf_row.append(f"{int(sub['confirmed'].fillna(False).astype(bool).sum())}/{len(sub)}" + (f" ({len(sub) - ran} not run)" if ran < len(sub) else ""))
    rows.append(conf_row)
    return _md(["language"] + models, rows)


def _interpret(hyp: str, df: pd.DataFrame, lower: bool, ref: str = "its reference condition") -> list[str]:
    ran = df[df["status"] == "ok"]
    total = len(df)
    conf = int(ran["confirmed"].astype(bool).sum()) if len(ran) else 0
    nr = total - len(ran)
    opp = int(((ran["p_holm"] < ALPHA) & ~ran["confirmed"].astype(bool)).sum()) if len(ran) else 0
    nonsig = len(ran) - conf - opp
    lines = [f"**{hyp} confirmed for {conf}/{total} model-language pairs**" + (f" ({nr} not run)." if nr else ".")]
    if len(ran) == 0:
        lines.append("No data yet for this family.")
        return lines
    word = "lower than" if lower else "higher than"
    lines.append(f"In the {conf} confirmed pairs, accuracy was significantly {word} {ref} (Holm-adjusted p < 0.05).")
    if nonsig:
        lines.append(f"In {nonsig} pairs the data do not show a significant difference in the predicted direction; "
                     "that is absence of evidence at this sample size, not evidence that there is no difference.")
    if opp:
        lines.append(f"In {opp} pairs the difference was significant in the *opposite* direction to the prediction.")
    return lines


def write_summary(out: Path, conf: pd.DataFrame, acc: pd.DataFrame, tokens: pd.DataFrame, explo: dict, models: list[str],
                  holm_planned: bool, runs_info: str) -> None:
    L = ["# Language and Script Gaps in Local Language Models for Indian Users: results summary", "",
         f"_Generated by `python -m evaluation.stats`. {runs_info}_", "",
         "## How to read this file", "",
         "* **Primary** sections (H1-H5) are the five main hypothesis tests described in the paper's Methods and nothing else. "
         "Every test is shown whether or not it is significant.",
         "* Cells show the paired accuracy gap in percentage points (condition minus reference; for H3 the Spearman rho) "
         "followed by the verdict: ✓ confirmed (Holm-adjusted p < 0.05 and gap in the predicted direction), "
         "✗ not confirmed, ⚠ significant but in the opposite direction, `not run` = no data for that test.",
         f"* Holm correction is applied separately inside each hypothesis family, over the {'fixed-size' if holm_planned else 'run'} family "
         "(5 models x languages)" + ("; tests that could not run count as p = 1." if holm_planned else "."),
         "* Gaps are accuracy(condition) - accuracy(reference) on the same items; invalid answers count as wrong. "
         "CIs, p-values, b/c discordant counts and n are in `confirmatory.csv`.", ""]
    fam = {h: conf[conf["hypothesis"] == h] for h in ("H1", "H2", "H3", "H4", "H5")}
    titles = {
        "H1": ("H1 (Belebele): native script vs English; predicted lower", True),
        "H2": ("H2 (Belebele): romanized vs native script, Hindi/Bengali/Urdu/Nepali; predicted lower", True),
        "H4": ("H4 (Belebele): IndicXlit-converted romanized input vs unconverted romanized; predicted higher", False),
        "H5": ("H5 (Belebele): translated to English (NLLB) vs native input; predicted higher", False),
    }
    refs = {"H1": "English", "H2": "the native-script version", "H4": "the unconverted romanized version",
            "H5": "the untranslated native-script version"}
    for h in ("H1", "H2"):
        df = fam[h]
        L += [f"## {titles[h][0]}", "", _family_pivot(df, models) if len(df) else "No data.", ""]
        L += _interpret(h, df, titles[h][1], refs[h]) + [""]
    # H3
    df = fam["H3"]
    L += ["## H3 (Belebele): token ratio vs accuracy gap, Spearman rho across 13 languages; predicted rho > 0 (one-sided)", ""]
    rows = []
    for _, r in df.iterrows():
        if r["status"] == "ok":
            rows.append([r["model"], int(r["n"]), f"{r['rho']:+.2f}", _pf(r["p"]), _pf(r["p_holm"]), "✓" if r["confirmed"] else "✗"])
        else:
            rows.append([r["model"], int(r["n"]), "", "", "", r["status"]])
    L += [_md(["model", "languages", "rho", "p (one-sided)", "p Holm", "verdict"], rows), ""]
    ran = df[df["status"] == "ok"]
    k = int(ran["confirmed"].astype(bool).sum()) if len(ran) else 0
    L += [f"**H3 confirmed for {k}/{len(df)} models**" + (f" ({len(df) - len(ran)} not run)." if len(ran) < len(df) else "."),
          "Where confirmed, languages that need more tokens than English tend to show a larger English-minus-native gap "
          "(an association across 13 languages, not evidence that tokens cause the gap). Elsewhere the data do not show a "
          "significant association.", ""]
    for h in ("H4", "H5"):
        df = fam[h]
        L += [f"## {titles[h][0]}", ""]
        if len(df) and (df["status"] == "ok").any():
            L += [_family_pivot(df, models), ""]
        else:
            L += ["**Not run:** no rows for the derived conditions yet.", ""]
        L += _interpret(h, df, titles[h][1], refs[h]) + [""]

    # Exploratory
    L += ["---", "", "# EXPLORATORY analyses (not among the five main hypotheses; no primary claims)", ""]
    if not acc.empty:
        bel = acc[acc["benchmark"] == "belebele"]
        L += ["## Accuracy overview (full table: `accuracy.csv`)", ""]
        rows = []
        for m in models + [q for q in sorted(set(acc["model"])) if q not in models]:
            sub = bel[bel["model"] == m]
            if sub.empty:
                continue
            e = sub[sub["condition"] == "eng_Latn"]
            nat = sub[sub["condition"].isin([c for _, c in native_languages()])]
            rom = sub[sub["condition"].isin([lt for _, _, lt in romanized_languages()])]
            rows.append([m, f"{_pp(e['accuracy'].iloc[0], False)}" if len(e) else "n/a",
                         f"{_pp(nat['accuracy'].mean(), False)} ({len(nat)})" if len(nat) else "n/a",
                         f"{_pp(rom['accuracy'].mean(), False)} ({len(rom)})" if len(rom) else "n/a",
                         f"{_pp(sub['invalid_rate'].mean(), False)}", f"{_pp(sub['invalid_rate'].max(), False)}"])
        L += [_md(["model", "English acc %", "mean native acc % (configs)", "mean romanized acc % (configs)", "mean invalid %", "max invalid % (any condition)"], rows), ""]
    for name, key, blurb in (
        ("Global-MMLU-Lite and Global-MMLU: gap vs English", "gmmlu", "Unadjusted exact McNemar p; Holm only within benchmark (column in CSV). Descriptive."),
    ):
        g = explo.get("gmmlu_gaps")
        L += [f"## {name} (`gmmlu_gaps.csv`)", ""]
        if g is not None and len(g):
            rows = [[r["benchmark"], r["model"], r["condition"], int(r["n"]), _pp(r["gap"]),
                     f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]", _pf(r["p_holm_within_benchmark"])] for _, r in g.iterrows()]
            L += [blurb, "", _md(["benchmark", "model", "condition", "n", "gap pp", "95% CI", "p Holm (in benchmark)"], rows), ""]
        else:
            L += ["No data.", ""]
    d = explo.get("cs_ca_diff")
    L += ["## Global-MMLU-Lite: culturally sensitive (CS) vs agnostic (CA) (`gmmlu_lite_cs_ca.csv`, `gmmlu_lite_cs_vs_ca_gap_difference.csv`)", ""]
    if d is not None and len(d):
        rows = [[r["model"], r["condition"], _pp(r["gap_cs"]), _pp(r["gap_ca"]), _pp(r["diff_cs_minus_ca"]),
                 f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]"] for _, r in d.iterrows()]
        L += ["Gap vs English (pp) within each label; difference = CS gap minus CA gap (negative = CS items lose more). "
              "Bootstrap CI over items, no multiplicity correction. If the CI includes 0 the data do not show a CS/CA difference.", "",
              _md(["model", "condition", "CS gap", "CA gap", "CS - CA", "95% CI of difference"], rows), ""]
    else:
        L += ["No data (or `cultural_sensitivity_label` parquet not found).", ""]
    q = explo.get("quantization")
    L += ["## Quantization sensitivity: Llama 3.1 8B Q3_K_M minus Q4_K_M (`quantization.csv`)", ""]
    if q is not None and len(q):
        rows = [[r["benchmark"], r["condition"], int(r["n"]), _pp(r["gap"]), f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]",
                 int(r["b"]), int(r["c"]), _pf(r["p"])] for _, r in q.iterrows()]
        L += ["Gap = q3 accuracy - q4 accuracy (pp), paired by item. Shows how much Q4 to Q3 alone moves accuracy (relevant to Sarvam, which runs at Q3).", "",
              _md(["benchmark", "condition", "n", "gap pp", "95% CI", "b", "c", "p (unadjusted)"], rows), ""]
    else:
        L += ["No data (needs both `llama3.1-8b` and `llama3.1-8b-q3` runs).", ""]
    s = explo.get("sarvam")
    L += ["## Sarvam-30B vs the mean of the other four models (`sarvam_vs_others.csv`; descriptive only)", ""]
    if s is not None and len(s):
        rows = [[r["language"], _pp(r["sarvam_gap"], False), _pp(r["others_mean_gap"], False), _pp(r["sarvam_minus_others_mean"])] for _, r in s.iterrows()]
        L += ["English-minus-native accuracy gap (pp; positive = native is worse). One India-built model cannot support a general claim about origin; "
              "no test is run. Sarvam also runs at Q3_K_M (a known confound).", "",
              _md(["language", "Sarvam gap", "mean gap, other models", "Sarvam minus others"], rows), ""]
    else:
        L += ["No data.", ""]
    rc = explo.get("resource_corr")
    L += ["## Accuracy gap vs language resource level (`gap_vs_resource.csv`, `resource_correlation.csv`)", ""]
    if rc is not None and len(rc):
        rows = [[r["model"], r["resource"], int(r["n_languages"]), "" if not np.isfinite(r["rho"]) else f"{r['rho']:+.2f}",
                 _pf(r["p_two_sided"])] for _, r in rc.iterrows()]
        L += ["Spearman rho between the English-minus-native gap and log10(resource) across the 13 languages "
              "(speakers: Census of India 2011; Wikipedia article count). Two-sided, unadjusted. A negative rho means "
              "better-resourced languages have smaller gaps.", "", _md(["model", "resource", "languages", "rho", "p"], rows), ""]
    else:
        L += ["No data or `data/external/language_stats.csv` missing.", ""]
    L += ["## Token ratio (`token_ratio.csv`)", "",
          "Net tokens per item (prompt tokens minus the model's empty-template tokens) relative to English, per model and condition. "
          "H3 uses `token_ratio_net`; `token_ratio_raw` (no template subtraction) is given for comparison.", ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------- driver


def run(runs: Path, out: Path, raw: Path | None = None, n_boot: int = N_BOOT, holm_planned: bool = True,
        langstats_csv: Path | None = None) -> dict:
    runs, out = Path(runs), Path(out)
    raw = Path(raw) if raw else ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    items, templates = load_runs(runs)
    tables = build_tables(items)
    models = MAIN_MODELS
    try:
        runs_shown = runs.resolve().relative_to(ROOT.resolve()).as_posix() + "/"
    except ValueError:
        runs_shown = runs.as_posix()
    info = (f"Runs read from `{runs_shown}`: {len(items):,} answer rows, "
            f"{items['model_key'].nunique() if len(items) else 0} models, {len(templates)} template calibration rows.")
    acc = accuracy_table(tables, n_boot) if tables else pd.DataFrame()
    tok = token_ratios(tables, templates) if tables else pd.DataFrame()
    conf, h3pts = confirmatory(tables, templates, models, n_boot, holm_planned)
    gaps = gap_table(tables, [m for m in sorted({m for m, _ in tables.get("belebele", {})})])
    from .langstats import load_language_stats
    ls = load_language_stats(langstats_csv)
    res_pts, res_corr = resource_tables(gaps, ls)
    labels = load_cs_labels(raw)
    cs_ca, cs_diff = gmmlu_lite_cs_ca(tables, labels, n_boot)
    explo = dict(
        gmmlu_gaps=gmmlu_gaps(tables, n_boot), cs_ca=cs_ca, cs_ca_diff=cs_diff,
        quantization=quantization(tables, n_boot), sarvam=sarvam_vs_others(gaps),
        resource_pts=res_pts, resource_corr=res_corr)

    def w(df, name):
        if df is not None and len(df):
            df.to_csv(out / name, index=False)

    w(acc, "accuracy.csv")
    w(conf, "confirmatory.csv")
    w(tok, "token_ratio.csv")
    w(h3pts, "gap_vs_token_ratio.csv")
    w(gaps, "belebele_gap_by_language.csv")
    w(explo["gmmlu_gaps"], "gmmlu_gaps.csv")
    w(cs_ca, "gmmlu_lite_cs_ca.csv")
    w(cs_diff, "gmmlu_lite_cs_vs_ca_gap_difference.csv")
    w(explo["quantization"], "quantization.csv")
    w(explo["sarvam"], "sarvam_vs_others.csv")
    w(res_pts, "gap_vs_resource.csv")
    w(res_corr, "resource_correlation.csv")
    write_summary(out, conf, acc, tok, explo, models, holm_planned, info)
    return dict(items=items, tables=tables, accuracy=acc, confirmatory=conf, token_ratio=tok, h3_points=h3pts,
                gaps=gaps, **explo)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "results")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw", help="for Global-MMLU-Lite CS/CA labels")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--holm-run-only", action="store_true",
                    help="Holm over tests that ran (default: fixed-size family, missing tests count as p=1)")
    ap.add_argument("--langstats", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a.runs, a.out, a.raw, a.n_boot, not a.holm_run_only, a.langstats)
    conf = res["confirmatory"]
    print(f"wrote results to {a.out}")
    for h, g in conf.groupby("hypothesis"):
        ran = int((g["status"] == "ok").sum())
        print(f"  {h}: confirmed {int(g['confirmed'].fillna(False).astype(bool).sum())}/{len(g)} ({len(g) - ran} not run)")


if __name__ == "__main__":
    main()

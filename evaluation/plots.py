"""Figures for the paper and slides. Reads the CSVs written by evaluation.stats; draws nothing from raw runs.

    .venv/bin/python -m evaluation.plots [--results reports/results] [--out reports/figures]

Every figure is saved as PNG (200 dpi) and SVG. Palette: Okabe-Ito (colour-blind safe). A figure whose input
CSV is missing or empty is skipped with a message.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from .config import LANGUAGES, MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
MODEL_COLOR = {"gemma3-12b": OI["blue"], "qwen3-8b": OI["orange"], "llama3.1-8b": OI["green"],
               "gpt-oss-20b": OI["vermillion"], "sarvam-30b": OI["purple"], "llama3.1-8b-q3": OI["sky"]}
MODEL_NAME = {m.key: m.display for m in MODELS} | {"llama3.1-8b-q3": "Llama 3.1 8B (Q3)"}
MODEL_ORDER = [m.key for m in MODELS]
LANG_NAME = {i: l.name for i, l in LANGUAGES.items()}
NATIVE_ORDER = [i for i, l in LANGUAGES.items() if i != "en" and l.belebele_native]
ROMAN = [(i, l.belebele_native, l.belebele_latn) for i, l in LANGUAGES.items() if l.belebele_latn]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.bbox": "tight",
})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    print(f"  wrote {out / name}.png/.svg")


def _read(results: Path, name: str) -> pd.DataFrame | None:
    p = results / name
    if not p.exists():
        print(f"  skip: {p} not found")
        return None
    df = pd.read_csv(p)
    if df.empty:
        print(f"  skip: {p} is empty")
        return None
    return df


def _models_in(df: pd.DataFrame, col: str = "model") -> list[str]:
    present = set(df[col])
    return [m for m in MODEL_ORDER if m in present]


def _text_color(rgba) -> str:
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


# 1 --------------------------------------------------------------------------------------------------------
def fig_heatmap(results: Path, out: Path) -> bool:
    conf = _read(results, "confirmatory.csv")
    if conf is None:
        return False
    h1 = conf[(conf["hypothesis"] == "H1") & (conf["status"] == "ok")]
    if h1.empty:
        print("  skip heatmap: no H1 rows")
        return False
    models = _models_in(h1)
    langs = [l for l in NATIVE_ORDER if l in set(h1["language"])]
    gap = np.full((len(models), len(langs)), np.nan)
    star = np.zeros_like(gap, dtype=bool)
    for _, r in h1.iterrows():
        i, j = models.index(r["model"]), langs.index(r["language"])
        gap[i, j] = -r["gap"] * 100  # English minus native, percentage points
        star[i, j] = bool(r["confirmed"])
    vmax = max(5.0, np.nanmax(np.abs(gap)))
    fig, ax = plt.subplots(figsize=(1.0 * len(langs) + 3, 0.8 * len(models) + 2.6))
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(gap, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(models)):
        for j in range(len(langs)):
            if np.isfinite(gap[i, j]):
                ax.text(j, i, f"{gap[i, j] + 0.0:+.1f}{'*' if star[i, j] else ''}".replace("-0.0", "+0.0"), ha="center", va="center", fontsize=11,
                        color=_text_color(cmap(norm(gap[i, j]))))
    ax.set_xticks(range(len(langs)), [LANG_NAME[l] for l in langs], rotation=45, ha="right")
    ax.set_yticks(range(len(models)), [MODEL_NAME[m] for m in models])
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("English minus native accuracy (percentage points)\npositive = model is worse in the Indian language")
    ax.set_title("Belebele: accuracy gap between English and each native-script language\n"
                 "(300 items; * = confirmed under H1, Holm-adjusted p < 0.05)", loc="left")
    _save(fig, out, "fig1_gap_heatmap")
    return True


# 2 --------------------------------------------------------------------------------------------------------
def fig_dumbbell(results: Path, out: Path) -> bool:
    acc = _read(results, "accuracy.csv")
    if acc is None:
        return False
    bel = acc[acc["benchmark"] == "belebele"].set_index(["model", "condition"])
    models = [m for m in MODEL_ORDER if m in {k[0] for k in bel.index}]
    panels = [(i, n, r) for i, n, r in ROMAN if any((m, n) in bel.index and (m, r) in bel.index for m in models)]
    if not panels:
        print("  skip dumbbell: no native+romanized pairs")
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1, 0.55 * len(models) + 2.4), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    ys = np.arange(len(models))[::-1]
    for ax, (iso, nat, rom) in zip(axes, panels):
        for y, m in zip(ys, models):
            if (m, nat) not in bel.index or (m, rom) not in bel.index:
                continue
            a, b = bel.loc[(m, nat)], bel.loc[(m, rom)]
            ax.plot([a["accuracy"] * 100, b["accuracy"] * 100], [y, y], color="0.6", lw=2, zorder=1)
            for row, mk, fc in ((a, "o", MODEL_COLOR[m]), (b, "D", "white")):
                ax.errorbar(row["accuracy"] * 100, y, xerr=[[(row["accuracy"] - row["ci_low"]) * 100], [(row["ci_high"] - row["accuracy"]) * 100]],
                            fmt=mk, ms=8, mfc=fc, mec=MODEL_COLOR[m], mew=2, ecolor=MODEL_COLOR[m], elinewidth=1, capsize=2, zorder=3)
        ax.set_title(LANG_NAME[iso])
        ax.set_yticks(ys, [MODEL_NAME[m] for m in models])
        ax.grid(axis="x", color="0.9", zorder=0)
        ax.set_xlabel("Accuracy (%)")
    handles = [Line2D([], [], marker="o", color="0.3", ls="", ms=8, label="native script (filled)"),
               Line2D([], [], marker="D", color="0.3", mfc="white", ls="", ms=8, label="romanized (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Belebele: accuracy in native script vs romanized (Latin) script, with 95% bootstrap CIs (300 items)", x=0.01, ha="left", y=1.02)
    _save(fig, out, "fig2_native_vs_romanized")
    return True


# 3 --------------------------------------------------------------------------------------------------------
def fig_token_scatter(results: Path, out: Path) -> bool:
    pts = _read(results, "gap_vs_token_ratio.csv")
    if pts is None:
        return False
    conf = _read(results, "confirmatory.csv")
    h3 = conf[conf["hypothesis"] == "H3"].set_index("model") if conf is not None else pd.DataFrame()
    models = _models_in(pts)
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), models):
        g = pts[pts["model"] == m]
        ax.axhline(0, color="0.7", lw=1)
        ax.scatter(g["token_ratio"], g["gap_eng_minus_native"] * 100, color=MODEL_COLOR[m], s=45, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r["language"], (r["token_ratio"], r["gap_eng_minus_native"] * 100), xytext=(4, 3), textcoords="offset points", fontsize=9)
        label = "rho not computed"
        if m in h3.index and h3.loc[m, "status"] == "ok":
            label = f"Spearman rho = {h3.loc[m, 'rho']:+.2f}  (Holm p = {h3.loc[m, 'p_holm']:.3f})"
        ax.legend(handles=[Line2D([], [], marker="o", color=MODEL_COLOR[m], ls="", label=label)], loc="upper left", frameon=False, fontsize=10)
        ax.set_title(MODEL_NAME[m])
        ax.grid(color="0.92")
    for k, ax in enumerate(axes.ravel()[:len(models)]):
        ax.tick_params(labelbottom=True)
        if k + ncol >= len(models):  # no panel below this one
            ax.set_xlabel("Token ratio vs English\n(net prompt tokens per item, template removed)")
    for ax in axes[:, 0]:
        ax.set_ylabel("English minus native accuracy (pp)")
    fig.suptitle("Belebele: do languages that need more tokens lose more accuracy? (one point per language; H3)", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    _save(fig, out, "fig3_token_ratio_vs_gap")
    return True


# 4 --------------------------------------------------------------------------------------------------------
def fig_fixes(results: Path, out: Path) -> bool:
    conf = _read(results, "confirmatory.csv")
    if conf is None:
        return False
    d = conf[conf["hypothesis"].isin(["H4", "H5"]) & (conf["status"] == "ok")]
    if d.empty:
        print("  skip fix-effect bars: no H4/H5 results yet (derived conditions not run)")
        return False
    present = [h for h in ("H4", "H5") if (d["hypothesis"] == h).any()]
    widths = {"H4": 7, "H5": 15}
    titles = {"H4": "H4: IndicXlit (romanized to native script) vs unconverted romanized input",
              "H5": "H5: translate to English (NLLB-200) vs native-script input"}
    fig, axes = plt.subplots(len(present), 1, figsize=(max(widths[h] for h in present), 4.4 * len(present)), squeeze=False)
    for ax, h in zip(axes[:, 0], present):
        g = d[d["hypothesis"] == h]
        langs = [l for l in NATIVE_ORDER if l in set(g["language"])]
        models = _models_in(g)
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            sub = g[g["model"] == m].set_index("language")
            xs, ys, lo, hi, st = [], [], [], [], []
            for j, l in enumerate(langs):
                if l in sub.index:
                    r = sub.loc[l]
                    xs.append(j - 0.4 + w * (k + 0.5))
                    ys.append(r["gap"] * 100)
                    lo.append((r["gap"] - r["ci_low"]) * 100)
                    hi.append((r["ci_high"] - r["gap"]) * 100)
                    st.append(bool(r["confirmed"]))
            ax.bar(xs, ys, width=w * 0.92, color=MODEL_COLOR[m], label=MODEL_NAME[m], zorder=2)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1, capsize=2, zorder=3)
            for x, y, h_, s in zip(xs, ys, hi, st):
                if s:
                    ax.text(x, y + h_ + 0.4, "*", ha="center", va="bottom", fontsize=13)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(langs)), [LANG_NAME[l] for l in langs], rotation=30, ha="right")
        ax.set_ylabel("Change in accuracy (pp)\nfixed minus unfixed")
        ax.set_title(titles[h] + "\nbars: paired gap; whiskers: 95% bootstrap CI; * = confirmed (Holm p < 0.05)", loc="left", fontsize=11, pad=16)
        ax.margins(y=0.12)
        ax.grid(axis="y", color="0.92", zorder=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, out, "fig4_fix_effects")
    return True


# 5 --------------------------------------------------------------------------------------------------------
def fig_cs_ca(results: Path, out: Path) -> bool:
    d = _read(results, "gmmlu_lite_cs_ca.csv")
    if d is None:
        return False
    d = d[d["label"].isin(["CS", "CA"])]
    if d.empty:
        return False
    models = _models_in(d)
    langs = [l for l in ("en", "hi", "bn", "or") if l in set(d["condition"])]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 1, 4.4), sharey=True, squeeze=False)
    col = {"CS": OI["vermillion"], "CA": OI["blue"]}
    for ax, m in zip(axes[0], models):
        g = d[d["model"] == m]
        for k, lab in enumerate(("CS", "CA")):
            xs, ys, lo, hi = [], [], [], []
            for j, l in enumerate(langs):
                r = g[(g["condition"] == l) & (g["label"] == lab)]
                if len(r):
                    r = r.iloc[0]
                    xs.append(j + (k - 0.5) * 0.38)
                    ys.append(r["accuracy"] * 100)
                    lo.append((r["accuracy"] - r["ci_low"]) * 100)
                    hi.append((r["ci_high"] - r["accuracy"]) * 100)
            ax.bar(xs, ys, width=0.36, color=col[lab], label="culturally sensitive (CS)" if lab == "CS" else "culturally agnostic (CA)", zorder=2)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(range(len(langs)), [LANG_NAME.get(l, l) for l in langs], rotation=30, ha="right")
        ax.set_title(MODEL_NAME[m], fontsize=12)
        ax.grid(axis="y", color="0.92", zorder=0)
        ax.set_ylim(0, 100)
    axes[0, 0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Global-MMLU-Lite (400 items, human-translated): accuracy by language, culturally sensitive vs agnostic questions (95% bootstrap CI)",
                 x=0.01, ha="left", y=1.08, fontsize=12)
    fig.tight_layout()
    _save(fig, out, "fig5_gmmlu_lite_cs_vs_ca")
    return True


# 6 --------------------------------------------------------------------------------------------------------
def fig_resource(results: Path, out: Path) -> bool:
    pts = _read(results, "gap_vs_resource.csv")
    if pts is None:
        return False
    corr = _read(results, "resource_correlation.csv")
    pts = pts[pts["model"].isin(MODEL_ORDER)]
    resources = [("speakers", "First-language speakers in India (Census 2011, log scale)"),
                 ("wikipedia_articles", "Wikipedia articles in the language (log scale)")]
    resources = [(c, l) for c, l in resources if pts[c].notna().any()]
    if not resources:
        print("  skip resource plot: no resource statistics")
        return False
    fig, axes = plt.subplots(1, len(resources), figsize=(6.6 * len(resources), 5.2), sharey=True, squeeze=False)
    for ax, (c, xl) in zip(axes[0], resources):
        ax.axhline(0, color="0.7", lw=1)
        for m in _models_in(pts):
            g = pts[pts["model"] == m]
            ax.scatter(g[c], g["gap_eng_minus_native"] * 100, s=22, alpha=0.55, color=MODEL_COLOR[m], label=MODEL_NAME[m], zorder=2)
        mean = pts.groupby("language").agg(gap=("gap_eng_minus_native", "mean"), x=(c, "first")).dropna()
        ax.scatter(mean["x"], mean["gap"] * 100, s=90, marker="D", color="black", zorder=4, label="mean of models")
        for l, r in mean.iterrows():
            ax.annotate(LANG_NAME.get(l, l), (r["x"], r["gap"] * 100), xytext=(5, 4), textcoords="offset points", fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xl)
        ax.grid(color="0.92", zorder=0)
        if corr is not None:
            row = corr[(corr["resource"] == c) & corr["model"].str.startswith("MEAN")]
            if len(row) and np.isfinite(row.iloc[0]["rho"]):
                ax.set_title(f"Spearman rho (mean of models) = {row.iloc[0]['rho']:+.2f}, p = {row.iloc[0]['p_two_sided']:.2f}", fontsize=11, loc="left")
    axes[0, 0].set_ylabel("English minus native accuracy (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=10)
    fig.suptitle("Belebele: accuracy gap vs language resource level, 13 Indian languages (exploratory)", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    _save(fig, out, "fig6_gap_vs_resource")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "reports" / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "figures")
    a = ap.parse_args(argv)
    for f in (fig_heatmap, fig_dumbbell, fig_token_scatter, fig_fixes, fig_cs_ca, fig_resource):
        print(f.__name__)
        f(a.results, a.out)


if __name__ == "__main__":
    main()

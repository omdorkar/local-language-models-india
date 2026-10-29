"""Matplotlib figures for the dashboard. Pure functions: DataFrames in, Figure out (the caller shows and closes it).

Palette: Okabe-Ito (colour-blind safe), taken from ``evaluation.plots`` so the dashboard and the paper figures agree.
Diverging maps use ColorBrewer RdBu and sequential maps use viridis, both colour-blind safe. Numbers are printed inside
heatmap cells and verdicts are written as text, so no information depends on colour alone.
Labels are English names only (no Indian-script glyphs inside figures), so no special font is needed here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from evaluation import plots  # noqa: E402  (sets the shared rcParams)

MODEL_COLOR = plots.MODEL_COLOR
MODEL_NAME = plots.MODEL_NAME
OI = plots.OI


def _color(m: str) -> str:
    return MODEL_COLOR.get(m, "#555555")


def _name(m: str) -> str:
    return MODEL_NAME.get(m, m)


def empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return fig


def annotated_heatmap(values: np.ndarray, texts: list[list[str]], row_labels: list[str], col_labels: list[str],
                      cbar_label: str, diverging: bool = False, vmin: float | None = None, vmax: float | None = None):
    """Rows x columns heatmap with the number printed in each cell. NaN cells are grey with a dash."""
    nr, nc = values.shape
    fig, ax = plt.subplots(figsize=(1.35 * nc + 3.6, 0.42 * nr + 1.9))
    if diverging:
        cmap = plt.get_cmap("RdBu_r").copy()
        lim = vmax if vmax is not None else max(5.0, float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 5.0)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    else:
        cmap = plt.get_cmap("viridis").copy()
        lo = vmin if vmin is not None else 0.0
        hi = vmax if vmax is not None else 100.0
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
    cmap = cmap.with_extremes(bad="#e6e6e6")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    for i in range(nr):
        for j in range(nc):
            if np.isfinite(values[i, j]):
                ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=10 if "\n" not in texts[i][j] else 8.5,
                        color=plots._text_color(cmap(norm(values[i, j]))))
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="#666666")
    ax.set_xticks(range(nc), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(nr), row_labels)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    fig.tight_layout()
    return fig


def dumbbell_panels(acc: pd.DataFrame, panels: list[tuple[str, str, str]], models: list[str], lang_name: dict[str, str]):
    """One panel per language: native-script accuracy (filled) joined to romanized accuracy (open), with 95% CIs.

    ``acc`` is ``stats.accuracy_table`` output for Belebele; ``panels`` = [(iso, native condition, romanized condition)].
    """
    bel = acc.set_index(["model", "condition"])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels) + 1.6, 0.55 * len(models) + 2.4), sharey=True, sharex=True, squeeze=False)
    ys = np.arange(len(models))[::-1]
    for ax, (iso, nat, rom) in zip(axes[0], panels):
        for y, m in zip(ys, models):
            if (m, nat) not in bel.index or (m, rom) not in bel.index:
                continue
            a, b = bel.loc[(m, nat)], bel.loc[(m, rom)]
            ax.plot([a["accuracy"] * 100, b["accuracy"] * 100], [y, y], color="0.6", lw=2, zorder=1)
            for row, mk, fc in ((a, "o", _color(m)), (b, "D", "white")):
                ax.errorbar(row["accuracy"] * 100, y, xerr=[[(row["accuracy"] - row["ci_low"]) * 100], [(row["ci_high"] - row["accuracy"]) * 100]],
                            fmt=mk, ms=8, mfc=fc, mec=_color(m), mew=2, ecolor=_color(m), elinewidth=1, capsize=2, zorder=3)
        ax.set_title(lang_name.get(iso, iso))
        ax.set_yticks(ys, [_name(m) for m in models])
        ax.grid(axis="x", color="0.9", zorder=0)
        ax.set_xlabel("Accuracy (%)")
    handles = [Line2D([], [], marker="o", color="0.3", ls="", ms=8, label="native script (filled)"),
               Line2D([], [], marker="D", color="0.3", mfc="white", ls="", ms=8, label="romanized (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return fig


def grouped_bars(df: pd.DataFrame, cats: list[str], cat_labels: dict[str, str], models: list[str], ylabel: str,
                 zero_line: bool = True, ylim: tuple[float, float] | None = None, rotate: int = 30):
    """Bars per category, one bar per model, with 95% CI whiskers and a * on confirmed results.

    ``df`` columns (percentage points or percent): model, cat, value, lo, hi (absolute CI bounds), star (bool, optional).
    """
    fig, ax = plt.subplots(figsize=(max(6.5, 1.15 * len(cats) + 3.0), 4.4))
    w = 0.8 / max(1, len(models))
    for k, m in enumerate(models):
        g = df[df["model"] == m].set_index("cat")
        xs, ys, lo, hi, st_ = [], [], [], [], []
        for j, c in enumerate(cats):
            if c in g.index:
                r = g.loc[c]
                xs.append(j - 0.4 + w * (k + 0.5))
                ys.append(r["value"])
                lo.append(r["value"] - r["lo"])
                hi.append(r["hi"] - r["value"])
                st_.append(bool(r["star"]) if "star" in g.columns and pd.notna(r["star"]) else False)
        if not xs:
            continue
        ax.bar(xs, ys, width=w * 0.92, color=_color(m), label=_name(m), zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1, capsize=2, zorder=3)
        for x, y, h, s in zip(xs, ys, hi, st_):
            if s:
                ax.text(x, y + h + 0.4, "*", ha="center", va="bottom", fontsize=13)
    if zero_line:
        ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(cats)), [cat_labels.get(c, c) for c in cats], rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.margins(y=0.12)
    ax.grid(axis="y", color="0.92", zorder=0)
    ax.legend(ncol=min(5, len(models)), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.tight_layout()
    return fig


def token_scatter(points: pd.DataFrame, h3: pd.DataFrame, models: list[str]):
    """One panel per model: token ratio vs English-minus-native gap, one point per language, with the H3 rho when computed."""
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    h3i = h3.set_index("model") if len(h3) else pd.DataFrame()
    for ax, m in zip(axes.ravel(), models):
        g = points[points["model"] == m]
        ax.axhline(0, color="0.7", lw=1)
        ax.scatter(g["token_ratio"], g["gap_eng_minus_native"] * 100, color=_color(m), s=45, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r["language"], (r["token_ratio"], r["gap_eng_minus_native"] * 100), xytext=(4, 3), textcoords="offset points", fontsize=9)
        label = "rho not computed"
        if m in h3i.index and h3i.loc[m, "status"] == "ok":
            label = f"Spearman rho = {h3i.loc[m, 'rho']:+.2f}  (Holm p = {h3i.loc[m, 'p_holm']:.3f})"
        ax.legend(handles=[Line2D([], [], marker="o", color=_color(m), ls="", label=label)], loc="upper left", frameon=False, fontsize=10)
        ax.set_title(_name(m))
        ax.grid(color="0.92")
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("Token ratio vs English (template removed)")
        ax.set_ylabel("English minus native accuracy (pp)")
    fig.tight_layout()
    return fig


def cs_ca_panels(acc: pd.DataFrame, models: list[str], langs: list[str], lang_name: dict[str, str]):
    """Global-MMLU-Lite: culturally sensitive (CS) vs agnostic (CA) accuracy by language, one panel per model."""
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 1, 4.4), sharey=True, squeeze=False)
    col = {"CS": OI["vermillion"], "CA": OI["blue"]}
    names = {"CS": "culturally sensitive (CS)", "CA": "culturally agnostic (CA)"}
    for ax, m in zip(axes[0], models):
        g = acc[acc["model"] == m]
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
            ax.bar(xs, ys, width=0.36, color=col[lab], label=names[lab], zorder=2, hatch="//" if lab == "CA" else None, edgecolor="white")
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(range(len(langs)), [lang_name.get(l, l) for l in langs], rotation=30, ha="right")
        ax.set_title(_name(m), fontsize=12)
        ax.grid(axis="y", color="0.92", zorder=0)
        ax.set_ylim(0, 100)
    axes[0, 0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout()
    return fig

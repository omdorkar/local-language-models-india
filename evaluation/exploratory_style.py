"""Exploratory (not among the five main hypotheses) comparisons for the machine-made style conditions on Belebele:
Indian English vs English, and Hinglish vs romanized Hindi and vs English. Same paired bootstrap and exact
McNemar as the primary tests, unadjusted.

    python -m evaluation.exploratory_style   # -> reports/results/exploratory_style.{csv,md}
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from . import REPORTS, ROOT
from .config import SEED

COMPARISONS = [("eng_Latn+inde", "eng_Latn"), ("hin_Latn+cm", "hin_Latn"), ("hin_Latn+cm", "eng_Latn")]


def main() -> None:
    rows = []
    for f in sorted((ROOT / "runs" / "belebele").glob("*.jsonl")):
        d = pd.read_json(f, lines=True)
        model = d.model_key.iloc[0]
        if model.endswith("-q3"):
            continue
        wide = d.pivot_table(index="item_id", columns="condition", values="correct", aggfunc="first")
        for cond, ref in COMPARISONS:
            if cond not in wide or ref not in wide:
                continue
            x = wide[[cond, ref]].dropna().astype(bool)
            b, c = int((x[ref] & ~x[cond]).sum()), int((~x[ref] & x[cond]).sum())
            diff = x[cond].astype(int).to_numpy() - x[ref].astype(int).to_numpy()
            rng = np.random.default_rng(SEED)
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10_000)]
            rows.append({
                "model": model, "condition": cond, "reference": ref, "n": len(x),
                "accuracy_pct": round(100 * x[cond].mean(), 1), "gap_pp": round(100 * diff.mean(), 1),
                "ci_low": round(100 * np.percentile(boot, 2.5), 1), "ci_high": round(100 * np.percentile(boot, 97.5), 1),
                "b": b, "c": c, "p_unadjusted": mcnemar([[0, b], [c, 0]], exact=True).pvalue,
            })
    t = pd.DataFrame(rows).sort_values(["condition", "reference", "model"])
    out = REPORTS / "results"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "exploratory_style.csv", index=False)
    shown = t.assign(p_unadjusted=t.p_unadjusted.map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"))
    (out / "exploratory_style.md").write_text(
        "# EXPLORATORY: Indian English and Hinglish (Belebele)\n\n"
        "Not among the five main hypotheses (H1-H5). Gap = accuracy(condition) - accuracy(reference) in percentage points on the "
        "same items; 95% bootstrap CI over items (10,000 resamples, seed 1909); exact McNemar p, unadjusted. "
        "Both conditions are machine-made approximations, not real users' writing.\n\n"
        + shown.to_markdown(index=False) + "\n")
    print(shown.to_string(index=False))


if __name__ == "__main__":
    main()

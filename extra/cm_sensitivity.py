"""Sensitivity of the Hinglish mixing level to the alignment-confidence filter (writes reports/extra/cm_sensitivity.csv).

Reuses the cached alignments; for each (sim_min, margin_min) it reports, over passage sentences + questions (>= 8 tokens), the mean
realised CMI, the mean 'ceiling' CMI (all candidates substituted), and the KS distance to COMI-LINGUA.
"""
import json

import pandas as pd
from scipy import stats

from . import cm
from .cmi import load_comi
from .common import REPORT_DIR, load_rows, sample_ids


def main():
    ids = sample_ids()
    en, hl = load_rows("eng_Latn", ids), load_rows("hin_Latn", ids)
    cache = json.loads(cm.ALIGN_CACHE.read_text())
    ref = load_comi().cmi.values
    out = []
    for s, m in [(0.70, 0.01), (0.68, 0.005), (0.66, 0.0), (0.0, 0.0)]:
        cm.SIM_MIN, cm.MARGIN_MIN = s, m
        U, nrep = [], 0
        for iid in ids:
            for f in ["flores_passage", "question"]:
                r = cm.mix_field(hl[iid][f], en[iid][f], f, ref.tolist(), cache)
                U += [u for u in r["units"] if u["n_tokens"] >= 8]
                nrep += len(r["replacements"])
        d = pd.DataFrame(U)
        out.append({"sim_min": s, "margin_min": m, "mean_cmi_out": d.out_cmi.mean(), "mean_cmi_ceiling": d.max_cmi.mean(),
                    "mean_cmi_comi": ref.mean(), "KS_out_vs_comi": stats.ks_2samp(d.out_cmi, ref).statistic,
                    "KS_ceiling_vs_comi": stats.ks_2samp(d.max_cmi, ref).statistic, "substitutions_passage_question": nrep})
    pd.DataFrame(out).to_csv(REPORT_DIR / "cm_sensitivity.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string())


if __name__ == "__main__":
    main()

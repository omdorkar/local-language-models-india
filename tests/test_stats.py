"""Tests for evaluation.stats / plots / langstats on synthetic data. Nothing here touches the real runs/ or reports/."""

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import synthetic_runs as syn  # noqa: E402

from evaluation import langstats, plots, stats  # noqa: E402

N_BOOT_TEST = 1000


# ---------------------------------------------------------------------------------------------- helpers
def s01(vals):
    return pd.Series(np.asarray(vals, dtype=int), index=[f"i{k}" for k in range(len(vals))])


def table(correct, tokens=None):
    n = len(correct)
    return pd.DataFrame({"correct": np.asarray(correct, dtype=int), "invalid": 0,
                         "tokens": np.nan if tokens is None else np.asarray(tokens, dtype=float)},
                        index=[f"i{k}" for k in range(n)])


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    paths = syn.write_synthetic_runs(root)
    lang_csv = root / "language_stats.csv"
    rows = [{"iso": i, "language": l.name, "census2011_first_language_speakers": 10 ** (6 + k % 3) * (k + 1),
             "wikipedia_articles": 1000 * (14 - k) ** 2} for k, (i, l) in enumerate(syn.LANGUAGES.items())]
    pd.DataFrame(rows).to_csv(lang_csv, index=False)
    out = root / "results"
    res = stats.run(paths["runs"], out, paths["raw"], n_boot=N_BOOT_TEST, langstats_csv=lang_csv)
    return {"root": root, "paths": paths, "out": out, "res": res}


# -------------------------------------------------------------------------------------- bootstrap / paired
def _planted_sample(rng, n, acc_ref, acc_cond, rho=0.7):
    v = rng.random(n)
    vr = np.where(rng.random(n) < rho, v, rng.random(n))
    vc = np.where(rng.random(n) < rho, v, rng.random(n))
    return s01(vc < acc_cond), s01(vr < acc_ref)


def test_bootstrap_ci_covers_planted_gap():
    true_gap = -0.10
    rng = np.random.default_rng(123)
    cover = 0
    reps = 150
    for _ in range(reps):
        c, r = _planted_sample(rng, 300, 0.80, 0.70)
        res = stats.paired_gap(c, r, n_boot=400)
        cover += res["ci_low"] <= true_gap <= res["ci_high"]
    assert cover / reps >= 0.88  # nominal 0.95; loose bound to keep the test stable

    c, r = _planted_sample(np.random.default_rng(1909), 300, 0.80, 0.70)
    res = stats.paired_gap(c, r, n_boot=stats.N_BOOT)
    assert res["ci_low"] < res["gap"] < res["ci_high"]
    assert res["ci_high"] - res["ci_low"] < 0.12  # sane width for n=300


def test_bootstrap_is_reproducible_and_order_independent():
    c, r = _planted_sample(np.random.default_rng(5), 200, 0.8, 0.7)
    a = stats.paired_gap(c, r, n_boot=500)
    b = stats.paired_gap(c, r, n_boot=500)
    assert a == b


def test_paired_gap_uses_only_common_items_and_counts_b_c():
    c = pd.Series([1, 1, 0, 0, 1], index=list("abcde"))
    r = pd.Series([1, 0, 1, 0, 1, 1], index=list("abcdef"))  # 'f' only in ref
    res = stats.paired_gap(c, r, n_boot=200)
    assert res["n"] == 5
    assert (res["b"], res["c"]) == (1, 1)  # b: cond right/ref wrong = 'b'; c: cond wrong/ref right = 'c'
    assert res["gap"] == pytest.approx(0.0)
    assert res["acc_condition"] == pytest.approx(3 / 5)


# ----------------------------------------------------------------------------- McNemar / Holm hand-computed
def test_mcnemar_exact_hand_computed():
    # b=8, c=2: p = 2 * P(Bin(10, .5) <= 2) = 2 * (1 + 10 + 45) / 1024
    assert stats.mcnemar_exact(8, 2) == pytest.approx(112 / 1024)
    assert stats.mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 3) == pytest.approx(1.0)

    cond = s01([1] * 8 + [0] * 2 + [1] * 10)
    ref = s01([0] * 8 + [1] * 2 + [1] * 10)
    res = stats.paired_gap(cond, ref, n_boot=100)
    assert (res["b"], res["c"], res["n"]) == (8, 2, 20)
    assert res["gap"] == pytest.approx(0.3)
    assert res["p"] == pytest.approx(112 / 1024)


def test_holm_hand_computed():
    # sorted .01,.03,.04 -> .03, .06, .06 (monotone)
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5]) == pytest.approx([0.5])
    assert stats.holm([0.001, 0.2, 0.9]) == pytest.approx([0.003, 0.4, 0.9])


# ------------------------------------------------------------------------------------------- direction rule
def test_direction_rule():
    assert stats.direction_ok(-0.1, "lower") and not stats.direction_ok(0.1, "lower")
    assert stats.direction_ok(0.1, "higher") and not stats.direction_ok(-0.1, "higher")
    assert not stats.direction_ok(0.0, "lower") and not stats.direction_ok(0.0, "higher")
    assert not stats.direction_ok(float("nan"), "lower")


def test_significant_but_wrong_direction_is_not_confirmed():
    eng = table([1] * 50 + [0] * 50)
    native_better = table([1] * 100)  # significantly HIGHER than English: opposite of H1
    # model A: native higher than English; model B: native lower (40 of 100 right vs English 100 of 100)
    eng_b = table([1] * 100)
    native_b = table([0] * 60 + [1] * 40)
    bel = {("A", "eng_Latn"): eng, ("A", "hin_Deva"): native_better, ("B", "eng_Latn"): eng_b, ("B", "hin_Deva"): native_b}
    df = stats._paired_family("H1", [("hi", "hin_Deva", "eng_Latn")], "lower", bel, ["A", "B"], 200, True)
    a, b = df.set_index("model").loc["A"], df.set_index("model").loc["B"]
    assert a["p_holm"] < 0.05 and a["gap"] > 0 and not a["confirmed"]
    assert b["p_holm"] < 0.05 and b["gap"] < 0 and b["confirmed"]


def test_holm_family_counts_missing_tests_as_p1_by_default():
    eng = table([1] * 60 + [0] * 40)
    nat = table([0] * 30 + [1] * 30 + [0] * 40)
    bel = {("A", "eng_Latn"): eng, ("A", "hin_Deva"): nat}
    planned = stats._paired_family("H1", [("hi", "hin_Deva", "eng_Latn")], "lower", bel, ["A", "B"], 100, True)
    run_only = stats._paired_family("H1", [("hi", "hin_Deva", "eng_Latn")], "lower", bel, ["A", "B"], 100, False)
    pa = planned.set_index("model").loc["A", "p"]
    assert planned.set_index("model").loc["A", "p_holm"] == pytest.approx(min(1, 2 * pa))  # m = 2 planned
    assert run_only.set_index("model").loc["A", "p_holm"] == pytest.approx(pa)  # m = 1
    assert planned.set_index("model").loc["B", "status"].startswith("not run")


# ------------------------------------------------------------------------------------ loading / templates
def test_template_rows_excluded_and_invalid_counts_wrong(tmp_path):
    d = tmp_path / "runs" / "belebele"
    d.mkdir(parents=True)
    base = dict(benchmark="belebele", model_key="m1", lang="en", script="Latn", source="human", gold="A",
                prompt_tokens=100, completion_tokens=1)
    rows = [
        dict(base, condition="__template__", item_id="__template__", gold="", answer=None, correct=False, prompt_tokens=40),
        dict(base, condition="eng_Latn", item_id="x1", answer="A", correct=True),
        dict(base, condition="eng_Latn", item_id="x2", answer=None, correct=False),
        dict(base, condition="eng_Latn", item_id="x3", answer="B", correct=False),
        dict(base, condition="eng_Latn", item_id="x4", answer="A", correct=True),
        dict(base, condition="eng_Latn", item_id="x4", answer="B", correct=False),  # duplicate key: last wins
    ]
    (d / "m1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{truncated\n")
    (tmp_path / "runs" / "stray.jsonl").write_text(json.dumps(rows[1]) + "\n")  # legacy root-level file ignored
    with pytest.warns(UserWarning):
        items, templates = stats.load_runs(tmp_path / "runs")
    assert "__template__" not in set(items["condition"])
    assert templates == {("belebele", "m1"): 40}
    assert len(items) == 4
    acc = stats.accuracy_table(stats.build_tables(items), n_boot=100).iloc[0]
    assert acc["n"] == 4 and acc["accuracy"] == pytest.approx(1 / 4)  # x1 right only; invalid counted wrong
    assert acc["invalid_rate"] == pytest.approx(1 / 4)


def test_synthetic_template_rows_not_in_accuracy(synth):
    acc = synth["res"]["accuracy"]
    assert "__template__" not in set(acc["condition"])
    assert (acc[acc["benchmark"] == "belebele"]["n"] == syn.N_ITEMS).all()


# ----------------------------------------------------------------------------------------- token ratios
def test_token_ratio_math():
    T = 50
    eng = table([1] * 4, tokens=[T + 100, T + 100, T + 100, T + 100])
    cond = table([1] * 4, tokens=[T + 300, T + 300, T + 300, T + 300])
    tb = {"belebele": {("m", "eng_Latn"): eng, ("m", "hin_Deva"): cond}}
    r = stats.token_ratios(tb, {("belebele", "m"): T}).set_index("condition")
    assert r.loc["hin_Deva", "token_ratio_net"] == pytest.approx(3.0)
    assert r.loc["hin_Deva", "token_ratio_raw"] == pytest.approx(350 / 150)
    assert r.loc["eng_Latn", "token_ratio_net"] == pytest.approx(1.0)
    # no template row -> net ratio unavailable and flagged, raw still given
    r2 = stats.token_ratios(tb, {}).set_index("condition")
    assert np.isnan(r2.loc["hin_Deva", "token_ratio_net"]) and "template" in r2.loc["hin_Deva", "status"]
    assert r2.loc["hin_Deva", "token_ratio_raw"] == pytest.approx(350 / 150)


def test_token_ratio_recovers_planted_values(synth):
    tok = synth["res"]["token_ratio"]
    nat = tok[(tok["model"] == "gemma3-12b") & tok["condition"].isin([c for _, c in stats.native_languages()])]
    for _, r in nat.iterrows():
        assert r["token_ratio_net"] == pytest.approx(syn.TOKEN_RATIO[r["language"]], rel=0.02)


def test_h3_not_run_without_template(tmp_path):
    paths = syn.write_synthetic_runs(tmp_path, n_items=60, with_gmmlu=False, drop_template_for=["qwen3-8b"])
    items, templates = stats.load_runs(paths["runs"])
    conf, _ = stats.confirmatory(stats.build_tables(items), templates, n_boot=100)
    h3 = conf[conf["hypothesis"] == "H3"].set_index("model")
    assert h3.loc["qwen3-8b", "status"].startswith("not run")
    assert pd.isna(h3.loc["qwen3-8b", "confirmed"])
    assert h3.loc["gemma3-12b", "status"] == "ok"


# -------------------------------------------------------------------------------- confirmatory on synthetic
def test_confirmatory_structure_and_family_sizes(synth):
    conf = synth["res"]["confirmatory"]
    for col in ["hypothesis", "model", "language", "n", "gap", "ci_low", "ci_high", "b", "c", "p", "p_holm",
                "predicted_direction", "confirmed", "status"]:
        assert col in conf.columns
    sizes = conf.groupby("hypothesis").size().to_dict()
    assert sizes == {"H1": 65, "H2": 20, "H3": 5, "H4": 20, "H5": 65}
    assert set(conf["model"]) == set(stats.MAIN_MODELS)  # sensitivity model is not confirmatory
    ok = conf[conf["status"] == "ok"]
    assert (ok["p_holm"] >= ok["p"] - 1e-12).all()
    assert (ok.loc[ok["hypothesis"] != "H3", "n"] == syn.N_ITEMS).all()


def test_planted_effects_recovered(synth):
    conf = synth["res"]["confirmatory"]
    h1 = conf[conf["hypothesis"] == "H1"].set_index(["model", "language"])
    # strong planted gap: llama Odia = 0.055 * 3.5 = 19.25 pp
    r = h1.loc[("llama3.1-8b", "or")]
    assert r["confirmed"] and r["gap"] == pytest.approx(-0.1925, abs=0.06)
    # planted NO gap (mr, gu): never confirmed
    for m in stats.MAIN_MODELS:
        for lang in syn.NO_GAP:
            assert not h1.loc[(m, lang), "confirmed"]
    # gpt-oss has no planted token-ratio gaps: H1 not confirmed anywhere
    assert not h1.loc["gpt-oss-20b"]["confirmed"].any()
    # H2: Urdu romanized 15 pp lower is detected for every model; Nepali (planted 0) is not
    h2 = conf[conf["hypothesis"] == "H2"].set_index(["model", "language"])
    assert all(h2.loc[(m, "ur"), "confirmed"] for m in ("gemma3-12b", "llama3.1-8b", "sarvam-30b"))
    assert not h2.xs("ne", level="language")["confirmed"].any()
    # H3: planted positive relation for gemma, none for gpt-oss
    h3 = conf[conf["hypothesis"] == "H3"].set_index("model")
    assert h3.loc["gemma3-12b", "confirmed"] and h3.loc["gemma3-12b", "rho"] > 0.5
    assert not h3.loc["gpt-oss-20b", "confirmed"]
    assert (h3["n"] == 13).all()
    # H5: mt_en recovers more for Odia (gap 0.055*3.5) than for Hindi (small)
    h5 = conf[conf["hypothesis"] == "H5"].set_index(["model", "language"])
    assert h5.loc[("llama3.1-8b", "or"), "gap"] > h5.loc[("llama3.1-8b", "hi"), "gap"]
    assert h5.loc[("llama3.1-8b", "or"), "confirmed"]


def test_fix_conditions_absent_gives_not_run(tmp_path):
    paths = syn.write_synthetic_runs(tmp_path, n_items=80, with_fixes=False, with_gmmlu=False)
    res = stats.run(paths["runs"], tmp_path / "res", paths["raw"], n_boot=100, langstats_csv=tmp_path / "none.csv")
    conf = res["confirmatory"]
    for h in ("H4", "H5"):
        g = conf[conf["hypothesis"] == h]
        assert (g["status"].str.startswith("not run")).all() and g["confirmed"].isna().all()
        assert g["gap"].isna().all()
    assert (conf[conf["hypothesis"] == "H1"]["status"] == "ok").all()
    summary = (tmp_path / "res" / "summary.md").read_text()
    assert "Not run" in summary
    # plots skip the fix figure gracefully
    plots.main(["--results", str(tmp_path / "res"), "--out", str(tmp_path / "figs")])
    assert not (tmp_path / "figs" / "fig4_fix_effects.png").exists()
    assert (tmp_path / "figs" / "fig1_gap_heatmap.png").exists()


def test_partial_fix_data_only_some_models(tmp_path):
    paths = syn.write_synthetic_runs(tmp_path, n_items=80, fix_models=["gemma3-12b"], with_gmmlu=False)
    items, templates = stats.load_runs(paths["runs"])
    conf, _ = stats.confirmatory(stats.build_tables(items), templates, n_boot=100)
    h4 = conf[conf["hypothesis"] == "H4"]
    assert (h4[h4["model"] == "gemma3-12b"]["status"] == "ok").all()
    assert (h4[h4["model"] != "gemma3-12b"]["status"].str.startswith("not run")).all()


# ----------------------------------------------------------------------------------------- exploratory
def test_cs_ca_difference_recovered(synth):
    d = synth["res"]["cs_ca_diff"]
    assert len(d) == 5 * 3
    assert d["diff_cs_minus_ca"].mean() < -0.04  # planted extra 8 pp loss on CS
    assert (d["ci_low"] <= d["diff_cs_minus_ca"]).all() and (d["diff_cs_minus_ca"] <= d["ci_high"]).all()
    assert set(synth["res"]["cs_ca"]["label"]) == {"all", "CS", "CA"}


def test_quantization_recovers_planted_shift(synth):
    q = synth["res"]["quantization"]
    pooled = q[q["condition"].str.startswith("ALL")].iloc[0]
    assert pooled["gap"] == pytest.approx(-0.03, abs=0.02)
    assert pooled["ci_low"] < pooled["gap"] < pooled["ci_high"]
    assert (q[~q["condition"].str.startswith("ALL")]["n"] == syn.N_ITEMS).all()


def test_gmmlu_gaps_and_sarvam_and_resource(synth):
    g = synth["res"]["gmmlu_gaps"]
    assert set(g["benchmark"]) == {"gmmlu_lite", "gmmlu"}
    assert (g["gap"] < 0).all()
    sv = synth["res"]["sarvam"]
    assert "MEAN over languages" in set(sv["language"]) and sv["n_other_models"].eq(4).all()
    rc = synth["res"]["resource_corr"]
    assert set(rc["resource"]) == {"speakers", "wikipedia_articles"}
    assert not any(str(m).startswith("MEAN") and m not in ("MEAN over the 5 main models",) for m in rc["model"])


# -------------------------------------------------------------------------------------------- end to end
def test_end_to_end_outputs_and_plots(synth, tmp_path):
    out = synth["out"]
    for name in ["accuracy.csv", "confirmatory.csv", "token_ratio.csv", "summary.md", "gmmlu_gaps.csv", "quantization.csv",
                 "sarvam_vs_others.csv", "gmmlu_lite_cs_ca.csv", "gap_vs_resource.csv", "gap_vs_token_ratio.csv"]:
        assert (out / name).exists(), name
    summary = (out / "summary.md").read_text()
    assert "H1 confirmed for" in summary and "/65 model-language pairs" in summary
    assert "EXPLORATORY" in summary and "✓" in summary and "✗" in summary
    figs = tmp_path / "figs"
    plots.main(["--results", str(out), "--out", str(figs)])
    for stem in ["fig1_gap_heatmap", "fig2_native_vs_romanized", "fig3_token_ratio_vs_gap", "fig4_fix_effects",
                 "fig5_gmmlu_lite_cs_vs_ca", "fig6_gap_vs_resource"]:
        assert (figs / f"{stem}.png").stat().st_size > 5_000 and (figs / f"{stem}.svg").exists(), stem
    # the real repo outputs were not touched by this test module
    assert not str(synth["root"]).startswith(str(stats.ROOT / "reports"))


def test_cli_runs_on_empty_runs_dir(tmp_path):
    (tmp_path / "runs").mkdir()
    stats.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out"), "--raw", str(tmp_path / "raw"),
                "--n-boot", "50", "--langstats", str(tmp_path / "nolang.csv")])
    conf = pd.read_csv(tmp_path / "out" / "confirmatory.csv")
    assert conf["status"].str.startswith("not run").all()
    assert (tmp_path / "out" / "summary.md").exists()


# ----------------------------------------------------------------------------------------------- langstats
CENSUS_HTML = """<html><body><table>
<tr><th rowspan=2>Language</th><th colspan=2>First language speakers</th></tr>
<tr><th>Figure[14]</th><th>% of total population</th></tr>
<tr><td>Hindi[note 2]</td><td>528,347,193</td><td>43.63%</td></tr>
<tr><td>Nepali</td><td>2,926,168</td><td>0.24%</td></tr>
</table></body></html>"""


def test_langstats_parsing_offline():
    def handler(request):
        return httpx.Response(200, text=CENSUS_HTML)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        census = langstats.fetch_census(client)
    assert census["Hindi"] == 528347193 and census["Nepali"] == 2926168

    def boom(request):
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(boom)) as client:
        assert langstats.fetch_wiki_articles(client, "hi") is None  # blank, never guessed

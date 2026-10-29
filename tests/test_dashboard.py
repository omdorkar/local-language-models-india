"""Smoke tests for dashboard/app.py with Streamlit's AppTest. Everything runs on synthetic data in a temp dir;
the real runs/ and data/ folders are never read or written (the app is pointed elsewhere through environment variables)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import synthetic_runs as syn  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"
sys.path.insert(0, str(ROOT / "dashboard"))
import dash_data as D  # noqa: E402

N_ITEMS = 40
HINDI = "नमस्ते दुनिया"  # Devanagari
URDU = "سلام دنیا"  # Arabic script
VIEWS = [
    "Run progress", "1. Overview: all languages vs English", "2. Native script vs romanized", "3. Token ratio vs gap",
    "4. Item browser", "5. Fixes and exploratory conditions", "6. Global-MMLU",
]


def _write_belebele_texts(root: Path, items: list[str], runs: Path) -> None:
    """Tiny stand-in for data/raw/belebele/data and data/derived/{samples,conditions}. Gold letters match the synthetic runs."""
    gold = {}
    with (runs / "belebele" / "gemma3-12b.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            gold[r["item_id"]] = r["gold"]

    def row(i, item, text):
        link, num = item.rsplit("#", 1)
        return {"link": link, "question_number": int(num), "flores_passage": f"Passage {i} {text}", "question": f"Question {i} {text}?",
                "mc_answer1": f"one {text}", "mc_answer2": f"two {text}", "mc_answer3": f"three {text}", "mc_answer4": f"four {text}",
                "correct_answer_num": str("ABCD".index(gold[item]) + 1)}

    raw = root / "raw" / "belebele" / "data"
    raw.mkdir(parents=True, exist_ok=True)
    for cond, text in (("eng_Latn", "english"), ("hin_Deva", HINDI), ("hin_Latn", "namaste"), ("urd_Arab", URDU)):
        (raw / f"{cond}.jsonl").write_text("\n".join(json.dumps(row(i, it, text), ensure_ascii=False) for i, it in enumerate(items)) + "\n", encoding="utf-8")
    der = root / "derived" / "conditions" / "belebele"
    der.mkdir(parents=True, exist_ok=True)
    (der / "hin_Latn+xlit.jsonl").write_text(
        "\n".join(json.dumps({**row(i, it, HINDI), "item_id": it, "source_condition": "hin_Latn", "method": "indicxlit"}, ensure_ascii=False)
                  for i, it in enumerate(items)) + "\n", encoding="utf-8")
    samples = root / "derived" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "belebele.json").write_text(json.dumps({"benchmark": "belebele", "seed": 1909, "items": items}))
    (samples / "gmmlu_lite.json").write_text(json.dumps({"benchmark": "gmmlu_lite", "seed": 1909, "items": [f"subj{i % 20}/test/{i}" for i in range(400)]}))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("dash")
    paths = syn.write_synthetic_runs(root, n_items=N_ITEMS)
    _write_belebele_texts(root, paths["items"], paths["runs"])
    return {"root": root, "runs": paths["runs"], "raw": paths["raw"], "derived": root / "derived", "items": paths["items"]}


@pytest.fixture
def env(synth, monkeypatch):
    monkeypatch.setenv("EVALUATION_RUNS", str(synth["runs"]))
    monkeypatch.setenv("EVALUATION_RAW", str(synth["raw"]))
    monkeypatch.setenv("EVALUATION_DERIVED", str(synth["derived"]))
    monkeypatch.setenv("EVALUATION_RESULTS", str(synth["root"] / "no_results"))
    monkeypatch.setenv("EVALUATION_FIGURES", str(synth["root"] / "no_figures"))
    monkeypatch.setenv("EVALUATION_NBOOT", "200")
    return synth


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180)


def _snapshot(runs: Path) -> dict:
    return {str(p.relative_to(runs)): (p.stat().st_size, p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [h.value for h in at.header] + [w.value for w in at.warning]
    parts += [i.value for i in at.info]
    return "\n".join(map(str, parts))


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_loads_without_exception(env, view):
    before = _snapshot(env["runs"])
    at = _app().run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio(key="view").set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.header) >= 1
    assert _snapshot(env["runs"]) == before, "the dashboard must not modify runs/"


def test_overview_shows_heatmaps_and_confirmatory_label(env):
    at = _app().run()
    at.sidebar.radio(key="view").set_value(VIEWS[1]).run()
    assert not at.exception
    text = _text(at)
    assert "PRIMARY" in text and "EXPLORATORY" in text
    assert "H1" in text
    assert "Share of Belebele questions answered correctly" in text  # every chart carries a caption
    assert "English minus native" in text
    assert len(at.dataframe) >= 1


def test_item_browser_shows_indian_script_and_model_answers(env):
    at = _app().run()
    at.sidebar.radio(key="view").set_value(VIEWS[4]).run()
    assert not at.exception
    at.multiselect(key="item_conds").set_value(["eng_Latn", "hin_Deva", "urd_Arab", "hin_Latn+xlit"]).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert HINDI in text and URDU in text
    assert "gold answer" in text
    assert "Gemma 3 12B" in text
    # the filter variants must run too
    for flt in ("Some model is right in English but wrong in a chosen condition", "Models disagree in a chosen condition"):
        at.selectbox(key="item_filter").set_value(flt).run()
        assert not at.exception, [e.value for e in at.exception]


def test_query_param_selects_runs_folder(env, tmp_path):
    empty = tmp_path / "empty_runs"
    empty.mkdir()
    at = _app()
    at.query_params["runs"] = str(empty)
    at.run()
    assert not at.exception
    assert at.sidebar.text_input(key="runs_dir").value == str(empty)
    assert any("No answer rows" in str(i.value) for i in at.info)


@pytest.mark.parametrize("view", VIEWS)
def test_empty_and_missing_runs_folder(monkeypatch, tmp_path, view):
    monkeypatch.setenv("EVALUATION_RUNS", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("EVALUATION_RAW", str(tmp_path / "raw"))
    monkeypatch.setenv("EVALUATION_DERIVED", str(tmp_path / "derived"))
    monkeypatch.setenv("EVALUATION_RESULTS", str(tmp_path / "res"))
    at = _app().run()
    assert not at.exception
    at.sidebar.radio(key="view").set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]


def test_partial_results_and_truncated_last_line(env, tmp_path, monkeypatch):
    """A run in progress: only some models, fewer rows, and a half-written last line must not break any view."""
    root = tmp_path / "partial"
    paths = syn.write_synthetic_runs(root, n_items=N_ITEMS, models=["gemma3-12b", "llama3.1-8b"], with_fixes=False, with_q3=False, with_gmmlu=False)
    f = paths["runs"] / "belebele" / "gemma3-12b.jsonl"
    lines = f.read_text().splitlines()[:700]
    f.write_text("\n".join(lines) + '\n{"benchmark": "belebele", "model_key": "gemma3-12b", "condi')  # cut mid-row
    monkeypatch.setenv("EVALUATION_RUNS", str(paths["runs"]))
    monkeypatch.setenv("EVALUATION_RAW", str(paths["raw"]))
    monkeypatch.setenv("EVALUATION_DERIVED", str(env["derived"]))
    at = _app().run()
    assert not at.exception
    for view in VIEWS:
        at.sidebar.radio(key="view").set_value(view).run()
        assert not at.exception, (view, [e.value for e in at.exception])
    at.sidebar.radio(key="view").set_value(VIEWS[1]).run()
    assert any("not finished" in str(w.value) for w in at.warning)


# ------------------------------------------------------------------------------------------ helper functions


def test_cond_label():
    assert D.cond_label("eng_Latn") == "English"
    assert D.cond_label("hin_Deva") == "Hindi (native script)"
    assert D.cond_label("hin_Latn") == "Hindi (romanized)"
    assert D.cond_label("hin_Latn+xlit").startswith("Hindi (romanized) + IndicXlit")
    assert "Multi-VALUE" in D.cond_label("eng_Latn+inde")
    assert D.cond_label("hi", "gmmlu_lite") == "Hindi"


def test_signature_changes_when_a_run_file_grows_and_loading_tolerates_a_partial_line(tmp_path):
    runs = tmp_path / "runs" / "belebele"
    runs.mkdir(parents=True)
    f = runs / "m.jsonl"
    good = {"benchmark": "belebele", "model_key": "m", "condition": "eng_Latn", "lang": "en", "script": "Latn", "source": "human",
            "item_id": "a#1", "gold": "A", "answer": "A", "prompt_tokens": 100}
    f.write_text(json.dumps(good) + "\n")
    s1 = D.runs_signature(str(tmp_path / "runs"))
    with f.open("a") as fh:
        fh.write(json.dumps({**good, "item_id": "a#2"}) + "\n" + '{"benchmark": "bele')
    s2 = D.runs_signature(str(tmp_path / "runs"))
    assert s1 != s2
    from evaluation import stats
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        items, _ = stats.load_runs(tmp_path / "runs")
    assert len(items) == 2 and items["correct"].all()


def test_completeness_counts_cells(tmp_path):
    import pandas as pd
    counts = pd.DataFrame([{"benchmark": "belebele", "condition": "eng_Latn", "model_key": "gemma3-12b", "n": 300},
                           {"benchmark": "belebele", "condition": "hin_Deva", "model_key": "gemma3-12b", "n": 120}])
    comp = D.completeness(counts, tmp_path, ["gemma3-12b", "qwen3-8b"])
    assert comp["belebele"]["cells_total"] == 2 * 18
    assert comp["belebele"]["cells_done"] == 1
    assert comp["belebele"]["rows"] == 420


def test_charts_return_figures():
    import numpy as np
    import pandas as pd
    import dash_charts as C
    fig = C.annotated_heatmap(np.array([[1.0, np.nan], [-2.0, 3.0]]), [["+1.0", ""], ["-2.0", "+3.0*"]], ["a", "b"], ["x", "y"], "pp", diverging=True)
    assert fig.axes
    df = pd.DataFrame({"model": ["gemma3-12b", "qwen3-8b"], "cat": ["hi", "hi"], "value": [1.0, -1.0], "lo": [-1.0, -3.0], "hi": [3.0, 1.0], "star": [True, False]})
    assert C.grouped_bars(df, ["hi", "bn"], {"hi": "Hindi", "bn": "Bengali"}, ["gemma3-12b", "qwen3-8b"], "pp").axes
    assert C.empty_figure("nothing").axes

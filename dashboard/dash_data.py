"""Data access for the dashboard: where the files are, cached loading, and thin wrappers over evaluation.stats.

Nothing here re-implements a statistic. Loading is ``stats.load_runs`` / ``stats.build_tables``; accuracy tables,
paired gaps, Holm-corrected families, token ratios and the Global-MMLU tables come from ``evaluation.stats``.

Everything under ``runs/`` is opened read-only. Paths are resolved on every script run (not at import time), so a test
or a second browser tab can point at a different folder.

Folder settings, highest priority first: the sidebar box (pre-filled from the ``?runs=`` query parameter), command-line
arguments after ``--`` (``streamlit run dashboard/app.py -- --runs /some/dir``), environment variables, defaults.

=====================  ===================  ===========================================
setting                environment variable default
=====================  ===================  ===========================================
runs folder            EVALUATION_RUNS       <repo>/runs
raw data               EVALUATION_RAW        <repo>/data/raw
derived data           EVALUATION_DERIVED    <repo>/data/derived
stored results         EVALUATION_RESULTS    <repo>/reports/results
stored figures         EVALUATION_FIGURES    <repo>/reports/figures
bootstrap resamples    EVALUATION_NBOOT      2000 (the paper's analysis uses 10000)
=====================  ===================  ===========================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from evaluation import stats
from evaluation.config import LANGUAGES

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ("belebele", "gmmlu_lite", "gmmlu")
BENCHMARK_NAME = {"belebele": "Belebele (reading comprehension)", "gmmlu_lite": "Global-MMLU-Lite (knowledge)",
                  "gmmlu": "Global-MMLU, full (knowledge)"}
NBOOT_CHOICES = [500, 2000, 10000]
DEFAULT_NBOOT = 2000

SCRIPT_NAME = {"Latn": "Latin", "Deva": "Devanagari", "Beng": "Bengali", "Arab": "Perso-Arabic", "Telu": "Telugu",
               "Taml": "Tamil", "Gujr": "Gujarati", "Knda": "Kannada", "Mlym": "Malayalam", "Guru": "Gurmukhi",
               "Orya": "Odia"}
METHOD_LABEL = {"xlit": "IndicXlit back to native script", "mt_en": "NLLB translation to English",
                "inde": "Indian English (Multi-VALUE)", "cm": "Hinglish (machine-made)"}
RTL_SCRIPTS = {"Arab"}


# ------------------------------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class Paths:
    runs: Path
    raw: Path
    derived: Path
    results: Path
    figures: Path


def _cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("runs", "raw", "derived", "results", "figures"):
        ap.add_argument(f"--{name}", default=None)
    try:
        args, _ = ap.parse_known_args(sys.argv[1:])
    except SystemExit:  # unrelated arguments (for example under pytest)
        return argparse.Namespace(**{n: None for n in ("runs", "raw", "derived", "results", "figures")})
    return args


def _pick(cli: str | None, env: str, default: Path) -> Path:
    return Path(cli or os.environ.get(env) or default).expanduser()


def default_runs_dir() -> str:
    return str(_pick(_cli_args().runs, "EVALUATION_RUNS", ROOT / "runs"))


def resolve_paths(runs: str | None = None) -> Paths:
    a = _cli_args()
    return Paths(
        runs=Path(runs).expanduser() if runs else _pick(a.runs, "EVALUATION_RUNS", ROOT / "runs"),
        raw=_pick(a.raw, "EVALUATION_RAW", ROOT / "data" / "raw"),
        derived=_pick(a.derived, "EVALUATION_DERIVED", ROOT / "data" / "derived"),
        results=_pick(a.results, "EVALUATION_RESULTS", ROOT / "reports" / "results"),
        figures=_pick(a.figures, "EVALUATION_FIGURES", ROOT / "reports" / "figures"),
    )


def default_nboot() -> int:
    try:
        return int(os.environ.get("EVALUATION_NBOOT", DEFAULT_NBOOT))
    except ValueError:
        return DEFAULT_NBOOT


# ------------------------------------------------------------------------------------- labels


def condition_script(cond: str) -> str:
    base = cond.split("+")[0]
    return base.split("_")[1] if "_" in base else ""


def cond_label(cond: str, benchmark: str = "belebele") -> str:
    """Plain-English name of a condition id, e.g. ``hin_Latn+xlit`` -> ``Hindi (romanized) + IndicXlit back to native script``."""
    base, _, method = cond.partition("+")
    iso = stats.condition_lang(base)
    name = LANGUAGES[iso].name if iso in LANGUAGES else base
    if benchmark != "belebele":
        label = name
    elif base == "eng_Latn":
        label = "English"
    elif condition_script(base) == "Latn":
        label = f"{name} (romanized)"
    else:
        label = f"{name} (native script)"
    if method:
        label += f" + {METHOD_LABEL.get(method, method)}"
    return label


def belebele_human_conditions() -> list[str]:
    """English, then 13 native-script languages, then the 4 romanized ones (the main experiment's 18 conditions)."""
    out = ["eng_Latn"] + [c for _, c in stats.native_languages()] + [lt for _, _, lt in stats.romanized_languages()]
    return out


def expected_human_conditions(benchmark: str) -> list[str]:
    if benchmark == "belebele":
        return belebele_human_conditions()
    attr = "global_mmlu_lite" if benchmark == "gmmlu_lite" else "global_mmlu"
    return [getattr(l, attr) for l in LANGUAGES.values() if getattr(l, attr)]


def fmt_pp(x, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "n/a"
    v = round(float(x) * 100, 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def fmt_ci(lo, hi, signed: bool = True) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "n/a"
    return f"[{fmt_pp(lo, signed)}, {fmt_pp(hi, signed)}]"


def fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def verdict(row) -> str:
    """Verdict of one row of ``stats.confirmatory`` output, in words (never colour only)."""
    if row["status"] != "ok":
        return "not run"
    if bool(row["confirmed"]):
        return "confirmed"
    if row["p_holm"] < stats.ALPHA:
        return "significant, opposite direction"
    return "not confirmed"


# ------------------------------------------------------------------------------------- loading (cached)


def runs_signature(runs_dir: str) -> tuple:
    """Cheap fingerprint of the run files (name, size, mtime). The files are append-only while the experiment runs."""
    runs = Path(runs_dir)
    sig = []
    if runs.is_dir():
        for p in sorted(runs.glob("*/*.jsonl")):
            try:
                s = p.stat()
            except OSError:
                continue
            sig.append((f"{p.parent.name}/{p.name}", s.st_size, s.st_mtime_ns))
    return tuple(sig)


def path_signature(*paths: Path) -> tuple:
    out = []
    for p in paths:
        try:
            s = Path(p).stat()
            out.append((str(p), s.st_size, s.st_mtime_ns))
        except OSError:
            out.append((str(p), -1, -1))
    return tuple(out)


@st.cache_resource(show_spinner="Reading the run files (read-only)...", max_entries=3)
def load_all(runs_dir: str, sig: tuple) -> dict:
    """Run rows -> (items, templates, tables). Read-only; a half-written last line is skipped, not fatal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        items, templates = stats.load_runs(Path(runs_dir))
    n_bad = sum("unreadable JSON" in str(w.message) for w in caught)
    tables = stats.build_tables(items) if len(items) else {}
    return dict(items=items, templates=templates, tables=tables, n_bad=n_bad)


@st.cache_data(show_spinner="Computing accuracy and bootstrap confidence intervals...", max_entries=6)
def accuracy(runs_dir: str, sig: tuple, n_boot: int) -> pd.DataFrame:
    d = load_all(runs_dir, sig)
    return stats.accuracy_table(d["tables"], n_boot) if d["tables"] else pd.DataFrame()


@st.cache_data(show_spinner="Running the main hypothesis tests (paired bootstrap, exact McNemar, Holm)...", max_entries=6)
def confirmatory(runs_dir: str, sig: tuple, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H1-H5 exactly as ``evaluation.stats`` defines them (fixed-size Holm families: a test that cannot run counts as p = 1)."""
    d = load_all(runs_dir, sig)
    return stats.confirmatory(d["tables"], d["templates"], None, n_boot, True)


@st.cache_data(show_spinner=False, max_entries=6)
def token_table(runs_dir: str, sig: tuple) -> pd.DataFrame:
    d = load_all(runs_dir, sig)
    return stats.token_ratios(d["tables"], d["templates"]) if d["tables"] else pd.DataFrame()


@st.cache_data(show_spinner="Computing exploratory comparisons...", max_entries=6)
def exploratory_gaps(runs_dir: str, sig: tuple, n_boot: int) -> pd.DataFrame:
    """Indian English and Hinglish vs their unmodified references; paired gap + CI + unadjusted exact McNemar."""
    d = load_all(runs_dir, sig)
    bel = d["tables"].get("belebele", {})
    pairs = [("eng_Latn+inde", "eng_Latn"), ("hin_Latn+cm", "hin_Latn")]
    rows = []
    for model in sorted({m for m, _ in bel}):
        for cond, ref in pairs:
            tc, tr = bel.get((model, cond)), bel.get((model, ref))
            if tc is None or tr is None:
                continue
            res = stats.paired_gap(tc["correct"], tr["correct"], n_boot)
            if res["n"]:
                rows.append(dict(model=model, condition=cond, reference=ref, **res))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing Global-MMLU comparisons...", max_entries=6)
def gmmlu_bundle(runs_dir: str, sig: tuple, raw_dir: str, raw_sig: tuple, n_boot: int) -> dict:
    d = load_all(runs_dir, sig)
    tables = d["tables"]
    labels = stats.load_cs_labels(Path(raw_dir))
    cs_ca, cs_diff = stats.gmmlu_lite_cs_ca(tables, labels, n_boot)
    return dict(gaps=stats.gmmlu_gaps(tables, n_boot), cs_ca=cs_ca, cs_diff=cs_diff, have_labels=labels is not None)


@st.cache_data(show_spinner="Computing the Q3 vs Q4 sensitivity check...", max_entries=6)
def quantization(runs_dir: str, sig: tuple, n_boot: int) -> pd.DataFrame:
    d = load_all(runs_dir, sig)
    return stats.quantization(d["tables"], n_boot) if d["tables"] else pd.DataFrame()


# ------------------------------------------------------------------------------------- progress


def sample_size(derived_dir: Path, benchmark: str, observed_max: int) -> tuple[int, bool]:
    """(expected items per condition, taken from the sample file?). Falls back to the largest count seen."""
    p = Path(derived_dir) / "samples" / f"{benchmark}.json"
    try:
        return len(json.loads(p.read_text())["items"]), True
    except (OSError, ValueError, KeyError):
        return observed_max, False


def row_counts(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=["benchmark", "condition", "model_key", "n"])
    return items.groupby(["benchmark", "condition", "model_key"], sort=False).size().rename("n").reset_index()


def ordered_models(models) -> list[str]:
    from evaluation import plots

    present = set(models)
    main = [m for m in plots.MODEL_ORDER if m in present]
    return main + sorted(present - set(main))


def ordered_conditions(benchmark: str, present) -> list[str]:
    """Expected human conditions first (in paper order), then derived ones, then anything unexpected."""
    present = set(present)
    base = expected_human_conditions(benchmark)
    derived = sorted(c for c in present if "+" in c)
    other = sorted(c for c in present if c not in base and "+" not in c)
    return base + derived + other


def completeness(counts: pd.DataFrame, derived_dir: Path, models: list[str]) -> dict:
    """How much of the planned main grid (5 models x human conditions) is done, per benchmark."""
    out = {}
    for b in BENCHMARKS:
        sub = counts[counts["benchmark"] == b]
        if sub.empty:
            out[b] = dict(expected_n=0, cells_total=0, cells_done=0, rows=0, rows_total=0)
            continue
        exp_n, _ = sample_size(derived_dir, b, int(sub["n"].max()))
        conds = expected_human_conditions(b)
        got = {(r.model_key, r.condition): r.n for r in sub.itertuples()}
        cells_total = len(models) * len(conds)
        done = sum(1 for m in models for c in conds if got.get((m, c), 0) >= exp_n)
        rows = sum(got.get((m, c), 0) for m in models for c in conds)
        out[b] = dict(expected_n=exp_n, cells_total=cells_total, cells_done=done, rows=rows, rows_total=cells_total * exp_n)
    return out


# ------------------------------------------------------------------------------------- item text


def _read_jsonl_by_item(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = r.get("item_id") or f"{r.get('link')}#{int(r.get('question_number'))}"
            rows[iid] = r
    return rows


def condition_path(raw_dir: Path, derived_dir: Path, cond: str) -> Path:
    if "+" in cond:
        return Path(derived_dir) / "conditions" / "belebele" / f"{cond}.jsonl"
    return Path(raw_dir) / "belebele" / "data" / f"{cond}.jsonl"


@st.cache_data(show_spinner=False, max_entries=40)
def belebele_texts(raw_dir: str, derived_dir: str, cond: str, sig: tuple) -> dict[str, dict]:
    """item_id -> {passage, question, options[4], gold (A-D)} for one Belebele condition (human or machine-made)."""
    p = condition_path(Path(raw_dir), Path(derived_dir), cond)
    if not p.exists():
        return {}
    out = {}
    for iid, r in _read_jsonl_by_item(p).items():
        try:
            gold = "ABCD"[int(r["correct_answer_num"]) - 1]
        except (KeyError, ValueError, IndexError):
            gold = ""
        out[iid] = dict(passage=r.get("flores_passage", ""), question=r.get("question", ""),
                        options=[r.get(f"mc_answer{i}", "") for i in range(1, 5)], gold=gold)
    return out


def available_item_conditions(raw_dir: Path, derived_dir: Path, in_runs) -> list[str]:
    found = set()
    raw = Path(raw_dir) / "belebele" / "data"
    if raw.is_dir():
        found |= {p.stem for p in raw.glob("*.jsonl")}
    der = Path(derived_dir) / "conditions" / "belebele"
    if der.is_dir():
        found |= {p.stem for p in der.glob("*.jsonl")}
    found |= set(in_runs)
    ordered = [c for c in belebele_human_conditions() if c in found]
    return ordered + sorted(c for c in found if c not in ordered)


def sample_items(derived_dir: Path, benchmark: str, fallback) -> list[str]:
    p = Path(derived_dir) / "samples" / f"{benchmark}.json"
    try:
        return list(json.loads(p.read_text())["items"])
    except (OSError, ValueError, KeyError):
        return sorted(set(fallback))


def item_title(item_id: str) -> str:
    link, _, num = item_id.rpartition("#")
    return f"{link.rstrip('/').rsplit('/', 1)[-1].replace('_', ' ')} / question {num}"

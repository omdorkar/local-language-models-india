"""Main experiment harness (models, prompt and decoding as described in paper/paper.md section 3; data formats in docs/CONTRACT.md).

    python -m evaluation.run --benchmark belebele --models all
    python -m evaluation.run --benchmark belebele --models llama3.1-8b --conditions eng_Latn hin_Deva --limit 3
    python -m evaluation.run --benchmark gmmlu_lite --models all --dry-run

Appends one row per (model, condition, item) to runs/<benchmark>/<model_key>.jsonl, flushed immediately.
Rows already present are never re-run, so an interrupted run is simply started again.
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import DATA_RAW, ROOT
from .client import ask
from .config import LANGUAGES, MODELS, SEED
from .prompts import (GMMLU_TEMPLATE_SHA, TEMPLATE_SHA, belebele_gold, belebele_prompt, gmmlu_gold, gmmlu_prompt)

RUNS = ROOT / "runs"
SAMPLES = ROOT / "data" / "derived" / "samples"
DERIVED = ROOT / "data" / "derived" / "conditions"
BENCHMARKS = ("belebele", "gmmlu_lite", "gmmlu")
OLLAMA_URL = "http://127.0.0.1:11434/v1"
LLAMA_URL = "http://127.0.0.1:8080/v1"
RETRIES = 3
TEMPLATE_ID = "__template__"


# ---------------------------------------------------------------- per-model settings

@dataclass(frozen=True)
class ModelSettings:
    key: str
    tag: str  # Ollama tag or llama-server alias
    runtime: str  # "ollama" | "llama-server"
    base_url: str
    max_tokens: int = 32
    reasoning_effort: str | None = None


MODEL_SETTINGS: dict[str, ModelSettings] = {
    "gemma3-12b": ModelSettings("gemma3-12b", "gemma3:12b", "ollama", OLLAMA_URL),
    "qwen3-8b": ModelSettings("qwen3-8b", "qwen3:8b", "ollama", OLLAMA_URL, 32, "none"),
    "llama3.1-8b": ModelSettings("llama3.1-8b", "llama3.1:8b", "ollama", OLLAMA_URL),
    "gpt-oss-20b": ModelSettings("gpt-oss-20b", "gpt-oss:20b", "ollama", OLLAMA_URL, 512, "low"),
    "sarvam-30b": ModelSettings("sarvam-30b", "sarvam-30b-q3", "llama-server", LLAMA_URL),
    # Sensitivity check for the Q3 confound (paper/paper.md section 3.3); not part of `--models all`.
    "llama3.1-8b-q3": ModelSettings("llama3.1-8b-q3", "llama3.1:8b-instruct-q3_K_M", "ollama", OLLAMA_URL),
}
MAIN_MODELS = [m.key for m in MODELS]


# ---------------------------------------------------------------- conditions

_BELEBELE_ISO = {c: iso for iso, l in LANGUAGES.items() for c in (l.belebele_native, l.belebele_latn) if c}
_BELEBELE_NATIVE = {iso: l.belebele_native for iso, l in LANGUAGES.items() if l.belebele_native}
_GMMLU_SCRIPT = {"en": "Latn", "hi": "Deva", "bn": "Beng", "or": "Orya", "te": "Telu", "ne": "Deva"}


def human_conditions(benchmark: str) -> list[str]:
    if benchmark == "belebele":
        return [c for l in LANGUAGES.values() for c in (l.belebele_native, l.belebele_latn) if c]
    attr = "global_mmlu_lite" if benchmark == "gmmlu_lite" else "global_mmlu"
    return [getattr(l, attr) for l in LANGUAGES.values() if getattr(l, attr)]


def is_derived(condition: str) -> bool:
    return "+" in condition


def derived_path(benchmark: str, condition: str, derived_dir: Path = DERIVED) -> Path:
    return derived_dir / benchmark / f"{condition}.jsonl"


def condition_meta(benchmark: str, condition: str) -> dict:
    """lang (ISO 639-1), script (ISO 15924), source (human/machine) for a condition id."""
    base, _, method = condition.partition("+")
    if benchmark == "belebele":
        lang, script = _BELEBELE_ISO[base], base.split("_")[1]
        native = _BELEBELE_NATIVE[lang].split("_")[1]
    else:
        lang, script = base, _GMMLU_SCRIPT[base]
        native = script
    if method == "xlit":  # romanized input converted back to the native script
        script = native
    elif method == "mt_en":  # machine-translated to English
        lang, script = "en", "Latn"
    return {"lang": lang, "script": script, "source": "machine" if method else "human"}


def sampled_items(benchmark: str, limit: int | None = None, samples_dir: Path = SAMPLES) -> list[str]:
    items = json.loads((samples_dir / f"{benchmark}.json").read_text())["items"]
    return items[:limit] if limit else items


def load_condition(benchmark: str, condition: str, item_ids: list[str], derived_dir: Path = DERIVED) -> dict[str, dict]:
    """item_id -> source row for the requested items (items absent from the condition are left out)."""
    wanted = set(item_ids)
    if is_derived(condition):
        path = derived_path(benchmark, condition, derived_dir)
        if not path.exists():
            raise FileNotFoundError(f"derived condition file missing: {path}")
        rows = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["item_id"] in wanted:
                    rows[row["item_id"]] = row
        return rows
    if benchmark == "belebele":
        from .inspect_belebele import load
        df = load(condition)
        rows = {}
        for item_id in item_ids:
            link, number = item_id.rsplit("#", 1)
            key = (link, int(number))
            if key in df.index:
                rows[item_id] = df.loc[key].to_dict()
        return rows
    root = "global_mmlu_lite" if benchmark == "gmmlu_lite" else "global_mmlu"
    df = pd.read_parquet(DATA_RAW / root / condition / "test-00000-of-00001.parquet")
    return {r["sample_id"]: r for r in df[df.sample_id.isin(wanted)].to_dict("records")}


def prompt_and_gold(benchmark: str, row) -> tuple[str, str]:
    if benchmark == "belebele":
        return belebele_prompt(row), belebele_gold(row)
    return gmmlu_prompt(row), gmmlu_gold(row)


def template_sha(benchmark: str) -> str:
    return TEMPLATE_SHA if benchmark == "belebele" else GMMLU_TEMPLATE_SHA


def calibration_prompt(benchmark: str) -> str:
    """The template with every field empty (for the token-tax baseline)."""
    if benchmark == "belebele":
        return belebele_prompt({k: "" for k in ("flores_passage", "question", "mc_answer1", "mc_answer2",
                                                "mc_answer3", "mc_answer4")} | {"correct_answer_num": "1"})
    return gmmlu_prompt({k: "" for k in ("question", "option_a", "option_b", "option_c", "option_d")})


# ---------------------------------------------------------------- output file

def run_path(benchmark: str, model_key: str, runs_dir: Path = RUNS) -> Path:
    return runs_dir / benchmark / f"{model_key}.jsonl"


def read_rows(path: Path) -> list[dict]:
    """All complete rows; a truncated last line from a killed run is ignored."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def done_keys(path: Path) -> set[tuple[str, str]]:
    """(condition, item_id) already present; the model is implied by the file name."""
    return {(r["condition"], r["item_id"]) for r in read_rows(path)}


def all_pairs(conditions: list[str], item_ids: list[str]) -> list[tuple[str, str]]:
    return [(c, i) for c in conditions for i in item_ids]


def shuffled(pairs: list, seed: int = SEED) -> list:
    """Fixed-seed order, independent of what is already done (so resumed runs keep the same order)."""
    order = np.random.default_rng(seed).permutation(len(pairs))
    return [pairs[i] for i in order]


# ---------------------------------------------------------------- requests

def ask_with_retry(client: httpx.Client, s: ModelSettings, prompt: str, sleep=time.sleep) -> dict:
    """Up to RETRIES retries with exponential backoff (2, 4, 8 s) on HTTP/timeout/malformed-response errors."""
    for attempt in range(RETRIES + 1):
        try:
            return ask(client, s.base_url, s.tag, prompt, SEED, s.max_tokens, s.reasoning_effort)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            if attempt == RETRIES:
                raise
            sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _row(benchmark: str, s: ModelSettings, meta: dict, condition: str, item_id: str, gold: str, out: dict) -> dict:
    return {
        "benchmark": benchmark, "model_key": s.key, "model_tag": s.tag, "condition": condition,
        **meta, "item_id": item_id, "gold": gold, "answer": out["answer"], "correct": out["answer"] == gold,
        "raw": out["raw"], "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
        "seconds": out["seconds"], "template_sha": template_sha(benchmark), "seed": SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _log_error(errors_log: Path, s: ModelSettings, benchmark: str, condition: str, item_id: str, err: Exception) -> None:
    errors_log.parent.mkdir(parents=True, exist_ok=True)
    with open(errors_log, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{benchmark}\t{s.key}\t{condition}\t"
                f"{item_id}\t{type(err).__name__}: {err}\n")


class _Appender:
    """Append-only JSONL writer that flushes every row."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = path.exists() and path.stat().st_size > 0 and path.read_bytes()[-1:] != b"\n"
        self.f = open(path, "a", encoding="utf-8")
        if needs_newline:  # previous run was killed mid-line
            self.f.write("\n")

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def run_model(benchmark: str, s: ModelSettings, data: dict[str, dict[str, dict]], conditions: list[str],
              item_ids: list[str], client: httpx.Client, *, runs_dir: Path = RUNS, errors_log: Path | None = None,
              sleep=time.sleep, progress: bool = True) -> dict:
    """Run every missing (condition, item) for one model. `data[condition][item_id]` is the source row."""
    path = run_path(benchmark, s.key, runs_dir)
    errors_log = errors_log or runs_dir / "errors.log"
    done = done_keys(path)
    pairs = [p for p in shuffled(all_pairs(conditions, item_ids)) if p not in done and p[1] in data[p[0]]]
    stats = {"written": 0, "failed": 0, "calibration": 0}
    out = _Appender(path)
    try:
        if (TEMPLATE_ID, TEMPLATE_ID) not in done:
            try:
                res = ask_with_retry(client, s, calibration_prompt(benchmark), sleep)
                out.write(_row(benchmark, s, {"lang": "", "script": "", "source": ""}, TEMPLATE_ID, TEMPLATE_ID, "", res))
                stats["calibration"] = 1
            except Exception as e:  # noqa: BLE001 - logged, rerun fills it
                _log_error(errors_log, s, benchmark, TEMPLATE_ID, TEMPLATE_ID, e)
                stats["failed"] += 1
        bar = tqdm(pairs, desc=f"{benchmark}/{s.key}", unit="row", disable=None if progress else True)
        t0 = time.time()
        for n, (condition, item_id) in enumerate(bar, 1):
            prompt, gold = prompt_and_gold(benchmark, data[condition][item_id])
            try:
                res = ask_with_retry(client, s, prompt, sleep)
            except Exception as e:  # noqa: BLE001 - logged, not written, so a rerun fills it
                _log_error(errors_log, s, benchmark, condition, item_id, e)
                stats["failed"] += 1
                continue
            out.write(_row(benchmark, s, condition_meta(benchmark, condition), condition, item_id, gold, res))
            stats["written"] += 1
            if progress and not sys.stderr.isatty() and n % 100 == 0:  # plain log lines when not a terminal
                rate = (time.time() - t0) / n
                print(f"  {benchmark}/{s.key}: {n}/{len(pairs)} rows, ETA {rate * (len(pairs) - n) / 60:.0f} min", flush=True)
    finally:
        out.close()
    return stats


def summary(benchmark: str, model_key: str, conditions: list[str], runs_dir: Path = RUNS) -> pd.DataFrame:
    rows = [r for r in read_rows(run_path(benchmark, model_key, runs_dir)) if r["condition"] in set(conditions)]
    if not rows:
        return pd.DataFrame(columns=["condition", "n", "accuracy", "invalid"])
    df = pd.DataFrame(rows)
    g = df.groupby("condition", sort=False)
    return pd.DataFrame({"n": g.size(), "accuracy": g.correct.mean().round(3),
                         "invalid": g.answer.apply(lambda s: int(s.isna().sum()))}).reset_index()


# ---------------------------------------------------------------- CLI

def resolve_models(names: list[str]) -> list[ModelSettings]:
    keys = MAIN_MODELS if names == ["all"] else names
    unknown = [k for k in keys if k not in MODEL_SETTINGS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(MODEL_SETTINGS)} or 'all'")
    return [MODEL_SETTINGS[k] for k in keys]


def resolve_conditions(benchmark: str, requested: list[str] | None) -> list[str]:
    human = human_conditions(benchmark)
    if not requested or requested == ["human"]:
        return human
    bad = [c for c in requested if not is_derived(c) and c not in human]
    if bad:
        raise SystemExit(f"unknown condition(s) {bad} for {benchmark}; human conditions: {human}")
    for c in requested:
        if is_derived(c):
            try:
                condition_meta(benchmark, c)
            except KeyError:
                raise SystemExit(f"cannot interpret derived condition {c!r}") from None
    return requested


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--models", nargs="+", required=True, help=f"keys from {list(MODEL_SETTINGS)} or 'all' (the 5 main)")
    ap.add_argument("--conditions", nargs="+", help="condition ids, or 'human' (default: all human conditions)")
    ap.add_argument("--limit", type=int, help="only the first N sampled items (pilot)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; call no model")
    args = ap.parse_args(argv)

    models = resolve_models(args.models)
    conditions = resolve_conditions(args.benchmark, args.conditions)
    item_ids = sampled_items(args.benchmark, args.limit)

    missing = [c for c in conditions if is_derived(c) and not derived_path(args.benchmark, c).exists()]
    if missing and not args.dry_run:
        raise SystemExit(f"derived condition files missing: {missing}")
    data = {c: load_condition(args.benchmark, c, item_ids) for c in conditions if c not in missing}
    absent = {c: len(item_ids) - len(d) for c, d in data.items() if len(d) < len(item_ids)}
    if absent:
        print(f"WARNING: items missing from conditions (will not be run): {absent}")
    runnable = [c for c in conditions if c in data]
    total = sum(len(d) for d in data.values())

    print(f"benchmark={args.benchmark} items={len(item_ids)} conditions={len(conditions)} "
          f"rows per model={total} template={template_sha(args.benchmark)}")
    if missing:
        print(f"  derived conditions without a file (skipped in dry run): {missing}")
    plan_todo = {}
    for s in models:
        done = done_keys(run_path(args.benchmark, s.key))
        todo = sum(1 for c in runnable for i in data[c] if (c, i) not in done)
        plan_todo[s.key] = todo
        have = sum(1 for c in runnable for i in data[c] if (c, i) in done)
        calib = "yes" if (TEMPLATE_ID, TEMPLATE_ID) in done else "no"
        print(f"  {s.key:15s} [{s.tag}] done={have} to_do={todo} calibration_row={calib}")
    if args.dry_run:
        return 0

    def _interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):  # make `finally` blocks (server shutdown) run
        signal.signal(sig, _interrupt)

    from .servers import ServerError, serve
    failures = 0
    for s in models:
        if not plan_todo[s.key] and (TEMPLATE_ID, TEMPLATE_ID) in done_keys(run_path(args.benchmark, s.key)):
            print(f"\n{s.key}: nothing to do")
            print(summary(args.benchmark, s.key, runnable).to_string(index=False))
            continue
        print(f"\n=== {s.key} ({s.tag}) ===", flush=True)
        try:
            with serve(s), httpx.Client(timeout=600) as client:
                stats = run_model(args.benchmark, s, data, runnable, item_ids, client)
        except ServerError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            failures += 1
            continue
        failures += stats["failed"]
        print(f"{s.key}: wrote {stats['written']} rows"
              f"{' + calibration row' if stats['calibration'] else ''}, {stats['failed']} failed (see runs/errors.log)")
        print(summary(args.benchmark, s.key, runnable).to_string(index=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

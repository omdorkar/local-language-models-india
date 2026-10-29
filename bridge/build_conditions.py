"""Regenerate the H4 (xlit) and H5 (mt_en) condition files from scratch, resumably.

    .venv-bridge/bin/python -m bridge.build_conditions [--only xlit|mt] [--langs hin_Latn,hin_Deva,...] [--no-validate]

Word/sentence caches live in data/derived/cache/ (xlit_<lang>.json, mt_<config>.json); a re-run with warm caches
loads no model and just rewrites the files. Delete the cache files to recompute from scratch.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time

from . import data as D
from . import mt as MT
from . import xlit as XL


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def build_xlit(configs: list[str], info: dict) -> None:
    t_all = time.time()
    x = XL.Xlit()
    for cfg in configs:
        t0 = time.time()
        lang = D.LATN_TO_XLIT[cfg]
        rows = D.load_config(cfg)
        flat = [r[f] for r in rows for f in D.TEXT_FIELDS]
        conv = x.transliterate_many(flat, lang)
        x.save()
        out, k = [], 0
        for r in rows:
            n = dict(r)
            for f in D.TEXT_FIELDS:
                n[f] = conv[k]
                k += 1
            n.update(source_condition=cfg, method=XL.METHOD, method_version=XL.METHOD_VERSION)
            out.append(n)
        D.write_condition(f"{cfg}+xlit", out)
        info["xlit"][cfg] = {"rows": len(out), "unique_words": len(x._words(lang)), "seconds": round(time.time() - t0, 1)}
        print(f"[xlit] {cfg}: {len(out)} rows, {info['xlit'][cfg]['seconds']}s", flush=True)
    info["xlit_total_seconds"] = round(time.time() - t_all, 1)
    info["xlit_peak_rss_gb"] = round(_peak_rss_gb(), 2)


def build_mt(configs: list[str], info: dict) -> None:
    t_all = time.time()
    tr = MT.Translator()
    for cfg in configs:
        t0, m0 = time.time(), tr.seconds_model
        rows = D.load_config(cfg)
        # unique passages once; questions/options per row
        uniq_pass = list(dict.fromkeys(r["flores_passage"] for r in rows))
        pas = dict(zip(uniq_pass, tr.translate_texts(uniq_pass, cfg, progress=True)))
        rest_fields = [f for f in D.TEXT_FIELDS if f != "flores_passage"]
        flat = [r[f] for r in rows for f in rest_fields]
        conv = tr.translate_texts(flat, cfg, progress=True)
        tr.save(cfg)
        out, k = [], 0
        for r in rows:
            n = dict(r)
            n["flores_passage"] = pas[r["flores_passage"]]
            for f in rest_fields:
                n[f] = conv[k]
                k += 1
            n.update(source_condition=cfg, method=MT.METHOD, method_version=MT.METHOD_VERSION)
            out.append(n)
        D.write_condition(f"{cfg}+mt_en", out)
        info["mt"][cfg] = {"rows": len(out), "seconds": round(time.time() - t0, 1), "generate_seconds": round(tr.seconds_model - m0, 1)}
        print(f"[mt] {cfg}: {len(out)} rows, {info['mt'][cfg]['seconds']}s", flush=True)
    info["mt_total_seconds"] = round(time.time() - t_all, 1)
    info["mt_peak_rss_gb"] = round(_peak_rss_gb(), 2)
    info["mt_device"], info["mt_dtype"] = tr.device_name, tr.dtype_name
    try:
        import torch

        if torch.backends.mps.is_available() and tr.device_name == "mps":
            info["mt_mps_driver_allocated_gb_at_end"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    except Exception:
        pass


def add_row_scores(kind: str, scores: dict, field: str) -> None:
    """Write per-row validation scores into the condition files."""
    by_cfg: dict[str, dict[str, float]] = {}
    for (cfg, iid), v in scores.items():
        by_cfg.setdefault(cfg, {})[iid] = v
    for cfg, d in by_cfg.items():
        p = D.COND_DIR / f"{cfg}+{kind}.jsonl"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r[field] = round(d[r["item_id"]], 4)
        D.write_condition(f"{cfg}+{kind}", rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["xlit", "mt"])
    ap.add_argument("--langs", help="comma-separated source configs (e.g. hin_Latn,hin_Deva); default all")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    want = set(a.langs.split(",")) if a.langs else None
    D.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    info_path = D.REPORT_DIR / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.setdefault("xlit", {})
    info.setdefault("mt", {})
    info["machine"] = platform.platform() + "; " + platform.machine()

    from . import validate as V

    if a.only in (None, "xlit"):
        cfgs = [c for c in D.LATN_TO_XLIT if want is None or c in want]
        build_xlit(cfgs, info)
        info["xlit_method_version"] = XL.METHOD_VERSION
    if a.only in (None, "mt"):
        cfgs = [c for c in D.NATIVE_CONFIGS if want is None or c in want]
        build_mt(cfgs, info)
        info["mt_method_version"] = MT.METHOD_VERSION
        info["mt_generation"] = MT.GEN
    info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False))

    if not a.no_validate:
        if a.only in (None, "xlit") and want is None:
            _, sc = V.xlit_validation()
            add_row_scores("xlit", sc, "cer_vs_native")
        if a.only in (None, "mt") and want is None:
            _, sc = V.mt_validation()
            add_row_scores("mt_en", sc, "chrf_vs_eng")
        print("validation reports written to", D.REPORT_DIR)


if __name__ == "__main__":
    main()

"""Check, without downloading any weights, whether each candidate model fits this Mac's GPU memory.

    python -m evaluation.model_fit   # writes reports/model_fit.md

Sizes come from the Ollama registry manifests and Hugging Face file listings. The GPU budget is
Metal's recommendedMaxWorkingSetSize: the most memory macOS lets the GPU hold by default.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import httpx

from . import REPORTS
from .config import MODELS

CONTEXT_TOKENS = 4096  # Belebele passage + question in an Indic script stays well under this
COMPUTE_BUFFER_GB = 1.0  # llama.cpp / Ollama scratch buffers; conservative
OLLAMA_KV_GB = 0.8  # upper bound for KV cache at 4k context for the 8-20B models here

SARVAM_QUANTS = [  # (label, repo, filename prefix)
    ("Q4_K_M (official)", "sarvamai/sarvam-30b-gguf", "sarvam-30b-Q4_K_M"),
    # Pre-merge community conversion: arch "sarvam_moe" is rejected by llama.cpp (tested 2026-10-20).
    # About the same size as a Q3_K_M made locally from Sumitc13/sarvam-30b-GGUF BF16 with `llama-quantize`.
    ("Q3_K_M (size reference)", "DevQuasar/sarvamai.sarvam-30b-GGUF", "sarvamai.sarvam-30b.f16.gguf.Q3_K_M"),
]


def metal_budget_gb() -> tuple[float, str]:
    swift = 'import Metal\nprint(MTLCreateSystemDefaultDevice()!.recommendedMaxWorkingSetSize)\n'
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
        f.write(swift)
    try:
        out = subprocess.run(["swift", f.name], capture_output=True, text=True, timeout=120)
        return int(out.stdout.strip()) / 1e9, "Metal recommendedMaxWorkingSetSize"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        return ram * 2 / 3 / 1e9, "estimate: 2/3 of RAM (Swift unavailable)"
    finally:
        Path(f.name).unlink(missing_ok=True)


def ollama_weights_gb(tag: str) -> float:
    name, size = tag.split(":")
    r = httpx.get(f"https://registry.ollama.ai/v2/library/{name}/manifests/{size}",
                  headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"}, timeout=30)
    r.raise_for_status()
    # The vision projector (Gemma 3) is not loaded for text-only prompts.
    return sum(l["size"] for l in r.json()["layers"] if l["mediaType"].endswith(".model")) / 1e9


def hf_files_gb(repo: str, prefix: str) -> float:
    r = httpx.get(f"https://huggingface.co/api/models/{repo}", params={"blobs": "true"}, timeout=30)
    r.raise_for_status()
    return sum(s.get("size") or 0 for s in r.json()["siblings"] if s["rfilename"].startswith(prefix)) / 1e9


def sarvam_kv_gb() -> float:
    c = httpx.get("https://huggingface.co/sarvamai/sarvam-30b/raw/main/config.json",
                  follow_redirects=True, timeout=30).json()
    per_token = c["num_hidden_layers"] * 2 * c["num_key_value_heads"] * c["head_dim"] * 2  # K and V, fp16
    return per_token * CONTEXT_TOKENS / 1e9


def main() -> None:
    budget, source = metal_budget_gb()
    rows = []
    for m in MODELS:
        if m.ollama_tag:
            w = ollama_weights_gb(m.ollama_tag)
            rows.append((m.display, f"Ollama `{m.ollama_tag}`", w, OLLAMA_KV_GB + COMPUTE_BUFFER_GB))
    kv = sarvam_kv_gb()
    for label, repo, prefix in SARVAM_QUANTS:
        rows.append(("Sarvam-30B", f"llama.cpp, {label}", hf_files_gb(repo, prefix), kv + COMPUTE_BUFFER_GB))

    lines = [
        "# Does each model fit on this Mac?",
        "",
        f"GPU memory budget: **{budget:.2f} GB** ({source}). Context assumed: {CONTEXT_TOKENS} tokens.",
        "Only one model is loaded at a time, so each row is checked on its own.",
        "",
        "| Model | Runtime / quantization | Weights (GB) | KV + buffers (GB) | Total (GB) | Headroom (GB) | Fits on GPU? |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name, runtime, w, extra in rows:
        total = w + extra
        lines.append(f"| {name} | {runtime} | {w:.2f} | {extra:.2f} | {total:.2f} | {budget - total:+.2f} | "
                     f"{'yes' if total <= budget else '**no**'} |")
    lines += [
        "",
        f"Sarvam-30B KV cache is small ({kv * 1000:.0f} MB at {CONTEXT_TOKENS} tokens) because it has 19 layers "
        "and 4 KV heads; the weights are what decide the fit.",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_fit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

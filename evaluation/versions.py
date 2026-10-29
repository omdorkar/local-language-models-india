"""Record the exact version of every model under test, so results can be tied to specific weights.

    python -m evaluation.versions   # writes reports/model_versions.{md,json}

Ollama models are identified by their manifest digest (the same tag can point to new weights later);
the Sarvam GGUF by its SHA-256.
"""

import hashlib
import json

import httpx

from . import REPORTS, ROOT
from .config import MODELS

OLLAMA = "http://127.0.0.1:11434"
SARVAM_GGUF = ROOT / "models" / "sarvam-30b" / "sarvam-30b-Q3_K_M.gguf"


def ollama_rows(client: httpx.Client) -> list[dict]:
    tags = {m["name"]: m for m in client.get(f"{OLLAMA}/api/tags").json()["models"]}
    rows = []
    for m in MODELS:
        if not m.ollama_tag:
            continue
        tag = tags.get(m.ollama_tag)
        if tag is None:
            rows.append({"model": m.display, "tag": m.ollama_tag, "status": "not pulled"})
            continue
        show = client.post(f"{OLLAMA}/api/show", json={"model": m.ollama_tag}).json()
        info = show.get("model_info", {})
        rows.append({
            "model": m.display,
            "tag": m.ollama_tag,
            "digest": tag["digest"],
            "size_gb": round(tag["size"] / 1e9, 2),
            "architecture": info.get("general.architecture"),
            "parameters": tag["details"].get("parameter_size"),
            "quantization": tag["details"].get("quantization_level"),
            "capabilities": show.get("capabilities", []),
            "licence": (show.get("license") or "").strip().splitlines()[0][:80] if show.get("license") else "",
            "modified_at": tag.get("modified_at"),
        })
    return rows


def sarvam_row() -> dict:
    if not SARVAM_GGUF.exists():
        return {"model": "Sarvam-30B", "status": "not built (see reports/sarvam_setup.md)"}
    h = hashlib.sha256()
    with SARVAM_GGUF.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return {
        "model": "Sarvam-30B", "tag": "llama.cpp local GGUF", "digest": f"sha256:{h.hexdigest()}",
        "size_gb": round(SARVAM_GGUF.stat().st_size / 1e9, 2), "architecture": "bailingmoe2",
        "parameters": "32B (2.4B active)", "quantization": "Q3_K_M", "capabilities": ["completion", "thinking"],
        "licence": "Apache-2.0",
    }


def main() -> None:
    with httpx.Client(timeout=60) as client:
        ollama_version = client.get(f"{OLLAMA}/api/version").json()["version"]
        rows = ollama_rows(client)
    rows.append(sarvam_row())
    import subprocess
    llama_cpp = subprocess.run(["llama-server", "--version"], capture_output=True, text=True).stderr.strip().splitlines()
    runtimes = {"ollama": ollama_version, "llama.cpp": next((l for l in llama_cpp if l.startswith("version")), "")}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_versions.json").write_text(json.dumps({"runtimes": runtimes, "models": rows}, indent=2))
    lines = [
        "# Exact model versions",
        "",
        f"Runtimes: Ollama {runtimes['ollama']}; llama.cpp {runtimes['llama.cpp']}.",
        "The paper (section 3.3, Table 2) lists the first 12 characters of each digest.",
        "",
        "| Model | Tag / runtime | Digest | Size (GB) | Arch | Params | Quant | Capabilities | Licence |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for r in rows:
        if "digest" not in r:
            lines.append(f"| {r['model']} | {r.get('tag', '')} | {r['status']} | | | | | | |")
            continue
        lines.append(f"| {r['model']} | `{r['tag']}` | `{r['digest'][:19]}…` | {r['size_gb']} | {r['architecture']} | "
                     f"{r['parameters']} | {r['quantization']} | {', '.join(r['capabilities'])} | {r['licence']} |")
    lines += ["", "Full digests are in `model_versions.json`."]
    (REPORTS / "model_versions.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

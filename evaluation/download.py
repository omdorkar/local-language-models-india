"""Download every public dataset the study uses into data/raw/, and record exactly what was fetched.

    python -m evaluation.download                 # all Hugging Face datasets
    python -m evaluation.download --dakshina      # also Dakshina (streams a 2 GB tar, keeps only romanized files)
    python -m evaluation.download --only belebele global_mmlu

data/raw/MANIFEST.json records the source, licence, revision and SHA-256 of every file,
so a reader can verify they are evaluating on identical data.
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from . import DATA_RAW
from .config import DATASETS, LANGUAGES, Dataset

MANIFEST = DATA_RAW / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(ds: Dataset) -> dict:
    target = DATA_RAW / ds.key
    revision = HfApi().dataset_info(ds.repo).sha
    snapshot_download(
        repo_id=ds.repo, repo_type="dataset", revision=revision,
        allow_patterns=ds.patterns, local_dir=target,
    )
    files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
    return {
        "source": f"https://huggingface.co/datasets/{ds.repo}",
        "revision": revision,
        "files": {str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }


def fetch_dakshina(ds: Dataset) -> dict:
    """Stream the tar and keep only romanized/ folders for Indian languages (~2 GB transfer, small on disk)."""
    target = DATA_RAW / ds.key
    target.mkdir(parents=True, exist_ok=True)
    wanted = {l.dakshina for l in LANGUAGES.values() if l.dakshina}
    with urllib.request.urlopen(ds.repo) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
        for member in tar:
            parts = Path(member.name).parts  # dakshina_dataset_v1.0/<lang>/romanized/<file>
            if member.isfile() and len(parts) >= 4 and parts[1] in wanted and parts[2] == "romanized":
                tar.extract(member, target, filter="data")
    files = sorted(p for p in target.rglob("*") if p.is_file())
    return {
        "source": ds.repo,
        "revision": "v1.0",
        "files": {str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(DATASETS), help="download just these datasets")
    ap.add_argument("--dakshina", action="store_true", help="include Dakshina (2 GB stream)")
    args = ap.parse_args()

    keys = args.only or [k for k in DATASETS if k != "dakshina" or args.dakshina]
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    for key in keys:
        ds = DATASETS[key]
        print(f"→ {key} ({ds.repo})")
        try:
            entry = fetch_dakshina(ds) if key == "dakshina" else fetch_hf(ds)
        except (GatedRepoError, HfHubHTTPError) as e:
            if ds.gated:
                print(f"  skipped: {ds.repo} is gated. Accept its terms at https://huggingface.co/datasets/{ds.repo} "
                      "while signed in, run `hf auth login`, then rerun with --only " + key)
                continue
            raise SystemExit(f"  failed: {e}")
        entry |= {"licence": ds.licence, "citation": ds.citation,
                  "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest[key] = entry
        total = sum(f["bytes"] for f in entry["files"].values())
        print(f"  {len(entry['files'])} files, {total / 1e6:.1f} MB")
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()

"""Environment shims for the separate .venv-extra (import this BEFORE multivalue/stanza/nltk).

* keeps NLTK / Stanza caches inside .venv-extra (nothing written to $HOME)
* value-nlp's __init__ calls `spacy download` and `nltk.download` on every import; we install those
  resources once (see README) and neutralise the calls so imports are offline and fast
* torch>=2.6 defaults to weights_only=True which cannot unpickle Stanza 1.8.2's (official) coref checkpoint;
  we load those official Stanza checkpoints with weights_only=False
"""
import functools
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-extra"
os.environ.setdefault("NLTK_DATA", str(VENV / "nltk_data"))
os.environ.setdefault("STANZA_RESOURCES_DIR", str(VENV / "stanza_resources"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(VENV / "hf_cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
SEED = 1909


def patch_for_multivalue():
    import nltk
    import spacy.cli
    import torch

    spacy.cli.download = lambda *a, **k: None
    nltk.download = lambda *a, **k: True
    _orig = torch.load

    @functools.wraps(_orig)
    def _load(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)

    torch.load = _load

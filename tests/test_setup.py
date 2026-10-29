import json

import pytest

from evaluation import DATA_RAW
from evaluation.config import DATASETS, LANGUAGES
from evaluation.inspect_belebele import latin_share

MANIFEST = DATA_RAW / "MANIFEST.json"


def test_latin_share():
    import pandas as pd

    assert latin_share(pd.Series(["aap kaise ho"])) == 1.0
    assert latin_share(pd.Series(["आप कैसे हो"])) == 0.0


def test_romanized_languages_also_have_native():
    for lang in LANGUAGES.values():
        if lang.belebele_latn:
            assert lang.belebele_native, lang.name


@pytest.mark.skipif(not MANIFEST.exists(), reason="run `python -m evaluation.download` first")
def test_every_belebele_config_was_downloaded():
    files = json.loads(MANIFEST.read_text())["belebele"]["files"]
    expected = [p for p in DATASETS["belebele"].patterns if p.endswith(".jsonl")]
    assert sorted(p for p in files if p.endswith(".jsonl")) == sorted(expected)

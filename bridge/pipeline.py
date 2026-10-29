"""Small reusable API.

    from bridge import normalize, to_english
    normalize("aap kaise hain?", "hi", "Latn")   # -> "आप कैसे हैं?"      (IndicXlit; Hindi/Bengali/Urdu/Nepali)
    to_english("आप कैसे हैं?", "hi")              # -> "How are you?"     (NLLB-200-distilled-1.3B)

`lang` is ISO 639-1 (`hi`, `bn`, ...) or a Belebele/NLLB code (`hin_Deva`). Models load lazily on first use.
"""
from __future__ import annotations

import warnings

from . import mt, xlit

_xlit: xlit.Xlit | None = None
_mt: mt.Translator | None = None


def normalize(text: str, lang: str, script: str = "Latn") -> str:
    """Return `text` in the language's native script. Latin-script input is transliterated word by word
    with IndicXlit (supported here: hi, bn, ur, ne); text already in a native script is returned unchanged."""
    global _xlit
    if script != "Latn":
        return text
    iso = {"hin": "hi", "ben": "bn", "urd": "ur", "npi": "ne"}.get(lang.split("_")[0], lang)
    if iso not in xlit.XLIT_LANGS:
        warnings.warn(f"No transliteration configured for {lang!r}; text returned unchanged.")
        return text
    if _xlit is None:
        _xlit = xlit.Xlit()
    return _xlit.transliterate(text, iso)


def to_english(text: str, lang: str) -> str:
    """Translate native-script `text` to English with NLLB-200-distilled-1.3B (beam 4, sentence-split)."""
    global _mt
    if _mt is None:
        _mt = mt.Translator()
    return _mt.translate(text, lang)

"""Roman -> native-script transliteration with AI4Bharat IndicXlit (word level).

Uses the *official* `ai4bharat-transliteration` package (fairseq transformer checkpoint
`indicxlit-en-indic-v1.0`, plus the word-probability rescoring dictionaries). Only the
text handling around the model is ours: IndicXlit is word-level, so a text is split into
Latin-letter words, each unique word is transliterated once (batched, beam 4, rescored
top-1, exactly the package's own `post_process`), and the words are put back in place.
Punctuation, digits, whitespace and non-Latin characters are left untouched.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from ._compat import patch_urduhack

XLIT_LANGS = {"hi": "Hindi", "bn": "Bengali", "ur": "Urdu", "ne": "Nepali"}  # target codes used by IndicXlit
METHOD = "indicxlit"
METHOD_VERSION = "ai4bharat-transliteration==1.1.3; model=indicxlit-en-indic-v1.0 (transformer); beam=4; rescore(alpha=0.9) top-1; word-level"

# A "word": Latin letters (incl. accented), optionally joined by an apostrophe (oceloto’er -> ocelotoer).
WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’][A-Za-zÀ-ɏ]+)*")


def _ascii_fold(word: str) -> str:
    w = word.replace("'", "").replace("’", "")
    w = unicodedata.normalize("NFKD", w)
    return "".join(c for c in w if not unicodedata.combining(c)).lower()


class Xlit:
    def __init__(self, langs=tuple(XLIT_LANGS), cache_dir: str | Path = "data/derived/cache", batch_size: int = 64, threads: int = 6):
        patch_urduhack()
        import torch

        torch.set_num_threads(threads)
        from ai4bharat.transliteration import XlitEngine

        self.langs = tuple(langs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.engine = XlitEngine(list(self.langs), beam_width=4, rescore=True)
        self._cache: dict[str, dict[str, str]] = {}

    # ---- word cache -------------------------------------------------------------
    def _cache_path(self, lang: str) -> Path:
        return self.cache_dir / f"xlit_{lang}.json"

    def _words(self, lang: str) -> dict[str, str]:
        if lang not in self._cache:
            p = self._cache_path(lang)
            self._cache[lang] = json.loads(p.read_text()) if p.exists() else {}
        return self._cache[lang]

    def save(self) -> None:
        for lang, d in self._cache.items():
            self._cache_path(lang).write_text(json.dumps(d, ensure_ascii=False, sort_keys=True))

    # ---- model call -------------------------------------------------------------
    def _run_model(self, words: list[str], lang: str) -> list[str]:
        """Transliterate already-folded lower-case ASCII words; returns one native word each."""
        eng = self.engine
        pre = eng.pre_process(words, "en", lang)
        raw = eng.transliterator.translate(pre, nbest=4)
        groups: dict[int, list[str]] = {}
        for line in raw.split("\n"):
            if line[:2] in ("S-", "H-"):
                groups.setdefault(int(line.split("\t")[0].split("-")[1]), []).append(line)
        out = []
        for i in range(len(words)):
            # reuse the package's own post-processing (rescoring) one word at a time
            cands = eng.post_process("\n".join(groups[i]), lang)
            out.append(cands[0] if cands else words[i])
        return out

    def translit_words(self, words: list[str], lang: str) -> dict[str, str]:
        cache = self._words(lang)
        todo = sorted({_ascii_fold(w) for w in words} - cache.keys())
        todo = [w for w in todo if w]
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i : i + self.batch_size]
            for w, t in zip(chunk, self._run_model(chunk, lang)):
                cache[w] = t
        return cache

    # ---- text -------------------------------------------------------------------
    def transliterate(self, text: str, lang: str) -> str:
        if lang not in self.langs:
            raise ValueError(f"IndicXlit engine not loaded for {lang!r}; have {self.langs}")
        cache = self.translit_words(WORD_RE.findall(text), lang)
        return WORD_RE.sub(lambda m: cache.get(_ascii_fold(m.group(0)), m.group(0)) or m.group(0), text)

    def transliterate_many(self, texts: list[str], lang: str) -> list[str]:
        """Warm the word cache for all texts in one go (efficient batching), then rewrite each."""
        self.translit_words([w for t in texts for w in WORD_RE.findall(t)], lang)
        return [self.transliterate(t, lang) for t in texts]

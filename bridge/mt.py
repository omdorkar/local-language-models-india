"""Translate-test: native-script text -> English with NLLB-200-distilled-1.3B (open, CC-BY-NC-4.0).

`src_lang` is a Belebele config code, which is also the NLLB language code (e.g. `hin_Deva`).
Passages are split into sentences (NLLB is trained on sentence-level data and drops content on long
inputs); each unique sentence is translated once with beam search and cached on disk.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
MODEL_REVISION = "7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
TGT_LANG = "eng_Latn"
METHOD = "nllb-200-distilled-1.3B"
GEN = dict(num_beams=4, max_input_tokens=256, max_new_tokens_cap=256, length_factor=2.0, length_slack=10)
METHOD_VERSION = (
    f"{MODEL_ID}@{MODEL_REVISION[:12]}; transformers==4.46.3; beam={GEN['num_beams']}; "
    "sentence-split; max_new_tokens=min(256, 2*src_len+10); fp16 on MPS"
)

# ISO 639-1 -> default NLLB/Belebele code (for the convenience API only)
ISO_TO_NLLB = {
    "hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab", "ne": "npi_Deva", "mr": "mar_Deva", "te": "tel_Telu",
    "ta": "tam_Taml", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru", "or": "ory_Orya",
    "as": "asm_Beng", "en": "eng_Latn",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?।॥؟۔])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def resolve_lang(lang: str) -> str:
    return ISO_TO_NLLB.get(lang, lang)


class Translator:
    def __init__(self, cache_dir: str | Path = "data/derived/cache", device: str | None = None, dtype: str | None = None, batch_size: int = 16):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device_name = device or os.environ.get("BRIDGE_DEVICE")
        self.dtype_name = dtype or os.environ.get("BRIDGE_DTYPE")
        self.batch_size = batch_size
        self.model = None
        self.tok = None
        self._cache: dict[str, dict[str, str]] = {}
        self.seconds_model = 0.0  # time spent in generate()
        self.n_segments = 0

    # ---- model ------------------------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if (self.device_name or "mps") == "mps":  # cap the MPS allocator (read at first MPS use); other jobs share the GPU
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
            os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
        dev = self.device_name or ("mps" if torch.backends.mps.is_available() else "cpu")
        dt = self.dtype_name or ("float16" if dev == "mps" else "float32")
        self.device_name, self.dtype_name = dev, dt
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        # The upstream checkpoint is a 5.5 GB fp32 pytorch_model.bin (loading it peaks near 10 GB RSS). On first
        # use we write an fp16 safetensors copy to the cache dir; later loads are mmap'd, ~2.7 GB, no spike.
        local = self.cache_dir / f"nllb-200-distilled-1.3B-{MODEL_REVISION[:12]}-{dt}"
        if (local / "model.safetensors").exists():
            model = AutoModelForSeq2SeqLM.from_pretrained(local, torch_dtype=getattr(torch, dt))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, torch_dtype=getattr(torch, dt), low_cpu_mem_usage=True
            )
            model.save_pretrained(local, safe_serialization=True)
        self.model = model.to(dev).eval()

    def unload(self) -> None:
        import gc

        import torch

        self.model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ---- cache ------------------------------------------------------------------
    def _cache_path(self, src_lang: str) -> Path:
        return self.cache_dir / f"mt_{src_lang}.json"

    def _seg_cache(self, src_lang: str) -> dict[str, str]:
        if src_lang not in self._cache:
            p = self._cache_path(src_lang)
            self._cache[src_lang] = json.loads(p.read_text()) if p.exists() else {}
        return self._cache[src_lang]

    def save(self, src_lang: str) -> None:
        tmp = self._cache_path(src_lang).with_suffix(".tmp")
        tmp.write_text(json.dumps(self._seg_cache(src_lang), ensure_ascii=False, sort_keys=True))
        tmp.replace(self._cache_path(src_lang))

    # ---- translation ------------------------------------------------------------
    def _generate(self, segs: list[str], src_lang: str) -> list[str]:
        import torch

        self.load()
        self.tok.src_lang = src_lang
        enc = self.tok(segs, return_tensors="pt", padding=True, truncation=True, max_length=GEN["max_input_tokens"]).to(self.device_name)
        src_len = int(enc["attention_mask"].sum(1).max())
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                forced_bos_token_id=self.tok.convert_tokens_to_ids(TGT_LANG),
                num_beams=GEN["num_beams"],
                do_sample=False,
                max_new_tokens=min(GEN["max_new_tokens_cap"], int(GEN["length_factor"] * src_len) + GEN["length_slack"]),
            )
        self.seconds_model += time.time() - t0
        if self.device_name == "mps":
            torch.mps.empty_cache()  # variable shapes otherwise make the MPS allocator balloon
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def translate_segments(self, segs: list[str], src_lang: str, progress: bool = False) -> list[str]:
        cache = self._seg_cache(src_lang)
        todo = sorted({s for s in segs if s not in cache}, key=len, reverse=True)
        i, nb = 0, 0
        while i < len(todo):
            # length-bucketed batches: at most `batch_size` segments and ~batch_size*120 characters (longest first)
            j = i + 1
            while j < len(todo) and j - i < 2 * self.batch_size and (j - i + 1) * len(todo[i]) <= self.batch_size * 120:
                j += 1
            chunk = todo[i:j]
            for s, t in zip(chunk, self._generate(chunk, src_lang)):
                cache[s] = t.strip()
            self.n_segments += len(chunk)
            i, nb = j, nb + 1
            if nb % 20 == 0:
                self.save(src_lang)  # resumable mid-language
            if progress and nb % 10 == 0:
                print(f"  [{src_lang}] {i}/{len(todo)} segments", flush=True)
        return [cache[s] for s in segs]

    def translate_texts(self, texts: list[str], src_lang: str, progress: bool = False) -> list[str]:
        """Translate whole fields (sentence-split, re-joined with a space)."""
        split = [split_sentences(t) or [t] for t in texts]
        flat = [s for ss in split for s in ss]
        done = iter(self.translate_segments(flat, src_lang, progress))
        return [" ".join(next(done) for _ in ss) for ss in split]

    def translate(self, text: str, src_lang: str) -> str:
        return self.translate_texts([text], resolve_lang(src_lang))[0]

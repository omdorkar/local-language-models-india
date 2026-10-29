# bridge: preprocessing fixes for H4 (transliteration) and H5 (translate-test)

Two fixes tested by the study, plus a small reusable API for the eventual open-source tool.

| Fix | Hypothesis | Method | Input -> output |
|---|---|---|---|
| Transliteration | H4 | AI4Bharat IndicXlit (`ai4bharat-transliteration` 1.1.3, model `indicxlit-en-indic-v1.0`) | romanized `hin_Latn`, `ben_Latn`, `urd_Latn`, `npi_Latn` -> native script |
| Translate-test | H5 | `facebook/nllb-200-distilled-1.3B` @ `7be3e24664b3`, beam 4 | the 13 native-script Indian configs -> English |

IndicTrans2 is gated and is deliberately **not** used.

## Setup (separate venv; never install into `.venv`)

```bash
cd /path/to/local-language-models-india
uv venv .venv-bridge --python 3.10          # uv may warn that 3.10 < the project's 3.11; harmless
P=.venv-bridge/bin/python
uv pip install --python $P "numpy<2" "cython<3" wheel "setuptools<70" "torch==2.5.1"

# fairseq 0.12.2: the PyPI sdist is missing fairseq/clib, so build from the git tag
git clone --depth 1 --branch v0.12.2 https://github.com/facebookresearch/fairseq.git /tmp/fairseq-src   # commit 4a388e64cd64
CXXFLAGS=-Wno-invalid-specialization CFLAGS=-Wno-invalid-specialization \
  uv pip install --python $P --no-build-isolation /tmp/fairseq-src

# IndicXlit WITHOUT its dependency tree (that tree pulls TensorFlow via urduhack; see bridge/_compat.py)
uv pip install --python $P --no-deps ai4bharat-transliteration==1.1.3 urduhack==1.1.1
uv pip install --python $P -r bridge/requirements-bridge.txt
```

The IndicXlit checkpoint (127 MB) and rescoring dictionaries (850 MB) are downloaded on first use from the
AI4Bharat GitHub release `v1.0` into the venv (`site-packages/ai4bharat/transliteration/.../models`). NLLB (5.5 GB fp32
`pytorch_model.bin`) is downloaded from Hugging Face on first use; an fp16 safetensors copy (2.7 GB) is then written to
`data/derived/cache/nllb-200-distilled-1.3B-*-float16/` so later loads are fast and stay under 4 GB RSS.

## Reproduce

```bash
.venv-bridge/bin/python -m bridge.build_conditions            # everything (xlit + mt), resumable, then validation
.venv-bridge/bin/python -m bridge.build_conditions --only xlit
.venv-bridge/bin/python -m bridge.build_conditions --only mt --langs hin_Deva,ben_Beng
```

Caches (`data/derived/cache/xlit_<lang>.json`, `mt_<config>.json`) make re-runs cheap (a fully warm run: ~35 s, no model
loaded). Delete them to recompute. Cold run on an Apple M4 Pro (GPU shared with other jobs): xlit 3.2 min on CPU; NLLB
4.2 h on MPS fp16 (11-34 min per language; 682-2066 s). Peak RSS 5.9 GB (cold load, fp32 -> fp16) and ~4.7 GB steady.
`/usr/bin/time -l` also reported a "peak memory footprint" of 46 GB for the MT run, which we believe counts cumulative
MPS/compressed allocations; the MPS allocator was capped at 60 % of the recommended working set
(`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6`) and `driver_allocated_memory` read 3.7 GB at the end. Timings and settings: `reports/bridge/run_info.json`.

## API

```python
from bridge import normalize, to_english
normalize("aap kaise hain?", "hi", "Latn")     # IndicXlit; hi, bn, ur, ne
to_english("आप कैसे हैं?", "hi")                 # NLLB; ISO 639-1 or Belebele/NLLB code ("hin_Deva")
```

## Outputs

`data/derived/conditions/belebele/` (300 rows each, per `docs/CONTRACT.md`):

* `hin_Latn+xlit.jsonl`, `ben_Latn+xlit.jsonl`, `urd_Latn+xlit.jsonl`, `npi_Latn+xlit.jsonl`
* `<config>+mt_en.jsonl` for hin_Deva, ben_Beng, urd_Arab, npi_Deva, mar_Deva, tel_Telu, tam_Taml, guj_Gujr, kan_Knda, mal_Mlym, pan_Guru, ory_Orya, asm_Beng

Rows keep all source fields (`link`, `question_number`, `flores_passage`, `question`, `mc_answer1..4`, `correct_answer_num`,
`dialect`, `ds`; `dialect` still names the *source* config) and add `item_id`, `source_condition`, `method`, `method_version`,
plus a per-row validation score (`cer_vs_native` for xlit, `chrf_vs_eng` for mt_en). `correct_answer_num` is unchanged: options are translated one by one in place.

Reports in `reports/bridge/`: `xlit_validation.{md,csv}`, `mt_validation.{md,csv}`, `run_info.json`, `build_mt.log`.

## How it works

* **xlit** (`bridge/xlit.py`): text is split into Latin-letter words (internal apostrophes dropped, accents folded, lower-cased);
  each unique word is run through the official model (fairseq transformer, beam 4, top-4 rescored with the package's word-probability
  dictionaries, alpha 0.9, top-1 kept) in batches of 64; words are put back in place. Punctuation, digits, whitespace stay as in the source
  (full stops are **not** converted to danda / Arabic full stop; digits stay ASCII). IndicXlit supports all four target languages (hi, bn, ur, ne).
* **mt** (`bridge/mt.py`): `src_lang` = Belebele config code (= NLLB code), `tgt_lang=eng_Latn`, beam 4, no sampling, fp16 on MPS.
  Fields are split into sentences on `. ! ? । ॥ ؟ ۔`+whitespace, unique sentences translated once (length-sorted batches), re-joined with a space.
  `max_new_tokens = min(256, 2*src_len+10)`, input truncated at 256 tokens. Passages shared by several questions are translated once.

## Validation (300 sampled items, vs Belebele's own human text)

CER / WER in %. **Read the question/options columns with care**: Belebele's romanized passage is a faithful romanization of the native passage,
but the romanized question and options are an *independent* Hinglish-style translation (e.g. `Passage ke hisab se` vs native `परिच्छेद के अनुसार`), so
their CER mostly measures lexical divergence between two human translations, not transliteration error. The passage CER is the clean number.
`CER` ignores punctuation, zero-width characters, Arabic diacritics; unifies Arabic/Urdu letter variants and native digits.

| Config | CER passage | CER question | CER options | CER all | WER passage |
|---|---|---|---|---|---|
| hin_Latn+xlit (vs hin_Deva) | 4.67 | 49.28 | 49.35 | 16.55 | 16.8 |
| ben_Latn+xlit (vs ben_Beng) | 4.89 | 45.68 | 44.08 | 15.63 | 16.57 |
| urd_Latn+xlit (vs urd_Arab) | 8.26 | 49.51 | 43.39 | 18.18 | 31.72 |
| npi_Latn+xlit (vs npi_Deva) | 5.76 | 45.43 | 46.91 | 16.56 | 20.4 |

NLLB vs Belebele English (sacrebleu 2.6.0, chrF default; BLEU 13a; corpus level; `passage` = 165 unique passages):

| Config | chrF passage | chrF question | chrF options | chrF all | BLEU passage | BLEU all |
|---|---|---|---|---|---|---|
| hin_Deva+mt_en | 67.9 | 64.61 | 62.44 | 66.09 | 39.8 | 37.97 |
| ben_Beng+mt_en | 62.51 | 58.14 | 56.21 | 60.32 | 31.17 | 29.42 |
| urd_Arab+mt_en | 64.05 | 67.15 | 59.73 | 63.58 | 33.99 | 33.96 |
| npi_Deva+mt_en | 68.9 | 65.06 | 54.02 | 64.82 | 41.8 | 34.16 |
| mar_Deva+mt_en | 63.08 | 61.18 | 55.95 | 61.13 | 30.19 | 29.09 |
| tel_Telu+mt_en | 64.67 | 59.34 | 57.36 | 62.11 | 31.16 | 29.85 |
| tam_Taml+mt_en | 61.41 | 61.34 | 53.86 | 59.7 | 27.25 | 27.04 |
| guj_Gujr+mt_en | 65.52 | 62.16 | 62.77 | 64.29 | 31.61 | 31.67 |
| kan_Knda+mt_en | 64.08 | 67.68 | 61.36 | 64.05 | 34.63 | 34.64 |
| mal_Mlym+mt_en | 63.64 | 61.41 | 63.37 | 63.14 | 30.5 | 30.67 |
| pan_Guru+mt_en | 67.48 | 65.05 | 61.22 | 65.64 | 39.47 | 37.43 |
| ory_Orya+mt_en | 65.9 | 62.16 | 63.13 | 64.59 | 37.18 | 36.46 |
| asm_Beng+mt_en | 56.07 | 61.91 | 53.71 | 56.51 | 20.16 | 21.4 |

## Licences

* IndicXlit / `ai4bharat-transliteration`: MIT. fairseq: MIT.
* NLLB-200-distilled-1.3B: **CC-BY-NC 4.0** (research use only; derived `+mt_en` files inherit this restriction).
* Belebele: CC-BY-SA 4.0 (derived files are share-alike).

## Known limitations

* Romanized Belebele is human-written with free spelling; IndicXlit expects fairly regular romanization, so Urdu is the weakest (passage WER 32 %; Roman Urdu drops vowels/ain/hamza distinctions).
* IndicXlit is word-level and context-free: English loanwords and names in Latin text are also transliterated (`community law centre` -> `कम्युनिटी लॉ सेंटर`, usually fine; sometimes wrong), words are lower-cased.
* NLLB mistranslates numbers and negation occasionally; Assamese (chrF 56.5) is clearly the weakest source, then Tamil and Bengali. Option strings are short fragments, which depresses BLEU a lot (13-32).
* NLLB is sentence-level; cross-sentence references in passages are lost, and a sentence-split heuristic can split after abbreviations.
* fp16 on MPS gave near-identical output to fp32 in a 19-sentence check (chrF 90.7 between them; differences were synonyms); not re-verified on the full set.
* `torchaudio` 2.11 gets pulled in by fairseq's metadata and does not match torch 2.5.1; it is unused.
* urduhack's TensorFlow-dependent top level is bypassed (`bridge/_compat.py`); this is safe because only the Indic->Roman direction uses its Urdu normaliser.

# `extra/` — exploratory input conditions (Indian English, machine-made Hinglish)

Two **exploratory** Belebele conditions for the 300 sampled items (`data/derived/samples/belebele.json`), built only with
published open methods and validated automatically against open human-made references. No LLM rewriting, no manual annotation,
no new data collection.

| Condition id | Output file | Method |
|---|---|---|
| `eng_Latn+inde` | `data/derived/conditions/belebele/eng_Latn+inde.jsonl` | Multi-VALUE `IndianDialect` (rule-based) on the English items |
| `hin_Latn+cm` | `data/derived/conditions/belebele/hin_Latn+cm.jsonl` | SimAlign word alignment (hin_Latn <-> eng_Latn) + single-token English substitution, CMI-calibrated to COMI-LINGUA |

Reports: `reports/extra/inde_validation.md`, `reports/extra/cm_validation.md` (+ CSVs, `cm_cmi_hist.png`), each with three
original-vs-transformed examples. Everything lives in `extra/` and `.venv-extra/`; nothing is installed in `.venv`.

## Setup (one-off, about 10 minutes plus about 1 GB of downloads)

```bash
# from the repo root
uv venv --python 3.10 .venv-extra                      # Multi-VALUE targets Python 3.10 (pins spacy 3.7.4, stanza 1.8.2)
uv pip install --python .venv-extra/bin/python -r extra/requirements-extra.txt
uv pip install --python .venv-extra/bin/python \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
NLTK_DATA=$PWD/.venv-extra/nltk_data .venv-extra/bin/python -m nltk.downloader -d .venv-extra/nltk_data wordnet cmudict
```

`extra/requirements-extra.lock.txt` is the exact `uv pip freeze` of the environment used. Stanza English models
(`stanza.Pipeline("en", ...coref)`, about 450 MB) and the Hugging Face models (MuRIL, LaBSE; Stanza's coref model uses
`google/electra-large-discriminator`) are downloaded automatically on first use into `.venv-extra/` (`extra/_env.py` redirects
`STANZA_RESOURCES_DIR`, `NLTK_DATA` and `HF_HOME` there). All downloads are ungated; no login or terms acceptance was needed.

Shims in `extra/_env.py` (needed because `value-nlp` 0.3 is old): (i) `value-nlp` calls `spacy download` and `nltk.download` on
every import — both are no-op'd after the one-off install above; (ii) torch >= 2.6 defaults to `weights_only=True`, which cannot load
Stanza 1.8.2's official coref checkpoint — `torch.load` is forced to `weights_only=False` inside this venv only; (iii) `numpy<2`.

## Commands (repo root)

```bash
.venv-extra/bin/python -m extra.inde            # Indian English: transform (about 40-60 min on CPU) + validate + report
.venv-extra/bin/python -m extra.inde --validate-only   # re-score / re-write outputs from the cached Multi-VALUE output
.venv-extra/bin/python -m extra.cm              # Hinglish: align + substitute + validate + report (about 3-5 min after model download)
.venv-extra/bin/python -m extra.cm --sim-min 0.66 --margin-min 0   # looser alignment-confidence filter (see below)
.venv-extra/bin/python -m extra.cm_sensitivity  # mixing level vs. filter strength -> reports/extra/cm_sensitivity.csv
.venv-extra/bin/python -m extra.cmi             # prints the COMI-LINGUA CMI summary
```

Seed 1909 everywhere. Randomness is keyed per text unit (`sha256("1909|item_id|field")` for Multi-VALUE; `sha256("1909|cm|<hindi unit>")`
for the Hinglish substitution), so outputs do not depend on processing order and the same passage gets identical treatment under every
question. Caches: `reports/extra/inde_field_cache_v2.json` (Multi-VALUE output per field), `reports/extra/cm_alignment_cache.json`.

## Output format (CONTRACT)

One row per sampled item, in the order of `belebele.json`, with the source row's fields (`link`, `question_number`, `flores_passage`,
`question`, `mc_answer1..4`, `correct_answer_num`, `ds`; `correct_answer_num` is copied unchanged) plus `dialect` (= the condition id),
`item_id`, `source_condition`, `method`, `method_version`, `options_distinct`, and a `validation` object. Extra fields:
`eng_Latn+inde` has `rules_fired`, `rules_detail` (per field: rule type, character span in the *original* string, produced value),
`n_rules_fired`, `mv_raw_fields` (unfiltered Multi-VALUE output), `gate_reverted_fields`, `fallback_status`;
`hin_Latn+cm` has `aligned_with`, `replacements` (per field: Hindi word, English replacement, aligned English token, POS, rule),
`n_replaced`, `reverted_option_fields`.

## Condition 1 — Indian English (`eng_Latn+inde`)

**Method.** Multi-VALUE (Ziems, Held, Yang, Dhamala, Gupta, Yang. *Multi-VALUE: A Framework for Cross-Dialectal English NLP*. ACL 2023, pp. 744-768;
https://aclanthology.org/2023.acl-long.44 ; code https://github.com/SALT-NLP/multi-value ; PyPI `value-nlp`) is a rule-based
system that maps Standard American English to synthetic dialects by applying morphosyntactic feature functions (189 features, 50 dialects)
attested in eWAVE; which functions apply, and with what probability, comes from the dialect's attestation vector. Its Indian English
setting is `multivalue.Dialects.IndianDialect` (eWAVE variety "Indian English", abbr. IndE; `Dialects.IndianSouthAfricanDialect` is a different
variety). We apply it to the passage, question and each option separately (spaCy `en_core_web_sm` 3.7.1 parses, Stanza 1.8.2 coreference),
re-seeding Python's RNG per field. The firing of every rule is recorded (type, span, output).
**Two additions of ours, both reported:** (1) `install_guard` in `extra/inde.py` stops the `progressives` rule ("is/are + V-ing") from
firing on verbs that are the sentence root without a grammatical subject — the stock rule turns imperatives ("Make sure ...") and noun-phrase
fragments (option "Meat") into "Are Making sure ...", "Are Meating"; every other rule is unchanged. (2) A **LaBSE gate**: a field whose
LaBSE cosine to the original is below 0.85 is reverted to the original English in the main fields (the unfiltered output stays in
`mv_raw_fields`; items with any such field carry `validation.flag_low_sim`). Without the gate, short options come out as e.g. "Polouses" for "Police".
Options are also checked to stay pairwise distinct (a colliding option would be reverted; none collided).

**Validation** (`reports/extra/inde_validation.md`): fraction of items/fields changed, mean character edit ratio (Levenshtein / max length),
LaBSE cosine per field, flagged items, rule counts. See the report for the numbers of the current build.

**Licence.** `value-nlp` is Apache-2.0 **with an added use clause**: users agree not to use Multi-VALUE for "malicious purposes including ... deception,
impersonation, mockery, discrimination, hate speech, targeted harassment and cultural appropriation", and the authors may rescind permission. Using it to test
whether models are robust to a dialect is the intended use, but the paper should describe the condition as synthetic dialect *stress-testing*, not as how Indians write.

## Condition 2 — machine-made Hinglish (`hin_Latn+cm`)

**Method (cite as: simplified alignment-based lexical substitution in the spirit of GCM / Equivalence Constraint theory).**
1. *Matrix text and parallel source.* The human romanized Hindi `hin_Latn` field is the matrix language text; `eng_Latn` is the parallel English.
2. *Word alignment.* SimAlign (Jalili Sabet et al., Findings of EMNLP 2020, pp. 1627-1643, https://aclanthology.org/2020.findings-emnlp.147), "Inter"
   (argmax-in-both-directions) matching on mean-pooled sub-word embeddings from layer 8 of **MuRIL** (`google/muril-base-cased`; Khanuja et al. 2021,
   arXiv:2103.10730), a multilingual encoder pre-trained with transliterated Indic text. mBERT and XLM-R, SimAlign's defaults, gave unusable alignments
   on romanized Hindi (e.g. "ke"<->"the", "baahar"<->"the"; MuRIL: "bum"<->"bomb", "karyalaya"<->"office", "baahar"<->"outside").
   Fields are aligned whole, tokens are regex words and punctuation, English tokens/POS/lemmas/entities come from spaCy.
3. *Alignment-confidence filter.* A pair is used only if its similarity is >= 0.70 and exceeds the best competing similarity in its row/column by >= 0.01
   (thresholds chosen by inspecting candidate pairs on the 300 items, no gold alignment exists; `--sim-min/--margin-min` change them).
4. *Candidates (single-token, same slot, so the Equivalence Constraint is respected trivially).* (a) Hindi token aligned to an English common NOUN or ADJ;
   (b) Hindi nominal immediately followed by a light verb *kar-* aligned to an English VERB, replaced by the verb's lemma ("istemal karein" -> "use karein");
   (c) a closed list of frequent single-word switches whose English partner is a conjunction/discourse adverb/quantifier/number word
   (and, or, but, so, because, if, when, while, although, however, also, very, only, often, almost, now, then, all, some, every, any, both, number words).
   Never replaced: proper nouns/named entities, tokens already English (identical, lemma-identical or spelling variants), Hindi function words, auxiliaries,
   postpositions and pronouns (system morphemes stay Hindi, as in Matrix Language Frame theory).
5. *Mixing control.* For each unit (one sentence of the passage, the question, or one option) a target CMI is drawn from the empirical CMI
   distribution of COMI-LINGUA; candidates are ordered at random (seeded) and substituted until the unit's CMI is closest to the target (zero
   substitutions if the plain hin_Latn unit is already closer).
6. *Safeguards.* Options must remain pairwise distinct (a colliding option would be reverted to hin_Latn; none did); `correct_answer_num` is untouched; digits and
   named entities are never changed (number *words* may switch to English, e.g. teen -> three).

**CMI** (Das & Gambäck 2014, ICON, pp. 378-387, https://aclanthology.org/W14-5152/, eq. 2): `CMI = 100 x (1 - max_i w_i / (n - u))` if `n > u`, else 0, with `n` tokens,
`u` language-independent tokens (here: punctuation, digits, named entities) and `w_i` the tokens of language `i`.
**Reference:** COMI-LINGUA (Sheth, Beniwal, Singh, Findings of EMNLP 2025, https://aclanthology.org/2025.findings-emnlp.422/, CC BY 4.0) `LID_test.csv`, 4,995 sentences
(5 dropped for annotator token-count mismatches) with 3 annotators' token-level `hi`/`en`/`ot` tags; majority vote per token (3-way splits: annotator 1).
Its test sentences are all code-mixed (CMI min about 7.7, mean 27.8, median 26.3). Our units are tagged by provenance, using the same function for the plain `hin_Latn`
text and the output: substituted tokens are `en`; other tokens are `en` if identical to their aligned English token or to a token/lemma of the parallel English field
(length >= 3, not a Hindi/English homograph), `ot` for punctuation/digits/proper nouns, otherwise `hi`.
**Related work to cite alongside:** GCM — Rizvi, Srinivasan, Ganu, Choudhury, Sitaram, *GCM: A Toolkit for Generating Synthetic Code-mixed Text*, EACL 2021 System Demonstrations,
pp. 205-211, https://aclanthology.org/2021.eacl-demos.24 (code: https://github.com/microsoft/CodeMixed-Text-Generator, MIT) — which generates code-mixed sentences from parallel text with word
alignments and a parser under the Equivalence Constraint and Matrix Language theories and then samples; Pratapa, Bhat, Choudhury, Sitaram, Dandapat, Bali, *Language Modeling for Code-Mixing: The
Role of Linguistic Theory based Synthetic Data*, ACL 2018, pp. 1543-1553, https://aclanthology.org/P18-1143 ; EC theory: Poplack 1980. **Our method is a simplified variant, not GCM**: no parse trees and
no full EC enumeration, only single-token substitutions.

**Validation** (`reports/extra/cm_validation.md`): CMI means/quantiles/KS vs COMI-LINGUA, LaBSE similarity (cm vs English, plain hin_Latn vs English, cm vs hin_Latn), fraction changed,
substitution counts, `cm_units.csv` (per-unit target / baseline / output / ceiling CMI), histogram.

## Licences

| Resource | Licence |
|---|---|
| Multi-VALUE / `value-nlp` 0.3 | Apache-2.0 plus a use clause (see above) |
| SimAlign 0.4 | MIT |
| MuRIL (`google/muril-base-cased`), LaBSE (`sentence-transformers/LaBSE`) | Apache-2.0, ungated |
| spaCy 3.7.4 / `en_core_web_sm` 3.7.1 | MIT |
| Stanza 1.8.2 | Apache-2.0 |
| COMI-LINGUA | CC BY 4.0 (reference distribution only) |
| Belebele (source of both conditions) | CC BY-SA 4.0 — **both output files are derivatives and must be released under CC BY-SA 4.0** |
| Code in `extra/` | MIT (repository default) |

## Limitations (state them in the paper)

* **Indian English is synthetic.** Multi-VALUE applies attested IndE features mechanically; outputs include features that read oddly in isolation (e.g. "y'all", article deletion,
  mass-noun plurals) and parser-driven errors. The authors' own gold Indian English (CoQA) is not Belebele. Treat results as robustness to a *rule-based approximation* of the dialect.
  The LaBSE gate reverts a share of option fields (see report), so the condition is a mixture of transformed and untouched fields; the guard on `progressives` is our modification.
* **Some Multi-VALUE rules can alter meaning.** The LaBSE gate cannot see small grammatical distortions: e.g. `verbal_ing_suffix` turns a passive participle into "-ing"
  ("He was arrested at the scene" -> "He have been arresting at scene"), and `null_prepositions` / `fronting_pobj` drop or move prepositional material. The gate only removes gross failures
  (field cosine < 0.85). If the study needs a stricter guarantee, either raise the gate (per-field cosines are in `reports/extra/inde_validation_fields.csv` and in each row's `validation`),
  or drop specific rules (rule spans are in `rules_detail`; `mv_raw_fields` is the unfiltered output), or spot-check a sample of items before running models.
* **Hinglish mixing is lower than real Hinglish.** The method reaches a mean unit CMI well below COMI-LINGUA's (see report: KS distance reduced from about 0.84 for plain hin_Latn to about 0.31,
  not to 0). The binding constraint is the number of safe single-token slots (the "ceiling" CMI with *every* candidate substituted is itself below the COMI mean); loosening the filter barely helps
  (`cm_sensitivity.csv`). Reaching COMI-level mixing would require phrase-level or function-word switching, i.e. the full GCM/EC machinery, or lower precision.
* **Comparison is approximate.** COMI-LINGUA is mostly Devanagari/Latin mixed news and web text (only 16% of its test sentences are Latin-dominant), tagged by humans; ours is romanized Wikipedia-style text
  tagged by provenance. Named entities are `ot` for us but `hi`/`en` for COMI annotators.
* **Alignment errors.** No gold alignments exist for hin_Latn-eng_Latn; precision is controlled by the confidence filter, not measured. Occasional substitutions can shift nuance
  (e.g. "kul sankhya" -> "final sankhya"). LaBSE is weak on romanized Hindi, so its similarity flags are noisy for this condition (see report).
* **English side is translation-ese.** Substituted words come from the English original, not from what Hinglish speakers would choose; morphology (plural forms, agreement) is not adapted.
* Exploratory only: not part of the five main hypothesis tests (H1-H5).

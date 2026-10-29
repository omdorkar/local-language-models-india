# IndicXlit validation: CER vs Belebele native-script text

Hypothesis: IndicXlit (Roman -> native) applied to the human-romanized Belebele item. Reference: Belebele's own native-script text for the same item.
CER / WER in percent (micro-averaged edit distance / reference length) on 300 sampled items. `CER` ignores punctuation/symbols, zero-width characters and Arabic diacritics and unifies Arabic/Urdu letter variants and native digits (২০ = 20); `CER_with_punct` only applies NFC. `all` pools the six text fields; `options` pools mc_answer1..4. `n_texts` counts field texts (passages repeat across questions of the same passage).
IMPORTANT: in Belebele the romanized *passage* is a faithful romanization of the native passage, but the romanized *question and options* are an independent (Hinglish-style) translation, not a transliteration of the native question/options (e.g. hin_Latn "Passage ke hisab se" vs hin_Deva "परिच्छेद के अनुसार"). So CER for `question`/`options` mostly measures lexical divergence between two human translations, not transliteration error; the `passage` CER is the clean transliteration-quality number.
Caveat: the Latin Belebele text is itself a human romanization (spelling is free), and the native reference is one of several valid spellings, so CER is an upper bound on the transliteration error of the pipeline.

| config | reference | field | n_texts | CER | CER_with_punct | WER |
|---|---|---|---|---|---|---|
| hin_Latn+xlit | hin_Deva | all | 1800 | 16.55 | 16.81 | 30.44 |
| hin_Latn+xlit | hin_Deva | passage | 300 | 4.67 | 5.22 | 16.8 |
| hin_Latn+xlit | hin_Deva | question | 300 | 49.28 | 48.78 | 66.51 |
| hin_Latn+xlit | hin_Deva | options | 1200 | 49.35 | 49.45 | 70.4 |
| ben_Latn+xlit | ben_Beng | all | 1800 | 15.63 | 16.02 | 31.23 |
| ben_Latn+xlit | ben_Beng | passage | 300 | 4.89 | 5.66 | 16.57 |
| ben_Latn+xlit | ben_Beng | question | 300 | 45.68 | 45.07 | 72.75 |
| ben_Latn+xlit | ben_Beng | options | 1200 | 44.08 | 44.23 | 70.43 |
| urd_Latn+xlit | urd_Arab | all | 1800 | 18.18 | 19.25 | 43.65 |
| urd_Latn+xlit | urd_Arab | passage | 300 | 8.26 | 9.62 | 31.72 |
| urd_Latn+xlit | urd_Arab | question | 300 | 49.51 | 50.69 | 82.94 |
| urd_Latn+xlit | urd_Arab | options | 1200 | 43.39 | 43.47 | 72.05 |
| npi_Latn+xlit | npi_Deva | all | 1800 | 16.56 | 17.33 | 37.12 |
| npi_Latn+xlit | npi_Deva | passage | 300 | 5.76 | 6.98 | 20.4 |
| npi_Latn+xlit | npi_Deva | question | 300 | 45.43 | 44.84 | 87.33 |
| npi_Latn+xlit | npi_Deva | options | 1200 | 46.91 | 46.96 | 83.08 |

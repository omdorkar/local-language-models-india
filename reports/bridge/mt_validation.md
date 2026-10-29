# NLLB-200-distilled-1.3B validation: chrF / BLEU vs Belebele English

Hypothesis: NLLB translation (native config -> eng_Latn). Reference: Belebele's eng_Latn text for the same item (the FLORES-200 English source).
sacrebleu 2.6.0: chrF = `CHRF()` default (char 6-gram, no word n-grams, beta=2); BLEU = `BLEU()` default (13a tokenizer, exp smoothing). Corpus-level over the 300 sampled items; `passage` uses each unique passage once (165), `options` pools mc_answer1..4, `all` pools the three groups.

| config | reference | field | n_texts | chrF | BLEU |
|---|---|---|---|---|---|
| hin_Deva+mt_en | eng_Latn | all | 1665 | 66.09 | 37.97 |
| hin_Deva+mt_en | eng_Latn | passage | 165 | 67.9 | 39.8 |
| hin_Deva+mt_en | eng_Latn | question | 300 | 64.61 | 38.88 |
| hin_Deva+mt_en | eng_Latn | options | 1200 | 62.44 | 29.49 |
| ben_Beng+mt_en | eng_Latn | all | 1665 | 60.32 | 29.42 |
| ben_Beng+mt_en | eng_Latn | passage | 165 | 62.51 | 31.17 |
| ben_Beng+mt_en | eng_Latn | question | 300 | 58.14 | 30.91 |
| ben_Beng+mt_en | eng_Latn | options | 1200 | 56.21 | 20.82 |
| urd_Arab+mt_en | eng_Latn | all | 1665 | 63.58 | 33.96 |
| urd_Arab+mt_en | eng_Latn | passage | 165 | 64.05 | 33.99 |
| urd_Arab+mt_en | eng_Latn | question | 300 | 67.15 | 42.17 |
| urd_Arab+mt_en | eng_Latn | options | 1200 | 59.73 | 26.56 |
| npi_Deva+mt_en | eng_Latn | all | 1665 | 64.82 | 34.16 |
| npi_Deva+mt_en | eng_Latn | passage | 165 | 68.9 | 41.8 |
| npi_Deva+mt_en | eng_Latn | question | 300 | 65.06 | 38.21 |
| npi_Deva+mt_en | eng_Latn | options | 1200 | 54.02 | 13.57 |
| mar_Deva+mt_en | eng_Latn | all | 1665 | 61.13 | 29.09 |
| mar_Deva+mt_en | eng_Latn | passage | 165 | 63.08 | 30.19 |
| mar_Deva+mt_en | eng_Latn | question | 300 | 61.18 | 36.27 |
| mar_Deva+mt_en | eng_Latn | options | 1200 | 55.95 | 19.27 |
| tel_Telu+mt_en | eng_Latn | all | 1665 | 62.11 | 29.85 |
| tel_Telu+mt_en | eng_Latn | passage | 165 | 64.67 | 31.16 |
| tel_Telu+mt_en | eng_Latn | question | 300 | 59.34 | 31.75 |
| tel_Telu+mt_en | eng_Latn | options | 1200 | 57.36 | 21.62 |
| tam_Taml+mt_en | eng_Latn | all | 1665 | 59.7 | 27.04 |
| tam_Taml+mt_en | eng_Latn | passage | 165 | 61.41 | 27.25 |
| tam_Taml+mt_en | eng_Latn | question | 300 | 61.34 | 34.41 |
| tam_Taml+mt_en | eng_Latn | options | 1200 | 53.86 | 19.25 |
| guj_Gujr+mt_en | eng_Latn | all | 1665 | 64.29 | 31.67 |
| guj_Gujr+mt_en | eng_Latn | passage | 165 | 65.52 | 31.61 |
| guj_Gujr+mt_en | eng_Latn | question | 300 | 62.16 | 37.11 |
| guj_Gujr+mt_en | eng_Latn | options | 1200 | 62.77 | 27.12 |
| kan_Knda+mt_en | eng_Latn | all | 1665 | 64.05 | 34.64 |
| kan_Knda+mt_en | eng_Latn | passage | 165 | 64.08 | 34.63 |
| kan_Knda+mt_en | eng_Latn | question | 300 | 67.68 | 43.75 |
| kan_Knda+mt_en | eng_Latn | options | 1200 | 61.36 | 26.3 |
| mal_Mlym+mt_en | eng_Latn | all | 1665 | 63.14 | 30.67 |
| mal_Mlym+mt_en | eng_Latn | passage | 165 | 63.64 | 30.5 |
| mal_Mlym+mt_en | eng_Latn | question | 300 | 61.41 | 33.88 |
| mal_Mlym+mt_en | eng_Latn | options | 1200 | 63.37 | 28.12 |
| pan_Guru+mt_en | eng_Latn | all | 1665 | 65.64 | 37.43 |
| pan_Guru+mt_en | eng_Latn | passage | 165 | 67.48 | 39.47 |
| pan_Guru+mt_en | eng_Latn | question | 300 | 65.05 | 40.04 |
| pan_Guru+mt_en | eng_Latn | options | 1200 | 61.22 | 27.75 |
| ory_Orya+mt_en | eng_Latn | all | 1665 | 64.59 | 36.46 |
| ory_Orya+mt_en | eng_Latn | passage | 165 | 65.9 | 37.18 |
| ory_Orya+mt_en | eng_Latn | question | 300 | 62.16 | 36.83 |
| ory_Orya+mt_en | eng_Latn | options | 1200 | 63.13 | 32.19 |
| asm_Beng+mt_en | eng_Latn | all | 1665 | 56.51 | 21.4 |
| asm_Beng+mt_en | eng_Latn | passage | 165 | 56.07 | 20.16 |
| asm_Beng+mt_en | eng_Latn | question | 300 | 61.91 | 34.34 |
| asm_Beng+mt_en | eng_Latn | options | 1200 | 53.71 | 15.81 |

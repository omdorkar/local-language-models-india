# EXPLORATORY: Indian English and Hinglish (Belebele)

Not among the five main hypotheses (H1-H5). Gap = accuracy(condition) - accuracy(reference) in percentage points on the same items; 95% bootstrap CI over items (10,000 resamples, seed 1909); exact McNemar p, unadjusted. Both conditions are machine-made approximations, not real users' writing.

| model       | condition     | reference   |   n |   accuracy_pct |   gap_pp |   ci_low |   ci_high |   b |   c | p_unadjusted   |
|:------------|:--------------|:------------|----:|---------------:|---------:|---------:|----------:|----:|----:|:---------------|
| gemma3-12b  | eng_Latn+inde | eng_Latn    | 300 |           87   |     -3.7 |     -6.7 |      -0.7 |  17 |   6 | 0.035          |
| gpt-oss-20b | eng_Latn+inde | eng_Latn    | 300 |           90.7 |     -1.3 |     -4.7 |       1.7 |  14 |  10 | 0.541          |
| llama3.1-8b | eng_Latn+inde | eng_Latn    | 300 |           89   |     -3.3 |     -6.3 |      -0.7 |  15 |   5 | 0.041          |
| qwen3-8b    | eng_Latn+inde | eng_Latn    | 300 |           90.3 |     -2.3 |     -4.7 |       0   |  10 |   3 | 0.092          |
| sarvam-30b  | eng_Latn+inde | eng_Latn    | 300 |           79.3 |     -7.3 |    -11   |      -3.7 |  28 |   6 | <0.001         |
| gemma3-12b  | hin_Latn+cm   | eng_Latn    | 300 |           75.3 |    -15.3 |    -19.7 |     -11   |  49 |   3 | <0.001         |
| gpt-oss-20b | hin_Latn+cm   | eng_Latn    | 300 |           78.3 |    -13.7 |    -18.3 |      -9.3 |  48 |   7 | <0.001         |
| llama3.1-8b | hin_Latn+cm   | eng_Latn    | 300 |           63.3 |    -29   |    -34.7 |     -23.7 |  92 |   5 | <0.001         |
| qwen3-8b    | hin_Latn+cm   | eng_Latn    | 300 |           70.3 |    -22.3 |    -27.7 |     -17   |  75 |   8 | <0.001         |
| sarvam-30b  | hin_Latn+cm   | eng_Latn    | 300 |           65.3 |    -21.3 |    -26.7 |     -16   |  72 |   8 | <0.001         |
| gemma3-12b  | hin_Latn+cm   | hin_Latn    | 300 |           75.3 |      0   |     -3.7 |       3.7 |  16 |  16 | 1.000          |
| gpt-oss-20b | hin_Latn+cm   | hin_Latn    | 300 |           78.3 |      1   |     -3   |       5   |  17 |  20 | 0.743          |
| llama3.1-8b | hin_Latn+cm   | hin_Latn    | 300 |           63.3 |      5   |      0.7 |       9.3 |  17 |  32 | 0.044          |
| qwen3-8b    | hin_Latn+cm   | hin_Latn    | 300 |           70.3 |      3.3 |     -1   |       8   |  18 |  28 | 0.184          |
| sarvam-30b  | hin_Latn+cm   | hin_Latn    | 300 |           65.3 |     -3   |     -7.3 |       1.3 |  25 |  16 | 0.211          |

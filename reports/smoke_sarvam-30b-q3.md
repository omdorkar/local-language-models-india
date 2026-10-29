# Smoke test: sarvam-30b-q3

10 Belebele items (seed 0), same items in every config. Template `d3ee3d7bd6e7`.
Too few items for any conclusion; this only shows the model loads and answers in the right format.

| config   |   items |   correct |   invalid |   mean_seconds |   mean_prompt_tokens |
|:---------|--------:|----------:|----------:|---------------:|---------------------:|
| eng_Latn |      10 |         8 |         0 |           0.64 |                213.6 |
| hin_Deva |      10 |         8 |         0 |           0.69 |                254.3 |
| hin_Latn |      10 |         6 |         0 |           0.74 |                295.2 |

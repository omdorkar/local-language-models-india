# Smoke test: gpt-oss:20b

10 Belebele items (seed 0), same items in every config. Template `d3ee3d7bd6e7`.
Settings: max_tokens=512, reasoning_effort=low, server http://127.0.0.1:11434/v1.
Too few items for any conclusion; this only shows the model loads and answers in the right format.

| config   |   items |   correct |   invalid |   mean_seconds |   mean_prompt_tokens |   mean_output_tokens |
|:---------|--------:|----------:|----------:|---------------:|---------------------:|---------------------:|
| eng_Latn |      10 |        10 |         0 |           3.53 |                302.7 |                    6 |
| hin_Deva |      10 |         7 |         0 |           4.53 |                609.1 |                    6 |
| hin_Latn |      10 |         8 |         0 |           3.79 |                436.3 |                    6 |

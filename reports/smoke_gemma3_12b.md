# Smoke test: gemma3:12b

10 Belebele items (seed 0), same items in every config. Template `d3ee3d7bd6e7`.
Settings: max_tokens=32, reasoning_effort=not sent, server http://127.0.0.1:11434/v1.
Too few items for any conclusion; this only shows the model loads and answers in the right format.

| config   |   items |   correct |   invalid |   mean_seconds |   mean_prompt_tokens |   mean_output_tokens |
|:---------|--------:|----------:|----------:|---------------:|---------------------:|---------------------:|
| eng_Latn |      10 |         9 |         0 |           2.31 |                212.6 |                    7 |
| hin_Deva |      10 |         9 |         0 |           2.36 |                253.3 |                    7 |
| hin_Latn |      10 |         9 |         0 |           2.65 |                294.2 |                    7 |

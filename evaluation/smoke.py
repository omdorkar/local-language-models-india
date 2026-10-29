"""Quick check that a model loads and answers Belebele items through an OpenAI-compatible server
(llama.cpp `llama-server` or Ollama's /v1). Not part of the analysis: too few items.

    python -m evaluation.smoke --model sarvam-30b-q3 --configs eng_Latn hin_Deva hin_Latn --n 10
    python -m evaluation.smoke --base-url http://127.0.0.1:11434/v1 --model gemma3:12b
"""

import argparse

import httpx
import pandas as pd

from . import REPORTS, ROOT
from .client import ask
from .inspect_belebele import load
from .prompts import TEMPLATE_SHA, belebele_gold, belebele_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", default=["eng_Latn", "hin_Deva", "hin_Latn"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--reasoning-effort", help="sent as reasoning_effort, e.g. none or low (Ollama)")
    args = ap.parse_args()

    # Same items in every config: sample keys from English, then look them up elsewhere.
    keys = load("eng_Latn").sample(args.n, random_state=args.seed).index
    rows = []
    with httpx.Client(timeout=600) as client:
        for config in args.configs:
            data = load(config)
            for key in keys:
                item = data.loc[key]
                out = ask(client, args.base_url, args.model, belebele_prompt(item), args.seed,
                          args.max_tokens, args.reasoning_effort)
                rows.append({"config": config, "link": key[0], "question_number": int(key[1]),
                             "gold": belebele_gold(item), **out})
                print(f"{config:9s} q{key[1]} gold={rows[-1]['gold']} got={out['answer']} "
                      f"{out['seconds']:5.1f}s prompt_tokens={out['prompt_tokens']}")

    df = pd.DataFrame(rows)
    df["correct"] = df.answer == df.gold
    summary = df.groupby("config", sort=False).agg(
        items=("correct", "size"), correct=("correct", "sum"), invalid=("answer", lambda s: s.isna().sum()),
        mean_seconds=("seconds", "mean"), mean_prompt_tokens=("prompt_tokens", "mean"),
        mean_output_tokens=("completion_tokens", "mean"),
    ).round(2).reset_index()

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    df.assign(model=args.model, template_sha=TEMPLATE_SHA).to_json(runs / f"smoke_{safe}.jsonl", orient="records",
                                                                   lines=True, force_ascii=False)
    report = [f"# Smoke test: {args.model}", "",
              f"{args.n} Belebele items (seed {args.seed}), same items in every config. Template `{TEMPLATE_SHA}`.",
              f"Settings: max_tokens={args.max_tokens}, reasoning_effort={args.reasoning_effort or 'not sent'}, "
              f"server {args.base_url}.",
              "Too few items for any conclusion; this only shows the model loads and answers in the right format.",
              "", summary.to_markdown(index=False), ""]
    (REPORTS / f"smoke_{safe}.md").write_text("\n".join(report))
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Language resource statistics for the 13 Indian languages (+ English).

    python -m evaluation.langstats            # fetch from the web and write data/external/language_stats.csv

Everything is fetched at run time; nothing is typed in from memory. Columns:
  census2011_first_language_speakers  Census of India 2011, Statement 1 (scheduled-language
                                      groupings of mother tongues), as reproduced in the
                                      Wikipedia article "List of languages by number of native
                                      speakers in India" (first table, "First language speakers").
  wikipedia_articles                  `articles` from the Wikimedia siteinfo API of <code>.wikipedia.org.
A value that cannot be fetched is left blank, never guessed.
Use `load_language_stats()` to read the CSV (returns an empty frame if it does not exist).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

from .config import LANGUAGES

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "external" / "language_stats.csv"
OUT_README = ROOT / "data" / "external" / "README.md"

CENSUS_URL = "https://en.wikipedia.org/wiki/List_of_languages_by_number_of_native_speakers_in_India"
WIKI_API = "https://{code}.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"
UA = {"User-Agent": "local-language-models-india/0.1 (academic benchmark audit by Om Dorkar)"}

# Name used for each language in the census table.
CENSUS_NAME = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ur": "Urdu", "ne": "Nepali", "mr": "Marathi",
               "te": "Telugu", "ta": "Tamil", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
               "pa": "Punjabi", "or": "Odia", "as": "Assamese"}


def _clean_name(s: str) -> str:
    return re.sub(r"\[.*?\]", "", str(s)).strip()


def fetch_census(client: httpx.Client) -> dict[str, int]:
    r = client.get(CENSUS_URL)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        if any("First language" in c for c in cols) and len(t) < 40:
            name_col = 0
            fig_col = next(i for i, c in enumerate(cols) if "First language speakers" in c and "Figure" in c)
            out = {}
            for _, row in t.iterrows():
                nm = _clean_name(row.iloc[name_col])
                val = re.sub(r"\[.*?\]|[,\s]", "", str(row.iloc[fig_col]))
                if val.isdigit():
                    out[nm] = int(val)
            return out
    raise RuntimeError("census table not found")


def fetch_wiki_articles(client: httpx.Client, code: str) -> int | None:
    try:
        r = client.get(WIKI_API.format(code=code))
        r.raise_for_status()
        return int(r.json()["query"]["statistics"]["articles"])
    except Exception as e:  # noqa: BLE001 - leave blank rather than guess
        print(f"  could not fetch Wikipedia articles for {code}: {e}")
        return None


def build() -> pd.DataFrame:
    today = dt.date.today().isoformat()
    with httpx.Client(headers=UA, follow_redirects=True, timeout=30) as client:
        try:
            census = fetch_census(client)
        except Exception as e:  # noqa: BLE001
            print(f"Census fetch failed: {e}")
            census = {}
        rows = []
        for code, lang in LANGUAGES.items():
            rows.append({
                "iso": code, "language": lang.name,
                "census2011_first_language_speakers": census.get(CENSUS_NAME[code]),
                "wikipedia_articles": fetch_wiki_articles(client, code),
            })
    df = pd.DataFrame(rows)
    df["census_source_url"] = CENSUS_URL
    df["wikipedia_source_url"] = df["iso"].map(lambda c: WIKI_API.format(code=c))
    df["retrieved"] = today
    for col in ["census2011_first_language_speakers", "wikipedia_articles"]:
        df[col] = df[col].astype("Int64")
    return df


README = """# External language statistics

`language_stats.csv` is produced by `python -m evaluation.langstats` (fetched live; retrieval date in the
`retrieved` column, source URL per column in `census_source_url` / `wikipedia_source_url`).

| Column | Source | Notes |
|---|---|---|
| `census2011_first_language_speakers` | Census of India 2011, Statement 1 (mother tongues grouped under the 22 scheduled languages), as reproduced in the first table ("First language speakers") of {census_url} | Hindi's figure (528 M) includes mother tongues the Census groups under Hindi (Bhojpuri, Rajasthani, ...). Nepali and Urdu are Indian-census counts, not worldwide speakers. English is the Census count of English as a *mother tongue* (very small; not a measure of English use in India). Blank = not found. |
| `wikipedia_articles` | `articles` field of the Wikimedia siteinfo API of `<code>.wikipedia.org` ({api}) | Snapshot on the retrieval date; Wikipedia article count is a proxy for digital resource level, not quality. Blank = fetch failed. |

Retrieved: {today}.
"""


def write_readme(today: str) -> None:
    OUT_README.write_text(README.format(census_url=CENSUS_URL, api=WIKI_API.format(code="<code>"), today=today))


def load_language_stats(path: Path | None = None) -> pd.DataFrame:
    path = path or OUT_CSV
    if not Path(path).exists():
        return pd.DataFrame(columns=["iso", "language", "census2011_first_language_speakers", "wikipedia_articles"])
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    df = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    write_readme(df["retrieved"].iloc[0])
    print(df[["iso", "language", "census2011_first_language_speakers", "wikipedia_articles"]].to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

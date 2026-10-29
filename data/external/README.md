# External language statistics

`language_stats.csv` is produced by `python -m evaluation.langstats` (fetched live; retrieval date in the
`retrieved` column, source URL per column in `census_source_url` / `wikipedia_source_url`).

| Column | Source | Notes |
|---|---|---|
| `census2011_first_language_speakers` | Census of India 2011, Statement 1 (mother tongues grouped under the 22 scheduled languages), as reproduced in the first table ("First language speakers") of https://en.wikipedia.org/wiki/List_of_languages_by_number_of_native_speakers_in_India | Hindi's figure (528 M) includes mother tongues the Census groups under Hindi (Bhojpuri, Rajasthani, ...). Nepali and Urdu are Indian-census counts, not worldwide speakers. English is the Census count of English as a *mother tongue* (very small; not a measure of English use in India). Blank = not found. |
| `wikipedia_articles` | `articles` field of the Wikimedia siteinfo API of `<code>.wikipedia.org` (https://<code>.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json) | Snapshot on the retrieval date; Wikipedia article count is a proxy for digital resource level, not quality. Blank = fetch failed. |

Retrieved: 2026-10-21.

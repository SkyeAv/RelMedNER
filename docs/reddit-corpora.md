# Reddit corpora

Seven general-Reddit dumps from the `tensorshield` hub org, all MIT-licensed. The
inclusion bar is corpus-wide `total_rows >= 100,000` in the org's stats.json: the kept
seven span 121,584 to 7,114,560 rows. `tensorshield/reddit_dataset_226` (22,572 rows)
carries the same license but fails the bar: after the health-community filter its
projected yield is noise, so it stays out of the pipeline.

None of these corpora are biomedical on their own, so every entry filters rows
client-side post-stream with `match_on`: exact-value membership on `communityName`
against a single yaml anchor `&biomed_communities` holding 42 provisional health
subreddits (`r/AskDocs`, `r/diabetes`, `r/CrohnsDisease`, ...; the full list lives in
`src/relmedner/data/ingests.yaml` and all seven entries alias the one anchor, so the
allowlists cannot drift apart). Casing must be the Reddit canonical form: `r/askdocs`
does not match, and neither does a non-health community like `r/madmen`. `columns_out`
is `text` alone.

The table's rows-in are stats.json `total_rows` for the whole corpus, before the
community filter. Mining runs the unchanged fullmap path documented under
[How fullmap mining works](fullmap-mining.md); informal reddit prose (slang,
misspellings, first-person narratives) is expected to mine at lower unigram precision
than the curated corpus, and the measured gates in `relmedner/constants.py` are
deliberately not loosened for it.

## Related

- [README index](../README.md) -- the ingest-table rows for the seven entries and the full docs index.
- [How fullmap mining works](fullmap-mining.md) -- the mining path every reddit entry runs unchanged.
- [Row-quality heuristics](quality-heuristics.md) -- the row filters the reddit entries ship with.

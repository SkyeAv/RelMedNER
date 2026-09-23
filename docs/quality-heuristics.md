# Row-quality heuristics (C4/Gopher-style QC filters)

The six opt-in `RowFilters` knobs beyond the length/regex basics: web-corpus QC heuristics
adapted from the document-level filters that large pretraining pipelines (C4, Gopher, and
their open reimplementation in HuggingFace datatrove) run over crawled corpora. They guard
training rows against the same failure family: degenerate language, markup noise, shouting
caps, repetition spam, and broken line structure.

## The knobs and their lineage

| knob | drop reason | guards against | lineage |
| --- | --- | --- | --- |
| `min_words` | `min_words` | rows below a word-count floor (titles, fragments, table rows) | C4/Gopher word-count floors |
| `min_stop_word_ratio` | `min_stop_word_ratio` | non-English or language-degenerate text (gene-symbol lists, tables, code) | C4's at-least-N-stopwords rule, Gopher's stop-word check |
| `max_symbol_ratio` | `max_symbol_ratio` | markup, code, punctuation noise | C4/Gopher symbol-to-letter ratios |
| `max_upper_ratio` | `max_upper_ratio` | shouting caps | datatrove's `uppercase_word_ratio` |
| `max_repeat_ngram_ratio` | `max_repeat_ngram_ratio` | repeated words/phrases (spam, template junk) | Gopher's duplicate 10-gram repetition filter |
| `max_short_line_ratio` | `max_short_line_ratio` | abnormal line breaks: newline spam, OCR fragments, bullet walls | C4's ellipsis/bullet/line-length rules, datatrove's `short_line_ratio_chars` |

Two deliberate deviations from the web-scale originals:

- **Language identification is a stop-word ratio, not a model.** FastText language ID was
  rejected: it needs a ~130MB pretrained model download, a native-lib dependency, and
  per-row inference cost, for a corpus that is English-biomedical by design. The
  `min_stop_word_ratio` check over the existing closed-class `FUNCTION_WORDS` stoplist
  (constants.py, the pipeline's ONLY stoplist) catches the same mislabeled-language and
  degenerate-text rows at zero cost.
- **No boilerplate/DOM extraction.** Web pipelines strip nav/ads/boilerplate before
  filtering; this corpus is curated (HF datasets, avro containers, TSV), so there is no
  HTML to strip. Revisit only if a raw-crawl source lands.

## Evaluation order and cost

The knobs evaluate at the single row-filter chokepoint (`row_filters.first_drop_reason`)
in the fixed order documented in [yaml-config](yaml-config.md): after the always-on token
cap, before the regexes; the first failing rule owns the attribution in the quality line.

Cost contract: an unset knob costs nothing (the evaluator does not even tokenize the row;
this is spy-tested), and with any knob set the row is tokenized ONCE and every enabled rule
reads the shared words. All rules are single-pass O(row length) with no regexes and no
`hash()`, so enabling them cannot dominate a Beam worker the way dedup's MinHash pass does.

## Thresholds: measure first, never guess

Every dropped row is supervised signal. The repo's convention (see constants.py: fullmap
gates "fixed by measurement") applies doubly here: a threshold is a data-composition
decision, and the only trusted value is one you measured against the actual source.

Workflow for enabling thresholds on a source:

1. Probe the source as declared (no filters) and note `rows_in`/`rows_out`:
   `bash .pi/skills/add-dataset/scripts/remote-gate.sh probe -- --declared <source-id> --limit 2000`
2. Add candidate thresholds to the declaration and re-probe. The quality line attributes
   every drop per reason: `dropped={min_words:812, max_symbol_ratio:104, ...}`.
3. Sanity-check the survivors: sample a few kept rows and a few dropped reasons against
   judgment. If a rule's drop count surprises you, fix the threshold, not the data.
4. Record the before/after table in the ingest's docs page or PR, then land the values.

Reading the numbers: a rule dropping < 2% of a source is usually noise-trimming; a rule
dropping > 20% deserves a sampled eyeball audit before it lands; a rule dropping 100%
raises `ZeroYieldError` and fails the run loudly instead of shipping an empty training set.

## Worked example

```yaml
- task: {type: fullmap, outputs: [entities, relations]}
  source: hf
  dataset: some/noisy-reddit-corpus
  split: train
  columns_out: [text]
  filters:
    drop_empty: true
    min_words: 20
    min_stop_word_ratio: 0.05
    max_symbol_ratio: 0.3
    max_upper_ratio: 0.5
    max_repeat_ngram_ratio: 0.3
    max_short_line_ratio: 0.5
```

Every value above is a placeholder shape, not a recommendation. Field semantics, bounds,
and the full 12-step drop order: [yaml-config](yaml-config.md). The quality line they
produce: [Reading ingest quality numbers](yaml-config.md#reading-ingest-quality-numbers).

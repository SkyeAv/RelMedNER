# Probe recipes: what to measure, and what each measurement decides

`scripts/probe.py` runs on wenceslaus through `remote-gate.sh probe -- <args>`. Every number that
reaches the README, every `LABEL_MAP` entry, and every drop rule comes from one of these recipes. An
unmeasured number is a guess wearing a citation.

## Sizing a probe

| goal | `--limit` | cost |
| --- | --- | --- |
| shape discovery (columns, dtypes, span encoding) | 20-200 | seconds |
| drop-rule rates | 1,000-5,000 | a minute |
| label census for a `LABEL_MAP` | the FULL split | minutes to an hour in tmux |
| mined-yield expectation | 200 docs | a minute (one batched redb round trip) |
| row counts | `--info`, no rows | seconds, no download |

A label map written from a sample is the single most expensive mistake in this repo's history: a
901-row sample of `knowledgator/biomed_NER` missed a 23-span label and understated two plural
variants by two orders of magnitude, and the correction was its own commit. Census the full split,
then check `PROBE_LABEL_HAPAX` (labels seen once) before deciding the tail is noise.

`--limit` streams and stops, so it is cheap for the first N rows. It is NOT cheap when a `match_on`
filter keeps a rare value: the reddit corpora carry 0.3-0.5% health-community rows, so the first 200
matches cost ~50k streamed rows. Declare the entry first, probe it with `--declared` (which applies
`match_on` and reports the real `rows_in`/`rows_out`), and run it in tmux.

## Recipe A: unlabeled text, candidate for a `fullmap` task

```bash
bash .pi/skills/add-dataset/scripts/remote-gate.sh --no-sync probe -- \
  --dataset <org>/<name> --info
bash .pi/skills/add-dataset/scripts/remote-gate.sh probe -- \
  --dataset <org>/<name> --subset <config> --split <split> --limit 200 \
  --text-column <column> --fullmap
```

Read:

- `PROBE_ROWS_TOTAL` / `PROBE_ROWS` -> the README `rows in` cell (name the endpoint as the source)
- `PROBE_TEXT_TOKENS_*` -> documents long enough for n-gram mining? a 20-token median makes
  `max_ngram=6` pointless
- `PROBE_FULLMAP_MENTIONS_PER_DOC`, `PROBE_FULLMAP_DOCS_WITH_MENTIONS`, `PROBE_FULLMAP_RELATIONS`
  -> the yield expectation and whether declaring `relations` is honest

Decides: mineable at all; `columns_out` is one column (the miner reads only the first projected
column); general-domain text whose surfaces will never match a biomedical fullmap (then the honest
add is a `script` with a trust-gold stance, or a deferral -- see `SuperGLUE ReCoRD`).

## Recipe B: gold spans, token-aligned (`[start, end_inclusive, label]`)

```bash
bash .pi/skills/add-dataset/scripts/remote-gate.sh probe -- \
  --dataset <org>/<name> --split train --limit 1000 \
  --text-column tokenized_text --span-column ner --label-column ner \
  --sample 3
```

Read: `PROBE_COLUMN` (list-of-int or stringified?), `PROBE_SPAN_ARITY` (healthy = one bar),
`PROBE_SPAN_ELEMENT_TYPES` (bools and floats are poison; stringified indices are the streaming
corruption tell), `PROBE_SPAN_OUT_OF_BOUNDS` (rate), `PROBE_LABEL_DISTINCT` + `PROBE_LABEL_TOP_*`
(census size), sample rows (eyeball the `[start, end, label]` convention and whether bounds are
inclusive against `len(tokens)`).

Decides: skip-don't-coerce rules (`coerced_span`-style), whether the loader needs the `hf_json`
route (`PROBE_SPAN_STRINGIFIED_INDICES > 0` on a cold cache is the PubMedAbstractsNER tell), the
`LABEL_MAP` breadth, and the span bounds convention the test fixtures must use.

## Recipe C: raw text + char offsets

Run Recipe B but with `--text-column text --span-column entities`, then read the sample rows: for one
span, compare `text[start:end]` with `text[start:end+1]` against the surface to settle exclusive vs
inclusive. The knowledgator corpus measured end-EXCLUSIVE on 18,685 sampled spans. Also:
`PROBE_TEXT_CHARS_*` (how many rows would exceed context length), empty-`ner` rate (those rows ship
text-only and fall out on the declared-outputs filter).

Decides: the `char_spans_to_token_spans` bridging, emitted `text` = re-joined token stream (raw-text
containment fails for ~26.7% of char slices once punctuation detaches), `LABEL_MAP`, and the
drop-rules for the docstring.

## Recipe D: gold relations, inline tags

`--text-column sentences --span-column sentences` does not fit; instead read the sample rows and run
a throwaway counting pass on wenceslaus in tmux (tag counts per row, self-loops, nested angle-bracket
markup). The landed sentence_rex numbers: 550 null, 521 wrong tag counts, 18 nested markup,
48 self-loops of 44,115 rows. Those four drop rules each got a negative test.

Decides: the parser's guard set, the `outputs: [relations]`-only declaration, and the README's
measured ship rate (42,978 / 44,115).

## Recipe E: classification corpora

```bash
... --text-column <text cols> --label-column <label> --limit 5000
```

Read: `PROBE_LABEL_DISTINCT` (must match the declared vocabulary exactly), `PROBE_LABEL` values
(ints? the `False`/`True` strings? a `-1` sentinel for unlabeled rows?), and the
bool-masquerading-as-int trap (probe reports `bool` in `PROBE_SPAN_ELEMENT_TYPES` when labels are
bools).

Decides: whether `run()` accepts both decode forms (ClassLabel int AND string), the exact
out-of-vocabulary behavior, and whether unlabeled rows ship text-only.

## Recipe F: chat-shaped QA (surfaces, not offsets)

Read: `PROBE_COLUMN` shapes for the conversations column, sample rows (the `Text: ` prefix and the
templated question), and run a throwaway tmux pass for the hallucinated-surface rate (gpt answers
with surfaces the document never contains; the landed rule recovers spans only where the surface is a
token subsequence, which is simultaneously the hallucination guard and the tokenization guard).

Decides: whether `ScriptUtils.parse_conversations` fits as-is or a variant is needed, and the
drop rule for unlocatable mentions.

## Recipe G: declared-ingest probe (post-declaration)

After the yaml entry and the class exist:

```bash
bash .pi/skills/add-dataset/scripts/remote-gate.sh probe -- \
  --declared '<org>/<name>[:<subset|file>]' --limit 200 \
  --script <ScriptName> --outputs <outputs> --text-column <column>
```

`PROBE_QUALITY` is the stream's own counters (`rows_in`, `rows_out`, `dropped={reason:n}`) -- the only
per-ingest rows_in/rows_out evidence a `--direct` Prism run does not print (`QUALITY_LINES:0` in the
smoke log is normal). `PROBE_DISPATCH_ROWS_EMITTING` / `PROBE_DISPATCH_EMIT_PCT` /
`PROBE_DISPATCH_SHAPES` are the nonzero-yield evidence the tests and the README claim.

Decides: the numbers for the README's format-notes paragraph, and whether the shipped fraction is
honest (a 2% emit rate is a bug or a wrong `columns_out`, not a property).

## The mapping: measurement -> README sentence -> code decision

| measurement | README sentence it becomes | code decision it drives |
| --- | --- | --- |
| `PROBE_ROWS(_TOTAL)` | "`rows in` cell, (source: datasets-server `size` endpoint)" | whether the split/subset declaration is right |
| `PROBE_TEXT_*` | "rows are long (median 233 / max 1,558 tokens, none truncated)" | `filters: {min_text_len, max_text_len}` needs |
| `PROBE_LABEL_DISTINCT` + top coverage | "the label tail is long: N distinct labels with the top-60 covering only X%" | `LABEL_MAP` breadth; direct-label vs fullmap strategy |
| `PROBE_LABEL_HAPAX` | "the residual tail stays unmapped on purpose" | where the map stops |
| `PROBE_SPAN_ARITY` / `ELEMENT_TYPES` | "the streaming loader delivers spans as stringified indices with quote-wrapped labels" | the coercion/drop function's shape list |
| `PROBE_SPAN_OUT_OF_BOUNDS` | "0.04-0.24% out-of-bounds dropped, ~0.4% whitespace slop normalized" | the bridge's drop-then-snap rules and tests |
| `PROBE_DISPATCH_EMIT_PCT` | "measured over the first 20,000 records: 83% ship with at least one entity" | whether `outputs` is declared honestly |
| `PROBE_FULLMAP_MENTIONS_PER_DOC` | "measured expectations: ~20 mentions/doc" | whether a `fullmap` task is the right lane at all |
| `PROBE_QUALITY` | "ingest quality ... rows_in=... rows_out=... dropped={...}" | whether `match_on`/`filters` need tuning |

## Ground rules

- Probes run on wenceslaus. Never stream a corpus on the laptop, not even once.
- `--info`/`--freeze` are the only no-download probes; everything else streams and belongs in tmux
  when `--limit` is over a few hundred.
- Save probe evidence before it is overwritten: each run truncates its log, and two probes in a row
  share the `probe` log name. Read the log through `tr ':' '~'`.
- A gated hub repo needs a token on the box first (none is configured); say so instead of reporting a
  zero-row census as a dataset property.
- Record the probe invocation in the README paragraph or the commit body next to the numbers, so the
  next agent can re-run the exact measurement instead of trusting it.

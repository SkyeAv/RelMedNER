# relmedner

Apache Beam pipeline that builds gliner2 training data from biomedical text corpora.
Two ingest types share one declarative pipeline:

- **script tasks** — datasets that already carry gold spans
  (`anthonyyazdaniml/gliner-biomed-pre-training`, the IOB-formatted
  `disi-unibo-nlp/Pile-NER-biomed-IOB`, and the multi-task
  `anthonyyazdaniml/gliner-biomed-post-training`); spans are relabeled to biolink classes
  via tablassert `Categories` and local fullmap resolution, and relations are
  distant-supervised through a biolink-predicate gazetteer matched between mention surfaces.
  The post-training corpus additionally carries native gold relations and sampled
  negatives, and splits into per-row task families (below).
- **fullmap tasks** — unlabeled text (e.g.
  `anthonyyazdaniml/gliner-biomed-curated-corpus` and the downsampled, class-balanced
  `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus`, a strict 158,890-row subset of
  the curated corpus with an identical `text`-only schema); entities are *mined* by enumerating
  n-grams, resolving them in one batched round trip against the local fullmap redb, and
  keeping only spans that exactly match a normalized preferred name. Mined spans also feed
  the gazetteer for `evidence="distant"` relations. The gazetteer additionally emits
  statement-qualifier relations (below).
- **local sources** — header-delimited TSV/CSV committed as package data
  (`src/relmedner/data/qualifiers/qualifier_corpus.tsv`, read by `LocalDataStream`),
  declared like any other task with `source: local` instead of a hub dataset; paths resolve
  against the CWD first, then package data.

## Ingests

| dataset | task | inputs | outputs | rows in |
| --- | --- | --- | --- | --- |
| `anthonyyazdaniml/gliner-biomed-pre-training` | `script` → `GlinerBiomedScript` | `tokenized_text`, `ner` | entities, relations | 98,659 |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | `script` → `PileNerBiomedScript` | `tokens`, `ner_tags` | entities | 58,861 |
| `anthonyyazdaniml/gliner-biomed-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 418,381 |
| `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 158,890 |
| `qualifier corpus` (`local` package data) | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 24 |
| `anthonyyazdaniml/gliner-biomed-post-training` | `script` → `GlinerBiomedPostScript` | `tokenized_text`, `ner`, `negatives` | entities, classifications, structures, relations | — |

All script tasks share one resolution chain — fullmap first, a shared lowercased
`FALLBACK_LABEL_MAP` second (dataset vocabularies ride on top via
`resolve_mentions(label_map=...)`), raw labels last — and two shared quality gates:

- `ResolutionGate` rejects fullmap hits contradicting the source corpus's own label
  (label↔category buckets, model-organism CURIE guard, acronym-over-catch-all guard);
  rejections fall through to fallback/raw, never dropping the mention. ~20% of fullmap
  hits rejected on the pile-ner corpus, nearly all true false positives.
- `PredicateRangeGate` rejects gazetteer relations whose head/tail biolink categories
  contradict the predicate's domain/range (raw labels impose no constraint). ~21% of
  candidate relations rejected, all sampled rejects genuinely wrong. Qualifier contexts are
  stricter: typed slots demand resolvable biolink ancestors and reject
  `JUNKY_CATEGORIES` (UMLS qualifier/indexing concepts).

## Statement qualifiers and negation

Gazetteer relations also carry the six DAKP-declared nullable qualifier slots
(`disease_context_qualifier`, `anatomical_context_qualifier`, `sex_qualifier`,
`population_context_qualifier`, `frequency_qualifier`, `temporal_context_qualifier`;
`species_context_qualifier` is deliberately absent per tablassert
`DISABLED_EDGE_FIELDS`). Attachment mirrors DAKP's `attach_qualifiers_with_scores`:
qualifiers fire only where a predicate relation fired (they qualify statements, not
entity pairs), the host is the nearest statement endpoint with the tail preferred, a
typed context never restates an endpoint, one value per slot per sentence, and
value-style slots (`frequency`/`temporal`) fall back to the multi-token phrase surface
("twice daily") when no mention follows the trigger. Word-bounded negation cues ("not",
"failed to", "without", ...) sharing a sentence with a fired predicate re-encode the
statement as `not_<predicate>` with `negated=true`.

Dataset-format notes (`PileNerBiomedScript`): `tokens`/`ner_tags` are python-repr strings
(`ast.literal_eval`, malformed rows skip); orphan `I-` tags promote to single-token spans
rather than dropping; raw labels PascalCase so the full 3,896-type tail stays
biolink-shaped. Measured full corpus: 100% of rows emit, ~188k entity mentions, and 6,058
gazetteer relations across 5,501 rows (9.3% relation-bearing, across 23 biolink predicates,
each carrying its biolink slot description as `relation_descriptions`).

## Output

Output is Avro records of `TrainingExample` (`text`, `entities`, `relations`, ...).
Relation provenance rides in the Avro records: `negated` (`true` only for sampled
negatives) and `evidence` (`asserted` for gold-span scripts, `distant` for fullmap-mined
spans, `sampled_negative` for grid-sampled non-observations from the post-training
corpus). The gliner2 JSONL projection (`to_output()`) emits mention fields only, because
gliner2 validates every relation value as a mention in the text — sampled negatives
therefore train under `not_<predicate>` relation names. Predicate slot definitions DO
ride the projection as `relation_descriptions` (the processor consumes them as label
prompts), mirroring `entity_descriptions`.

The declared-outputs filter is a **permitted-shapes contract**: a row ships when it
produced at least one declared shape. Rows may produce fewer shapes than declared
(entity-only mined rows flow) and may produce extra shapes (the `[entities]`-only
pile-ner declaration keeps rows whose gazetteer also fired — relations are free signal,
not a contract violation).

## How post-training row families work

`src/relmedner/families.py` splits the multi-task corpus into disjoint families on the NER
label set alone (`RowFamily` ABC, self-registering, consulted in priority order):

| family | share | output shape |
| --- | --- | --- |
| native relations (`head <> predicate <> tail` span labels) | 2.4% | `relations` |
| classification option lists (`label`/`category`/`class`/`tag`) | 9.0% | `classifications` |
| `match` span extraction | 16.0% | `structures` |
| open-vocabulary NER (anything else) | 68.1% | `entities` + gazetteer `relations` |
| empty `ner` | 4.4% | dropped |

Zero-shot breadth rules on this corpus:

- predicates map onto `tablassert.biolink.Predicates` when one matches (`associated with`
  → `associated_with`); everything else keeps a biolink-shaped `snake_case` native form;
- sampled negatives from the dataset's `negatives` column train under `not_<predicate>`
  names after guards: malformed, self-loop, duplicate, positive-colliding, and
  not-in-text triples drop, capped at 2x the row's positive count;
- relation surfaces that tokenization tore away from the text (`CC-chemokines` vs tokens
  `CC`, `-`, `chemokines`) are filtered at extraction — gliner2 would drop them anyway;
- `GlinerBiomedPostScript.LABEL_MAP` extends the shared `FALLBACK_LABEL_MAP` with 25
  validated biolink classes; values are validated loudly at import.

## How fullmap mining works

Per batch of documents (Beam `BatchElements`, one redb round trip per batch):

1. tokenize with gliner2's own `WhitespaceTokenSplitter` (imported by file path so torch
   never loads), strip per-token edge punctuation — which also makes sentence-crossing
   n-grams impossible because `.` tokenizes alone and cleans to empty;
2. enumerate contiguous n-grams up to `max_ngram`, dropping all-numeric and
   all-function-word grams; add hyphen/slash folds, Greek/unicode folds, and one-way
   in-document acronym bridges (`body mass index (BMI)` resolves the expansion, keeps the
   `BMI` span);
3. one `rs.normalize_terms` + `lookup_rows` + `filter_and_rank` per batch — fullmap keys
   are **byte-sorted bags of Porter2 stems**, so word order is already irrelevant and
   permutation-style augmentation is a proven no-op;
4. accept a row only when `normalize(PREFERRED_NAME) == term` (EXACT). The published PR
   tiers are *not* a quality dial: PR=50 just means a preferred name already in
   sorted-stem order; PR=250/500 are stopword collisions;
5. gates (constants in `relmedner/constants.py`, each fixed by measurement — see
   `PLAN.md`): exclude model-organism CURIE prefixes (`FB`, `ZFIN`, `MGI`, ... — human
   genes only), junk-category gate (UMLS qualifier/indexing concepts), unigram minimum
   length (digit-bearing exempt), gene/protein casing rule (rejects `in`→`NCBIGene:3630
   INS` collisions), strict unigram name agreement (case-insensitive equality or simple
   plural — kills Porter2 derivational collisions like `oxidative`→`oxide`), and
   rejection of spans that begin or end with a function word;
6. greedy longest-match non-overlap selection, then group by biolink category with
   class-definition + `[fullmap: CURIE | name]` descriptions.

Measured expectations on the curated corpus: **~20 mentions/doc** (~8.5M projected over
418,381 docs), unigram precision ≈ 78%, multi-token precision ≈ 90%. Yield by gram
length per 200 docs: n=1 2,962 · n=2 918 · n=3 137 · n=4 29 · n≥5 4. Distinct candidate
surfaces grow ~54k per 100 docs with no cross-document saturation, so per-batch dedup is
the only and sufficient lever (~11.5 µs/key lookup).

A future teacher-distillation pass (gliner-biomed-large agreeing with mined spans) is
deliberately deferred; `fullmap_mine.resolve_batch` is the interception point.

## Install

    uv sync

## Build the dataset

Smoke run over 5 sampled rows per dataset:

    uv run relmedner build-dataset -t -o ./relmedner-test.avro

Full run on the local DirectRunner — no cluster, no VPN needed (`--direct` skips cluster
deployment, watching, and shard collection):

    uv run relmedner build-dataset --direct -o ./relmedner.avro

Full run on the LAN flink cluster (requires the VPN route to every `cluster.yaml` worker — the SSH
gateway alone is not enough, because flink taskmanagers connect back to the laptop's jobmanager
ports and `YamlClusterParser.jobmanager()` resolves the laptop via a route probe toward the first
remote):

    uv run relmedner build-dataset -o ./relmedner.avro

## Testing

    uv run pytest -q
    uv run ruff check src tests

Entity resolution reads a local fullmap database from the hardcoded
`FULLMAP_DIR` path in `src/relmedner/constants.py`; fullmap-dependent tests skip when that
mount is absent, and miner unit tests inject fake `lookup_rows` rows so they never need it.

## Cluster

`make deploy` targets the LAN Flink cluster declared in
`src/relmedner/data/cluster.yaml`; workers bind-mount the fullmap bundle read-only at
`/opt/fullmap`. The compute node is only reachable through the gateway SSH hop — from
off-VPN, add a `ProxyJump` through the gateway in `~/.ssh/config`, or run without the
cluster entirely: `build-dataset --direct` (no pipeline options) keeps output on the local
filesystem.

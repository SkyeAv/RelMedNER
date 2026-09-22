# relmedner

Apache Beam pipeline that builds gliner2 training data from biomedical text corpora.
Datasets come from the HuggingFace hub (`source: hf`) or from a local avro container built
out-of-band (`source: local`, see [CTKP interventions](#ctkp-interventions)). Two ingest
types share one declarative pipeline:

- **script tasks** — datasets that already carry gold spans
  (`anthonyyazdaniml/gliner-biomed-pre-training`, the IOB-formatted
  `disi-unibo-nlp/Pile-NER-biomed-IOB`, the multi-task
  `anthonyyazdaniml/gliner-biomed-post-training`, and the relation-tagged
  `knowledgator/sentence_rex`); spans are relabeled to biolink classes
  via tablassert `Categories` and local fullmap resolution, and relations are
  distant-supervised through a biolink-predicate gazetteer matched between mention surfaces.
  The post-training corpus additionally carries native gold relations and sampled
  negatives, and splits into per-row task families (below). `knowledgator/sentence_rex`
  instead carries gold relations with both participant spans marked inline (`<e1>`/`<e2>`
  tags) and labels kept native.
- **fullmap tasks** — unlabeled text (e.g.
  `anthonyyazdaniml/gliner-biomed-curated-corpus` and the downsampled, class-balanced
  `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus`, a strict 158,890-row subset of
  the curated corpus with an identical `text`-only schema); entities are *mined* by enumerating
  n-grams, resolving them in one batched round trip against the local fullmap redb, and
  keeping only spans that exactly match a normalized preferred name. Mined spans also feed
  the gazetteer for `evidence="distant"` relations.

## Ingests

| dataset | task | inputs | outputs | rows in |
| --- | --- | --- | --- | --- |
| `anthonyyazdaniml/gliner-biomed-pre-training` | `script` → `GlinerBiomedScript` | `tokenized_text`, `ner` | entities, relations | 98,659 |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | `script` → `PileNerBiomedScript` | `tokens`, `ner_tags` | entities | 58,861 |
| `knowledgator/sentence_rex` | `script` -> `SentenceRexScript` | `sentences`, `labels` | relations | 44,115 |
| `anthonyyazdaniml/gliner-biomed-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 418,381 |
| `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 158,890 |
| `anthonyyazdaniml/gliner-biomed-post-training` | `script` → `GlinerBiomedPostScript` | `tokenized_text`, `ner`, `negatives` | entities, classifications, structures, relations | — |
| `~/Desktop/interventions.avro` (local) | `script` → `CtkpInterventionsScript` | whole avro record | entities | 1,020,749 |

All script tasks share one resolution chain — fullmap first, a shared lowercased
`FALLBACK_LABEL_MAP` second (dataset vocabularies ride on top via
`resolve_mentions(label_map=...)`), raw labels last — and two shared quality gates:

- `ResolutionGate` rejects fullmap hits contradicting the source corpus's own label
  (label↔category buckets, model-organism CURIE guard, acronym-over-catch-all guard);
  rejections fall through to fallback/raw, never dropping the mention. ~20% of fullmap
  hits rejected on the pile-ner corpus, nearly all true false positives.
- `PredicateRangeGate` rejects gazetteer relations whose head/tail biolink categories
  contradict the predicate's domain/range (raw labels impose no constraint). ~21% of
  candidate relations rejected, all sampled rejects genuinely wrong.

Dataset-format notes (`PileNerBiomedScript`): `tokens`/`ner_tags` are python-repr strings
(`ast.literal_eval`, malformed rows skip); orphan `I-` tags promote to single-token spans
rather than dropping; raw labels PascalCase so the full 3,896-type tail stays
biolink-shaped. Measured full corpus: 100% of rows emit, ~188k entity mentions, and 6,058
gazetteer relations across 5,501 rows (9.3% relation-bearing, across 23 biolink predicates,
each carrying its biolink slot description as `relation_descriptions`).

Dataset-format notes (`SentenceRexScript`): each `sentences` row wraps its two relation
participants in `<e1>`/`<e2>` tags and `labels` carries the gold predicate; the text ships
with ONLY the four tag literals stripped (no whitespace normalization), measured true on all
43,044 well-formed rows. Of the 44,115 train rows, 550 are null and drop; 521 violate the
one-pair-per-tag invariant, 18 well-formed-count rows carry nested angle-bracket markup
inside a surface (`< sub>`), and 48 rows have case-insensitively identical head and tail
surfaces (self-loops), so 42,978 rows ship. Labels: 846 distinct raw strings, 837 after
snake_case normalization, of which 17 are biolink `Predicates` members covering 1,263 rows
(2.9%); only 4.2% of rows carry a biomedical-marker label, and general-domain rows are kept.
Formatting the strip-only rule deliberately tolerates: ~99% of rows wrap tag surfaces in
internal whitespace (`<e1> Myristica fragrans </e1>`) and 95.8% use spaced punctuation
(` ,`, ` .`).

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

## CTKP interventions

The one `source: local` ingest. `LocalAvroDataStream` reads an avro container off disk and
ships each whole record to the declared script — there is no `columns_out` projection,
because the file's own schema is the contract. This keeps a dataset that cannot live on
the hub (rebuilt per AACT snapshot) on the same declarative path as everything else.

The file is built out-of-band from the clinical trials KP (CTKP) build on the Hypatia box:
the raw AACT `interventions` + `intervention_other_names` tables joined against the KP's
NameResolver output (`interventions_mapped`, `interventions_unmapped`,
`interventions_synonyms`, `interventions_synonyms_restored`). Record
`relmedner.ingests.Intervention`:

```
nct_id, intervention_type, name, description
matches: array<Match{curie, category, preferred_name, source, matched_text, unmapped}>
other_names: array<string>, synonym_curies: array<string>, unmapped: boolean
```

`matches` is an array because 177,979 of the 1,020,749 interventions carry more than one
normalization hit ("Nab-paclitaxel plus Gemcitabine" → `CHEBI:175901` + `MESH:C520255`),
and each hit keeps its own `matched_text` — the actual mention span, which is what makes
the record supervision rather than just text.

`CtkpInterventionsScript` therefore does **not** re-resolve through fullmap: these spans
are the KP's own gold CURIEs. It only enforces the shared contracts — biolink-class
membership (`biolink:Procedure` and `Procedure` collapse to one label) and gliner2's
occurs-in-text rule. Records the KP never normalized fall back to their AACT
`intervention_type` keyed on the trial's own name, so the 41k DEVICE / 36k PROCEDURE /
39k BEHAVIORAL rows still teach something; `OTHER` has no defensible biolink class and is
deliberately absent from `LABEL_MAP`, so those rows ship text with no entity and the
declared-outputs filter drops them.

Measured over the first 20,000 records: 83% ship with at least one entity, across 12
biolink classes (SmallMolecule, Procedure, Drug, Device, BehavioralFeature,
ChemicalEntity, DiagnosticAid, Protein, BiologicalEntity, MolecularMixture, Food,
GenomicEntity).

The declared `path` is expanded with `expanduser` at stream time. To point at a different
snapshot, edit `path` in `src/relmedner/data/ingests.yaml`; the rebuild scripts live on
wenceslaus at `/users/sgoetz/ctkp-staging/`.

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

    make test        # full gate: every test, measured, 90% coverage floor
    make test-fast   # iteration loop: parallel, no coverage measurement
    make lint        # ruff check + format check

Pre-commit runs `make lint` only (sub-second); the suite runs at the pre-push boundary, so it gates
once per pushed batch instead of taxing every commit. Install both hook types once:

    uv run pre-commit install -t pre-commit -t pre-push

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

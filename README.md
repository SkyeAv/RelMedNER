# relmedner

Apache Beam pipeline that builds gliner2 training data from biomedical text corpora.
Datasets come from the HuggingFace hub (`source: hf`) or from a local avro container built
out-of-band (`source: local`, see [CTKP interventions](#ctkp-interventions)). Two ingest
types share one declarative pipeline:

- **script tasks** -- datasets that already carry gold spans
  (every `script` row in the [ingest table](#ingests), which is the list that stays current as
  corpora are added); spans are relabeled to biolink classes
  via tablassert `Categories` and local fullmap resolution, and relations are
  distant-supervised through a biolink-predicate gazetteer matched between mention surfaces.
  Two corpora deviate: the multi-task `anthonyyazdaniml/gliner-biomed-post-training`
  additionally carries native gold relations and sampled negatives, and splits into per-row
  task families (below), while `knowledgator/sentence_rex` carries gold relations with both
  participant spans marked inline (`<e1>`/`<e2>` tags) and labels kept native.
- **fullmap tasks** -- unlabeled text (e.g.
  `anthonyyazdaniml/gliner-biomed-curated-corpus` and the downsampled, class-balanced
  `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus`, a strict 158,890-row subset of
  the curated corpus with an identical `text`-only schema); entities are *mined* by enumerating
  n-grams, resolving them in one batched round trip against the local fullmap redb, and
  keeping only spans that exactly match a normalized preferred name. Mined spans also feed
  the gazetteer for `evidence="distant"` relations. The gazetteer additionally emits
  statement-qualifier relations (below).
- **local sources** -- files on disk instead of a hub dataset. `source: local` reads an avro
  container and ships each record whole (see [CTKP interventions](#ctkp-interventions));
  `source: local_delimited` reads a header-delimited TSV/CSV with a `columns_out` projection
  (`src/relmedner/data/qualifiers/qualifier_corpus.tsv`, read by `LocalDelimitedDataStream`).
  Relative paths resolve against the CWD first, then the package data dir, so in-repo corpora
  work from any CWD and inside the worker container.

## Ingests

Field-by-field schema, discriminator rules, and copy-paste templates for `ingests.yaml`: [docs/yaml-config.md](docs/yaml-config.md).
Machine-readable JSON Schemas for editor autocomplete and pre-validation: [schemas/ingests.schema.json](schemas/ingests.schema.json) (+ `schemas/cluster.schema.json`), regenerated with `uv run relmedner schema`.

| dataset | task | inputs | outputs | rows in |
| --- | --- | --- | --- | --- |
| `anthonyyazdaniml/gliner-biomed-pre-training` | `script` -> `GlinerBiomedScript` | `tokenized_text`, `ner` | entities, relations | 98,659 |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | `script` -> `PileNerBiomedScript` | `tokens`, `ner_tags` | entities | 58,861 |
| `Universal-NER/Pile-NER-type` | `script` -> `PileNerTypeScript` | `conversations` | entities | 45,889 |
| `knowledgator/sentence_rex` | `script` -> `SentenceRexScript` | `sentences`, `labels` | relations | 44,115 |
| `anthonyyazdaniml/gliner-biomed-pre-training` | `script` -> `GlinerBiomedScript` | `tokenized_text`, `ner` | entities, relations | 98,659 |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | `script` -> `PileNerBiomedScript` | `tokens`, `ner_tags` | entities | 58,861 |
| `knowledgator/biomed_NER` | `script` -> `KnowledgatorBiomedScript` | `text`, `entities` | entities | 4,840 |
| `knowledgator/gliner-multilingual-synthetic` | `script` -> `GlinerMultilingualScript` | `tokenized_text`, `ner` | entities | 96,606 |
| `anthonyyazdaniml/gliner-biomed-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 418,381 |
| `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus` | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 158,890 |
| `TrialPanorama/TrialPanorama-database` (`studies`) | `fullmap` (max_ngram=6, taxon=9606) | `abstract` | entities, relations | 1,332,141 |
| `anthonyyazdaniml/gliner-biomed-post-training` | `script` -> `GlinerBiomedPostScript` | `tokenized_text`, `ner`, `negatives` | entities, classifications, structures, relations | -- |
| `interventions/interventions.avro` (local package data) | `script` -> `CtkpInterventionsScript` | whole avro record | entities | 1,020,749 |
| `qualifiers/qualifier_corpus.tsv` (local package data) | `fullmap` (max_ngram=6, taxon=9606) | `text` | entities, relations | 24 |
| `aps/super_glue` (`multirc`) | `script` -> `SuperGlueMultiRCScript` | `paragraph`, `question`, `answer`, `label` | classifications | 27,243 |
| `aps/super_glue` (`record`) | `script` -> `SuperGlueRecordScript` | `passage`, `query`, `entities`, `entity_spans`, `answers` | entities, classifications | 100,730 |
| `knowledgator/PubMedAbstractsNER` | `script` -> `PubmedAbstractsScript` | `tokenized_text`, `ner` | entities, relations | 35,000 |
| `nvidia/Nemotron-PII` (train + test) | `script` -> `NemotronPiiScript` | `text`, `spans` | entities | 200,000 |
| `bigbio/chemprot` (`chemprot_full_source`, 3 splits) | `script` -> `ChemprotScript` | `text`, `entities`, `relations` | entities, relations | 2,432 |

All script tasks except the multilingual ingest (which labels directly, see its
notes below) share one resolution chain -- fullmap first, a shared lowercased
`FALLBACK_LABEL_MAP` second (dataset vocabularies ride on top via
`resolve_mentions(label_map=...)`), raw labels last -- and two shared quality gates:

- `ResolutionGate` rejects fullmap hits contradicting the source corpus's own label
  (label<->category buckets, model-organism CURIE guard, acronym-over-catch-all guard);
  rejections fall through to fallback/raw, never dropping the mention. ~20% of fullmap
  hits rejected on the pile-ner corpus, nearly all true false positives.
- `PredicateRangeGate` rejects gazetteer relations whose head/tail biolink categories
  contradict the predicate's domain/range (raw labels impose no constraint). ~21% of
  candidate relations rejected, all sampled rejects genuinely wrong. Qualifier contexts are
  stricter: typed slots demand resolvable biolink ancestors and reject
  `JUNKY_CATEGORIES` (UMLS qualifier/indexing concepts).

Dataset-format notes (`ChemprotScript`, dataset `bigbio/chemprot`): biomedical literature NER plus protein-chemical interaction extraction; each row is a title + abstract `text`, and the hub parquet auto-conversion delivers `entities` and `relations` as dicts of parallel lists that the script transposes (a ragged or non-dict table yields zero rows of that shape, never a crash). Gold char offsets are END-EXCLUSIVE, verified on all 62,147 spans with 0 out-of-bounds, 0 whitespace slop, and 0 discontinuous spans, so the `KnowledgatorBiomedScript` char-offset bridge is reused and the shipped `text` is the re-joined token stream (emitted mentions dedup to distinct surfaces per label within a row, the shared `group_entities` rule, so mention counts sit below the span census). Split sizes are train 1,020 / validation 612 / test 800 (source: datasets-server size endpoint via `probe.py --dataset bigbio/chemprot --info`); per-split declared probes through the production path (`remote-gate.sh probe -- --declared 'bigbio/chemprot:chemprot_full_source:<split>' --limit <rows> --script ChemprotScript --outputs entities,relations --text-column text`) measured `rows_in=rows_out` on every split and 100% of rows emitting entities, with a relations shape on 766 / 436 / 617 rows (~75%); the 602 rows with empty relations (253 / 169 / 180) ship entities-only under the permitted-shapes contract. Entity census: 62,147 spans in exactly three classes, CHEMICAL 31,831 -> `ChemicalEntity`, GENE-Y 20,157 -> `Gene`, and GENE-N 10,159 -> `GeneFamily` (-N marks nonspecific mentions such as "kinase" or "tumor necrosis factor" that name gene families/classes; labeling them `Gene` would train a family name as a specific gene, and dropping 16% of the gold spans was rejected). Relation census: 15,739 gold triples over the 11 labels CPR:0..CPR:10, arg1 the CHEMICAL and arg2 the gene in 100% of them (0 self-loops, 0 dangling args), so direction is always chemical -> gene; predicates map to biolink members where one honestly exists (CPR:1 `part_of`, CPR:2 `regulates`, CPR:3 and CPR:5 `increases_amount_or_activity_of`, CPR:4 and CPR:6 `decreases_amount_or_activity_of`, CPR:9 `is_substrate_of`) and stay native snake_case where biolink has no slot (CPR:7 `modulator`, CPR:8 `cofactor`; `regulates` would overcommit a direction the corpus deliberately leaves unspecified). CPR:0 (Undefined, 3 relations, the loader's own annotation defect) and CPR:10 ("Not", 683 asserted negatives; this pipeline never asserts negations) drop, 686 of 15,739 measured relations (4.4%), the rows' entities still shipping; 15,030 relations emit in total (6,188 / 3,376 / 5,466 per split), the residual ~23 lost when an arg's span dropped in the token bridge. Label semantics verified against the bigbio loading script that parsed the original corpus (hub revision 24ca8c5daece, chemprot.py `_GROUP_LABELS`, whose CPR:0 comment matches the measured 3 rows exactly).

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

Dataset-format notes (`PileNerTypeScript`): the general-domain pile-ner sibling packages NER
as chat rather than IOB columns. One human turn carries the document behind a `Text: `
prefix, then each entity type is asked for with the templated question
`What describes <type> in the text?` and answered by the next gpt turn with a JSON list of
surface mentions (`[]` for the negative-sampled types, whose rows fall out on the outputs
gate). Answers are surfaces, not offsets, so spans are recovered by matching the mention's
token subsequence case-insensitively against the document's whitespace tokens: that single
rule is both the hallucination guard (gpt occasionally answers with surfaces the document
never contains) and the tokenization guard, and it keeps entities and relation spans
aligned. Emitted mentions are the document's own token slice, so gliner2's sanitizer can
always find them, and repeated mentions contribute every occurrence as a relation span
while appearing once in the entity list. The type vocabulary is open-ended GPT output
(1,666 distinct types in a 1k-row probe, with `person`/`Person`/`PERSON` casing variants
collapsing on the lowercased fallback lookup); head labels ride the dataset-local
`LABEL_MAP` and the tail stays PascalCased raw. `Organization`, `Product` and `CreativeWork`
are not biolink classes, so `organization` maps to `Agent` and `product` stays a raw tail.

Dataset-format notes (`KnowledgatorBiomedScript`, dataset `knowledgator/biomed_NER`): rows are raw untokenized text plus
character-offset entity structs (`{start, end, class}`, end exclusive); char spans bridge to token
spans through `ScriptUtils.char_spans_to_token_spans` with drop-then-snap (measured: 0.04-0.24%
out-of-bounds dropped, ~0.4% whitespace slop normalized, 3.43% mid-token snaps); emitted `text` is
the re-joined token stream so every mention surface stays findable (26.7% of raw-text surfaces
would fail gliner2's validator); the 29-entry `LABEL_MAP` covers the 35 distinct raw class strings
measured on the full 4,840-row train split (21 canonical classes with an honest biolink target plus
8 plural/legacy variants; LANGUAGE, REGULATION OR LAW, MONEY, Unlabelled,
and the bare INTELLECTUAL variant stay unmapped -> raw PascalCase tails; gazetteer relations may
still fire as free signal beyond the declared shapes).

Dataset-format notes (`GlinerMultilingualScript`): the streaming loader delivers this
corpus's `ner` spans as stringified indices with quote-wrapped labels
(`[["18", "21", "\"organization\""]]`) while `tokenized_text` arrives as a real list, so the
script coerces defensively and skips malformed spans. The label tail is long and
multilingual: 21,640 distinct labels with the top-60 covering only 43.6% of spans, and the
labels themselves are multilingual (`person`/`Person`/`personne`/`Persona`/`Pessoa`/`Osoba`),
so the cross-lingual head labels map onto biolink classes (Human, GeographicLocation, Agent,
Disease, Drug, Food, Plant, OrganismTaxon) and everything else rides PascalCase for
zero-shot breadth. Fullmap is bypassed by design: its keys are byte-sorted bags of Porter2
English stems over a biomedical vocabulary, so general-domain non-English surfaces never
match it and label directly off `LABEL_MAP` plus PascalCase. Only `entities` is declared:
the gazetteer's 129 triggers are English biomedical phrases, so relations are not declared.

Dataset-format notes (`SuperGlueMultiRCScript`): general-domain English true/false reading-comprehension QA
(SuperGLUE MultiRC), mapped classification-only -- the corpus carries no entity or relation annotations. The
`label` column arrives as a ClassLabel index or its decoded string; the test split also carries unlabeled rows
(`label` -1), which this pipeline never ingests (train split only, per repo convention) and would skip rather
than coerce. The nested `idx` column is deliberately excluded: row provenance, not training signal.

Dataset-format notes (`PubmedAbstractsScript`): the hub file is one 35,000-object JSON
array (151,668,527 bytes); `tokenized_text` is a token list and `ner` carries end-inclusive
`[start, end, "<MeSH heading> - <definition>"]` spans -- 383,721 of them over 5,850
distinct labels, every label shaped `heading - definition`. Rows are long (median 233 /
max 1,558 tokens, none truncated) and 13 rows have empty `ner`, exiting as text-only
examples that the permitted-shapes contract drops. The repo ships a broken old-style
`dataset_infos.json`: under `datasets` 5.0.1 the hub path raises `KeyError: 'feature'`
inside `Features.from_dict`, with and without `data_files=`, so the ingest reads the raw
file non-streaming through the `hf_json` source (`HuggingFaceJsonDataStream`). Never read
it streaming: on a cold cache the streaming JSON builder promotes every `ner` cell to utf8
straight from the raw JSON text, `mention_spans` would then drop every span, and the
ingest would silently yield nothing. Each label is split on its first `" - "` and only the
bare heading reaches resolution: with the definition riding along, the shared gate's
label-word buckets false-reject 45,975 of 383,721 full-corpus spans (12.0%: gate accepts
the bare heading, rejects the full label -- 14.5% of the 316,600 fullmap-hit spans; e.g.
'ankle' loses a correct Disease hit because the definition contains 'region'/'leg').
Unresolved headings surface PascalCased (`Abdominal Core` -> `AbdominalCore`). Measured on
the FULL 35,000-row corpus with one batched fullmap round trip and the shared gates:
fullmap 316,600 spans (82.5%), fallback 15,437 (4.0%), raw 51,684 (13.5%) under the
US-002 seed map. The `LABEL_MAP` coverage rule maps every heading whose actual mention
surfaces fit one real biolink class -- every clean heading with >=30 raw-origin spans is
mapped, and the residual >=30 tail is mixed/junk or borderline -- rather than chasing a
>=90% share that is unreachable without mislabeling data; the US-004 map holds 211 entries
(19 seed + 192 measured), and the honest measured coverage is 44.2% (origin shares after:
82.5% / 10.0% / 7.5%) because the residual raw tail is dominated by headings with no
faithful class -- `Investigative Techniques` is 94% the surfaces 'methods'/'METHODS',
`Group Processes` is 99.7% 'role' and biolink has no Role class, `Chemical Phenomena` and
`Genetic Phenomena` mix categories -- which stay unmapped on purpose (a wrong bucket
silently mislabels training data). Re-measuring never re-downloads the 152MB file: cached
full file + one batched fullmap round trip (`rs.normalize_terms` then `_fullmap_best`) +
the shared gates make every candidate map pure set arithmetic over headings, so the harness
stays throwaway and is deliberately not committed. Gazetteer relations over a 5,000-row
sample through the production `run()` path with the expanded map: 18.4% of rows carry >=1
relation (1,107 relations, 22 distinct predicates); 100% of rows emit entities.

## Output

Output is Avro records of `TrainingExample` (`text`, `entities`, `relations`, ...).
Relation provenance rides in the Avro records: `negated` (`true` only for sampled
negatives) and `evidence` (`asserted` for gold-span scripts, `distant` for fullmap-mined
spans, `sampled_negative` for grid-sampled non-observations from the post-training
corpus). The gliner2 JSONL projection (`to_output()`) emits mention fields only, because
gliner2 validates every relation value as a mention in the text -- sampled negatives
therefore train under `not_<predicate>` relation names. Predicate slot definitions DO
ride the projection as `relation_descriptions` (the processor consumes them as label
prompts), mirroring `entity_descriptions`.

The declared-outputs filter is a **permitted-shapes contract**: a row ships when it
produced at least one declared shape. Rows may produce fewer shapes than declared
(entity-only mined rows flow) and may produce extra shapes (the `[entities]`-only
pile-ner declaration keeps rows whose gazetteer also fired -- relations are free signal,
not a contract violation).

## Deduplication

Before Avro write, every merged `TrainingExample` passes through a dedup stage so
repeated text does not reach training data twice. Exact dedup keeps one record per
whitespace-normalized text (a blake2b fingerprint of the normalized form is the key);
near dedup then groups records by LSH band keys over MinHash signatures and collapses
each group to a single winner. Signatures use 128 fixed-seed permutations over
lowercased word 5-gram shingles, banded 8 x 16, so the estimated-Jaccard bar for a merge
sits near 0.88 (the S-curve 50% point at s ~= 0.878): only "same abstract re-ingested"
texts merge, half-overlapping ones never do. Texts under 10 tokens bypass near-dedup
entirely, because below that a 5-gram shingle set has too few members for a stable
Jaccard estimate, but they still get exact dedup.

Where a duplicate group collapses, the survivor is the highest-weight record, ties
broken by canonical JSON so the winner is deterministic. Weights are not summed; the
export step owns weighted duplication.

`build-dataset --dedup-mode` controls the stage: `near` (the default) runs exact then
near, `exact` runs exact only, and `off` adds no dedup transforms at all. Drop counts
land in Beam counters under the `relmedner.dedup` namespace; on DirectRunner runs the
pipeline logs one summary line afterwards, `dedup: exact -<n> near -<n> of <total>
records`. On the Flink cluster the same counters surface in the Flink UI / REST job
metrics.

## CTKP interventions

The one `source: local` ingest. `LocalAvroDataStream` reads an avro container off disk and
ships each whole record to the declared script -- there is no `columns_out` projection,
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
normalization hit ("Nab-paclitaxel plus Gemcitabine" -> `CHEBI:175901` + `MESH:C520255`),
and each hit keeps its own `matched_text` -- the actual mention span, which is what makes
the record supervision rather than just text.

`CtkpInterventionsScript` therefore does **not** re-resolve through fullmap: these spans
are the KP's own gold CURIEs. It only enforces the shared contracts -- biolink-class
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

The declared `path` is `interventions/interventions.avro`: used as declared when it names an
existing file (absolute, `~`-expanded, or relative to the caller's CWD), otherwise resolved
against the package data dir (`relmedner.constants.DATA`), the same rule `local_delimited`
follows. The container itself is gitignored -- 92.6MB of already-compressed avro, past GitHub's
50MB warning, rebuilt per AACT snapshot -- and is dropped at
`src/relmedner/data/interventions/interventions.avro`. A fresh clone therefore has no blob, and
the stream fails with `FileNotFoundError` rather than silently yielding nothing. `uv_build`
ships every file under the package dir, so a built wheel (and the worker image built from
`dist/`) carries the corpus -- that is what makes `source: local` runnable on the Flink
cluster. Rebuilding: `build_ctkp2.sh` joins the AACT `interventions` and
`intervention_other_names` tables against the KP's `interventions_mapped`,
`interventions_unmapped`, `interventions_synonyms`, and `interventions_synonyms_restored` into
`combined2.tsv`, then `tsv_to_avro2.py` converts `combined2.tsv` into `interventions.avro`;
both scripts live on wenceslaus at `/users/sgoetz/ctkp-staging/`. To point at a different
snapshot, edit `path` in `src/relmedner/data/ingests.yaml`.

## TrialPanorama database

`TrialPanorama/TrialPanorama-database` ships 11 subsets (~27.4M rows total) keyed by
`study_id`. This ingest takes the `studies` subset (1,332,141 rows) and mines the
`abstract` column, the only long free text in the database. The structured subsets carry
MeSH / MedDRA / RxNorm identifiers the pipeline cannot yet translate into biolink classes,
so those subsets are deferred until an ontology-mapping pass exists. Empty or null
abstracts degrade to bare examples and are dropped by the declared-outputs filter,
matching fullmap behavior on textless rows.

Measured row count: 1,332,141 (source: HF datasets-server info endpoint, split `all`).

Mined rows flow through the same resolution chain and quality gates as the
`anthonyyazdaniml/gliner-biomed-curated-corpus` fullmap entries above; the miner itself
is documented under [How fullmap mining works](#how-fullmap-mining-works) and not
re-described here.

## SuperGLUE ReCoRD

The one ingest whose script ships general-domain text: `aps/super_glue` subset `record` is a
CNN news passage per row (100,730 training rows), annotated with gold entity spans
(`entity_spans` is a dict of parallel `text`/`start`/`end` lists) and a cloze-style query whose
answers name which spans fill its placeholder.

`SuperGlueRecordScript` trusts the dataset's gold and re-resolves nothing: spans ship verbatim
under the single catch-all biolink class `NamedThing`, the same trust-gold stance as
`CtkpInterventionsScript`, because the passages are news text unrelated to the biomedical
vocabularies fullmap resolves against -- re-categorizing "Dallas" or "the Rams" through those
would be noise, and ReCoRD declares no gold relations either. A span survives only when its
table is well-formed (equal-length lists) and `passage[start:end]` matches `text` exactly;
malformed or ragged span tables yield zero spans rather than a crash.

The cloze task ships as one `Classification(task='cloze entity resolution')`: labels are the
surviving span surfaces, `true_label` the gold answers that occur in that label set, and
`prompt` the raw query. Answers whose surface never survives span validation drop rather than
corrupting the label set; when no gold answer survives, the row ships entities only (subset
semantics of the declared-outputs filter).

## NVIDIA Nemotron-PII

The one general-domain ingest: NVIDIA's synthetic, persona-grounded PII/PHI corpus
(CC-BY-4.0, generated with NeMo Data Designer over personas grounded in U.S. Census data),
100,000 train + 100,000 test rows, 55+ span labels, `document_format` structured/unstructured,
`locale` us/intl, 30 domains. It earns its place next to the biomedical corpora two ways: the
PHI labels the biomedical sources never carry (`medical_record_number`,
`health_plan_beneficiary_number`, `date_of_birth`, `blood_type`) ride healthcare-domain
documents, and the person/place/employer labels widen the de-identification repertoire the
gliner2 model can name on clinical text.

Format notes: `spans` arrives as a python-repr string of span dicts (`ast.literal_eval`,
skip-don't-coerce, malformed entries drop individually). The offsets are end-exclusive and
slice-authoritative: the span dict's own `text` field drifts in case and type (int-typed for
`age`/`cvv`, lowercased surfaces like `spanish` against the real `Spanish`), so mention
surfaces come from `text[start:end]` and the field is never read.

Label policy: 14 labels map onto exact biolink classes (`first_name`/`last_name` -> `Human`,
seven location labels -> `GeographicLocation`, `company_name` -> `Agent`, `gender` ->
`BiologicalSex`, `occupation`/`education_level`/`employment_status` -> `SocioeconomicAttribute`);
the ~40-label PII tail stays raw PascalCase (`Ssn`, `Ipv4`, `MedicalRecordNumber`, ...), the
Pile-NER precedent, for zero-shot de-identification breadth. Entity-only: the gazetteer's
predicates are biomedical-mined, so emitting relations here would fabricate edges between PII
mentions on general-domain documents.

Gate note: `ResolutionGate` gained a `pii-person-name` bucket plus person/place/measure key
extensions, so a surname that fullmap-hits as a gene is rejected (only
`Human`/`IndividualOrganism` ancestors survive); rejections fall through to fallback/raw like
every other gate rejection.

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
  -> `associated_with`); everything else keeps a biolink-shaped `snake_case` native form;
- sampled negatives from the dataset's `negatives` column train under `not_<predicate>`
  names after guards: malformed, self-loop, duplicate, positive-colliding, and
  not-in-text triples drop, capped at 2x the row's positive count;
- relation surfaces that tokenization tore away from the text (`CC-chemokines` vs tokens
  `CC`, `-`, `chemokines`) are filtered at extraction -- gliner2 would drop them anyway;
- `GlinerBiomedPostScript.LABEL_MAP` extends the shared `FALLBACK_LABEL_MAP` with 25
  validated biolink classes; values are validated loudly at import.

## How fullmap mining works

Per batch of documents (Beam `BatchElements`, one redb round trip per batch):

1. tokenize with gliner2's own `WhitespaceTokenSplitter` (imported by file path so torch
   never loads), strip per-token edge punctuation -- which also makes sentence-crossing
   n-grams impossible because `.` tokenizes alone and cleans to empty;
2. enumerate contiguous n-grams up to `max_ngram`, dropping all-numeric and
   all-function-word grams; add hyphen/slash folds, Greek/unicode folds, and one-way
   in-document acronym bridges (`body mass index (BMI)` resolves the expansion, keeps the
   `BMI` span);
3. one `rs.normalize_terms` + `lookup_rows` + `filter_and_rank` per batch -- fullmap keys
   are **byte-sorted bags of Porter2 stems**, so word order is already irrelevant and
   permutation-style augmentation is a proven no-op;
4. accept a row only when `normalize(PREFERRED_NAME) == term` (EXACT). The published PR
   tiers are *not* a quality dial: PR=50 just means a preferred name already in
   sorted-stem order; PR=250/500 are stopword collisions;
5. gates (constants in `relmedner/constants.py`, each fixed by measurement -- see
   `PLAN.md`): exclude model-organism CURIE prefixes (`FB`, `ZFIN`, `MGI`, ... -- human
   genes only), junk-category gate (UMLS qualifier/indexing concepts), unigram minimum
   length (digit-bearing exempt), gene/protein casing rule (rejects `in`->`NCBIGene:3630
   INS` collisions), strict unigram name agreement (case-insensitive equality or simple
   plural -- kills Porter2 derivational collisions like `oxidative`->`oxide`), and
   rejection of spans that begin or end with a function word;
6. greedy longest-match non-overlap selection, then group by biolink category with
   class-definition + `[fullmap: CURIE | name]` descriptions.

Measured expectations on the curated corpus: **~20 mentions/doc** (~8.5M projected over
418,381 docs), unigram precision ~ 78%, multi-token precision ~ 90%. Yield by gram
length per 200 docs: n=1 2,962; n=2 918; n=3 137; n=4 29; n>=5 4. Distinct candidate
surfaces grow ~54k per 100 docs with no cross-document saturation, so per-batch dedup is
the only and sufficient lever (~11.5 us/key lookup).

A future teacher-distillation pass (gliner-biomed-large agreeing with mined spans) is
deliberately deferred; `fullmap_mine.resolve_batch` is the interception point.

## Install

    uv sync

## Build the dataset

Smoke run over 5 sampled rows per dataset:

    uv run relmedner build-dataset -t -o ./relmedner-test.avro

Full run on the local DirectRunner -- no cluster, no VPN needed (`--direct` skips cluster
deployment, watching, and shard collection):

    uv run relmedner build-dataset --direct -o ./relmedner.avro

Full run on the LAN flink cluster (requires the VPN route to every `cluster.yaml` worker -- the SSH
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

The default suite is fully offline: the ReCoRD live smoke (`tests/test_super_glue_record_live.py`)
streams one real hub row and runs only when `RELMEDNER_LIVE_HF=1` is set, so hub downloads never
happen in the default run.

The suite can also run on the wenceslaus box to keep the laptop unloaded. The remote shell is
bash-only and the snap `uv` on its PATH is broken, so sync with rsync and invoke uv by absolute
path; `PYTHONUTF8=1` is required for the remote pytest (UTF-8 locale gaps):

    ssh wenceslaus 'mkdir -p ~/Code/RelMedNER-worktrees/super-glue-record'
    rsync -a --delete --exclude .venv --exclude .ralph --exclude .git \
      /home/skyeav/Code/ISB/RelMedNER-worktrees/super-glue-record/ \
      wenceslaus:~/Code/RelMedNER-worktrees/super-glue-record/
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && ~/.local/bin/uv sync'
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && PYTHONUTF8=1 ~/.local/bin/uv run pytest -q'
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/super-glue-record && ~/.local/bin/uv run ruff check ./src ./tests && ~/.local/bin/uv run ruff format --check ./src ./tests'

### Agent skills

`.pi/skills/add-dataset/` is a repo-local skill for the pi coding agent: describe a corpus ("add
<nvidia/Nemotron-PII> to the training data", or `/skill:add-dataset`), and it runs the whole ingest
procedure -- remote probes and gates on wenceslaus, the `ingests.yaml`/`Script`/tuple-lock wiring,
README row plus measured format notes, one PR to `main`. It never streams data, builds, or runs the
full suite on the laptop; declaration-only adds run inline and adds needing new code hand off to the
`ralph` skill with a pre-filled constraint block. The remote loop it drives is exactly the
wenceslaus chain documented above, packaged as committed scripts
(`.pi/skills/add-dataset/scripts/remote-gate.sh`) with receipt lines.

One-time setup: `.pi/skills/` is a trust-requiring project resource, so pi asks whether to trust the
folder on first load. Answering "Trust parent folder" once covers this checkout and every
`RelMedNER-worktrees/*` worktree (pi resolves the nearest saved entry; the choice is written to
`~/.pi/agent/trust.json`). Headless runs (`pi -p`, and aoe/fleet workers) show no prompt and silently
ignore project resources unless that saved entry exists, `defaultProjectTrust` is set to `always` in
the global settings, or the invocation passes `--approve`.

Entity resolution reads a local fullmap database from the hardcoded
`FULLMAP_DIR` path in `src/relmedner/constants.py`; fullmap-dependent tests skip when that
mount is absent, and miner unit tests inject fake `lookup_rows` rows so they never need it.

## Cluster

`make deploy` targets the LAN Flink cluster declared in
`src/relmedner/data/cluster.yaml`; workers bind-mount the fullmap bundle read-only at
`/opt/fullmap`. The compute node is only reachable through the gateway SSH hop -- from
off-VPN, add a `ProxyJump` through the gateway in `~/.ssh/config`, or run without the
cluster entirely: `build-dataset --direct` (no pipeline options) keeps output on the local
filesystem.

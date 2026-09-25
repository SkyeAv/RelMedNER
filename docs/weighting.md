# Weighting guide

How to pick and adjust dataset weights in `src/relmedner/data/ingests.yaml`: what the
numbers mean, how the offline `validate-trust` step turns literature evidence into a
`trust:` value, and a concrete reweighting schema for the datasets already declared.

Field reference lives in `docs/yaml-config.md`; this file is the practice guide.

## What `weight` means here

- Declared per dataset entry, stamped onto **every** record the source emits as
  `TrainingExample.weight` (avro provenance).
- Stock gliner2 has no per-example weight channel, so training-time consumption is
  **weighted duplication** at the avro->JSONL export step: a record with weight 0.7 is
  emitted ~0.7x as often as a 1.0 record. The avro corpus always contains every record
  exactly once -- weights are mixing ratios, not keep/drop switches.
- Weight also decides **dedup survival**: when two near-duplicate records collide, the
  higher-weight one wins (`priority = (-weight, content_id)`, where `content_id` is a
  128-bit blake2b fingerprint of the record's canonical json). After trust adjustment
  (below) that means the higher-*trust* source's copy survives -- usually what you want.
  The tie-break only decides between records of **equal** weight, and it is a content hash
  so the pick stays deterministic across runners, shard counts, and arrival order while
  the near-dedup band shuffle carries 40 bytes per record instead of the whole payload.

## Two knobs, two intents -- don't conflate them

| knob | expresses | set by |
| --- | --- | --- |
| `weight` | mixing **intent**: how much this source should shape the model relative to others | you, by judgment |
| `trust` | **evidence**: how true the source's labels turned out to be under literature validation | `relmedner validate-trust`, reviewed by you |

`trust` folds into the stamped weight as `weight * trust`, clamped to a **fixed +/-20%
band** around the declared weight. Consequences:

- `trust` can only *nudge* inside the band. It can never overturn a declared weight, and
  it can never push a record above 1.0.
- To move a source's effective weight by more than +/-20%, change `weight` -- that is a
  mixing decision, not a trust decision.
- The one exception: `trust: 0` bypasses the band entirely and soft-drops every record
  from the source (record stays in avro, duplicates zero times at export). Use only when
  a source must be kept in the corpus for provenance but excluded from training.

## Edge-level heuristics: `trust_edges`

A source-level `trust:` treats the whole dataset uniformly. Often the sample says
something sharper: **specific predicates** are unreliable -- `treats` edges mined by
trigger proximity were 30% verifiable, `precedes` edges 95%. For that case, declare
`trust_edges` -- relation name -> trust in [0, 1]:

```yaml
  - task: *fullmap-task
    <<: *hf-train
    dataset: anthonyyazdaniml/gliner-biomed-balanced-curated-corpus
    columns_out: [text]
    trust_edges:
      treats: 0.30      # sampled verdicts were mostly unverified
      phosphorylates: 0.0  # soft-drops every record carrying this edge
```

How the sample extrapolates to the whole dataset (the heuristic):

1. `validate-trust` aggregates every relation verdict **by predicate name** per source
   (feature-keyed in `trust-report.jsonl`) and averages them into a per-predicate score.
2. The suggested yaml lists every predicate scored below 0.99 in a `trust_edges:` block.
3. At pipeline stamp time, each record's weight is scaled by the **weakest** flagged
   predicate it carries (`validators.record_edge_factor`). The min is deliberate: one
   unreliable asserted edge poisons the record's training value more than several
   reliable ones redeem it.
4. Records with no relations, or only predicates the sample cleared, are **untouched**  -- 
   edge weighting applies to a SUBSET of the dataset by design. The source weight stays
   byte-identical for them.

Edge trust SCALES directly (no +/-20% band): a targeted heuristic on a known-bad predicate
subset is evidence, not a mixing-intent question. `edge_trust: 0.3` on a weight-1.0
source stamps 0.3. Two entries sharing one row key may each flag different predicates
(their maps merge); the same predicate twice with different values is a parse error.

## The validate-trust workflow

```bash
# one source, quick smoke (5 rows):
relmedner validate-trust --source knowledgator/PubMedAbstractsNER --sample-size 5 -t

# a full source, default 50 records:
relmedner validate-trust --source knowledgator/PubMedAbstractsNER

# everything (slow; 3 req/s against PubMed without an NCBI key):
relmedner validate-trust
```

1. **Sample** -- records stream through the real dispatch path (same scripts, same
   gazetteer, same row filters as `build-dataset`), so validation sees exactly what
   training would see.
2. **Query** -- every emitted entity becomes a PubMed query
   (`"mention"[All Fields] AND "label"[All Fields]`); every relation becomes
   `"head"[All Fields] AND "tail"[All Fields] AND "predicate"[All Fields]`. Spans are
   binary (>=1 hit verifies). Relations are graded: >=5 hits verified, 1-4 partial
   (half credit), 0 unverified -- because two entities co-occurring once can be
   coincidence.
3. **Review** -- `trust-report.jsonl` lists every sampled record: text excerpt, each
   query, its hit count, verdict, and the record's trust. **Read it.** A weird trust
   score is usually a weird query (a generic head/tail pair), not a bad dataset -- and a
   per-predicate outlier (one predicate's verdicts dragging the average) is the signal
   to move that predicate into `trust_edges` instead of lowering the source `trust`.
4. **Commit** -- the command prints a suggested `trust: <value>` snippet per source.
   Hand-edit it into `ingests.yaml`. Nothing auto-applies; the pipeline stays offline
   and deterministic.

Driver settings live in the optional top-level `x-trust:` section of `ingests.yaml`
(`sample_size`, `report`), overridden flag-by-flag -- see `docs/yaml-config.md`.
Secrets (NCBI keys) never go in the yaml, env only:

Notes:

- Records with no entities/relations (classification-only sources) score nothing and are
  excluded from the average -- a classification source can't be literature-validated, so
  it gets no suggestion rather than a misleading 1.0.
- Query/network failures are recorded with `hits: null` and **no verdict**; they neither
  help nor hurt the source. The summary line counts them (`queries`, `errored`), and a
  source that scored nothing says which failure it hit: `all N queries errored` (fix the
  network or the key and re-run) versus `sampled records had no entities/relations to
  validate` (the source is not literature-validatable, keep its declared prior).
- Backend: PubMed E-utilities only, the Firecrawl web-search fallback is gone. NCBI allows
  3 req/s unauthenticated and 10 with a free `NCBI_API_KEY`; the client throttles itself to
  about 3.4 and about 9 req/s to stay inside those windows.
- Unauthenticated PubMed rate-limits at 3 req/s, and a small sample hits it: a 2-record run
  over `Pennlaine/Medical-Entity-JSON-Extraction` recorded `HTTP Error 429: Too Many
  Requests` on three of its queries (wenceslaus, 2026-09-23). Set `NCBI_API_KEY`, or lower
  `--sample-size`, before believing a score built on that many silent no-verdicts.
- On synthetic or fictional text the measurement scores the FICTION, not the labels. That
  same run suggested `trust: 0.23` for a corpus whose spans are correct, because invented
  details (`Name`, `Specialty`, `FocusArea`) return zero PubMed hits and count as
  unverified, while the numeric span `"39"` matched 275,775 unrelated records and counted
  as verified. Keep the declared prior for such a source; do not commit the measurement.
  Short numeric and generic spans are the usual culprit, so check the per-query rows before
  lowering anything.

## Picking a weight for a NEW dataset

1. **Start at 1.0.** Only deviate with a reason.
2. **Scale by volume.** The corpus self-normalizes at export duplication, but very large
   low-value sources still crowd out small gold ones -- drop them toward 0.5, not to 0.
3. **Scale by label quality.** Expert-annotated -> leave high. LLM-generated/synthetic ->
   0.5-0.8. Distant/noisy labels -> 0.3-0.7.
4. **Scale by domain fit.** Biomedical prose -> leave high. Non-medical task-transfer
   data (reading comprehension, PII) -> 0.3-0.5. Its entities/relations still teach span
   mechanics, just not medical semantics.
5. **Let trust refine, not decide.** After the source runs through `validate-trust`,
   set `trust` to what the evidence says. Expect +/-20% movement, nothing more.
6. **Re-check dedup collisions.** A lowered weight loses near-duplicate collisions
   against other sources. If two sources are near-mirrors (e.g. train/test splits of one
   repo), keep their weights equal or the same key must not disagree -- two entries
   sharing one row key must declare the same weight *and* the same trust, or parse
   raises.

## Reweighting the existing datasets

The tier scheme below is DECLARED: `x-defaults` in `src/relmedner/data/ingests.yaml` carries
the tier anchors (`&weight-gold 1.0`, `&weight-silver 0.7`, `&weight-general 0.4`, plus the
0.5 and 0.3 general bands), every entry aliases the tier it belongs to, and a drift lock
test in `tests/test_docs.py` fails when this table's Weight column drifts from the declared
yaml value for any row key. `trust` values remain PRIORS to confirm with `validate-trust`
-- run the validation, then replace priors with measured values.

Every row key in `src/relmedner/data/ingests.yaml` must appear in this table
(`tests/test_docs.py` fails otherwise), so adding an ingest means adding its tier here in
the same PR. Rows sharing one row key (two splits of one repo, or one local avro plus its
fullmap-mine tsv) are listed once, with the split note, because `weights_by_source` keys
on the row key and refuses to stamp two entries that disagree.

| Dataset(s) | Tier | Weight | Trust prior | Why |
| --- | --- | --- | --- | --- |
| `knowledgator/PubMedAbstractsNER` | gold | 1.0 | 1.0 | expert-labeled PubMed NER |
| `anthonyyazdaniml/gliner-biomed-pre-training` | gold | 1.0 | 1.0 | curated biomed pre-training |
| `knowledgator/biomed_NER` | gold | 1.0 | 1.0 | curated biomed NER |
| `anthonyyazdaniml/gliner-biomed-curated-corpus` | gold | 1.0 | 1.0 | curated fullmap mining base |
| `anthonyyazdaniml/gliner-biomed-balanced-curated-corpus` | gold | 1.0 | 1.0 | curated, class-balanced variant |
| `interventions/interventions.avro` | gold | 1.0 | 1.0 | local curated intervention gazetteer |
| `thunlp/docred` (three declared ingests, one row key) | gold | 1.0 | 1.0 | human-annotated document-level entity clusters and gold relations |
| `fewrel/*` (six declared ingests, six row keys, one wildcard like the reddit anchor) | gold | 1.0 | 1.0 | human-annotated gold token spans and relations; wiki/nyt/semeval general domain, pubmed splits biomedical |
| `disi-unibo-nlp/Pile-NER-biomed-IOB` | silver | 0.7 | 0.8 | silver IOB annotations |
| `knowledgator/sentence_rex` | silver | 0.7 | 0.8 | sentence-level RE, distant labels |
| `Universal-NER/Pile-NER-type` | silver | 0.7 | 0.8 | LLM-generated type annotations |
| `anthonyyazdaniml/gliner-biomed-post-training` | silver | 0.7 | 0.8 | synthetic post-training corpus |
| `knowledgator/gliner-multilingual-synthetic` | silver | 0.7 | 0.8 | synthetic, multilingual |
| `qualifiers/qualifier_corpus.tsv` | silver | 0.7 | 0.8 | local qualifier text, unverified provenance |
| `synthetic-ner-ade-tweets/ade_tweets_unannotated.tsv` | silver | 0.7 | 0.8 | same tweet texts with no gold spans, fullmap-mined, so distant labels |
| `ruslan/bioleaflets-biomedical-ner` (train + test, one row key) | silver | 0.7 | 0.8 | EMA regulatory leaflets; the card does not state annotation provenance |
| `Pennlaine/Medical-Entity-JSON-Extraction` | silver | 0.7 | 0.8 | 50 instruction-tuned vignettes, hub card body empty |
| `bc5cdr/train.avro` | gold | 1.0 | 1.0 | BioCreative V CDR, human-annotated gold chemical and disease spans |
| `bc5cdr/dev.avro` | gold | 1.0 | 1.0 | same corpus, dev split |
| `bc5cdr/test.avro` | gold | 1.0 | 1.0 | same corpus, test split |
| `medmentions/train.avro` | gold | 1.0 | 1.0 | MedMentions ST21pv, human-annotated gold UMLS-linked spans (CC0) |
| `medmentions/dev.avro` | gold | 1.0 | 1.0 | same corpus, dev split |
| `medmentions/test.avro` | gold | 1.0 | 1.0 | same corpus, test split |
| `bigbio/chemprot` (train, validation, test; one row key) | gold | 1.0 | 1.0 | BioCreative VI ChemProt, expert-annotated chemical-protein relations |
| `bigbio/chia` (five declared ingests, one row key) | gold | 1.0 | 1.0 | CHIA, expert-annotated clinical-trial eligibility criteria (Kury et al., 2020) |
| `wcole3/biored-parquet` (train, validation, test; one row key) | gold | 1.0 | 1.0 | BioRED, expert-annotated biomedical relations |
| `OpenMed/drugprot-parquet` (train, validation; one row key) | gold | 1.0 | 1.0 | DrugProt, expert-annotated chemical-gene relations (cc-by-4.0 mirror) |
| `AGBonnet/augmented-clinical-notes` | silver | 0.7 | 0.8 | LLM-augmented clinical-note paragraphs, unlabeled text, fullmap-mined distant labels (MIT) |
| `openlifescienceai/medmcqa` | silver | 0.7 | 0.8 | entrance-exam explanation prose, fullmap-mined distant labels (apache-2.0) |
| `OpenMed/MedDialog` | silver | 0.7 | 0.8 | patient-doctor consultation replies, fullmap-mined distant labels (apache-2.0) |
| `commanderstrife/jnlpba` | gold | 1.0 | 1.0 | JNLPBA biomedical abstract IOB gold spans (apache-2.0) |
| `ncbi/ncbi_disease` (train, validation, test; one row key) | gold | 1.0 | 1.0 | NCBI Disease corpus, dual-annotator gold disease spans (public domain) |
| `bigbio/linnaeus` | gold | 1.0 | 1.0 | LINNAEUS species corpus, gold species mentions over PMC full text (cc-by-4.0) |
| `bigbio/osiris` | gold | 1.0 | 1.0 | OSIRIS variant corpus, gold gene/variant mentions over PubMed abstracts (cc-by-3.0) |
| `bigbio/pubmed_qa` (labeled folds) | gold | 1.0 | 1.0 | PubMedQA expert-labeled folds, yes/no/maybe over abstract contexts (mit; artificial split undeclared, one-weight row key) |
| `rjac/clinicaltrials.gov-summary_and_eligibility` | silver | 0.7 | 0.8 | trial eligibility criteria prose, fullmap-mined distant labels (MIT) |
| `hackint0sh/Text-Clinical-Records` | silver | 0.7 | 0.8 | clinical-record prose, fullmap-mined distant labels (MIT) |
| `dakp-ner-export/examples.avro` | silver | 0.7 | 0.8 | DAKP GLiNER-mined DailyMed / FAERS / EMA spans and assertion relations, model output not human gold |
| `chanzuckerberg/MedMentions` (ST21pv) | gold | 1.0 | 1.0 | PubMed abstract gold UMLS-linked entity spans, 21 semantic types (CC0) |
| `agentlans/json-extraction` (six declared ingests, one row key) | silver | 0.7 | 0.8 | harvested structured-extraction tasks; the card does not state label provenance |
| `bigbio/gad` (train, validation, test; one row key) | silver | 0.7 | 0.8 | GAD gene-disease association labels, derived from database curation rather than span annotation |
| `synthetic-ner-ade-tweets/ade_tweets.avro` | gold | 1.0 | 1.0 | declared gold: human BRAT standoff ADE spans over the tweet text |
| `TrialPanorama/TrialPanorama-database` | general | 0.5 | 0.7 | clinical-trial records, structured not prose |
| `bigbio/ehr_rel` (four declared ingests, one row key) | general | 0.5 | 0.7 | clinician-rated SNOMED concept-pair relatedness, no text context; 0.5 is the declared weight |
| `nvidia/Nemotron-PII` (train + test, one row key -- weights must stay equal) | general | 0.4 | 0.6 | PII not biomedical, synthetic; kept for span diversity |
| `aps/super_glue` multirc | general | 0.3 | 0.6 | classification task transfer, non-med |
| `aps/super_glue` record | general | 0.3 | 0.6 | reading-comprehension transfer, non-med |
| `tensorshield/reddit_dataset_*` (all 7 entries, one shared anchor like the shared `match_on`) | general | 0.3 | 0.5 | noisy social text, health communities only |

### Effective corpus share (computed from README row counts x declared weights)

| tier | declared rows | weight | effective rows |
| --- | --- | --- | --- |
| gold | 1,840,386 | 1.0 | 1,840,386 |
| silver | 946,484 | 0.7 | 662,539 |
| TrialPanorama | 1,332,141 | 0.5 | 666,071 |
| ehr_rel | 11,223 | 0.5 | 5,612 |
| Nemotron-PII | 200,000 | 0.4 | 80,000 |
| super_glue | 127,973 | 0.3 | 38,392 |
| reddit (pre-health-filter) | 10,917,725 | 0.3 x 0.25 sample | 818,829 |
| total | 15,375,932 | | 4,111,828 |

The declared-anthonyyazdaniml balanced-curated rows (158,890) are a strict subset of the
curated corpus and are not added twice. The post-training corpus row count is unmeasured
and excluded from both columns. Gold's effective share of the training mix rises from 12.0
percent (the old everything-at-1.0 world) to 44.8 percent; the reddit number is an upper
bound because the health-community `match_on` filter shrinks it further, so the true gold
share is higher.

Worked example (one entry, gold tier):

```yaml
  - task: {type: script, name: PubmedAbstractsScript, outputs: [entities, relations]}
    source: hf_json
    split: train
    weight: *weight-gold        # 1.0
    trust: 0.97                 # measured by validate-trust, confirmed by review
    dataset: knowledgator/PubMedAbstractsNER
    file: train.json
    columns_out: [tokenized_text, ner]
```

With `weight: 1.0` and `trust: 0.97`, the stamped weight is `1.0 * 0.97 = 0.97`
(inside the band [0.8, 1.0]). With silver tier `weight: 0.7`, `trust: 0.8` stamps
`0.56` (band [0.56, 0.84] -- the clamp only bites when measured trust disagrees with the
tier prior by more than the band allows).

## Sampling ratios: `sample_rate`

| knob | expresses | effect |
| --- | --- | --- |
| `sample_rate` | how much of a source's PASSING rows enter the run at all | real keep/drop, before dispatch and dedup |

`weight` is a mixing ratio over rows that all exist; `sample_rate` decides how many rows
exist. Declare it per dataset entry (`sample_rate: 0.25` keeps a quarter). Use it when a
source is too big for its intended influence: the seven reddit fullmap ingests (7M raw rows
before the health-community `match_on` filter) mined end to end would dominate the run;
`sample_rate: 0.25` (declared on all seven via the shared `&reddit-sample` anchor) cuts
their mining, dispatch, and dedup cost by 4x AND their share of the corpus by 4x, which
`weight` cannot do (weights duplicate records at export, they never remove mining work).

- Deterministic per row CONTENT: keep iff `blake2b(source key + row repr)` lands under
  `rate * 2**64`. A resumed or re-parallelized pass re-makes the identical decision per
  row, and two sources never sample each other's rows.
- Runs AFTER the row filters: a row dropped by quality never also attributes to
  `sample_rate`, so the quality line keeps meaning quality. Sampled-out rows attribute to
  the `sample_rate` reason in the same per-source stats line.
- Sampling sits before the test-run `sample_limit` cut, so a smoke run slices the SAMPLE,
  not the raw stream.
- A sampled source can legitimately keep zero rows on a small input; that is a zero-yield
  source like any other, not an error.

Prefer `sample_rate` over `weight`-style mixing when the goal is compute AND share
reduction together; prefer `weight` when every row must stay in the corpus but the
source should shape the model less.

## Soft drops: `trust: 0` vs `weight: 0` vs `exclude_regex`

| tool | use when |
| --- | --- |
| `exclude_regex` / row filters | the DATA is unwanted (off-topic, wrong language, junk rows) -- removes rows before dispatch |
| `trust: 0` | the source must stay in the avro corpus (provenance, audit) but must not train on -- the normal soft drop |
| `weight: 0` | same effect as `trust: 0` but declared as intent; discouraged -- use `trust: 0` so `weight` keeps meaning "mixing intent" |

Both zero forms keep records in avro and exclude them at export duplication. Neither
removes them from dedup inputs -- a 0-weight record can still *lose* a collision (it just
never wins one against anything heavier).

## Related

- [README index](../README.md) -- the full docs index
- [yaml config](yaml-config.md) -- the field reference for `weight` and `trust`
- [ingests](ingests.md) -- the declared row keys the priors table covers

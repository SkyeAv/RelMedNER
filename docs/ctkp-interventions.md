# CTKP interventions

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

Multi-class mentions (one surface under two labels) are supported end to end -- the script
fans out per match row, `group_entities` groups by category, and gliner2's per-query boundary
targets train overlapping spans natively; the fan-out contract is pinned by
`test_the_same_surface_under_two_categories_ships_multiclass`. The snapshot never triggers it:
measured over all 1,020,749 records (20260920 build, wenceslaus), every `matched_text` carries
exactly one KP category -- the 177,979 multi-hit interventions are distinct surfaces, so one
surface resolves to one category upstream in the KP NameResolver, not in this build
(`build_ctkp2.sh` keeps all hits; `sort -u` removes only exact duplicate rows). Ancestor
propagation (span also ships under its biolink parent) was measured and rejected: all 451,234
mentions would gain a parent label and the propagated labels are structural middle classes
(MolecularEntity 221,842, ChemicalEntityOrProteinOrPolypeptide 87,772, ActivityAndBehavior
65,450) that fire ~100% correlated with their child -- bloat without disambiguation signal.
Genuinely multi-class spans (a peptide drug under two sibling vocabularies) therefore need a
second curated mapping source joined per surface, not a policy over this snapshot.

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
both scripts live on wenceslaus at `/local_raid1/sgoetz/ctkp-staging/`, next to the
`interventions.avro` snapshot they build. Operator-built corpus artifacts stay on the local
RAID array rather than the NFS home dir: the array is the faster disk for gate runs, and every
remote gate stages the blob from there into the package data dir. To point at a different
snapshot, edit `path` in `src/relmedner/data/ingests.yaml`.

## Related

- [README index](../README.md) -- the ingest-table row and the full docs index
- [bc5cdr](bc5cdr.md) -- the other out-of-band local-avro ingest
- [ingests](ingests.md) -- the resolution chain its rows enter

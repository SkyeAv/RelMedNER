# BC5CDR

The BioCreative V CDR corpus: 1,500 PubMed abstracts (500 train / 500 dev / 500 test)
annotated with MeSH-normalized chemicals and diseases plus chemical-induced-disease
(CID) relations. Shipped as three local avro containers under
`src/relmedner/data/bc5cdr/`, built out-of-band like the CTKP interventions --
nothing streams from the hub at pipeline time.

## Build

The builder lives on wenceslaus at `/users/sgoetz/bc5cdr-staging/build_bc5cdr.py`
(stdlib `xml.etree.ElementTree` plus `fastavro.writer` only). It maps
`CDR_TrainingSet.BioC.xml` -> `train.avro`, `CDR_DevelopmentSet.BioC.xml` ->
`dev.avro`, `CDR_TestSet.BioC.xml` -> `test.avro`, then re-checks every receipt in
the table below and exits nonzero naming the first contradicted receipt if any check
fails, so a silently-wrong container can never be written.

Rebuild (the corpus is copied out of volatile `/tmp` on wenceslaus once):

    ssh wenceslaus 'mkdir -p /users/sgoetz/bc5cdr-staging/corpus && cp -r /tmp/bc5cdr/CDR_Data/CDR.Corpus.v010516 /users/sgoetz/bc5cdr-staging/corpus/'
    ssh wenceslaus 'cd ~/Code/RelMedNER-worktrees/bc5cdr && /home/sgoetz/bin/uv run python /users/sgoetz/bc5cdr-staging/build_bc5cdr.py --corpus /users/sgoetz/bc5cdr-staging/corpus/CDR.Corpus.v010516 --out src/relmedner/data/bc5cdr'

## Record schema

Avro record `relmedner.ingests.Bc5CdrDocument`, one per `<document>`:

    pmid: string
    title: string
    abstract: string
    entities: array of Bc5CdrEntity
    relations: array of Bc5CdrRelation

    Bc5CdrEntity:   {type: string, mesh: string, offset: long, length: long, text: string}
    Bc5CdrRelation: {chemical_mesh: string, disease_mesh: string}

- `title` and `abstract` come from the passages whose `type` infon is `title` /
  `abstract`.
- `entity.type` and `entity.mesh` are the annotation's `type` and `MESH` infons, raw
  values (`mesh: "-1"` is the unnormalized sentinel and is kept as-is).
- `entity.offset` is document-absolute: the passage's own `<offset>` plus the
  annotation `<location offset>`; `length` is the location length; `text` is the
  annotation `<text>` verbatim.
- Every `<relation>` whose infon `relation` is `CID` becomes one `Bc5CdrRelation`
  from its `Chemical` and `Disease` infons. No participant matching happens in the
  builder; the consuming script owns it.

## Document text contract

`text = title + "\n" + abstract`. Entity offsets are document-absolute over that
exact string. The builder verifies each annotation against the contract string as a
census only (match / mismatch / out-of-bounds counts) and never drops anything: the
per-split slice mismatches below are corpus noise the containers keep.

## Measured receipts

Re-verified by the builder on every run. Slice match is `100 * match / (match +
mismatch)` over the annotations that land in bounds.

| receipt | train | dev | test |
| --- | --- | --- | --- |
| docs | 500 | 500 | 500 |
| annotations | 9570 | 9773 | 9928 |
| Chemical | 5207 | 5352 | 5394 |
| Disease | 4363 | 4421 | 4534 |
| CID relations | 1038 | 1012 | 1066 |
| MESH = -1 | 76 | 60 | 91 |
| slice match pct | 99.23 | 99.56 | 99.63 |
| slice mismatch (kept) | 74 | 43 | 37 |
| out-of-bounds | 0 | 0 | 0 |

Exactly two annotation types exist (Chemical, Disease). Document text lengths
(`title + "\n" + abstract`) run min / median / max 204 / 1326 / 3907 chars, pooled
over all 1,500 documents.

## Script behavior (Bc5CdrScript)

`Bc5CdrScript` trusts the corpus gold and never re-resolves through fullmap (the
`CtkpInterventionsScript` stance): the annotation's own MESH id rides as a `MESH:` curie
(origin `fullmap`) and the `-1` sentinel ships raw (origin `raw`). Labels map
Chemical -> ChemicalEntity, Disease -> Disease.

Drop rules (skip-don't-coerce, measured over all 1,500 docs in the declared probes):

- a span ships only when `text[offset:offset+length] == annotation text` exactly; the
  measured 74 / 43 / 37 slice mismatches and any out-of-bounds offset drop here
- unknown annotation types and malformed fields drop without raising
- relations resolve each `chemical_mesh` / `disease_mesh` against the SURVIVING spans only;
  unresolved meshes drop (measured 1 / 2 / 2 per split) and head == tail self-loops drop (0)
- surviving relations emit one asserted, non-negated biolink `causes` relation per distinct
  surface pair, cross-producted over same-mesh mentions and deduplicated

Dispatch yield (declared probes, `--script Bc5CdrScript --outputs entities,relations --limit 500`,
run 2026-09-23 on wenceslaus):

| probe | train | dev | test |
| --- | --- | --- | --- |
| rows_in / rows_out | 500 / 500 | 500 / 500 | 500 / 500 |
| rows emitting | 500 (100%) | 500 (100%) | 500 (100%) |
| entity mentions | 4,628 | 4,625 | 4,712 |
| relations | 2,329 | 2,380 | 2,495 |
| shapes | entities=500 relations=499 | entities=500 relations=499 | entities=500 relations=500 |

The 2,329 / 2,380 / 2,495 emitted relations are the surface-pair expansion of the 1,038 /
1,012 / 1,066 CID triples (mean ~2.3 pairs per CID, max 30) minus pairs whose participant
span dropped a drop rule.

## License and provenance

The CDR corpus is public domain under the NCBI/NLM PUBLIC DOMAIN NOTICE. The hub
mirror `bigbio/bc5cdr` simply re-ships the original `CDR_Data.zip`. The release also
carries PubTator `.txt` files with identical annotations; they are ignored here and
the BioC XML is canonical for these containers.

## Related

- [README index](../README.md) -- the ingest-table rows for the three containers and the full docs index
- [ctkp interventions](ctkp-interventions.md) -- the other out-of-band local-avro ingest
- [output](output.md) -- the TrainingExample records these rows emit

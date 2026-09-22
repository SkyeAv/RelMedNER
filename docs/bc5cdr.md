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

## License and provenance

The CDR corpus is public domain under the NCBI/NLM PUBLIC DOMAIN NOTICE. The hub
mirror `bigbio/bc5cdr` simply re-ships the original `CDR_Data.zip`. The release also
carries PubTator `.txt` files with identical annotations; they are ignored here and
the BioC XML is canonical for these containers.

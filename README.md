# relmedner

Apache Beam pipeline that builds gliner2 training data from the
`anthonyyazdaniml/gliner-biomed-pre-training` dataset. Entities are labeled with biolink
classes via tablassert `Categories` and local fullmap resolution; relations are
distant-supervised through a biolink-predicate gazetteer that matches trigger
phrases between mention surfaces.

## Install

    uv sync

## Build the dataset

Smoke run over 5 sampled rows:

    uv run relmedner build-dataset -t -o ./relmedner-test.avro

Full run:

    uv run relmedner build-dataset -o ./relmedner.avro

Output is Avro records with an `input` text and an `output` object. Entities
are grouped by biolink label with class-definition descriptions. Relations
carry SILVER labels inferred from trigger phrases; their head and tail fields
are mention surfaces. Rows without extracted relations are dropped by the
declared-outputs filter.

## Testing

    uv run pytest -q
    uv run ruff check src tests

Entity resolution reads a local fullmap database from the hardcoded
`FULLMAP_DIR` path in `src/relmedner/constants.py`; fullmap-dependent tests
skip when that mount is absent.

from __future__ import annotations

import os

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams real bigbio/chemprot rows from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)

# first-N rows per declared split: a handful proves entities AND relations emit through the
# production path on every split without a census-scale download (the full-split numbers live
# in the wenceslaus census log and the ChemprotScript docstring, not here)
SPLIT_LIMITS: dict[str, int] = {"train": 5, "validation": 3, "test": 3}


@requires_live_hf
@pytest.mark.parametrize("split", sorted(SPLIT_LIMITS))
def test_live_streamed_chemprot_rows_dispatch_end_to_end(split: str) -> None:
    """US-004 live smoke: the exact production path -- declared chemprot split tuple ->
    build_stream -> HuggingFaceDataStream streaming -> Script.REGISTRY dispatch -- must turn
    real hub rows into nonzero training examples carrying gold entities and gold relations
    with every surface occurring in the shipped text. Hand-built rows cannot catch hub-side
    schema drift, so this runs against the real streamed first rows of each declared split.
    Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-only: the hub download must never happen
    in the default offline suite.

    Everything that transitively imports `datasets` (registry -> huggingface, pipeline) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.huggingface import HuggingFaceDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_row
    from relmedner.registry import build_stream
    from relmedner.scripts import ChemprotScript  # noqa: F401  imports populate Script.REGISTRY

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    # select on the declared script NAME plus the hf payload's split slot (position 4 of
    # tuple_fields): the chemprot base key is declared once per split, so the name alone
    # cannot pick a stream
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1])
        for dataset in Ingests.datasets
        if getattr(dataset.task, "name", None) == "ChemprotScript" and dataset.to_tuple()[1][4] == split
    )
    stream: DataStream = build_stream(source, payload)
    assert isinstance(stream, HuggingFaceDataStream)

    Name: str
    Task: tuple
    Values: tuple
    relation_rows = 0
    weights = Ingests.weights_by_source()
    # one rows() iterator: calling next(stream.rows()) per row would restart the stream and
    # dispatch the first row over and over
    Rows = stream.rows()
    for _row_index in range(SPLIT_LIMITS[split]):
        Name, (Task, Values) = next(Rows)
        assert Name == "bigbio/chemprot"
        assert Task == ("script", "ChemprotScript", ("entities", "relations"))
        assert len(Values) == 3  # the declared columns_out projection: text, entities, relations

        Outputs, Example = Script.dispatch("ChemprotScript", (tuple(Task[2]), Values))
        assert Outputs == ("entities", "relations")
        assert isinstance(Example, TrainingExample)
        # every row carries at least one entity (census: 100% of rows); a tiny prefix sample
        # may legitimately contain relations-less rows (602 of 2,432 measured), so the
        # relation claim is per-sample, not per-row
        assert "entities" in Example.populated()
        relation_rows += 1 if Example.relations else 0
        # the shipped text is the re-joined token stream, and every gold surface must occur in it
        for entity in Example.entities:
            for mention in entity.mentions:
                assert mention in Example.text, f"mention {mention!r} does not occur in the text"
        for relation in Example.relations:
            for field in relation.fields:
                assert field.value in Example.text, f"{field.name} {field.value!r} does not occur in the text"
            # CPR:0 (Undefined) and CPR:10 ("Not") are deliberately absent from PREDICATE_MAP,
            # so asserting the emitted predicate is a mapped value doubles as the
            # dropped-labels-never-emit check
            assert relation.name in set(ChemprotScript.PREDICATE_MAP.values()), f"unmapped predicate {relation.name!r}"
            assert relation.negated is False  # this pipeline never asserts negations
            assert relation.evidence == "asserted"  # gold relations, native-gold stance

        # the pipeline's own dispatch wrapper must agree with the direct registry call above
        PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), weights)
        assert PipelineOutputs == Outputs
        assert PipelineExample == Example

    assert relation_rows >= 1, f"no relations emitted in the first {SPLIT_LIMITS[split]} {split} rows"

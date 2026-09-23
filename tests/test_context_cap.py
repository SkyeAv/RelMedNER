from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastavro import parse_schema, writer

from relmedner.constants import CHARS_PER_TOKEN, MAX_TEXT_TOKENS
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser
from relmedner.local import LocalAvroDataStream, LocalDelimitedDataStream
from relmedner.models import HuggingFaceDataset, RowFilters, RunConfig
from relmedner.registry import build_stream
from relmedner.row_filters import joined_text

# US-003: end-to-end synthetic-corpus coverage of the ALWAYS-ON context cap. Every fixture here
# is a real local file streamed through stream(RunConfig()) -- no network, no doubles -- so the
# cap is proven at the same chokepoint every declared ingest (the post-training hf corpus
# included) actually passes through.
CAP: int = MAX_TEXT_TOKENS * CHARS_PER_TOKEN
"""32768 joined chars: the strict > boundary means exactly-at keeps and one char more drops"""

POST_TRAINING_TASK: tuple[Any, ...] = ("script", "GlinerBiomedPostScript", ("entities",))
POST_TRAINING_DATASET: str = "anthonyyazdaniml/gliner-biomed-post-training"

# fixture row shapes: under 100 chars kept, exactly CAP chars kept, CAP + 1 chars dropped
UNDER: str = "aspirin trial"
BOUNDARY: str = "a" * CAP
OVER: str = "a" * (CAP + 1)
# post-training shape: tokenized_text rides as a LIST of token strings, so the cap must measure
# the joined text, not detect the column; ["tok"] * 8193 joins to 32771 chars, over the cap
TOKENS: list[str] = ["tok"] * 8193

SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "CapRow",
    "namespace": "relmedner.tests",
    "fields": [
        {"name": "name", "type": ["null", "string"], "default": None},
        {"name": "tokenized_text", "type": ["null", {"type": "array", "items": "string"}], "default": None},
    ],
}


def write_avro(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("wb") as handle:
        writer(handle, parse_schema(SCHEMA), records)
    return path


def write_tsv(path: Path, texts: tuple[str, ...]) -> str:
    path.write_text("\n".join(("text", *texts)) + "\n", encoding="utf-8")
    return str(path)


def cap_avro_stream(tmp_path: Path) -> LocalAvroDataStream:
    Target: Path = write_avro(
        tmp_path / "cap.avro",
        [
            {"name": UNDER, "tokenized_text": None},
            {"name": BOUNDARY, "tokenized_text": None},
            {"name": OVER, "tokenized_text": None},
            {"name": None, "tokenized_text": TOKENS},
        ],
    )
    return LocalAvroDataStream(POST_TRAINING_TASK, 1.0, str(Target))


def test_avro_stream_keeps_under_and_boundary_rows_and_drops_over_cap_ones(tmp_path: Path) -> None:
    """end-to-end over a real avro container: the sub-100-char row and the row whose joined text
    is EXACTLY CAP chars must yield unchanged, while the CAP+1-char row and the list-shaped
    tokenized_text row (GlinerBiomedPostScript's post-training corpus shape) drop with reason
    max_tokens -- pinning both the strict > boundary and the list-join text rule at stream level"""
    Stream: LocalAvroDataStream = cap_avro_stream(tmp_path)

    Yielded: list[Any] = list(Stream.stream(RunConfig()))

    assert len(UNDER) < 100 and len(BOUNDARY) == CAP and len(OVER) == CAP + 1
    assert [values[0]["name"] for _source, (_task, values) in Yielded] == [UNDER, BOUNDARY]
    for _source, (_task, values) in Yielded:
        # no yielded row may exceed the cap under the exact text rule the evaluator applies
        assert len(joined_text(tuple(values[0].values()))) <= CAP
    assert Stream.stats.rows_in == 4 and Stream.stats.rows_out == 2
    # exact fixture counts: the CAP+1 string row plus the tokenized list row
    assert Stream.stats.dropped_by == {"max_tokens": 2}


def test_delimited_stream_keeps_under_and_boundary_rows_and_drops_over_cap_ones(tmp_path: Path) -> None:
    """same end-to-end contract over a real TSV: the delimited projection rides the same
    always-on evaluator, so the exact-boundary row keeps and the CAP+1 row attributes to
    max_tokens -- a bare text column here hits the identical chokepoint the hf text columns do"""
    Path_: str = write_tsv(tmp_path / "cap.tsv", (UNDER, BOUNDARY, OVER))
    Stream: LocalDelimitedDataStream = LocalDelimitedDataStream(POST_TRAINING_TASK, 1.0, Path_, ("text",))

    Yielded: list[Any] = list(Stream.stream(RunConfig()))

    assert [values[0] for _source, (_task, values) in Yielded] == [UNDER, BOUNDARY]
    for _source, (_task, values) in Yielded:
        assert len(joined_text(values)) <= CAP
    assert Stream.stats.rows_in == 3 and Stream.stats.rows_out == 2
    assert Stream.stats.dropped_by == {"max_tokens": 1}


def test_quality_line_reports_max_tokens_on_both_local_sources(tmp_path: Path, caplog: Any) -> None:
    """the US-009 quality line must count the cap's drops like any declared filter's: after a
    stream(RunConfig()) pass over each local source, the relmedner.quality line mentions
    max_tokens with the exact fixture counts (the line lands in stream()'s finally, so a
    partially-dropped pass still reports)"""
    Avro: LocalAvroDataStream = cap_avro_stream(tmp_path)
    Path_: str = write_tsv(tmp_path / "cap.tsv", (UNDER, BOUNDARY, OVER))
    Delimited: LocalDelimitedDataStream = LocalDelimitedDataStream(POST_TRAINING_TASK, 1.0, Path_, ("text",))

    with caplog.at_level(logging.INFO, logger="relmedner.quality"):
        list(Avro.stream(RunConfig()))
        list(Delimited.stream(RunConfig()))

    Quality: list[Any] = [record for record in caplog.records if record.name == "relmedner.quality"]
    assert len(Quality) == 2
    assert "dropped={max_tokens:2}" in Quality[0].getMessage()
    assert "dropped={max_tokens:1}" in Quality[1].getMessage()


def test_the_post_training_ingest_rides_the_same_hf_cap_chokepoint() -> None:
    """coverage note (US-003): GlinerBiomedPostScript's corpus (dataset
    anthonyyazdaniml/gliner-biomed-post-training in ingests.yaml) declares source hf, so it rides
    HuggingFaceDataStream and its tokenized_text column hits the SAME always-on
    first_drop_reason chokepoint the synthetic list-shaped avro row exercises -- no per-script
    exclusion exists in code. Construction-only asserts pin that wiring offline (no hub contact);
    the drop behavior itself is proven above by the tokenized fixture, not by the network."""
    Ingests = YamlIngestsParser().parse_ingests()
    Matches: list[HuggingFaceDataset] = [
        dataset for dataset in Ingests.datasets if isinstance(dataset, HuggingFaceDataset) and dataset.dataset == POST_TRAINING_DATASET
    ]

    assert len(Matches) == 1
    Dataset: HuggingFaceDataset = Matches[0]
    assert Dataset.source == "hf" and "tokenized_text" in Dataset.columns_out
    Stream = build_stream(*Dataset.to_stream_args())

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.columns_out == ("tokenized_text", "ner", "negatives")
    # filters stays the declared None while the evaluator sees the default RowFilters, so the
    # ALWAYS-ON cap applies to this ingest exactly as to every other source
    assert Stream.filters is None and Stream.effective_filters == RowFilters()

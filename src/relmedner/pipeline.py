from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import apache_beam as beam
from apache_beam.io.avroio import WriteToAvro
from apache_beam.options.pipeline_options import PipelineOptions

from relmedner.constants import MAX_BATCH_ROWS, MIN_BATCH_ROWS, OUTPUTS_MOUNT
from relmedner.fullmap_mine import FullmapMiner
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig, TrainingExample
from relmedner.registry import build_stream
from relmedner.streams import DataStream, StreamedRow, rebuild_task
from relmedner.types import DispatchedExample, Script

FULLMAP_TYPE: str = "fullmap"


def stream_rows(stream: DataStream, config: RunConfig) -> Iterator[StreamedRow]:
    return stream.stream(config)


def dispatch_row(row: StreamedRow) -> DispatchedExample:
    """script tasks dispatch through the registry; the leading task value discriminates"""
    _source, (task, values) = row
    task_model = rebuild_task(task)
    outputs = tuple(task_model.outputs)
    return Script.dispatch(task_model.name, (outputs, values))


def resolve_rows(rows: list[StreamedRow]) -> Iterator[DispatchedExample]:
    """one shared redb round trip per batch of fullmap rows (batched upstream by BatchElements)"""
    tasks = [rebuild_task(task) for _source, (task, _values) in rows]
    pairs = [(values[0], task) for (_source, (_task, values)), task in zip(rows, tasks, strict=True)]
    for task, example in zip(tasks, FullmapMiner.resolve_batch(pairs), strict=True):
        yield (tuple(task.outputs), example)


def matches_declared_outputs(dispatched: DispatchedExample) -> bool:
    """permitted-shapes contract: a row ships when it produced at least one declared shape.
    Subset semantics in both directions -- rows may produce fewer shapes than declared (87% of
    mined rows carry no gazetteer relation; the post-training corpus keeps its entity-only NER
    rows too) and may produce extra shapes (the [entities]-only pile-ner declaration keeps rows
    whose gazetteer also fired; relations are free signal, not a contract violation)."""
    outputs, example = dispatched
    return bool(example.populated() & frozenset(outputs))


def to_record(example: TrainingExample) -> dict[str, Any]:
    return example.asdict()


class BeamPipeline:
    def __init__(self: Self, options: PipelineOptions | None = None) -> None:
        self.options: PipelineOptions | None = options

    def run(self: Self, config: RunConfig) -> None:
        IngestsParser: YamlIngestsParser = YamlIngestsParser()
        # Flink user code runs in the sdkworker, so external runs write to its durable output mount.
        # DirectRunner keeps honoring the caller's ordinary local path for development and unit tests.
        Output: Path = Path(config.output) if self.options is None else Path(OUTPUTS_MOUNT) / config.artifact_name()

        with beam.Pipeline(options=self.options) as new_pipeline:
            rows = (
                new_pipeline
                | "load declarative ingests" >> beam.Create(IngestsParser.generate_tuples())
                | "initialize datastream classes" >> beam.MapTuple(build_stream)
                | "stream declared data" >> beam.FlatMap(stream_rows, config=config)
            )
            # one branch per task type: script dispatch is per-row, fullmap mining needs
            # BatchElements so one redb round trip serves many documents
            script_rows, fullmap_rows = rows | "split by task type" >> beam.Partition(lambda row, count: 1 if row[1][0][0] == FULLMAP_TYPE else 0, 2)
            dispatched = (
                script_rows
                | "dispatch rows to declared scripts" >> beam.Map(dispatch_row)
                | "keep examples matching declared outputs" >> beam.Filter(matches_declared_outputs)
            )
            mined = (
                fullmap_rows
                | "buffer fullmap rows into batches" >> beam.BatchElements(min_batch_size=MIN_BATCH_ROWS, max_batch_size=MAX_BATCH_ROWS)
                | "resolve mined batches" >> beam.FlatMap(resolve_rows)
                | "keep mined examples matching declared outputs" >> beam.Filter(matches_declared_outputs)
            )
            (
                (dispatched, mined)
                | "merge task branches" >> beam.Flatten()
                | "drop the declared output key" >> beam.Values()
                | "shape examples into avro records" >> beam.Map(to_record)
                | "write training data to avro"
                >> WriteToAvro(
                    file_path_prefix=str(Output.with_suffix("")),
                    file_name_suffix=Output.suffix,
                    num_shards=1,
                    shard_name_template="",
                    schema=TrainingExample.avro_schema_to_python(),
                )
            )

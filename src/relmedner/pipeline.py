from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import apache_beam as beam
from apache_beam.io.avroio import WriteToAvro
from apache_beam.options.pipeline_options import PipelineOptions

from relmedner.constants import OUTPUTS_MOUNT
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig, TrainingExample
from relmedner.registry import build_stream
from relmedner.streams import DataStream, StreamedRow
from relmedner.types import DispatchedExample, Script


def stream_rows(stream: DataStream, config: RunConfig) -> Iterator[StreamedRow]:
    return stream.stream(config)


def matches_declared_outputs(dispatched: DispatchedExample) -> bool:
    outputs, example = dispatched
    return example.populated() == frozenset(outputs)


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
            (
                new_pipeline
                | "load declarative ingests" >> beam.Create(IngestsParser.generate_tuples())
                | "initialize datastream classes" >> beam.MapTuple(build_stream)
                | "stream declared data" >> beam.FlatMap(stream_rows, config=config)
                | "dispatch rows to declared scripts" >> beam.MapTuple(Script.dispatch)
                | "keep examples matching declared outputs" >> beam.Filter(matches_declared_outputs)
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

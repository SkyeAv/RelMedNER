from __future__ import annotations

from typing import Self

import apache_beam as beam

from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig
from relmedner.registry import build_stream
from relmedner.streams import DataStream


class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def run(self: Self, config: RunConfig) -> None:
        IngestsParser: YamlIngestsParser = YamlIngestsParser()

        def stream_rows(stream: DataStream):
            return stream.stream(config)

        with beam.Pipeline() as new_pipeline:
            (
                new_pipeline
                | "load declarative ingests" >> beam.Create(IngestsParser.generate_tuples())
                | "initialize datastream classes" >> beam.MapTuple(build_stream)
                | "stream declared data" >> beam.FlatMap(stream_rows)
                | "print data to debug" >> beam.Map(print)
            )

        return None

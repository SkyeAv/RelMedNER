from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from typing import Self

import apache_beam as beam

from relmedner.constants import TEST_ROW_LIMIT
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser


def limit_rows(generated: Iterator, test_run: bool) -> Iterator:
    return islice(generated, TEST_ROW_LIMIT) if test_run else generated


class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def hfgenerator(self: Self, hfops, test_run: bool = False):
        def rows(stream: HuggingFaceDataStream):
            return limit_rows(stream.generate_rows(), test_run)

        return (
            hfops
            | "initialize datastream classes" >> beam.MapTuple(HuggingFaceDataStream)
            | "stream declared data from hugging face" >> beam.FlatMap(rows)
        )

    def run(self: Self, test_run: bool = False) -> None:
        IngestsParser: YamlIngestsParser = YamlIngestsParser()

        with beam.Pipeline() as new_pipeline:
            dcode = new_pipeline | "load declarative ingests" >> beam.Create(IngestsParser.generate_tuples())

            hfops = dcode | "isolate hugging face ingests" >> beam.Filter(lambda ingest: ingest[0] == "hf") | "drop source keys" >> beam.Values()
            hfstream = self.hfgenerator(hfops, test_run=test_run)

            hfstream | "print data to debug" >> beam.Map(print)

        return None

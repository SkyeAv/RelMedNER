from __future__ import annotations

from relmedner.ingests import YamlIngestsParser
from relmedner.huggingface import HuggingFaceDataStream

from typing import Self
from itertools import starmap

import apache_beam as beam


class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def hfgenerator(self: Self, hfops):
        return (
            hfops
            | "initialize datastream classes" >> beam.Map(starmap, HuggingFaceDataStream)
            | "stream declared data from hugging face" >> beam.FlatMap(HuggingFaceDataStream.row_generator)
        )

    def run(self: Self) -> None:
        IngestsParser: YamlIngestsParser = YamlIngestsParser()

        with beam.Pipeline() as new_pipeline:
            dcode = new_pipeline | "load declarative ingests" >> beam.Create(IngestsParser.generate_tuples())

            hfops = (
                dcode
                | "isolate hugging face ingests" >> beam.Filter(lambda ingest: ingest[0] == "hf")
                | "drop source keys" >> beam.Values()
            )
            hfstream = self.hfgenerator(hfops)

            hfstream | "print data to debug" >> beam.Map(print)

        return None

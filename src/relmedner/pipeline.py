from __future__ import annotations

from relmedner.constants import INGESTS_AVRO
from relmedner.huggingface import HuggingFaceDataStream

from typing import Self
from itertools import starmap

from apache_beam.io.avroio import ReadFromAvro
import apache_beam as beam


class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def hfgenerator(self: Self, hfops):
        hfdata = hfops | "initialize datastream class" >> beam.Map(starmap, HuggingFaceDataStream) | "stream declared dataset data" >> beam.FlatMap(HuggingFaceDataStream.row_generator)
        return hfdata

    def run(self: Self) -> None:
        with beam.Pipeline() as new_pipeline:
            dcode = new_pipeline | "load declarative ingests" >> ReadFromAvro(INGESTS_AVRO.as_posix())

            hfops = dcode | "isolate hugging face ingests" >> beam.Filter(lambda ingest: ingest["source"] == "hf")
            hfstream = self.hfgenerator(hfops)

            hfstream | "print data to debug" >> beam.Map(print)

        return None

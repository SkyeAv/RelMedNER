from __future__ import annotations

from relmedner.constants import INGESTS_AVRO

from importlib.resources import files
from typing import Self

from apache_beam.io.avroio import ReadFromAvro
import apache_beam as beam


class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def run(self: Self) -> None:
        with beam.Pipeline() as new_pipeline:
            (
                new_pipeline
                | "load declarative ingests" >> ReadFromAvro(INGESTS_AVRO.as_posix())
                | "print ingests" >> beam.Map(print)
            )

        return None

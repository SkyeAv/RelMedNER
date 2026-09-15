import apache_beam as beam

from typing import Self

class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def run() -> None:
        with beam.Pipeline as new_pipeline:
            (
                new_pipeline
                | "" >> beam.Create()
            )

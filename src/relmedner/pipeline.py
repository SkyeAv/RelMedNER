from __future__ import annotations

from importlib.resources import files
from typing import Self

import apache_beam as beam


class BeamPipeline:
    def __init__(self: Self) -> None:
        return None

    def run(self: Self) -> None:
        with beam.Pipeline as new_pipeline:
            (new_pipeline | "" >> beam.Create())

        return None

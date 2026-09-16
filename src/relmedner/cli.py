from __future__ import annotations

from typing import Annotated

import cyclopts

from relmedner.pipeline import BeamPipeline

APP: cyclopts.App = cyclopts.App()


@APP.command(name="build-dataset")
def build_dataset(
    test_run: Annotated[bool, cyclopts.Parameter(alias="-t")] = False,
) -> None:
    BuildPipeline: BeamPipeline = BeamPipeline()
    BuildPipeline.run(test_run=test_run)

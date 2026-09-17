from __future__ import annotations

from typing import Annotated

import cyclopts

from relmedner.models import RunConfig
from relmedner.pipeline import BeamPipeline

APP: cyclopts.App = cyclopts.App()


@APP.command(name="build-dataset")
def build_dataset(
    test_run: Annotated[bool, cyclopts.Parameter(alias="-t")] = False,
) -> None:
    Config: RunConfig = RunConfig.from_flags(test_run)
    BuildPipeline: BeamPipeline = BeamPipeline()
    BuildPipeline.run(Config)

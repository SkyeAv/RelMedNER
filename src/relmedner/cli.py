from __future__ import annotations

from typing import Annotated

import cyclopts

from relmedner.constants import DEFAULT_OUTPUT
from relmedner.models import RunConfig
from relmedner.pipeline import BeamPipeline

APP: cyclopts.App = cyclopts.App()


@APP.command(name="build-dataset")
def build_dataset(
    test_run: Annotated[bool, cyclopts.Parameter(alias="-t")] = False,
    output: Annotated[str, cyclopts.Parameter(alias="-o")] = DEFAULT_OUTPUT,
) -> None:
    Config: RunConfig = RunConfig.from_flags(test_run, output)
    BuildPipeline: BeamPipeline = BeamPipeline()
    BuildPipeline.run(Config)

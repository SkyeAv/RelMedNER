from __future__ import annotations

from relmedner.pipeline import BeamPipeline

import cyclopts

APP: cyclopts.App = cyclopts.App()


@APP.command(name="build-dataset")
def build_dataset() -> None:
    BuildPipeline: BeamPipeline = BeamPipeline()
    BuildPipeline.run()


@APP.command(name="reset-ingests")
def reset_ingests() -> None:
    from relmedner.ingests import YamlIngestsParser

    IngestsParser: YamlIngestsParser = YamlIngestsParser()
    IngestsParser.write_ingests_arvo()

from relmedner.pipeline import BeamPipeline
import cyclopts

APP: cyclopts.App = cyclopts.App()


@APP.command(name="trigger")
def trigger() -> None:
    build_pipeline: BeamPipeline = BeamPipeline()
    build_pipeline.run()

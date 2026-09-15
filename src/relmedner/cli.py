from relmedner.pipeline import build_pipeline
import cyclopts

APP: cyclopts.App = cyclopts.App()


@APP.command(name="trigger")
def trigger() -> None:
    build_pipeline()

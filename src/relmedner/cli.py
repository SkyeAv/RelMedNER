import cyclopts

APP: cyclopts.App = cyclopts.App()

@APP.command(name="trigger")
def trigger() -> None:
    print("Hello from relmedner!")

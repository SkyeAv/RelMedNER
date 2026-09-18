from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    try:
        return version("relmedner")
    except PackageNotFoundError:
        return "dev"

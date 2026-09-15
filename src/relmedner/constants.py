from importlib.resources.abc import Traversable
from importlib.resources import files

DATA: Traversable = files("relmedner") / "data"

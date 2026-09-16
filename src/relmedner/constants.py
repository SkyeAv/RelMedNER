from importlib.resources.abc import Traversable
from importlib.resources import files

DATA: Traversable = files("relmedner") / "data"
INGESTS_AVRO: Traversable = DATA / "ingests.avro"
EXPANSION_SERVICE: str = "localhost:9097"

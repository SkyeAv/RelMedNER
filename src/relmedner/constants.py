from importlib.resources.abc import Traversable
from importlib.resources import files

DATA: Traversable = files("relmedner") / "data"
INGESTS_YAML: Traversable = DATA / "ingests.yaml"
EXPANSION_SERVICE: str = "localhost:9097"
TEST_ROW_LIMIT: int = 10

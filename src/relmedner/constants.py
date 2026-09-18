from importlib.resources import files
from importlib.resources.abc import Traversable

DATA: Traversable = files("relmedner") / "data"
COMPOSE_DIR: Traversable = files("relmedner") / "compose"
INGESTS_YAML: Traversable = DATA / "ingests.yaml"
CLUSTER_YAML: Traversable = DATA / "cluster.yaml"
EXPANSION_SERVICE: str = "localhost:9097"
TEST_ROW_LIMIT: int = 5
DEFAULT_OUTPUT: str = "./relmedner.avro"

FLINK_VERSION: str = "1.20"

FLINK_REST_PORT: int = 18081
JOBMANAGER_RPC_PORT: int = 16123
BLOB_SERVER_PORT: int = 16124
TASKMANAGER_DATA_PORT: int = 16125

WORKER_IMAGE_NAME: str = "localhost/relmedner-worker"
FLINK_IMAGE_NAME: str = "localhost/relmedner-flink"
LOCAL_HOST: str = "local"

PROJECT: str = "relmedner"
JOBMANAGER_COMPOSE: Traversable = COMPOSE_DIR / "docker-compose.jobmanager.yml"
TASKMANAGER_COMPOSE: Traversable = COMPOSE_DIR / "docker-compose.taskmanager.yml"
WORKER_DOCKERFILE: Traversable = COMPOSE_DIR / "Dockerfile.worker"
FLINK_DOCKERFILE: Traversable = COMPOSE_DIR / "Dockerfile.flink"

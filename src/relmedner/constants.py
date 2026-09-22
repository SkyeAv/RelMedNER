from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

DATA: Traversable = files("relmedner") / "data"
COMPOSE_DIR: Traversable = files("relmedner") / "compose"
INGESTS_YAML: Traversable = DATA / "ingests.yaml"
CLUSTER_YAML: Traversable = DATA / "cluster.yaml"
EXPANSION_SERVICE: str = "localhost:9097"
TEST_ROW_LIMIT: int = 5
DEFAULT_OUTPUT: str = "relmedner.avro"

FLINK_VERSION: str = "1.20"

FLINK_REST_PORT: int = 18081
JOBMANAGER_RPC_PORT: int = 16123
BLOB_SERVER_PORT: int = 16124
TASKMANAGER_DATA_PORT: int = 16125
WORKER_POOL_PORT: int = 50000  # fixed by beam's container boot.go (--service_port=50000)
FULLMAP_MOUNT: str = "/opt/fullmap"  # in-container mount target for the per-worker fullmap bundle
OUTPUTS_MOUNT: str = "/opt/outputs"  # in-container mount target the sdkworker writes avro shards into

WORKER_IMAGE_NAME: str = "localhost/relmedner-worker"
FLINK_IMAGE_NAME: str = "localhost/relmedner-flink"
LOCAL_HOST: str = "local"

PROJECT: str = "relmedner"
JOBMANAGER_COMPOSE: Traversable = COMPOSE_DIR / "docker-compose.jobmanager.yml"
TASKMANAGER_COMPOSE: Traversable = COMPOSE_DIR / "docker-compose.taskmanager.yml"
WORKER_DOCKERFILE: Traversable = COMPOSE_DIR / "Dockerfile.worker"
FLINK_DOCKERFILE: Traversable = COMPOSE_DIR / "Dockerfile.flink"

# Local fullmap database directory for resolution on the driver (no cluster mount).
FULLMAP_DIR: Path = Path("/home/skyeav/Desktop/fullmap")

# ---------------------------------------------------------------- fullmap mining knobs ----
# Every constant below was fixed by measurement against the live fullmap (2026jul22,
# Biolink 4.4.3) over ~1000 curated-corpus docs; see PLAN.md for the harness and numbers.

# CURIE namespaces whose gene/protein entries are model-organism-specific; excluded so
# gene hits are human-centric. TAXON_ID filtering alone cannot do this because fullmap
# retains TAXON_ID 0 rows (e.g. FB:FBgn0004015 "with", ZFIN "not", MGI "can").
NONHUMAN_PREFIXES: frozenset[str] = frozenset({"FB", "ZFIN", "MGI", "WB", "SGD", "POMBASE", "DICTYBASE", "RGD", "TAIR"})

# Closed-class English function words, the ONLY stoplist in the pipeline. Measured: zero
# real-entity losses when all-function-word n-grams and function-word unigrams are
# dropped. Frequency-derived stoplists were REJECTED -- they conflate common English with
# frequent domain heads and delete "chronic kidney disease" / "lung cancer" / "cell
# membrane". Never extend this list with corpus-frequency-derived words.
FUNCTION_WORDS: frozenset[str] = frozenset(
    """a an the and or but if then else when while of in on at by for with from to as is are
    was were be been being have has had do does did will would shall should can could may
    might must not no nor so than that this these those it its itself they them their theirs
    we us our ours you your yours he him his she her hers i me my mine who whom whose which
    what where why how all any both each few more most other some such only own same too very
    just also however therefore thus about above after again against all almost along already
    although always among around because before behind below beneath beside between beyond
    during either enough every except first further here himself however into latter least
    less like many much neither next nothing now once onto other others outside over per
    rather since still through throughout till toward towards under until up upon via whenever
    wherever whether within without yet""".split()
)

# biolink categories where an all-lowercase digit-free unigram surface is almost always an
# English word colliding with a symbol ("in" -> NCBIGene:3630 INS, "was" -> NCBIGene:7454);
# gene/protein symbols are conventionally not lowercase-only, so those surfaces are rejected.
GENELIKE_CATEGORIES: frozenset[str] = frozenset(
    {
        "Gene",
        "Protein",
        "Polypeptide",
        "GeneFamily",
        "ProteinFamily",
        "ProteinIsoform",
        "MicroRNA",
        "NoncodingRNAProduct",
        "RNAProduct",
        "GenomicEntity",
        "SequenceVariant",
        "Transcript",
        "CodingSequence",
    }
)

# biolink categories that fullmap populates with qualifier/indexing concepts (UMLS C* etc.)
# rather than the named entities NER training wants. Measured junk share at unigram level:
# InformationContentEntity 41% of all selections. Kept as a GATE only -- the categories
# remain legal in output whenever a mention clears the gates (user decision: no category
# allow-list baked into the data model).
JUNKY_CATEGORIES: frozenset[str] = frozenset(
    {
        "InformationContentEntity",
        "Publication",
        "Phenomenon",
        "Activity",
        "Behavior",
        "NamedThing",
        "PopulationOfIndividualOrganisms",
        "Cohort",
        "PhysicalEntity",
        "Agent",
        "Attribute",
        "Thing",
        "OntologyClass",
        "BiologicalEntity",
        "OrganismalEntity",
        "MolecularEntity",
        "ChemicalRole",
        "Event",
        "Process",
        "Occurrent",
        "Device",
        "DiagnosticAid",
        "ConfidenceLevel",
        "Onset",
        "SeverityValue",
        "Zygosity",
        "LifeStage",
        "Study",
        "Dataset",
        "Book",
        "Article",
        "JournalArticle",
        "Patent",
        "WebPage",
        "ClinicalIntervention",
        "Procedure",
        "ClinicalAttribute",
        "ClinicalMeasurement",
        "ClinicalCourse",
        "ClinicalModifier",
        "ClinicalTrial",
    }
)

# shortest accepted unigram mention unless it carries a digit ("CD63", "T2")
MIN_UNIGRAM_LENGTH: int = 3

# beam BatchElements bounds for one redb round trip per batch; ~54k distinct candidate
# surfaces per 100 docs and ~11.5us/key lookup make batches of rows (not terms) the unit.
MIN_BATCH_ROWS: int = 200
MAX_BATCH_ROWS: int = 2000

# ---------------------------------------------------------------- near-dedup knobs ----
# MinHash LSH constants for near-duplicate detection, fixed by the LSH S-curve
# P(pair shares >= 1 band) = 1 - (1 - s**r)**b for true shingle Jaccard s, r rows per band,
# b bands. At 8 bands x 16 rows the 50% point sits at (1/b)**(1/r) = 8**(-1/16) ~= 0.878:
# s = 0.95 collides with probability ~0.99, s = 0.98 with ~0.9999, s = 0.50 with ~0.0001.
# Deliberately conservative/high-precision: every dropped record is supervised signal, so
# near-dedup may only merge "super super similar" texts (same abstract re-ingested, trivial
# rewording), never half-overlapping ones.

# Fixed seed for the affine-permutation draw (a_j, b_j): signatures must be byte-identical
# across processes, runners, and workers. CPython's builtin hash() is salted per process and
# is forbidden anywhere in the dedup module; blake2b + random.Random(42) are stable.
DEDUP_SEED: int = 42

# text-dedup's recommended starting permutation count (datasketch default is 128 too);
# signature cost is O(num_perm) per record (~1 KB/record at 8-byte rows), negligible at the
# ingest registry's ~1.45M rows.
DEDUP_NUM_PERM: int = 128

# Band split of the 128-row signature; see the S-curve above: 8 bands x 16 rows puts the
# similarity bar for a band collision at s ~= 0.878 (50% point), i.e. high precision.
DEDUP_BANDS: int = 8

# Must stay DEDUP_NUM_PERM // DEDUP_BANDS; raising r shifts the whole S-curve right (fewer
# false merges, more misses). Each band key hashes these 16 rows together.
DEDUP_ROWS_PER_BAND: int = 16

# Texts shorter than this many tokens bypass near-dedup entirely: below ~10 tokens a word
# 5-gram shingle set has <6 members, so its MinHash Jaccard estimate is noise, not signal.
MIN_NEAR_TOKENS: int = 10

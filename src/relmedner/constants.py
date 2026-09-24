from importlib.resources import files
from importlib.resources.abc import Traversable
from os import environ
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

PROJECT: str = "relmedner"
JOBMANAGER_COMPOSE: Traversable = COMPOSE_DIR / "docker-compose.jobmanager.yml"
TASKMANAGER_COMPOSE: Traversable = COMPOSE_DIR / "docker-compose.taskmanager.yml"
WORKER_DOCKERFILE: Traversable = COMPOSE_DIR / "Dockerfile.worker"
FLINK_DOCKERFILE: Traversable = COMPOSE_DIR / "Dockerfile.flink"

# Local fullmap database directory for resolution on the driver (no cluster mount). Workers always
# read the mounted bundle instead: the sdkworker containers get RELMEDNER_FULLMAP_DIR=/opt/fullmap
# from the compose files, and the driver on the head host exports RELMEDNER_FULLMAP_DIR itself.
FULLMAP_DIR: Path = Path(environ.get("RELMEDNER_FULLMAP_DIR") or "/home/skyeav/Desktop/fullmap")

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

# gliner2-safe encoding of a negated statement: the relation NAME carries the negation
# (not_<predicate>) alongside negated=True, because the validator has no slot for a boolean
# flag. Shared by families (sampled dataset negatives) and the gazetteer (negation cues), so
# it lives here rather than in either: families imports gazetteer, and neither may define it.
NEGATIVE_NAME_PREFIX: str = "not_"

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
# script-path LRU bound on cached best fullmap rows (ScriptUtils._best_cache): each entry holds a
# normalized term plus three short strings (~0.3 KB), so 200k terms stays near 60 MB per worker
# while covering the recurring biomedical vocabulary a source repeats across rows
FULLMAP_BEST_CACHE_TERMS: int = 200_000
# miner-path LRU bound on per-token fold results (fullmap_mine._FOLD_CACHE): one entry is a token
# plus its folded parts (well under 0.2 KB), so 200k tokens stays near 40 MB per worker while
# covering every token the miner re-folds across the n-grams of millions of documents
FOLD_CACHE_TOKENS: int = 200_000
MAX_BATCH_ROWS: int = 2000

# ------------------------------------------------------- training-row token cap knobs ----
# Platform-wide cap on one training row's joined text, applied ALWAYS-ON by
# row_filters.first_drop_reason (independent of any declared RowFilters rules): the cap
# is a platform constant concern, not per-dataset config, so no ingests.yaml setting may waive it.

# Largest allowed text, in TOKENS. Chars-to-tokens conversion uses the OpenAI rule of thumb
# of ~4 characters per token for English
# (https://help.openai.com/en/articles/4936856-understanding-and-counting-tokens).
MAX_TEXT_TOKENS: int = 8192

# Chars assumed per token for the conversion above. Medical text runs longer words than
# general English, so its real chars/token sits above 4 and len(text) // CHARS_PER_TOKEN
# OVERestimates the true token count; the resulting early fire (a few rows a real tokenizer
# would keep) is the accepted medical-text error margin -- the cap is a cost guard, not
# tokenization. Compared in CHARS with strict > (never floor-divide first), so a 1-char
# overshoot still drops: floor division would round 32769 chars back to 8192 "tokens".
CHARS_PER_TOKEN: int = 4

# ------------------------------------------------- row-quality heuristic knobs ----
# Web-corpus QC heuristics (C4 / Gopher document-level filter family), wired as opt-in
# RowFilters rules by row_filters.first_drop_reason. The two knobs below are the shape
# constants of the pure ratio primitives; every THRESHOLD is a per-dataset decision in
# ingests.yaml, because every dropped record is supervised signal and repo convention fixes
# defaults only by measurement. See docs/quality-heuristics.md for measured starting values.

# Window (in words) of the duplicate-ngram repetition check (repeat_ngram_ratio). Gopher's
# document-level repetition rule uses 10-gram duplicate fraction; short texts under one
# window always pass.
REPEAT_NGRAM_WORDS: int = 10

# Line length under which a line counts as "short" for short_line_ratio (abnormal line
# breaks: newline spam, OCR fragments, bullet walls). A 30-char bar separates prose lines
# from fragments while leaving ordinary wrapped paragraphs at ratio 0.
SHORT_LINE_CHARS: int = 30

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

# text-dedup's recommended starting permutation count (datasketch default is 128 too). One
# signature costs O(num_perm * shingles), NOT O(num_perm): every permutation takes a minimum
# over every shingle hash. Measured in pure python on this repo's stack -- 0.60 ms for a
# 20-token record (16 shingles), 7.45 ms at the 233-token pubmed median (229 shingles), 49 ms
# at its 1,558-token max (1,554 shingles) -- so a full pass over the ~1.45M-row registry costs
# on the order of 180 core-minutes at median length, spread across Beam workers. The inner
# loop is now vectorized in numpy (dedup._affine_mod) with exact 32-bit-limb arithmetic that
# reproduces these signature bytes; tests/test_dedup.py pins equality against the pure-python
# reference, since a uint64 wraparound would silently change every near-dedup decision.
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

# ---------------------------------------------------------------- trust knobs ----
# Source-level trust (US-011): an offline sampled validation (validate.py, NEVER inside the
# Beam graph) scores a source's entities/relations against PubMed E-utilities and suggests a
# trust in [0, 1] that folds into the stamped weight as weight * trust, clamped to the fixed
# symmetric band below. Trust nudges; it cannot overturn a declared weight.

# Half-width of the trust adjustment band around the declared weight: trust-scaled weight is
# clamped to [w*(1-TRUST_RANGE), min(1.0, w*(1+TRUST_RANGE))]. Fixed for every source -- a
# wider or narrower band is a declared-weight change, not a trust change. 0.2 means the best
# a fully-trusted (trust=1.0) source gains is +20%, the worst a distrusted one loses is -20%.
TRUST_RANGE: float = 0.2

# Records sampled per source by `relmedner validate-trust` when the caller passes no size.
# Large enough that a source with ~20% bad rows shows a visibly sub-1.0 trust, small enough
# that 3 rps of PubMed E-utilities finishes a source in well under a minute.
TRUST_SAMPLE_SIZE: int = 50

# Relation hit grading (validators.grade_relation): >= 5 co-occurring PubMed documents is
# strong attestation, 1-4 is a real but possibly coincidental co-occurrence (half credit),
# 0 is unverified. Spans stay binary (one hit verifies) because a surface+label pair has no
# coincidental-middle case the way entity pairs do.
TRUST_RELATION_VERIFIED_HITS: int = 5
TRUST_RELATION_PARTIAL_HITS: int = 1

# NCBI E-utilities esearch endpoint (no key: 3 rps; NCBI_API_KEY env raises to 10 rps).
PUBMED_ESEARCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
# 3.4 requests/second ceiling under the no-key 3 rps policy (the 0.1 headroom absorbs jitter
# in NCBI's window accounting); with NCBI_API_KEY set the client uses a 0.11s delay for 9 rps.
PUBMED_THROTTLE_SECONDS: float = 0.29
PUBMED_THROTTLE_SECONDS_KEYED: float = 0.11

# ----------------------------------------------------------------------------- secondary labels --
# Morphological rules that attach a SECONDARY (never biolink) label to a mention whose surface
# carries the cue, teaching the model pharmacological / morphological classes the biolink
# categories cannot express. Every rule was measured against MedMentions ST21pv gold
# (203,282 spans): kept rules fire on >= 0.90 dominant-type concentration, dropped ones
# (-ase generic: 'disease'/'database'/'Embase' all fire it, 0.705; -algia 0.714; -cept 0.700)
# are absent by design. All checks are END-ANCHORED suffix/tail-word membership tests -- no
# regex -- because a 37-branch regex loop measures ~46us/span vs ~1.8us/span for endswith
# (measured, wenceslaus 2026-09-25). See docs/secondary-labels.md.

# last token (lowercased) endswith key -> secondary label; plural variants are explicit keys.
# measured concentration in parentheses (MedMentions gold dominant type).
SECONDARY_SUFFIX_LABELS: dict[str, str] = {
    "mab": "MonoclonalAntibodyDrug",
    "mabs": "MonoclonalAntibodyDrug",  # 0.989
    "nib": "KinaseInhibitorDrug",
    "nibs": "KinaseInhibitorDrug",  # 1.000
    "pril": "AceInhibitorDrug",
    "prils": "AceInhibitorDrug",  # 1.000
    "sartan": "AngiotensinAntagonist",
    "sartans": "AngiotensinAntagonist",  # 1.000
    "olol": "BetaBlockerDrug",
    "olols": "BetaBlockerDrug",  # 1.000
    "statin": "StatinDrug",
    "statins": "StatinDrug",  # 1.000
    "coxib": "Cox2InhibitorDrug",
    "coxibs": "Cox2InhibitorDrug",  # 1.000
    "gliptin": "GliptinDrug",
    "gliptins": "GliptinDrug",  # 1.000
    "caine": "LocalAnestheticDrug",
    "caines": "LocalAnestheticDrug",  # 0.935
    "dipine": "CalciumChannelBlocker",
    "dipines": "CalciumChannelBlocker",  # 1.000
    "prazole": "ProtonPumpInhibitor",
    "prazoles": "ProtonPumpInhibitor",  # 1.000
    "tidine": "H2AntagonistDrug",
    "tidines": "H2AntagonistDrug",  # 1.000
    "floxacin": "QuinoloneAntibiotic",
    "floxacins": "QuinoloneAntibiotic",  # 1.000
    "mycin": "AntibioticMycin",
    "mycins": "AntibioticMycin",  # 0.982
    "cillin": "PenicillinAntibiotic",
    "cillins": "PenicillinAntibiotic",  # 1.000
    "cycline": "TetracyclineAntibiotic",
    "cyclines": "TetracyclineAntibiotic",  # 1.000
    "tide": "PeptideDrug",
    "tides": "PeptideDrug",  # 0.975
    "itis": "InflammatoryDisease",  # 0.991
    "oma": "NeoplasticProcess",
    "omas": "NeoplasticProcess",  # 0.960
    "emia": "BloodCellDisease",
    "emias": "BloodCellDisease",  # 0.924
    "pathy": "OrganDisease",
    "pathys": "OrganDisease",  # 0.958
    "ectomy": "SurgicalRemoval",
    "ectomys": "SurgicalRemoval",  # 0.992
    "oscopy": "Endoscopy",
    "oscopies": "Endoscopy",  # 0.953
    "plasty": "SurgicalRepair",
    "plasties": "SurgicalRepair",  # 0.988
    "therapy": "TherapyProcedure",  # 0.971 ('therapies' is a different word, kept off)
    "cyte": "CellType",
    "cytes": "CellType",  # 0.985
}

# exact last-token / last-two-word phrases -> secondary label; no length gate, no suffix scan.
# 'insulin' etc. are LAST-word checks so 'insulin resistance' (a disease span) never fires.
SECONDARY_TAIL_LABELS: dict[str, str] = {
    "insulin": "InsulinDrug",
    "insulins": "InsulinDrug",  # 0.966
    "vaccine": "Vaccine",
    "vaccines": "Vaccine",  # 0.979
    "interferon": "InterferonDrug",
    "interferons": "InterferonDrug",  # 1.000
    "biopsy": "BiopsyAssay",
    "biopsies": "BiopsyAssay",  # 0.981
    "receptor": "ReceptorProtein",
    "receptors": "ReceptorProtein",  # 0.895
    "gene": "GeneMention",
    "genes": "GeneMention",  # 0.937
    "growth factor": "GrowthFactorProtein",
    "growth factors": "GrowthFactorProtein",  # 0.917
}

# tokens that must never fire a suffix rule: measured off-type producers (MedMentions FP samples)
SECONDARY_SUFFIX_STOP: frozenset[str] = frozenset(
    "disease diseases ease increase decrease release lease crease tease case base phase "
    "chase erase vase purchase rease embase database databases myostatin myostatins trauma "
    "comma dogma drama aroma enigma karma llama magma glycemia glycaemia academia empathy "
    "telepathy nucleotide nucleotides oligonucleotide encephalitides".split()
)

# minimum last-token length for a suffix rule (the measured rules carried two or more characters
# before the stem, so a 6-char floor keeps 'cocaine'-class hits while 'asa'-style fragments die)
SECONDARY_MIN_TOKEN: int = 6

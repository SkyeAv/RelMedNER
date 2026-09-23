from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self
from uuid import uuid4

from dataclasses_avroschema.pydantic import AvroBaseModel
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tablassert.biolink import Predicates

from relmedner.constants import DEFAULT_OUTPUT, TEST_ROW_LIMIT, TRUST_SAMPLE_SIZE
from relmedner.enums import DedupMode, OutputShapes, ProcessingTypes

TrustScore = Annotated[float, Field(ge=0.0, le=1.0)]
"""one trust value in [0, 1]; a named alias so trust/trust_edges carry the bound in the
schema without duplicated Annotated expressions"""


class StrictBase(AvroBaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    class Meta:
        namespace: str = "relmedner.ingests"

    @staticmethod
    def freeze(value: Any) -> Any:
        if isinstance(value, StrictBase):
            return value.to_tuple()
        if isinstance(value, list):
            return tuple(StrictBase.freeze(item) for item in value)
        return value

    def to_tuple(self: Self) -> tuple[Any, ...]:
        return tuple(self.freeze(getattr(self, name)) for name in type(self).model_fields)


class RunConfig(StrictBase):
    sample_limit: int | None = Field(None)
    output: str = Field(DEFAULT_OUTPUT)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    dedup_mode: DedupMode = Field(DedupMode.NEAR)
    """repeats must not reach training data unless dedup is explicitly disabled (default ON).
    use_enum_values stores the StrEnum's str value, which still compares equal to the member"""

    def artifact_name(self: Self) -> str:
        Output: Path = Path(self.output)
        return f"{Output.stem}-{self.run_id}{Output.suffix}"

    @classmethod
    def from_flags(cls, test_run: bool, output: str = DEFAULT_OUTPUT, dedup_mode: DedupMode = DedupMode.NEAR) -> Self:
        """new flags append at the end; the first two positional args keep their established
        order so existing callers (tests, cli) cannot silently shift meaning"""
        return cls(sample_limit=TEST_ROW_LIMIT if test_run else None, output=output, dedup_mode=dedup_mode)


class TaskBase(StrictBase):
    type: ProcessingTypes = Field(...)


class ScriptTask(TaskBase):
    type: Literal[ProcessingTypes.SCRIPT] = Field(...)
    name: str = Field(...)
    outputs: list[OutputShapes] = Field(..., min_length=1)


class FullmapTask(TaskBase):
    """distant-supervision entity/relation mining over unlabeled text via the fullmap redb

    Measured-good gates (function-word guards, GENELIKE casing rule, strict unigram name
    agreement, junk-category gate, NONHUMAN_PREFIXES exclusion) live as constants in
    relmedner.constants / relmedner.fullmap_mine with the measurements that fixed them;
    add fields here later to make any of them tunable.
    """

    type: Literal[ProcessingTypes.FULLMAP] = Field(...)
    max_ngram: int = Field(6, ge=1, le=10)
    taxon: str = Field("9606")
    relations: bool = Field(True)
    outputs: list[OutputShapes] = Field(..., min_length=1)


Task: Annotated = Annotated[
    ScriptTask | FullmapTask,
    Field(discriminator="type"),
]


class RowFilters(StrictBase):
    """declarative per-dataset row filters; kept OUT of the frozen payload tuple (NON_PAYLOAD_FIELDS)
    and handed to the stream ctor as a keyword so filter changes never shift tuple positions.
    The drop decision itself is the pure evaluator relmedner.row_filters.first_drop_reason"""

    drop_empty: bool = Field(False)
    """drop rows where every projected value is None, "", or an empty list/tuple/dict"""
    min_text_len: int | None = Field(None, ge=0)
    """drop rows whose joined text is shorter than this"""
    max_text_len: int | None = Field(None, ge=0)
    """drop rows whose joined text is longer than this"""
    include_regex: str | None = Field(None)
    """drop rows whose joined text does NOT match this pattern"""
    exclude_regex: str | None = Field(None)
    """drop rows whose joined text DOES match this pattern"""

    @model_validator(mode="after")
    def min_within_max(self: Self) -> Self:
        if self.min_text_len is not None and self.max_text_len is not None and self.min_text_len > self.max_text_len:
            raise ValueError(f"min_text_len {self.min_text_len} exceeds max_text_len {self.max_text_len}")
        return self


class DatasetBase(StrictBase):
    NON_PAYLOAD_FIELDS: ClassVar[frozenset[str]] = frozenset({"source", "filters", "trust", "trust_edges"})
    """names that never enter the packed payload: "source" is the dict key today; "filters",
    "trust", and "trust_edges" are validation-time keyword-only fields, so none can shift an
    existing tuple position"""

    tuple_fields: ClassVar[tuple[str, ...]]
    """the explicit field packing order frozen by tests/test_ingests.py EXPECTED locks; concrete
    dataset models declare it so a new model field is an opt-in tuple change, never an accidental
    position shift"""

    task: Task = Field(...)
    weight: float = Field(1.0, ge=0.0)
    """per-source mixing weight stamped onto every TrainingExample the source emits. Stock
    gliner2 has no per-example weight channel (InputExample/from_dict/ExtractorDataset all
    drop it), so consumption is weighted duplication at avro->JSONL export, not in-training.
    0 is legal and means soft drop: the record still reaches the avro provenance but
    duplicates zero times at export -- discouraged (docs/weighting.md), prefer trust: 0 or
    row filters for unwanted data"""

    trust: float = Field(1.0, ge=0.0, le=1.0)
    """source-level trust score in [0, 1] suggested by the offline validation step
    (`relmedner validate-trust`, see docs/weighting.md); folds into the stamped weight as
    weight * trust clamped to the fixed +-TRUST_RANGE band (validators.adjust_weight).
    1.0 (default) = no adjustment; 0 = explicit soft drop bypassing the band. Kept OUT of
    the frozen payload tuple (NON_PAYLOAD_FIELDS) like filters, so it shifts no position"""

    trust_edges: dict[str, TrustScore] | None = Field(None)
    """per-predicate edge trust: relation NAME -> trust in [0, 1], the heuristic application
    of sampled validation to the WHOLE dataset -- a predicate the sample found unreliable is
    down-weighted on every record carrying it, untouched records keep the source weight
    (validators.record_edge_factor takes the weakest flagged predicate on the record). 0
    soft-drops records carrying that edge. Keyword-only, never in the frozen payload tuple"""

    filters: RowFilters | None = Field(None)
    """declarative row filters applied by the stream after match_on; appended LAST (after trust)
    and excluded from tuple_fields, so it cannot shift any frozen payload position"""

    @property
    def row_key(self: Self) -> str:
        """the name streamed rows are stamped with (DataStream.rows yields it per row); the
        pipeline maps it to the declared weight before the source key is dropped"""
        raise NotImplementedError

    def to_tuple(self: Self) -> tuple[str, tuple[Any, ...]]:
        return (self.source, tuple(self.freeze(getattr(self, name)) for name in type(self).tuple_fields))

    def to_stream_args(self: Self) -> tuple[str, tuple[Any, ...], RowFilters | None]:
        """the (source, payload, filters) envelope build_stream unpacks; the payload stays the
        frozen 2-tuple shape cli.py and the EXPECTED locks depend on, filters ride keyword-only"""
        source, payload = self.to_tuple()
        return (source, payload, self.filters)


class MatchOn(StrictBase):
    column: str = Field(...)
    values: list[str] = Field(...)


class HuggingFaceDataset(DatasetBase):
    tuple_fields: ClassVar[tuple[str, ...]] = ("task", "weight", "dataset", "subset", "split", "match_on", "columns_out")

    source: Literal["hf"] = Field(...)
    dataset: str = Field(...)
    subset: str | None = Field(None)
    split: str | None = Field(None)
    match_on: list[MatchOn] | None = Field(None)
    columns_out: list[str] = Field(...)

    @property
    def row_key(self: Self) -> str:
        return self.dataset


class LocalAvroDataset(DatasetBase):
    """an avro container built out-of-band and read from disk by LocalAvroDataStream

    path is resolved with expanduser at stream time; the whole avro record ships to the declared
    script, which owns the record shape (there is no columns_out projection like the hf streams
    have, because the file's own schema is already the contract)
    """

    tuple_fields: ClassVar[tuple[str, ...]] = ("task", "weight", "path")

    source: Literal["local"] = Field(...)
    path: str = Field(...)

    @property
    def row_key(self: Self) -> str:
        return self.path


class LocalDelimitedDataset(DatasetBase):
    """a header-delimited file (TSV/CSV) read from disk by LocalDelimitedDataStream

    A separate source kind from "local" because the two contracts differ: avro ships whole
    records and the file's schema is the contract, while a delimited file's header row makes
    columns_out a real projection. Relative paths resolve against the caller's CWD when they
    exist there, else against the package data dir, so in-repo corpora work from any CWD.
    """

    tuple_fields: ClassVar[tuple[str, ...]] = ("task", "weight", "path", "columns_out", "match_on")

    source: Literal["local_delimited"] = Field(...)
    path: str = Field(...)
    columns_out: list[str] = Field(..., min_length=1)
    match_on: list[MatchOn] | None = Field(None)

    @property
    def row_key(self: Self) -> str:
        return self.path


class HuggingFaceJsonDataset(DatasetBase):
    """json-builder ingest over an hf:// URL inside one hub repo file; see relmedner.hf_json for the
    two measured blockers (old-style dataset_infos.json, cold-cache streaming corruption) that keep
    this route out of the "hf" source"""

    tuple_fields: ClassVar[tuple[str, ...]] = ("task", "weight", "dataset", "file", "split", "match_on", "columns_out")

    source: Literal["hf_json"] = Field(...)
    dataset: str = Field(...)
    file: str = Field(..., min_length=1)
    """file name inside the hub repo; min_length keeps the hf://datasets/{dataset}/{file} URL well formed"""
    split: str | None = Field(None)
    match_on: list[MatchOn] | None = Field(None)
    columns_out: list[str] = Field(...)

    @property
    def row_key(self: Self) -> str:
        """the stream stamps rows with the repo id alone, so two hf_json entries over the same
        repo (different files) share one weight slot; weights_by_source raises if they disagree"""
        return self.dataset


Dataset: Annotated = Annotated[
    HuggingFaceDataset | LocalAvroDataset | LocalDelimitedDataset | HuggingFaceJsonDataset,
    Field(discriminator="source"),
]


def _reject_malformed_phrases(owner: str, phrases: list[list[str]]) -> None:
    """shared phrase rules for every gazetteer trigger/cue table: an empty phrase or token can
    never match and only masks an authoring bug, and an uppercase token can never match the
    lowercased scan input (mirrors gazetteer.validate_trigger_table)"""
    for phrase in phrases:
        if not phrase:
            raise ValueError(f"{owner} has an empty phrase")
        for token in phrase:
            if not token:
                raise ValueError(f"{owner} phrase {phrase!r} contains an empty token")
            if token != token.lower():
                raise ValueError(f"{owner} phrase {phrase!r} contains uppercase token {token!r}")


class GazetteerPredicate(StrictBase):
    """one YAML-declared predicate arm of the relation gazetteer (US-010). WHY a model instead
    of raw dicts: the name must be a tablassert.biolink.Predicates member (mirroring
    gazetteer.validate_trigger_table) so a typo'd predicate fails at parse time instead of
    emitting KGX edges that fail Biolink validation downstream"""

    name: str = Field(...)
    triggers: list[list[str]] = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    def name_is_a_biolink_predicate(cls, value: str) -> str:
        valid: frozenset[str] = frozenset(predicate.value for predicate in Predicates)
        if value not in valid:
            raise ValueError(f"predicate {value!r} is not a tablassert.biolink.Predicates member")
        return value

    @field_validator("triggers")
    @classmethod
    def triggers_are_wellformed(cls, value: list[list[str]]) -> list[list[str]]:
        _reject_malformed_phrases("a YAML-declared predicate", value)
        return value


class GazetteerQualifier(StrictBase):
    """one qualifier arm; ONLY structurally validated in this tree. The qualifier scanner
    machinery (QUALIFIER_TRIGGERS, QUALIFIER_RANGES, DISABLED_QUALIFIERS) lives on the
    add-qualifiers-to-relationship-pipelines branch and lands with PR #22, so declaring
    qualifiers in ingests.yaml raises a structured NotImplementedError at configure time
    (gazetteer.configure_gazetteer) instead of being silently accepted and ignored"""

    slot: str = Field(...)
    range: str | None = Field(None)
    triggers: list[list[str]] = Field(default_factory=list)

    @field_validator("slot")
    @classmethod
    def slot_is_named(cls, value: str) -> str:
        if not value:
            raise ValueError("a YAML-declared qualifier has an empty slot")
        return value

    @field_validator("triggers")
    @classmethod
    def triggers_are_wellformed(cls, value: list[list[str]]) -> list[list[str]]:
        _reject_malformed_phrases("a YAML-declared qualifier", value)
        return value


class GazetteerSpec(StrictBase):
    """the optional top-level `gazetteer:` section of ingests.yaml (US-010). Predicates merge
    additively over the builtin trigger table (see gazetteer.configure_gazetteer); qualifiers
    and negation_cues parse and validate structurally but raise at configure time until PR #22
    lands the scanner -- fail-loud, never silent accept-and-ignore"""

    predicates: list[GazetteerPredicate] | None = Field(None)
    qualifiers: list[GazetteerQualifier] | None = Field(None)
    negation_cues: list[list[str]] | None = Field(None)

    @field_validator("negation_cues")
    @classmethod
    def negation_cues_are_wellformed(cls, value: list[list[str]] | None) -> list[list[str]] | None:
        if value is not None:
            _reject_malformed_phrases("YAML-declared negation_cues", value)
        return value

    @model_validator(mode="after")
    def the_section_declares_something(self: Self) -> Self:
        if not (self.predicates or self.qualifiers or self.negation_cues):
            raise ValueError("gazetteer section is empty: declare predicates, qualifiers, or negation_cues")
        return self

    @model_validator(mode="after")
    def yaml_phrases_have_one_owner(self: Self) -> Self:
        """two YAML predicates claiming one phrase would make the longest-match winner depend on
        merge order; builtin-vs-YAML ownership is rejected later, at configure time, where the
        builtin table is reachable without a circular models<->gazetteer import"""
        owners: dict[tuple[str, ...], str] = {}
        for entry in self.predicates or []:
            for phrase in entry.triggers:
                key = tuple(phrase)
                owner = owners.setdefault(key, entry.name)
                if owner != entry.name:
                    raise ValueError(f"phrase {key!r} is claimed by both {owner!r} and {entry.name!r}")
        return self


class ValidateTrustConfig(StrictBase):
    """the optional top-level `x-trust:` section of ingests.yaml: driver settings for the
    offline literature validation (relmedner validate-trust, docs/weighting.md). CLI flags
    override these one-for-one, so the yaml holds the per-repo default and the command line
    holds the one-off experiment. Secrets (NCBI/Firecrawl keys) NEVER ride here -- env only,
    ingests.yaml is a committed artifact"""

    sample_size: int = Field(TRUST_SAMPLE_SIZE, ge=1)
    """records sampled per source"""
    backend: str = Field("pubmed")
    """'pubmed' (E-utilities esearch, primary) or 'firecrawl' (self-hosted general-web fallback)"""
    report: str = Field("trust-report.jsonl")
    """JSONL report path"""

    @field_validator("backend")
    @classmethod
    def backend_is_known(cls, value: str) -> str:
        # a plain str with an explicit membership check, not Literal: Literal renders fine in
        # the pydantic JSON Schema but dataclasses-avroschema cannot map it for avro schema
        # generation (mirrors GazetteerPredicate.name_is_a_biolink_predicate)
        if value not in ("pubmed", "firecrawl"):
            raise ValueError(f"backend {value!r} is not 'pubmed' or 'firecrawl'")
        return value


class YamlIngests(StrictBase):
    """compose-spec x- extension namespace: a top-level "x-defaults" map hosts reusable YAML anchor
    definitions document-wide; CSafeLoader resolves the anchors before pydantic sees anything, so the
    parsed value is stored but never read by loader code (extra="forbid" stays intact otherwise)"""

    datasets: list[Dataset] = Field(...)
    x_defaults: dict[str, Any] | None = Field(None, alias="x-defaults")
    gazetteer: GazetteerSpec | None = Field(None)
    """optional relation-gazetteer overlay (US-010); appended LAST, after x_defaults, and kept
    out of generate_tuples/stream_args, so it shifts no frozen dataset payload position"""
    x_trust: ValidateTrustConfig | None = Field(None, alias="x-trust")
    """optional validate-trust driver settings (US-011); appended LAST, after gazetteer, kept
    out of generate_tuples/stream_args. CLI flags override these values one-for-one"""

    class Meta(StrictBase.Meta):
        # dataclasses-avroschema cannot map dict[str, Any] (no typing.Any arm exists, schema
        # generation raises "unknown type"), but x_defaults is a validation-only YAML
        # convenience and YamlIngests is never avro-serialized, so opt it out of the
        # generated schema instead of narrowing the annotation
        exclude = ["x_defaults"]

    def generate_tuples(self: Self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        """KEPT at the frozen 2-tuple shape: cli.py parallelism count and the test_ingests.py
        entry[1][2] helper index the payload positionally"""
        return tuple(dataset.to_tuple() for dataset in self.datasets)

    def stream_args(self: Self) -> tuple[tuple[str, tuple[Any, ...], RowFilters | None], ...]:
        """the pipeline's Create stage feeds build_stream, which unpacks each 3-tuple as
        (source, payload, filters)"""
        return tuple(dataset.to_stream_args() for dataset in self.datasets)

    def weights_by_source(self: Self) -> dict[str, float]:
        """row key -> declared mixing weight; rows key on the source's repo id (not the "hf"
        discriminator), so two entries sharing a repo id must agree on the weight or the stamp
        would be ambiguous"""
        weights: dict[str, float] = {}
        for dataset in self.datasets:
            if weights.setdefault(dataset.row_key, dataset.weight) != dataset.weight:
                raise ValueError(f"dataset {dataset.row_key!r} is declared twice with conflicting weights")
        return weights

    def trusts_by_source(self: Self) -> dict[str, float]:
        """row key -> declared trust; mirrors weights_by_source's shared-key agreement raise:
        two entries over one row key must agree on trust or the stamped weight would be
        ambiguous (weight * trust is computed once per row key, not per entry)"""
        trusts: dict[str, float] = {}
        for dataset in self.datasets:
            if trusts.setdefault(dataset.row_key, dataset.trust) != dataset.trust:
                raise ValueError(f"dataset {dataset.row_key!r} is declared twice with conflicting trusts")
        return trusts

    def trust_edges_by_source(self: Self) -> dict[str, dict[str, float]]:
        """row key -> {predicate: trust}; entries sharing one row key MERGE their maps (the two
        Nemotron splits may each flag different predicates) but the same predicate twice with
        different values raises, matching weights_by_source's ambiguity rule"""
        edges: dict[str, dict[str, float]] = {}
        for dataset in self.datasets:
            if dataset.trust_edges is None:
                continue
            slot: dict[str, float] = edges.setdefault(dataset.row_key, {})
            for predicate, score in dataset.trust_edges.items():
                if predicate in slot and slot[predicate] != score:
                    raise ValueError(f"dataset {dataset.row_key!r} declares conflicting trust for predicate {predicate!r}")
                slot[predicate] = score
        return edges


class Entity(StrictBase):
    label: str = Field(...)
    mentions: list[str] = Field(...)
    description: str | None = Field(None)


class Description(StrictBase):
    key: str = Field(...)
    description: str = Field(...)


class ChoiceField(StrictBase):
    value: str = Field(...)
    choices: list[str] = Field(...)

    def to_output(self: Self) -> dict[str, Any]:
        return {"value": self.value, "choices": self.choices}


class StructureField(StrictBase):
    name: str = Field(...)
    value: str | list[str] | ChoiceField = Field(...)
    description: str | None = Field(None)

    def to_value(self: Self) -> Any:
        return self.value.to_output() if isinstance(self.value, ChoiceField) else self.value


class Structure(StrictBase):
    name: str = Field(...)
    fields: list[StructureField] = Field(...)


class Classification(StrictBase):
    task: str = Field(...)
    labels: list[str] = Field(...)
    true_label: list[str] = Field(..., min_length=1)
    multi_label: bool = Field(False)
    prompt: str | None = Field(None)
    label_descriptions: list[Description] | None = Field(None)


class RelationField(StrictBase):
    name: str = Field(...)
    value: str = Field(...)


class Relation(StrictBase):
    name: str = Field(...)
    fields: list[RelationField] = Field(...)
    description: str | None = Field(None)
    """biolink slot definition for the predicate (ScriptUtils.predicate_description); emitted as
    relation_descriptions and consumed by gliner2's processor as a label prompt"""
    negated: bool = Field(False)
    """biolink Association.negated: True asserts the relation is false; this pipeline never
    asserts negations, so every emitted relation carries False (post-training-plan decision)"""
    evidence: str = Field("asserted")
    """how the triple was observed: asserted (gold spans), distant (fullmap-mined spans),
    sampled_negative (grid-sampled non-observation from the post-training corpus)"""


def describe(descriptions: list[Description] | None) -> dict[str, str]:
    return {entry.key: entry.description for entry in descriptions or []}


class TrainingExample(StrictBase):
    text: str = Field(...)
    weight: float = Field(1.0, ge=0.0)
    """effective (trust-adjusted) mixing weight; rides the avro record as provenance and stays out of the
    gliner2 to_output() projection -- stock gliner2 silently drops extra keys, so the actual
    training-time consumption is weighted duplication at the avro->JSONL export step. ge=0.0
    because trust == 0 soft-drops a record: it still ships to avro but duplicates zero times"""
    entities: list[Entity] = Field(default_factory=list)
    classifications: list[Classification] = Field(default_factory=list)
    structures: list[Structure] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def populated(self: Self) -> frozenset[str]:
        return frozenset(shape for shape in OutputShapes if getattr(self, shape))

    def entities_out(self: Self) -> dict[str, Any]:
        described: list[Description] = [
            Description(key=entity.label, description=entity.description) for entity in self.entities if entity.description
        ]
        output: dict[str, Any] = {"entities": {entity.label: entity.mentions for entity in self.entities}}
        return output | ({"entity_descriptions": describe(described)} if described else {})

    def classifications_out(self: Self) -> dict[str, Any]:
        return {
            "classifications": [
                {"task": task.task, "labels": task.labels, "true_label": task.true_label}
                | ({"multi_label": True} if task.multi_label else {})
                | ({"prompt": task.prompt} if task.prompt else {})
                | ({"label_descriptions": describe(task.label_descriptions)} if task.label_descriptions else {})
                for task in self.classifications
            ]
        }

    def structures_out(self: Self) -> dict[str, Any]:
        described: dict[str, dict[str, str]] = {
            structure.name: describe([Description(key=field.name, description=field.description) for field in structure.fields if field.description])
            for structure in self.structures
            if any(field.description for field in structure.fields)
        }
        output: dict[str, Any] = {
            "json_structures": [{structure.name: {field.name: field.to_value() for field in structure.fields}} for structure in self.structures]
        }
        return output | ({"json_descriptions": described} if described else {})

    def relations_out(self: Self) -> dict[str, Any]:
        # negated/evidence deliberately stay OUT of the gliner2 projection: gliner2's
        # Relation(name, **fields) swallows extra keys into _fields (dropping head/tail on
        # round-trip) and InputExample.validate() requires every relation value to occur in
        # the text. Provenance rides in the avro records (asdict) instead. Descriptions DO
        # ride the projection: the processor consumes them as label prompts.
        described: list[Description] = [
            Description(key=relation.name, description=relation.description) for relation in self.relations if relation.description
        ]
        output: dict[str, Any] = {
            "relations": [{relation.name: {field.name: field.value for field in relation.fields}} for relation in self.relations]
        }
        return output | ({"relation_descriptions": describe(described)} if described else {})

    def to_output(self: Self) -> dict[str, Any]:
        populated: frozenset[str] = self.populated()
        output: dict[str, Any] = {}
        for shape in OutputShapes:
            if shape in populated:
                output |= getattr(self, f"{shape}_out")()
        return {"input": self.text, "output": output}


class WorkerNode(StrictBase):
    host: str = Field(...)
    slots: int = Field(..., ge=1)
    memory: str = Field(...)

    fullmap: str = Field(...)
    """host directory holding the fullmap redb bundle (primary + shards), mounted read-only into the sdkworker"""

    outputs: str = Field(...)
    """host directory the sdkworker writes avro shards into; collected back to the laptop after a run"""

    polars_runtime: Literal["32", "64", "compat"] = Field("32")
    """POLARS_FORCE_PKG for this host's sdkworker: "compat" on CPUs without AVX2/FMA/BMI2, where the
    default runtime dies with SIGILL; the image ships both via tablassert[rt]"""


class Cluster(StrictBase):
    ssh_user: str = Field(...)
    jobmanager: str = Field(...)
    """head host running the jobmanager stack; deploy, the beam driver, and shard collection all
    run from this host's checkout, and it also carries a taskmanager + sdkworker of its own"""
    workers: list[WorkerNode] = Field(...)

    @model_validator(mode="after")
    def _jobmanager_is_a_worker(self: Self) -> Self:
        if self.jobmanager not in {worker.host for worker in self.workers}:
            raise ValueError(f"jobmanager host {self.jobmanager!r} is not a declared worker")
        return self


class FlinkJob(BaseModel):
    """one entry of the jobmanager's /jobs/overview payload; extra fields are ignored on purpose"""

    model_config: ConfigDict = ConfigDict(frozen=True, extra="ignore")

    jid: str = Field(...)
    name: str = Field("")
    state: str = Field(...)

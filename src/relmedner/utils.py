from __future__ import annotations

import ast
import json
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Self

from tablassert import rs
from tablassert.biolink import Categories, Predicates
from tablassert.fullmap import fullmap_db_path, lookup_rows

from relmedner.constants import FULLMAP_BEST_CACHE_TERMS, FULLMAP_DIR
from relmedner.models import Entity


def package_version() -> str:
    try:
        return version("relmedner")
    except PackageNotFoundError:
        return "dev"


BIOLINK_PREFIX: str = "biolink:"


def strip_biolink_prefix(value: str) -> str:
    return value[len(BIOLINK_PREFIX) :] if value.startswith(BIOLINK_PREFIX) else value


@dataclass(frozen=True)
class ResolvedMention:
    """one NER span after the fullmap/fallback/raw resolution chain"""

    mention: str
    category: str
    curie: str | None = None
    preferred_name: str | None = None
    origin: str = "raw"  # "fullmap" | "fallback" | "raw"


def _bucket(keys: str, categories: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """parse a ResolutionGate BUCKETS entry: space-separated substrings + allowed biolink ancestors"""
    return tuple(keys.split()), frozenset(categories.split())


LABEL_SPLIT: re.Pattern[str] = re.compile(r"[\s_/\-]+")
ACRONYM_MENTION: re.Pattern[str] = re.compile(r"^[A-Z0-9][A-Z0-9\-/]{0,3}$")
# Pile-NER-type conversation turns: the document rides a "Text: " prefix and every entity type is
# asked for with one templated question whose answer turn carries the JSON mention list
CONVERSATION_TEXT_PREFIX: str = "Text: "
CONVERSATION_QUESTION: re.Pattern[str] = re.compile(r"^What describes (.+?) in the text\?$")
HUMAN_TURN: str = "human"
GPT_TURN: str = "gpt"
MODEL_ORGANISM_PREFIXES: tuple[str, ...] = ("FB:", "ZFIN:", "WB:", "MGI:", "SGD:", "RGD:", "DICTYBASE:", "POMBASE:", "TAIR:", "XENBASE:")


class ResolutionGate:
    """rejects fullmap hits that contradict the source corpus's own label;
    shared by every Script so the gate improves all ingests at once.

    Rejections fall through to the fallback map / raw label (the mention is never dropped).
    Three independent checks, each measured against the Pile-NER-biomed-IOB probe:

    1. model-organism CURIE guard -- human biomed text rarely means the Drosophila gene "flu"
    2. label<->category bucket compatibility -- "organization" must not resolve to SmallMolecule
    3. acronym guard -- short uppercase mentions over catch-all labels are fullmap collision bait
    """

    # catch-all labels the corpus uses as placeholders; they carry no semantic signal in either
    # direction, so they never trigger the bucket check and only gate the acronym check
    OPEN_LABELS: ClassVar[frozenset[str]] = frozenset(
        {
            "concept",
            "entity",
            "entity type",
            "other",
            "biological",
            "biological entity",
            "biomolecule",
            "medical",
            "medical term",
            "medical concept",
            "thing",
            "item",
            "term",
            "object",
            "type",
            "name",
            "structure",
            "component",
            "factor",
            "element",
            "system",
            "field",
            "biological_entity",
            "material",
            "product",
            "technology",
            "method",
            "activity",
            "event",
            "data",
            "value",
            "level",
            "number",
            "time",
            "date",
            "age",
            "study",
            "class",
            "category",
            "attribute value",
        }
    )

    # label-substring -> allowed biolink ancestors; an unknown label is ungated (no opinion).
    # entries are space-separated for readability and split at import time
    BUCKETS: ClassVar[dict[str, tuple[tuple[str, ...], frozenset[str]]]] = {
        "chemical": _bucket(
            "chemical drug compound substance molecule medication nutrient metabolite reagent hormone toxin ingredient pharmac",
            "ChemicalEntity MolecularEntity Drug Food ChemicalMixture Treatment Protein Polypeptide",
        ),
        # generic/brand (BioLeaflets generic_name/brand_name drug labels) added by measurement:
        # 'brand name' triggered no bucket and let Protein (368 spans) and other non-chemical
        # classes land on drug labels; this bucket demands chemical compatibility. 'generic name'
        # keeps a residual hole: the gene bucket's 'gene' key substring-matches inside 'generic'
        # and is_label_compatible accepts when ANY triggered bucket allows, so Protein/GeneFamily
        # stay reachable there (585 measured spans) -- closing it needs an any->all change to the
        # bucket composition rule, which is not an additive bucket change
        "drugname": _bucket(
            "generic brand",
            "Drug ChemicalMixture MolecularEntity ChemicalEntity Food Treatment",
        ),
        "gene": _bucket(
            "gene protein enzyme peptide receptor transcript rna dna cytokine antibody antigen kinase mutation genetic variation variant allele",
            "GenomicEntity Gene Protein Polypeptide NucleicAcidEntity MacromolecularComplex"
            " ChemicalEntity GeneFamily ProteinFamily ProteinDomain SequenceVariant",
        ),
        "disease": _bucket(
            # dx/problem (BioLeaflets dx_name/problem, normalized) added by measurement: with no
            # bucket opinion those labels let fullmap land InformationContentEntity (2,274 spans),
            # AnatomicalEntity (220), Gene (257: 226 dx + 31 problem), and Publication (87) on
            # disease-labeled spans
            "disease disorder condition syndrome symptom illness pathology injury infection cancer tumor psychopathology phenotype sign dx problem",
            "DiseaseOrPhenotypicFeature Disease PhenotypicFeature PathologicalProcess"
            " ClinicalFinding BiologicalProcess OrganismTaxon Phenomenon ClinicalAttribute",
        ),
        "anatomy": _bucket(
            "anatom body part organ tissue cell region body fluid gland muscle bone",
            "AnatomicalEntity Cell CellLine GrossAnatomicalStructure CellularComponent OrganismalEntity MacromolecularComplex",
        ),
        "organism": _bucket(
            "organism species animal plant bacteri virus fungus fungi pathogen microorganism taxon strain",
            "OrganismTaxon OrganismalEntity IndividualOrganism Virus Bacterium Fungus Plant Cell ChemicalEntity PopulationOfIndividualOrganisms",
        ),
        "process": _bucket(
            "process pathway mechanism function physiolog metabolism",
            "BiologicalProcessOrActivity BiologicalProcess Pathway MolecularActivity"
            " PhysiologicalProcess Activity Behavior Phenomenon PathologicalProcess",
        ),
        "procedure": _bucket(
            "procedure treatment therapy test intervention surgery assay technique diagnos screening examination imaging vaccination",
            "Procedure Treatment ClinicalIntervention Device DiagnosticAid Activity Study ClinicalEntity ChemicalEntity",
        ),
        "person": _bucket(
            "person patient group population cohort demographic occupation profession job ethnicity nationality people age group gender"
            " education employment race religious religion political sexuality language",
            "PopulationOfIndividualOrganisms Cohort StudyPopulation IndividualOrganism Human Agent Attribute BiologicalSex OrganismTaxon Behavior",
        ),
        # Nemotron-PII person-name labels: a surname fullmap-hits as every kind of named entity, so
        # only Human/IndividualOrganism ancestors survive; everything else falls through to fallback/raw
        "pii-person-name": _bucket(
            "first name last name user name given name surname family name middle name",
            "Human IndividualOrganism",
        ),
        "place": _bucket(
            "location country city place geograph facility state county postcode street address coordinate",
            "GeographicLocation PlanetaryEntity EnvironmentalFeature AdministrativeEntity AnatomicalEntity",
        ),
        "org": _bucket(
            "organization institution company agency university hospital publisher",
            "Agent AdministrativeEntity InformationContentEntity PopulationOfIndividualOrganisms",
        ),
        "info": _bucket(
            "publication study document database article journal report dataset abbreviation topic",
            "InformationContentEntity Publication Study Dataset Attribute Activity",
        ),
        "measure": _bucket(
            "measurement quantity unit percentage statistic parameter score rate dose attribute property characteristic trait blood type",
            "Attribute ClinicalAttribute ClinicalMeasurement OrganismAttribute PhenotypicQuality"
            " SocioeconomicAttribute StudyVariable InformationContentEntity PhenotypicFeature"
            " Procedure",
        ),
        "device": _bucket(
            "device equipment instrument tool machine",
            "Device ProcessedMaterial ChemicalEntity MaterialSample InformationContentEntity Procedure",
        ),
    }

    # labels carrying genomic signal are allowed to keep model-organism CURIE hits (KRT6 -> MGI Krt6)
    GENEISH_LABEL_MARKERS: ClassVar[tuple[str, ...]] = ("gene", "protein", "enzyme", "biomarker", "rna", "dna", "allele", "locus", "transcript")

    def __init__(self: Self) -> None:
        raise NotImplementedError("ResolutionGate is a namespace of classmethods")

    @classmethod
    def accept(cls, mention: str, raw_label: str, curie: str | None, category: str) -> bool:
        """True when a fullmap hit is semantically compatible with the corpus's own IOB label"""
        label: str = ScriptUtils.normalize_iob_label(raw_label)
        if cls.is_model_organism_mismatch(curie, label):
            return False
        if not cls.is_label_compatible(label, category):
            return False
        if cls.is_acronym_over_open_label(mention, label):
            return False
        return True

    @classmethod
    def is_model_organism_mismatch(cls, curie: str | None, normalized_label: str) -> bool:
        """model-organism gene CURIEs are systematic false positives on non-genomic labels"""
        if not curie:
            return False
        prefix_hit = curie.upper().startswith(MODEL_ORGANISM_PREFIXES)
        genomic_label = any(marker in normalized_label for marker in cls.GENEISH_LABEL_MARKERS)
        return prefix_hit and not genomic_label

    @classmethod
    def is_label_compatible(cls, normalized_label: str, category: str) -> bool:
        """every bucket the label triggers must be consistent with the resolved category's ancestors;
        OPEN labels, labels matching no bucket, and non-biolink categories carry no opinion"""
        if normalized_label in cls.OPEN_LABELS:
            return True
        triggered: list[frozenset[str]] = [allowed for keys, allowed in cls.BUCKETS.values() if any(key in normalized_label for key in keys)]
        if not triggered:
            return True
        ancestors: frozenset[str] = cls.ancestors(category)
        return not ancestors or any(ancestors & allowed for allowed in triggered)

    @classmethod
    def is_acronym_over_open_label(cls, mention: str, normalized_label: str) -> bool:
        """short uppercase mentions (HVA, US, CP) over catch-all labels are fullmap collision bait"""
        return bool(ACRONYM_MENTION.match(mention)) and normalized_label in cls.OPEN_LABELS

    @staticmethod
    @cache
    def ancestors(category: str) -> frozenset[str]:
        """biolink class mro names; a non-biolink category has empty ancestors (no opinion downstream)"""
        from biolink_model.datamodel import pydanticmodel_v2

        model = getattr(pydanticmodel_v2, category, None)
        if not isinstance(model, type):
            return frozenset()
        return frozenset(ancestor.__name__ for ancestor in model.__mro__)


def _names(names: str) -> frozenset[str]:
    """parse a space-separated biolink ancestor list into a frozenset"""
    return frozenset(names.split())


class PredicateRangeGate:
    """rejects gazetteer relations whose head/tail biolink categories contradict the predicate's
    biolink domain/range; hand-authored from biolink-model association classes and slot ranges
    (the pure YAML derivation is too brittle: one association subclass per predicate).

    No-opinion rules, both measured against the Pile-NER-biomed-IOB probe:
    - ANY (None) sides are unconstrained
    - a side whose category is not a biolink class (PascalCased raw label) has empty ancestors
      and imposes no constraint -- without this, correct edges like
      caused_by: Respiratory insufficiency -> dapsone-induced methemoglobinemia die on a raw tail
    """

    CHEM: ClassVar[frozenset[str]] = _names("ChemicalEntity MolecularEntity Drug ChemicalMixture Food Treatment Protein Polypeptide")
    MOL: ClassVar[frozenset[str]] = _names(
        "ChemicalEntity MolecularEntity Drug GenomicEntity Protein Polypeptide MacromolecularComplex NucleicAcidEntity GeneFamily ProteinFamily"
    )
    GENEP: ClassVar[frozenset[str]] = _names(
        "GenomicEntity Gene Protein Polypeptide ProteinDomain ProteinFamily GeneFamily MacromolecularComplex NucleicAcidEntity"
    )
    DIS: ClassVar[frozenset[str]] = _names(
        "DiseaseOrPhenotypicFeature Disease PhenotypicFeature ClinicalFinding PathologicalProcess OrganismTaxon OrganismalEntity"
    )
    ANAT: ClassVar[frozenset[str]] = _names(
        "AnatomicalEntity Cell CellLine GrossAnatomicalStructure CellularComponent OrganismalEntity MacromolecularComplex"
    )
    PROC: ClassVar[frozenset[str]] = _names(
        "BiologicalProcessOrActivity BiologicalProcess Pathway MolecularActivity"
        " PhysiologicalProcess PathologicalProcess Activity Behavior Phenomenon"
    )
    TAXON: ClassVar[frozenset[str]] = _names(
        "OrganismTaxon IndividualOrganism CellularOrganism Virus Bacterium Fungus Plant Mammal"
        " Vertebrate Invertebrate Cell CellLine PopulationOfIndividualOrganisms"
    )
    SEX: ClassVar[frozenset[str]] = _names("BiologicalSex")
    POP: ClassVar[frozenset[str]] = _names("PopulationOfIndividualOrganisms Cohort Human IndividualOrganism")

    # predicate -> (allowed head-ancestor sets joined by "|", allowed tail-ancestor sets joined by "|");
    # None means ANY. Ancestor groups are ORs; head and tail are ANDs.
    DOMRANGE: ClassVar[dict[str, tuple[str | None, str | None]]] = {
        "treats": ("CHEM|Treatment|Procedure", "DIS"),
        "treated_by": ("DIS|OrganismalEntity|PopulationOfIndividualOrganisms|IndividualOrganism|Human", "CHEM|Treatment|Procedure"),
        "preventative_for_condition": ("CHEM|Treatment|Procedure", "DIS"),
        "causes": ("CHEM|MOL|GENEP|PROC|DIS|TAXON", "DIS|PROC"),
        "caused_by": ("DIS|PROC", "CHEM|MOL|GENEP|PROC|DIS|TAXON"),
        "associated_with": (None, None),
        "correlated_with": (None, None),
        "interacts_with": ("MOL", "MOL"),
        "binds": ("MOL", "MOL"),
        "biomarker_for": ("MOL|ClinicalMeasurement|Attribute|InformationContentEntity", "DIS|PROC"),
        "expressed_in": ("GENEP", "ANAT|TAXON"),
        "located_in": ("MOL|ANAT|TAXON|PROC", "ANAT|GeographicLocation|PlanetaryEntity|CellularComponent|Cell"),
        "decreases_amount_or_activity_of": ("CHEM|MOL|GENEP|PROC|Treatment", "MOL|GENEP|PROC|DIS|ClinicalMeasurement|Attribute"),
        "increases_amount_or_activity_of": ("CHEM|MOL|GENEP|PROC|Treatment", "MOL|GENEP|PROC|DIS|ClinicalMeasurement|Attribute"),
        "has_adverse_event": ("CHEM|Treatment|Procedure|Drug", "DIS"),
        "diagnoses": ("Procedure|DiagnosticAid|ClinicalMeasurement|InformationContentEntity|Device|Treatment|ChemicalEntity", "DIS|ClinicalFinding"),
        "has_phenotype": ("DIS|Genotype|IndividualOrganism|PopulationOfIndividualOrganisms|OrganismalEntity|Human", "DIS"),
        "part_of": (None, None),
        "in_taxon": ("MOL|ANAT|SequenceVariant|Gene|Protein|GenomicEntity|NucleicAcidEntity|ChemicalEntity|Cell|CellLine", "TAXON"),
        "superclass_of": (None, None),
        "participates_in": ("GENEP|MOL|ANAT|Cell", "PROC"),
        "precedes": ("PROC|ANAT|MOL|GENEP", "PROC|ANAT|MOL|GENEP|DIS"),
        "occurs_in": ("PROC|MOL|GENEP|DIS|ANAT", "ANAT|TAXON|PROC|GeographicLocation|PopulationOfIndividualOrganisms|IndividualOrganism"),
    }

    def __init__(self: Self) -> None:
        raise NotImplementedError("PredicateRangeGate is a namespace of classmethods")

    @classmethod
    def accept(cls, predicate: str, head_category: str, tail_category: str) -> bool:
        """True when head/tail biolink categories are compatible with the predicate's domain/range;
        a side that is not a biolink class (empty ancestors) imposes no constraint on its side"""
        entry: tuple[str | None, str | None] | None = cls.DOMRANGE.get(predicate)
        if entry is None:
            return True
        head_ancestors: frozenset[str] = ResolutionGate.ancestors(head_category)
        tail_ancestors: frozenset[str] = ResolutionGate.ancestors(tail_category)
        return cls._side_ok(entry[0], head_ancestors) and cls._side_ok(entry[1], tail_ancestors)

    @classmethod
    def category_ok(cls, allowed: str | None, category: str) -> bool:
        """True when one category clears an OR-group expression (qualifier range gates);
        None or a non-biolink category means no opinion"""
        return cls._side_ok(allowed, ResolutionGate.ancestors(category))

    @classmethod
    def category_ok_strict(cls, allowed: str | None, category: str) -> bool:
        """category_ok without the no-opinion escape: DAKP's type-driven qualifier field map
        is strictly typed, so a mention with no resolvable biolink ancestors can never
        become a typed qualifier context (it may still be a predicate argument)"""
        ancestors: frozenset[str] = ResolutionGate.ancestors(category)
        return bool(ancestors) and cls._side_ok(allowed, ancestors)

    @classmethod
    def _side_ok(cls, allowed: str | None, ancestors: frozenset[str]) -> bool:
        if allowed is None or not ancestors:
            return True
        return any(ancestors & cls._group(group) for group in allowed.split("|"))

    @classmethod
    @cache
    def _group(cls, group: str) -> frozenset[str]:
        """resolve one OR-group name to its ancestor set; bare class names form a singleton"""
        named: frozenset[str] | None = getattr(cls, group, None)
        return named if named is not None else frozenset({group})


class ScriptUtils:
    """helpers shared across Script classes; factor out later if this grows"""

    BIOLINK_PREFIX: ClassVar[str] = BIOLINK_PREFIX
    FULLMAP_TAXON: ClassVar[str] = "9606"
    # raw gliner-biomed labels -> biolink classes, consulted only after fullmap misses. Keys are
    # lowercase and looked up on raw_label.lower(); values must be members of tablassert Categories
    # (validated by test_every_fallback_label_maps_to_a_biolink_category); dataset-specific
    # vocabularies ride on top via resolve_mentions(label_map=...) and live with their dataset
    FALLBACK_LABEL_MAP: ClassVar[dict[str, str]] = {
        "drug": "Drug",
        "drug product": "Drug",
        "drug form": "Drug",
        "drug formulation": "Drug",
        "dosage form": "Drug",
        "gene": "Gene",
        "protein": "Protein",
        "condition": "Disease",
        "disease": "Disease",
        "symptom": "PhenotypicFeature",
        "adverse event": "PhenotypicFeature",
        "cell type": "Cell",
        "virus": "OrganismTaxon",
        "pathogen": "OrganismTaxon",
        "biological process": "BiologicalProcess",
        "biological pathway": "BiologicalProcess",
        "gene family": "GeneFamily",
    }

    _fullmap_db: ClassVar[Path | None] = None

    def __init__(self: Self) -> None:
        raise NotImplementedError("ScriptUtils is a namespace of classmethods")

    @staticmethod
    def join_tokens(tokens: list[str]) -> str:
        return " ".join(tokens)

    @classmethod
    def parse_literal_list(cls, value: Any) -> list[str]:
        """safely decode python-repr string columns (ast.literal_eval, no code execution);
        malformed rows yield [] so callers skip them instead of crashing (skip-don't-coerce)"""
        decoded: Any = cls._decode_container(value)
        if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
            return []
        return decoded

    @classmethod
    def parse_literal_dict(cls, value: Any) -> dict[str, Any]:
        """safely decode python-repr string columns that carry a dict (ast.literal_eval, no code
        execution); real dicts pass through and anything else (None, junk strings, lists, quoted
        strings, malformed reprs) yields {} so callers skip the section instead of crashing or
        coercing (skip-don't-coerce, mirroring parse_literal_list)"""
        decoded: Any = cls._decode_container(value)
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _decode_container(value: Any) -> Any:
        """one decode posture for every external column: real containers pass through, python-repr
        strings go through ast.literal_eval (no code execution), malformed input yields None"""
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None

    @classmethod
    def parse_conversations(cls, value: Any) -> tuple[str, list[tuple[str, list[str]]]]:
        """decode a Pile-NER-type conversations column into (text, [(raw_label, mentions), ...]).

        The corpus packages NER as chat: one human turn carries the document behind a 'Text: '
        prefix, then every entity type is asked for with the templated question
        'What describes <type> in the text?' and answered by the next gpt turn with a JSON list of
        surface mentions ('[]' for the negative-sampled types). Mentions are surfaces, not offsets.

        Skip-don't-coerce throughout: a row without a 'Text: ' turn, or one whose answer turn does
        not decode to a list of strings, yields ('', []) / an empty mention list so the caller emits
        an empty example instead of crashing."""
        turns = cls._decode_container(value)
        if not isinstance(turns, list):
            return ("", [])
        pairs: list[tuple[str, str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker, spoken = turn.get("from"), turn.get("value")
            if isinstance(speaker, str) and isinstance(spoken, str):
                pairs.append((speaker, spoken))
        prefixed = (spoken for speaker, spoken in pairs if speaker == HUMAN_TURN and spoken.startswith(CONVERSATION_TEXT_PREFIX))
        text: str = next((spoken[len(CONVERSATION_TEXT_PREFIX) :] for spoken in prefixed), "")
        if not text:
            return ("", [])
        answered: list[tuple[str, list[str]]] = []
        for (speaker, spoken), (next_speaker, next_spoken) in zip(pairs, pairs[1:], strict=False):
            question = CONVERSATION_QUESTION.match(spoken) if speaker == HUMAN_TURN else None
            if question is None or next_speaker != GPT_TURN:
                continue
            answered.append((question.group(1), cls.parse_mention_list(next_spoken)))
        return (text, answered)

    @classmethod
    def parse_mention_list(cls, value: str) -> list[str]:
        """decode one gpt answer turn: JSON first (the corpus writes JSON arrays), python-repr
        second; non-list answers and non-string items are dropped rather than coerced"""
        try:
            decoded: Any = json.loads(value)
        except (ValueError, RecursionError):
            decoded = cls._decode_container(value)
        if not isinstance(decoded, list):
            return []
        return [stripped for item in decoded if isinstance(item, str) and (stripped := item.strip())]

    @staticmethod
    def lowered_tokens(tokens: list[str]) -> list[str]:
        """the case-folded haystack token_occurrences matches against, folded once per document"""
        return [token.lower() for token in tokens]

    @classmethod
    def token_occurrences(cls, tokens: list[str], mention: str, lowered: list[str] | None = None) -> list[tuple[int, int]]:
        """every (start, end_inclusive) token span whose surface matches the mention.

        Pile-NER-type answers with surfaces rather than offsets, so mention spans are recovered by
        matching the mention's whitespace-split token subsequence case-insensitively against the
        document's whitespace tokens. An absent mention yields [] -- that is the hallucination guard
        (gpt answers occasionally name surfaces the document never contains) and simultaneously the
        tokenization guard, so entities and relation spans stay aligned on one rule.

        `lowered` is lowered_tokens(tokens) for the SAME document, letting a caller fold the haystack
        once instead of once per mention: a row carries dozens of answered mentions over hundreds of
        tokens, and the corpus has ~46k rows, so folding per mention dominated the script."""
        needle: list[str] = [token.lower() for token in mention.split()]
        if not needle or len(needle) > len(tokens):
            return []
        haystack: list[str] = cls.lowered_tokens(tokens) if lowered is None else lowered
        width: int = len(needle)
        return [(start, start + width - 1) for start in range(len(haystack) - width + 1) if haystack[start : start + width] == needle]

    @staticmethod
    def iob_spans(tags: list[str]) -> list[tuple[int, int, str]]:
        """decode IOB2 tags into (start, end_inclusive, label) triples; orphan I- tags
        (no matching B- of the same type) are promoted to single-token spans rather than dropped"""
        spans: list[list[int]] = []
        labels: list[str] = []
        for index, tag in enumerate(tags):
            kind, _, name = tag.partition("-")
            if kind not in ("B", "I") or not name:
                continue
            if kind == "I" and labels and labels[-1] == name:
                spans[-1][1] = index
                continue
            spans.append([index, index])
            labels.append(name)
        return [(start, end, label) for (start, end), label in zip(spans, labels, strict=True)]

    @staticmethod
    def normalize_iob_label(label: str) -> str:
        """collapse underscore/case variants (anatomical_structure == anatomical structure)"""
        return label.lower().replace("_", " ").strip()

    @staticmethod
    def pascal_label(label: str) -> str:
        """case a raw IOB label the way biolink classes are named (medical condition -> MedicalCondition)"""
        return "".join(part[:1].upper() + part[1:] for part in LABEL_SPLIT.split(label.lower()) if part)

    @classmethod
    def lookup_label(cls, label_map: Mapping[str, str], raw_label: str) -> str | None:
        """the shared two-shot label-map probe: the lowercased raw label first, then its
        IOB-normalized form, so one map may key either 'disease' or 'b-disease'/'anatomical_structure'
        style variants and callers need not spell out both probes (or forget the second)"""
        return label_map.get(raw_label.lower()) or label_map.get(cls.normalize_iob_label(raw_label))

    @classmethod
    def pascal_raw_labels(cls, resolved: list[ResolvedMention]) -> list[ResolvedMention]:
        """PascalCase the category of every raw-origin mention, leaving fullmap/fallback hits alone.

        A raw label is dataset vocabulary no ontology confirmed, so it trains under a biolink-shaped
        name rather than an arbitrary string, while a mapped entry already names a biolink class.
        No parallel label sequence is needed: _resolve copies the raw label into
        ResolvedMention.category on the raw path, so for those items the category IS the label.
        """
        return [
            item if item.origin != "raw" else ResolvedMention(mention=item.mention, category=cls.pascal_label(item.category), origin=item.origin)
            for item in resolved
        ]

    @staticmethod
    def mention_spans(tokens: list[str], ner: list[Any]) -> list[tuple[int, int, str]]:
        """filter valid GLiNER spans ([start, end, label], end inclusive) down to in-bounds triples"""
        spans: list[tuple[int, int, str]] = []
        for span in ner:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 3
                or isinstance(span[0], bool)
                or not isinstance(span[0], int)
                or isinstance(span[1], bool)
                or not isinstance(span[1], int)
                or not isinstance(span[2], str)
            ):
                continue
            start, end, label = span
            if start < 0 or end < start or end >= len(tokens):
                continue
            spans.append((start, end, label))
        return spans

    @classmethod
    def mentions(cls, tokens: list[str], ner: list[Any]) -> list[tuple[str, str]]:
        """slice mention text out of GLiNER token spans; bounds checks live in mention_spans"""
        return [(cls.join_tokens(tokens[start : end + 1]), label) for start, end, label in cls.mention_spans(tokens, ner)]

    @staticmethod
    def char_spans_to_token_spans(triples: list[tuple[str, int, int]], char_spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        """bridge char-offset entity spans to the repo-wide (start_token, end_token_inclusive, label)
        contract iob_spans/mention_spans emit; char ends are exclusive, token ends inclusive.

        Per-span pipeline, in order: drop out-of-bounds (start < 0 or end past the text; 0.04-0.24%
        measured), normalize boundary slop by advancing start past and retreating end across
        whitespace (~0.4%), drop degenerate spans (start >= end: zero-length and inverted), snap
        boundaries landing mid-token out to the full tokens they overlap (3.43% = 351/10,237), and
        drop spans left with no overlapping token (whitespace-only). Order-preserving: spans are
        never reordered, deduped, or merged (the dataset is measured overlap-free, 0/18,685).

        The raw text is not a parameter: the triples carry the text extent as the largest token
        end_char (the splitter's tokens tile the text), so the bounds check reads that maximum
        instead of len(raw_text).
        """
        text_extent: int = max((token_end for _token, _start, token_end in triples), default=0)
        token_spans: list[tuple[int, int, str]] = []
        for start, end, label in char_spans:
            if start < 0 or end > text_extent:
                continue
            # whitespace slop: chars outside the token tile sit exactly at whitespace runs, so clamp
            # each boundary to the nearest token char the span reaches (tokens are offset-ordered)
            for _token, token_start, token_end in triples:
                if token_end > start:
                    start = max(start, token_start)
                    break
            for _token, token_start, token_end in reversed(triples):
                if token_start < end:
                    end = min(end, token_end)
                    break
            if start >= end:
                continue
            first = last = -1
            for index, (_token, token_start, token_end) in enumerate(triples):
                if token_start >= end:
                    break
                if token_end > start:
                    if first < 0:
                        first = index
                    last = index
            if first < 0:
                continue
            token_spans.append((first, last, label))
        return token_spans

    @classmethod
    @cache
    def biolink_categories(cls) -> frozenset[str]:
        return frozenset(strip_biolink_prefix(category.value) for category in Categories)

    @classmethod
    @cache
    def biolink_predicates(cls) -> frozenset[str]:
        return frozenset(predicate.value for predicate in Predicates)

    @classmethod
    def is_biolink_category(cls, label: str) -> bool:
        return label in cls.biolink_categories()

    @staticmethod
    def normalize_predicate(raw: str) -> str:
        """biolink-shaped snake_case: the one normalizer for mapped and native predicates alike"""
        return raw.strip().lower().replace(" ", "_").replace("-", "_")

    @classmethod
    def resolve_predicate(cls, raw: str) -> tuple[str, bool]:
        """biolink member when one matches, biolink-shaped native snake_case otherwise (zero-shot breadth:
        only ~7% of the post-training corpus's predicates are biolink members, and discarding the rest
        would gut the relation family)"""
        normalized: str = cls.normalize_predicate(raw)
        return normalized, normalized in cls.biolink_predicates()

    @classmethod
    def fullmap_db(cls) -> Path:
        db: Path | None = cls._fullmap_db
        if db is None:
            db = Path(fullmap_db_path(FULLMAP_DIR))
            cls._fullmap_db = db
        return db

    @classmethod
    def fullmap_available(cls) -> bool:
        try:
            return cls.fullmap_db().is_file()
        except OSError:
            return False

    @classmethod
    def resolve_mentions(cls, spans: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        """fullmap first, static fallback second, raw label last (kept for zero-shot training);
        label_map is the caller's dataset-specific vocabulary merged over the shared FALLBACK_LABEL_MAP"""
        fallback_map: dict[str, str] = {**cls.FALLBACK_LABEL_MAP, **label_map} if label_map else cls.FALLBACK_LABEL_MAP
        distinct_mentions: list[str] = list(dict.fromkeys(mention for mention, _ in spans))
        normalized: dict[str, str] = dict(zip(distinct_mentions, rs.normalize_terms(distinct_mentions), strict=True))
        best: dict[str, dict[str, object]] = cls._fullmap_best(normalized)
        return [cls._resolve(mention, raw_label, normalized.get(mention), best, fallback_map) for mention, raw_label in spans]

    _best_cache: ClassVar[OrderedDict[str, dict[str, object] | None]] = OrderedDict()
    """process-wide LRU of normalized term -> its best fullmap row (None = no accepted row), so a
    term seen in an earlier row never pays another redb + polars round trip. Exact, not a
    heuristic: filter_and_rank with column_context=False ranks every term independently of the
    other terms in its batch, so a per-term memo returns what a fresh call would. Bounded by
    FULLMAP_BEST_CACHE_TERMS and holding only the three fields _resolve reads, so worker memory
    stays flat on long runs"""

    _BEST_FIELDS: ClassVar[tuple[str, ...]] = ("CATEGORY_NAME", "CURIE", "PREFERRED_NAME")

    @classmethod
    def _fullmap_best(cls, normalized: dict[str, str]) -> dict[str, dict[str, object]]:
        """one best-ranked fullmap row per normalized mention, opened read-only/shared-lock;
        cached terms skip the lookup, only the misses go to redb in one batch"""
        terms: set[str] = {norm for norm in normalized.values() if norm}
        if not terms:
            return {}
        cache = cls._best_cache
        misses: list[str] = sorted(term for term in terms if term not in cache)
        if misses:
            fetched: dict[str, dict[str, object]] = cls._fetch_best(misses)
            for term in misses:
                cache[term] = fetched.get(term)
            while len(cache) > FULLMAP_BEST_CACHE_TERMS:
                cache.popitem(last=False)
        best: dict[str, dict[str, object]] = {}
        for term in terms:
            # a term evicted between the store and this read (miss batch larger than the bound)
            # is re-fetched rather than silently reported as unresolved
            row = cache[term] if term in cache else cls._fetch_best([term]).get(term)
            if term in cache:
                cache.move_to_end(term)
            if row is not None:
                best[term] = row
        return best

    @classmethod
    def _fetch_best(cls, terms: list[str]) -> dict[str, dict[str, object]]:
        """the uncached path: one lookup_rows round trip, then the ranking below"""
        rows: list[dict[str, object]] = lookup_rows(cls.fullmap_db(), terms)
        if not rows:
            return {}
        return cls._rank_best(rows)

    @classmethod
    def _rank_best(cls, rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        """one best-ranked lookup_rows row per term, in plain python.

        This is filter_and_rank(column_context=False) + "keep the first row per term" expressed
        directly: polars builds and collects a lazy plan per call, and at one call per script row
        over a handful of rows that fixed overhead dominated dispatch (6.9s of a 13.5s profile for
        480 calls). The equivalence, term by term:

        - taxon filter: keep TAXON_ID == FULLMAP_TAXON or 0
        - PR: 1 when PREFERRED_NAME == term, else 5 when the level-one normalizations agree
          (NLP_LEVEL is 1 for every row, so the filter_and_rank guard always holds), else 10
        - best per term: min (PR, NLP_LEVEL, CURIE), which is exactly the head of
          deduplicate_result's sort for that term

        Ties that differ only in SOURCE_NAME/VERSION are resolved arbitrarily by polars too, and
        the returned fields are CURIE-derived, so the result is unchanged. Requires distinct terms
        (the caller passes a sorted set): a repeated term would multiply rows in the polars join.
        """
        taxon: int = int(cls.FULLMAP_TAXON)
        kept: list[dict[str, object]] = [row for row in rows if int(row["TAXON_ID"]) in (taxon, 0)]
        if not kept:
            return {}
        # one Rust batch call for both sides of the level-one comparison
        surfaces: list[str] = sorted({str(row["PREFERRED_NAME"]) for row in kept} | {str(row["term"]) for row in kept})
        normalized: dict[str, str] = dict(zip(surfaces, rs.normalize_terms(surfaces), strict=True))
        best: dict[str, tuple[tuple[int, int, str], dict[str, object]]] = {}
        for row in kept:
            term: str = str(row["term"])
            name: str = str(row["PREFERRED_NAME"])
            curie: str = str(row["CURIE"])
            rank = 1 if name == term else (5 if normalized[name] == normalized[term] else 10)
            order: tuple[int, int, str] = (rank, 1, curie)
            current = best.get(term)
            if current is None or order < current[0]:
                best[term] = (order, {field: row[field] for field in cls._BEST_FIELDS})
        return {term: payload for term, (_order, payload) in best.items()}

    @classmethod
    def _resolve(
        cls,
        mention: str,
        raw_label: str,
        normalized: str | None,
        best: dict[str, dict[str, object]],
        fallback_map: dict[str, str],
    ) -> ResolvedMention:
        row: dict[str, object] | None = best.get(normalized) if normalized else None
        if row is not None:
            category = strip_biolink_prefix(str(row["CATEGORY_NAME"]))
            curie = str(row["CURIE"])
            # the shared gate falls through on rejection: the mention keeps its fallback/raw path
            # instead of keeping a semantically incompatible fullmap category
            if cls.is_biolink_category(category) and ResolutionGate.accept(mention, raw_label, curie, category):
                return ResolvedMention(
                    mention=mention,
                    category=category,
                    curie=curie,
                    preferred_name=str(row["PREFERRED_NAME"]),
                    origin="fullmap",
                )
        fallback: str | None = cls.lookup_label(fallback_map, raw_label)
        if fallback is not None and cls.is_biolink_category(fallback):
            return ResolvedMention(mention=mention, category=fallback, origin="fallback")
        return ResolvedMention(mention=mention, category=raw_label, origin="raw")

    @classmethod
    def group_entities(cls, resolved: list[ResolvedMention]) -> list[Entity]:
        """group resolved mentions by category with one fullmap evidence string per label --
        the shared entity shape every Script emits"""
        mentions_by_label: dict[str, list[str]] = {}
        evidence_by_label: dict[str, tuple[str | None, str | None]] = {}
        for item in resolved:
            mentions = mentions_by_label.setdefault(item.category, [])
            if item.mention not in mentions:
                mentions.append(item.mention)
            if item.curie is not None and item.category not in evidence_by_label:
                evidence_by_label[item.category] = (item.curie, item.preferred_name)
        return [
            Entity(
                label=label,
                mentions=mentions,
                description=cls.biolink_category_description(label, *evidence_by_label.get(label, (None, None))),
            )
            for label, mentions in mentions_by_label.items()
            if mentions
        ]

    _predicate_descriptions: ClassVar[dict[str, str] | None] = None

    @classmethod
    def predicate_description(cls, predicate: str) -> str | None:
        """biolink slot definition for a predicate (underscore-normalized lookup, is_a walk);
        all 23 declared gazetteer predicates carry one -- mirrors biolink_category_description"""
        if cls._predicate_descriptions is None:
            cls._predicate_descriptions = cls._load_slot_descriptions()
        return cls._predicate_descriptions.get(predicate)

    @staticmethod
    def _load_slot_descriptions() -> dict[str, str]:
        """flattened biolink-model slot definitions keyed snake_case, inheriting along is_a"""
        from importlib.resources import files

        import yaml

        # explicit encoding: biolink_model.yaml carries non-ascii, and a non-utf-8 locale would
        # mis-decode it into control chars that the yaml reader rejects
        schema = files("biolink_model").joinpath("schema/biolink_model.yaml").read_text(encoding="utf-8")
        # libyaml's CSafeLoader (the loader parsers.py already uses for ingests.yaml): same safe
        # constructor set and an equal result, ~12x faster than the pure-python safe_load on this
        # multi-MB schema, which every worker parses once before its first predicate lookup
        slots: dict[str, dict[str, Any]] = yaml.load(schema, Loader=yaml.CSafeLoader)["slots"]

        def definition(name: str, seen: frozenset[str] = frozenset()) -> str | None:
            slot = slots.get(name)
            if slot is None or name in seen:
                return None
            described: str | None = slot.get("description")
            if described:
                return " ".join(str(described).split())
            return definition(str(slot.get("is_a")), seen | {name}) if slot.get("is_a") else None

        flattened: dict[str, str] = {}
        for name in slots:
            described = definition(name)
            if described:
                flattened[str(name).replace(" ", "_")] = described
        return flattened

    @classmethod
    @cache
    def biolink_category_description(cls, category: str, curie: str | None = None, preferred_name: str | None = None) -> str | None:
        """biolink class definition plus resolved fullmap evidence for GLiNER label prompts"""
        definition: str | None = cls._class_definition(category)
        evidence: str | None = f"[fullmap: {curie} | {preferred_name}]" if curie and preferred_name else None
        if definition and evidence:
            return f"{definition} {evidence}"
        return definition or evidence

    @staticmethod
    @cache
    def _class_definition(category: str) -> str | None:
        if category not in ScriptUtils.biolink_categories():
            return None
        from biolink_model.datamodel import pydanticmodel_v2

        model = getattr(pydanticmodel_v2, category, None)
        docstring = getattr(model, "__doc__", None)
        if not docstring:
            return None
        definition = " ".join(docstring.split())
        return definition if len(definition) <= 300 else f"{definition[:297]}..."

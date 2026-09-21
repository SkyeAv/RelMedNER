from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Self

from tablassert import rs
from tablassert.biolink import Categories, Predicates
from tablassert.fullmap import fullmap_db_path, lookup_rows

from relmedner.constants import FULLMAP_DIR
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


class ScriptUtils:
    """helpers shared across Script classes; factor out later if this grows"""

    BIOLINK_PREFIX: ClassVar[str] = BIOLINK_PREFIX
    FULLMAP_TAXON: ClassVar[str] = "9606"
    # raw gliner-biomed labels -> biolink classes, consulted only after fullmap misses;
    # keys are lowercase and looked up on raw_label.lower(); values must be members of
    # tablassert Categories (validated by test_every_fallback_label_maps_to_a_biolink_category);
    # dataset-specific vocabularies ride on top via resolve_mentions(label_map=...)
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

    @classmethod
    def _fullmap_best(cls, normalized: dict[str, str]) -> dict[str, dict[str, object]]:
        """one best-ranked fullmap row per normalized mention, opened read-only/shared-lock"""
        valid: dict[str, str] = {mention: norm for mention, norm in normalized.items() if norm}
        if not valid:
            return {}
        import polars as pl
        from tablassert.fullmap import filter_and_rank

        rows: list[dict[str, object]] = lookup_rows(cls.fullmap_db(), sorted(set(valid.values())))
        if not rows:
            return {}
        terms = pl.DataFrame({"term": list(valid.values()), "nlp_level": [1] * len(valid)})
        matches = filter_and_rank(pl.DataFrame(rows), terms, cls.FULLMAP_TAXON, None, None, False)
        best: dict[str, dict[str, object]] = {}
        for row in matches.iter_rows(named=True):
            best.setdefault(str(row["term"]), row)  # sorted best-first; keep the first
        return best

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
            if cls.is_biolink_category(category):
                return ResolvedMention(
                    mention=mention,
                    category=category,
                    curie=str(row["CURIE"]),
                    preferred_name=str(row["PREFERRED_NAME"]),
                    origin="fullmap",
                )
        fallback: str | None = fallback_map.get(raw_label.lower())
        if fallback is not None and cls.is_biolink_category(fallback):
            return ResolvedMention(mention=mention, category=fallback, origin="fallback")
        return ResolvedMention(mention=mention, category=raw_label, origin="raw")

    @staticmethod
    def group_entities(resolved: list[ResolvedMention]) -> list[Entity]:
        """group resolved mentions by category, attaching the first fullmap evidence found per label
        (shared by every NER-shaped dataset; the curated-corpus worktree needs exactly this)"""
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
                description=ScriptUtils.biolink_category_description(label, *evidence_by_label.get(label, (None, None))),
            )
            for label, mentions in mentions_by_label.items()
            if mentions
        ]

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

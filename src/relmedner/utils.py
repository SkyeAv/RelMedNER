
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    try:
        return version("relmedner")
    except PackageNotFoundError:
        return "dev"


from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, Self

from tablassert import rs
from tablassert.biolink import Categories
from tablassert.fullmap import fullmap_db_path, lookup_rows

from relmedner.constants import FULLMAP_DIR

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
    # values must be members of tablassert Categories (checked at resolution time)
    FALLBACK_LABEL_MAP: ClassVar[dict[str, str]] = {
        "Drug": "Drug",
        "Drug product": "Drug",
        "Drug form": "Drug",
        "Drug formulation": "Drug",
        "Dosage form": "Drug",
        "Gene": "Gene",
        "Protein": "Protein",
        "Condition": "Disease",
        "Disease": "Disease",
        "Symptom": "PhenotypicFeature",
        "Adverse event": "PhenotypicFeature",
        "Cell type": "Cell",
        "Virus": "OrganismTaxon",
        "Pathogen": "OrganismTaxon",
        "Biological process": "BiologicalProcess",
        "Biological pathway": "BiologicalProcess",
        "Gene family": "GeneFamily",
    }

    _fullmap_db: ClassVar[Path | None] = None

    def __init__(self: Self) -> None:
        raise NotImplementedError("ScriptUtils is a namespace of classmethods")

    @staticmethod
    def join_tokens(tokens: list[str]) -> str:
        return " ".join(tokens)

    @staticmethod
    def mentions(tokens: list[str], ner: list[list[Any]]) -> list[tuple[str, str]]:
        """slice mention text out of GLiNER token spans ([start, end, label], end inclusive)"""
        spans: list[tuple[str, str]] = []
        for span in ner:
            start, end, label = int(span[0]), int(span[1]), str(span[2])
            if start < 0 or end < start or end >= len(tokens):
                continue
            spans.append((" ".join(tokens[start : end + 1]), label))
        return spans

    @classmethod
    @cache
    def biolink_categories(cls) -> frozenset[str]:
        return frozenset(strip_biolink_prefix(category.value) for category in Categories)

    @classmethod
    def is_biolink_category(cls, label: str) -> bool:
        return label in cls.biolink_categories()

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
    def resolve_mentions(cls, spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        """fullmap first, static fallback second, raw label last (kept for zero-shot training)"""
        distinct_mentions: list[str] = list(dict.fromkeys(mention for mention, _ in spans))
        normalized: dict[str, str] = dict(zip(distinct_mentions, rs.normalize_terms(distinct_mentions), strict=True))
        best: dict[str, dict[str, object]] = cls._fullmap_best(normalized)
        return [cls._resolve(mention, raw_label, normalized.get(mention), best) for mention, raw_label in spans]

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
        fallback: str | None = cls.FALLBACK_LABEL_MAP.get(raw_label)
        if fallback is not None and cls.is_biolink_category(fallback):
            return ResolvedMention(mention=mention, category=fallback, origin="fallback")
        return ResolvedMention(mention=mention, category=raw_label, origin="raw")

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

from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class MedMentionsScript(Script):
    """turns one MedMentions (ST21pv) avro document into one entity training example

    The records are built out-of-band from the chanzuckerberg/MedMentions ST21pv PubTator corpus
    (CC0, 4,392 PubMed documents, 203,282 gold UMLS-linked entity spans over the 21 semantic
    types) with the bc5cdr builder contract: entity offsets are document-absolute over exactly
    "title + newline + abstract", and the converter's census slice-matched 203,282 of 203,282
    annotations, so unlike the zameji mirror (whose token-index re-encoding left 33.6 percent of
    its spans unrecoverable on any tokenization, measured on wenceslaus 2026-09-24) every span
    here is char-exact gold.

    This script does NOT re-resolve through fullmap -- it trusts the annotation's own UMLS CUI
    exactly the way Bc5CdrScript trusts its MESH (trust-gold philosophy): the record's
    'UMLS:C...' id rides as the curie verbatim (the corpus ships it pre-namespaced), and the
    entity carries origin 'fullmap' when a curie exists, 'raw' otherwise.

    Spans must satisfy gliner2's occurs-in-text contract verbatim: out-of-bounds offsets and
    document-absolute slice mismatches drop right here instead of shipping spans the downstream
    sanitizer would silently discard (the measured slice-match rate is 100 percent, so a mismatch
    is a converter bug, not corpus noise). Unknown semantic types and malformed fields drop too,
    and a document whose annotations all drop still ships its text for the declared-outputs
    filter. MedMentions ships no relations, so relations stay empty.

    LABEL_MAP covers the corpus's complete measured vocabulary (all 21 semantic types, census on
    wenceslaus 2026-09-24); T022 Body System, T031 Body Substance and T017 Anatomical Structure
    all group under AnatomicalEntity (the pile_ner_type conventions).
    """

    NAME: ClassVar[str] = "MedMentionsScript"

    # UMLS semantic-type code -> biolink class; every target validated against tablassert
    # Categories at import below. Coverage measured over all 203,282 annotations:
    # T038 41,422, T103 37,401, T058 24,306, T017 20,497, T033 16,227, T082 12,500,
    # T170 10,324, T062 9,172, T204 8,543, T098 6,154, T092 2,143, T007 2,050, T074 2,018,
    # T037 1,854, T097 1,779, T201 1,773, T168 1,352, T031 1,256, T005 1,100, T091 916, T022 495.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "T005": "Virus",
        "T007": "Bacterium",
        "T017": "AnatomicalEntity",
        "T022": "AnatomicalEntity",
        "T031": "AnatomicalEntity",
        "T033": "ClinicalFinding",
        "T037": "Disease",
        "T038": "BiologicalProcess",
        "T058": "ClinicalIntervention",
        "T062": "Study",
        "T074": "Device",
        "T082": "GeographicLocation",
        "T091": "Activity",
        "T092": "Agent",
        "T097": "PopulationOfIndividualOrganisms",
        "T098": "PopulationOfIndividualOrganisms",
        "T103": "ChemicalEntity",
        "T168": "Food",
        "T170": "InformationContentEntity",
        "T201": "ClinicalAttribute",
        "T204": "OrganismTaxon",
    }

    def text_of(self: Self, record: dict[str, Any]) -> str:
        """the builder's document text contract: title, one newline, abstract (the converter's
        census slice-matched every annotation against exactly this assembly)"""
        title: str = record["title"] if isinstance(record.get("title"), str) else ""
        abstract: str = record["abstract"] if isinstance(record.get("abstract"), str) else ""
        return f"{title}\n{abstract}"

    def spans_of(self: Self, record: dict[str, Any], text: str) -> list[tuple[str, str, str]]:
        """(surface, biolink label, raw UMLS cui) triples for gold annotations that survive every
        contract; a malformed table yields zero spans rather than a crash (skip-don't-coerce)

        A span survives only when its semantic type maps through LABEL_MAP, cui/text are
        non-empty strings, offset/length are non-bool non-negative ints, and the
        document-absolute slice equals the declared text exactly -- every out-of-bounds
        annotation and every slice mismatch drops right here (the corpus's measured slice-match
        rate is 100 percent, so a mismatch means a converter bug, not corpus noise).
        """
        entities: Any = record.get("entities")
        if not isinstance(entities, list):
            return []
        spans: list[tuple[str, str, str]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            ann_type: Any = entity.get("type")
            label: str | None = self.LABEL_MAP.get(ann_type) if isinstance(ann_type, str) else None
            cui: Any = entity.get("cui")
            surface: Any = entity.get("text")
            offset: Any = entity.get("offset")
            length: Any = entity.get("length")
            if label is None or not isinstance(cui, str) or not cui or not isinstance(surface, str) or not surface:
                continue
            if isinstance(offset, bool) or isinstance(length, bool) or not isinstance(offset, int) or not isinstance(length, int):
                continue
            if offset < 0 or length < 0 or text[offset : offset + length] != surface:
                continue
            spans.append((surface, label, cui))
        return spans

    def mentions_of(self: Self, record: dict[str, Any], text: str) -> list[ResolvedMention]:
        """one resolved mention per surviving gold span, carrying its own UMLS CUI as the curie
        when it arrives pre-namespaced ('UMLS:C...'); gold is never re-resolved through fullmap"""
        return [
            ResolvedMention(
                mention=surface,
                category=label,
                curie=cui if cui.startswith("UMLS:") else None,
                origin="fullmap" if cui.startswith("UMLS:") else "raw",
            )
            for surface, label, cui in self.spans_of(record, text)
        ]

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """LocalAvroDataStream ships the whole avro record as one value; a malformed record
        degrades to the empty example the pipeline filters downstream (skip-don't-coerce).
        MedMentions ships no relations, so relations stay empty."""
        (record,) = values
        if not isinstance(record, dict):
            return TrainingExample(text="")
        text: str = self.text_of(record)
        if not text.strip():
            return TrainingExample(text="")
        resolved: list[ResolvedMention] = self.mentions_of(record, text)
        return TrainingExample(
            text=text,
            entities=ScriptUtils.group_entities(resolved) if resolved else [],
            relations=[],
        )


validate_label_map(MedMentionsScript.LABEL_MAP, MedMentionsScript.NAME)

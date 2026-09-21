from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import in_text, validate_label_map
from relmedner.models import Entity, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils, strip_biolink_prefix


class CtkpInterventionsScript(Script):
    """turns CTKP intervention records into entity examples carrying the KP's own gold CURIEs

    Every record is one clinicaltrials.gov/AACT intervention joined against the clinical trials
    KP normalization output. Unlike the hub corpora, the spans here are not model-predicted: the
    KP's NameResolver pass already recorded, per mention, which surface form it matched
    (matched_text) and what it resolved to (curie/category/preferred_name). So this script does
    NOT re-resolve through fullmap -- it trusts the KP and only enforces the shared contracts:
    biolink-class membership and gliner2's occurs-in-text rule.

    Unmapped records still ship: an intervention whose mention never normalized (94,732 match
    rows in the 20260920 build, plus the wholly non-normalized DEVICE/PROCEDURE/BEHAVIORAL/OTHER
    types) falls back to its AACT intervention_type, so DEVICE rows still teach Device. Records
    that reach neither path -- OTHER with no match -- produce no entity and the pipeline's
    declared-outputs filter drops them, which is the intended outcome for a row with no label.
    """

    NAME: ClassVar[str] = "CtkpInterventionsScript"

    # AACT intervention_type -> biolink class, consulted for records the KP never normalized so
    # their mentions still carry a defensible label instead of a raw AACT enum. Types whose
    # biolink class would be a guess (OTHER) are deliberately absent and drop to no entity.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "drug": "Drug",
        "biological": "BiologicalEntity",
        "dietary_supplement": "Food",
        "combination_product": "Drug",
        "device": "Device",
        "procedure": "Procedure",
        "radiation": "Procedure",
        "diagnostic_test": "DiagnosticAid",
        "behavioral": "BehavioralFeature",
        "genetic": "GenomicEntity",
    }

    def text_of(self: Self, record: dict[str, Any]) -> str:
        """name first, then description -- the description repeats the name often enough that the
        mention stays findable, and the pair is what a reader sees on the trial record"""
        name: str = str(record.get("name") or "").strip()
        description: str = str(record.get("description") or "").strip()
        return f"{name}. {description}".strip() if description else name

    def mentions_of(self: Self, record: dict[str, Any], text: str) -> list[ResolvedMention]:
        """one resolved mention per KP match that kept a biolink category and occurs in the text"""
        resolved: list[ResolvedMention] = []
        for match in record.get("matches") or []:
            category: str | None = match.get("category")
            surface: str = str(match.get("matched_text") or "").strip()
            if not category or not surface or not in_text(surface, text):
                continue
            stripped: str = strip_biolink_prefix(str(category))
            if not ScriptUtils.is_biolink_category(stripped):
                continue
            resolved.append(
                ResolvedMention(
                    mention=surface,
                    category=stripped,
                    curie=match.get("curie"),
                    preferred_name=match.get("preferred_name"),
                    origin="fullmap" if match.get("curie") else "raw",
                )
            )
        return resolved

    def fallback_mentions(self: Self, record: dict[str, Any], text: str) -> list[ResolvedMention]:
        """records the KP never normalized fall back to their AACT type, keyed on the trial's own
        intervention name so the span is still a real surface in the text"""
        category: str | None = self.LABEL_MAP.get(str(record.get("intervention_type") or "").lower())
        name: str = str(record.get("name") or "").strip()
        if category is None or not in_text(name, text):
            return []
        return [ResolvedMention(mention=name, category=category, origin="fallback")]

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        (record,) = values
        text: str = self.text_of(record)
        if not text:
            return TrainingExample(text="")
        resolved: list[ResolvedMention] = self.mentions_of(record, text) or self.fallback_mentions(record, text)
        entities: list[Entity] = ScriptUtils.group_entities(resolved) if resolved else []
        return TrainingExample(text=text, entities=entities)


validate_label_map(CtkpInterventionsScript.LABEL_MAP, CtkpInterventionsScript.NAME)

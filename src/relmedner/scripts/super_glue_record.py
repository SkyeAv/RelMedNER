from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_category
from relmedner.models import Classification, Entity, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class SuperGlueRecordScript(Script):
    """turns one SuperGLUE ReCoRD row into one training example: gold spans under NamedThing plus
    the cloze query as a classification

    Every record is one CNN news passage annotated with gold entity spans (a dict of parallel
    lists) and a cloze query whose answers are candidate surfaces. Unlike the gliner scripts,
    nothing is re-resolved through fullmap: the spans here are dataset gold, not model-predicted,
    so all gold mentions ride under the single catch-all biolink class NamedThing (the same
    trust-gold philosophy as CtkpInterventionsScript). General-domain news text means fullmap
    categories would be noise, and ReCoRD declares no gold relations.

    The cloze task ships as one Classification(task="cloze entity resolution") whose labels are
    the surviving span surfaces and whose true labels are the gold answers that occur in that
    label set. Answers that never survive span validation drop rather than corrupting the label
    set; an example with no surviving gold answer ships entities only (subset semantics of the
    declared-outputs filter).
    """

    NAME: ClassVar[str] = "SuperGlueRecordScript"

    # ReCoRD annotates no entity types, so every gold span shares one catch-all biolink class;
    # validated at import by families.validate_category below
    CATEGORY: ClassVar[str] = "NamedThing"

    CLOZE_TASK: ClassVar[str] = "cloze entity resolution"

    def spans_of(self: Self, passage: str, entity_spans: Any) -> list[str]:
        """deduped surfaces of the gold spans that satisfy every survival rule; a malformed
        table yields zero spans rather than a crash (skip-don't-coerce)

        A span survives only when entity_spans is a dict of equal-length text/start/end lists,
        both offsets are non-bool ints with 0 <= start < end <= len(passage), and
        passage[start:end] == text exactly (char-exact; verified against live hub rows).
        Case differences stay distinct surfaces: consumers see verbatim gold text.
        """
        if not isinstance(entity_spans, dict):
            return []
        texts: Any = entity_spans.get("text")
        starts: Any = entity_spans.get("start")
        ends: Any = entity_spans.get("end")
        if not isinstance(texts, list) or not isinstance(starts, list) or not isinstance(ends, list):
            return []
        if not (len(texts) == len(starts) == len(ends)):
            return []
        surfaces: list[str] = []
        for text, start, end in zip(texts, starts, ends, strict=True):
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                continue
            if not isinstance(text, str) or not (0 <= start < end <= len(passage)):
                continue
            if passage[start:end] != text:
                continue
            if text not in surfaces:
                surfaces.append(text)
        return surfaces

    def classification_of(self: Self, query: Any, surfaces: list[str], answers: Any) -> Classification | None:
        """the cloze classification, or None when no gold answer lands in the label set"""
        if not surfaces or not isinstance(answers, list):
            return None
        true_label: list[str] = [answer for answer in answers if isinstance(answer, str) and answer in surfaces]
        if not true_label:
            return None
        return Classification(
            task=self.CLOZE_TASK,
            labels=surfaces,
            true_label=true_label,
            multi_label=len(true_label) > 1,
            prompt=query if isinstance(query, str) else None,
        )

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        passage, query, _candidates, entity_spans, answers = values
        if not passage:
            return TrainingExample(text="")
        surfaces: list[str] = self.spans_of(passage, entity_spans)
        resolved: list[ResolvedMention] = [ResolvedMention(mention=surface, category=self.CATEGORY, origin="raw") for surface in surfaces]
        entities: list[Entity] = ScriptUtils.group_entities(resolved) if resolved else []
        classification: Classification | None = self.classification_of(query, surfaces, answers)
        return TrainingExample(text=passage, entities=entities, classifications=[classification] if classification else [])


validate_category(SuperGlueRecordScript.CATEGORY, SuperGlueRecordScript.NAME)

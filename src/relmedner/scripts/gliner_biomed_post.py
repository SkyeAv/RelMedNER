from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import ParsedRow, RowFamily, validate_label_map
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils


class GlinerBiomedPostScript(Script):
    """streams gliner-biomed-post-training rows into zero-shot multi-task examples; the corpus splits
    into disjoint row families on its NER label set alone (native relations, classification option lists,
    'match' extraction, open-vocabulary NER, empty), so dispatch goes through RowFamily.REGISTRY"""

    NAME: ClassVar[str] = "GlinerBiomedPostScript"

    # this corpus's generic-English label vocabulary, merged over ScriptUtils.FALLBACK_LABEL_MAP at
    # resolution time; it lives with the dataset that needs it so sibling dataset worktrees extend
    # their own vocabulary without colliding on the shared base
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "person": "Agent",
        "politician": "Agent",
        "athlete": "Agent",
        "author": "Agent",
        "organization": "Agent",
        "medical condition": "Disease",
        "tumor type": "Disease",
        "location": "GeographicLocation",
        "city": "GeographicLocation",
        "protein": "Protein",
        "amino acid residue": "Protein",
        "quantity": "Attribute",
        "time": "Attribute",
        "date": "Attribute",
        "number": "Attribute",
        "money": "Attribute",
        "measurement": "Attribute",
        "property": "Attribute",
        "software": "InformationContentEntity",
        "technology": "InformationContentEntity",
        "book": "Publication",
        "event": "Event",
        "facility": "EnvironmentalFeature",
        "assessment": "Study",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, ner_value, negatives_value = values
        tokens: list[str] = [str(token) for token in tokens_value] if isinstance(tokens_value, list) else []
        ner: list[Any] = ner_value if isinstance(ner_value, list) else []
        negatives: tuple[str, ...] = tuple(str(item) for item in negatives_value) if isinstance(negatives_value, list) else ()
        if not tokens:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans = tuple(ScriptUtils.mention_spans(tokens, ner))
        row = ParsedRow(text=ScriptUtils.join_tokens(tokens), tokens=tokens, spans=spans, negatives=negatives, label_map=self.LABEL_MAP)
        labels: tuple[str, ...] = tuple(label for _, _, label in spans)
        for family in RowFamily.REGISTRY:
            if family.matches(labels):
                return family.build(row)
        return TrainingExample(text=row.text)


validate_label_map(GlinerBiomedPostScript.LABEL_MAP, GlinerBiomedPostScript.NAME)

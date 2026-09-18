from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from relmedner.gazetteer import extract_relations
from relmedner.models import Entity, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils

if TYPE_CHECKING:
    from relmedner.utils import ResolvedMention


class GlinerBiomedScript(Script):
    """streams gliner-biomed-pre-training rows into biolink-labeled entity examples"""

    NAME: ClassVar[str] = "GlinerBiomedScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, ner_value = values
        tokens: list[str] = [str(token) for token in tokens_value] if isinstance(tokens_value, list) else []
        ner: list[Any] = ner_value if isinstance(ner_value, list) else []
        if not tokens or not ner:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans: list[tuple[int, int, str]] = ScriptUtils.mention_spans(tokens, ner)
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(ScriptUtils.mentions(tokens, ner))
        # spans and mentions filter identically, so zip pairs each span with its resolution positionally
        resolved_spans: list[tuple[int, int, str]] = [(start, end, item.category) for (start, end, _), item in zip(spans, resolved, strict=True)]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=self._entities(resolved),
            relations=extract_relations(tokens, resolved_spans),
        )

    @staticmethod
    def _entities(resolved: list[ResolvedMention]) -> list[Entity]:
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

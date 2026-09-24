from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
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
        # the multi-class fan-out makes resolved longer than spans, so pair_spans re-pairs by
        # span_index (items[0] is the primary) and every fan-out row extends resolved_spans
        resolved_spans: list[tuple[int, int, str]] = [
            (start, end, item.category) for (start, end, _), items in ScriptUtils.pair_spans(spans, resolved) for item in items
        ]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(resolved),
            relations=extract_relations(tokens, resolved_spans),
        )

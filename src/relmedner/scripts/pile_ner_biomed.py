from __future__ import annotations

from typing import ClassVar, Self

from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class PileNerBiomedScript(Script):
    """streams Pile-NER-biomed-IOB rows (python-repr token/tag string columns) into biolink-labeled
    entity examples; unresolved labels are kept as PascalCased raw categories for zero-shot breadth"""

    NAME: ClassVar[str] = "PileNerBiomedScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, tags_value = values
        tokens: list[str] = ScriptUtils.parse_literal_list(tokens_value)
        tags: list[str] = ScriptUtils.parse_literal_list(tags_value)
        if not tokens or len(tokens) != len(tags):
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans: list[tuple[int, int, str]] = ScriptUtils.iob_spans(tags)
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions)
        # spans and mentions filter identically, so zip pairs each span with its resolution positionally;
        # fullmap hits the shared gate rejects fall through to fallback/raw, so no mention is dropped --
        # raw labels surface PascalCased (biolink-style casing) while fallback entries already name a
        # biolink class and stay untouched
        labeled: list[ResolvedMention] = [
            item if item.origin != "raw" else ResolvedMention(mention=item.mention, category=ScriptUtils.pascal_label(label), origin=item.origin)
            for item, (_, label) in zip(resolved, mentions, strict=True)
        ]
        resolved_spans: list[tuple[int, int, str]] = [(start, end, item.category) for (start, end, _), item in zip(spans, labeled, strict=True)]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )

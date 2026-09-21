from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from relmedner.gazetteer import extract_relations
from relmedner.models import Classification, Relation, RelationField, Structure, StructureField, TrainingExample
from relmedner.utils import ResolvedMention, ScriptUtils

RELATION_DELIMITER: str = " <> "
CLASSIFICATION_LABELS: frozenset[str] = frozenset({"label", "category", "class", "tag"})
EXTRACTION_LABELS: frozenset[str] = frozenset({"match"})
NEGATIVE_CAP_RATIO: int = 2
NEGATIVE_NAME_PREFIX: str = "not_"


@dataclass(frozen=True)
class ParsedRow:
    """one streamed post-training row after span validation; the shared currency is
    (start, end_inclusive, label) triples, which every sibling dataset format collapses into"""

    text: str
    tokens: list[str]
    spans: tuple[tuple[int, int, str], ...]
    negatives: tuple[str, ...] = ()
    label_map: dict[str, str] | None = None

    def surfaces(self: Self) -> list[str]:
        return [ScriptUtils.join_tokens(self.tokens[start : end + 1]) for start, end, _ in self.spans]


def in_text(surface: str, text: str) -> bool:
    """gliner2's sanitizer drops any relation whose field value is absent from the text, so surfaces
    that tokenization mangled (e.g. the dataset's 'CC-chemokines' against tokens 'CC', '-', 'chemokines')
    are filtered here rather than shipped to be silently discarded downstream"""
    return bool(surface) and surface.lower() in text.lower()


def validate_label_map(label_map: dict[str, str], owner: str) -> None:
    """fail loudly on any fallback label whose target is not a biolink class
    (mirrors gazetteer.validate_trigger_table and test_every_fallback_label_maps_to_a_biolink_category)"""
    for raw_label, category in label_map.items():
        if not ScriptUtils.is_biolink_category(category):
            raise ValueError(f"{owner} fallback {raw_label!r} -> {category!r} is not a biolink class")


class RowFamily(ABC):
    """one row shape of a multi-task corpus; subclasses self-register and are consulted in PRIORITY order,
    mirroring the Script.REGISTRY paradigm one level down. matches() is pure label-set dispatch -- the
    post-training corpus separates into five disjoint families on the NER label set alone"""

    PRIORITY: ClassVar[int]
    REGISTRY: ClassVar[list[RowFamily]] = []

    def __init_subclass__(cls: type[RowFamily], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        RowFamily.REGISTRY.append(cls())
        RowFamily.REGISTRY.sort(key=lambda instance: instance.PRIORITY)

    @abstractmethod
    def matches(self: Self, labels: tuple[str, ...]) -> bool:
        """labels is span-ordered; a row can carry many spans sharing a label, and which span carries
        the relation delimiter matters, so this is a tuple rather than a set"""

    @abstractmethod
    def build(self: Self, row: ParsedRow) -> TrainingExample:
        """turns one dispatched row into one gliner2 training example"""


class RelationFamily(RowFamily):
    """native gold relations -- span labels shaped 'head <> predicate <> tail', where the span marks the
    tail entity and the label prefix carries the head -- plus guarded sampled negatives"""

    PRIORITY: ClassVar[int] = 1

    def matches(self: Self, labels: tuple[str, ...]) -> bool:
        return any(RELATION_DELIMITER in label for label in labels)

    def build(self: Self, row: ParsedRow) -> TrainingExample:
        parsed: list[tuple[str, str, str]] = []
        for span in row.spans:
            if RELATION_DELIMITER not in span[2]:
                continue
            candidate = self._parse_positive(row, span)
            if candidate is not None and candidate not in parsed:
                parsed.append(candidate)
        relations = [
            Relation(name=predicate, fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)], evidence="asserted")
            for predicate, head, tail in parsed
        ]
        relations.extend(self._negatives(row, parsed))
        return TrainingExample(text=row.text, relations=relations)

    def _parse_positive(self: Self, row: ParsedRow, span: tuple[int, int, str]) -> tuple[str, str, str] | None:
        start, end, label = span
        head, _, raw_predicate = label.partition(RELATION_DELIMITER)
        head = head.strip()
        tail = ScriptUtils.join_tokens(row.tokens[start : end + 1])
        if not head or not raw_predicate.strip() or not in_text(head, row.text):
            return None
        predicate, _ = ScriptUtils.resolve_predicate(raw_predicate)
        return predicate, head, tail

    def _negatives(self: Self, row: ParsedRow, parsed: list[tuple[str, str, str]]) -> list[Relation]:
        """sampled hard negatives, guarded before they become training signal: the measured rows are a
        near-cartesian product of the row's entities x predicates minus positives (8.5% self-loops, up to
        99 per row), so drop self-loops, dedup on the resolved triple, keep the never-collide-with-positives
        invariant, and cap at 2x the row's positive count. They train under the biolink-shaped
        not_<predicate> name because gliner2 cannot carry a negation field alongside head/tail."""
        cap: int = NEGATIVE_CAP_RATIO * len(parsed)
        asserted: set[tuple[str, str, str]] = set(parsed)
        seen: set[tuple[str, str, str]] = set()
        negatives: list[Relation] = []
        for raw in row.negatives:
            if len(negatives) >= cap:
                break
            parts: list[str] = [part.strip() for part in raw.split(RELATION_DELIMITER)]
            if len(parts) != 3:
                continue
            head, raw_predicate, tail = parts
            if not head or not raw_predicate or not tail or head == tail or not in_text(head, row.text) or not in_text(tail, row.text):
                continue
            predicate, _ = ScriptUtils.resolve_predicate(raw_predicate)
            key = (predicate, head, tail)
            if key in asserted or key in seen:
                continue
            seen.add(key)
            negatives.append(
                Relation(
                    name=f"{NEGATIVE_NAME_PREFIX}{predicate}",
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    negated=True,
                    evidence="sampled_negative",
                )
            )
        return negatives


class ClassificationFamily(RowFamily):
    """option-list rows: spans mark the true labels inside the instruction's choice list"""

    PRIORITY: ClassVar[int] = 2

    def matches(self: Self, labels: tuple[str, ...]) -> bool:
        return bool(labels) and set(labels) <= CLASSIFICATION_LABELS

    def build(self: Self, row: ParsedRow) -> TrainingExample:
        # spans only ever mark true labels, so labels == true_label; the negative options live in the
        # prompt prose and are not separately addressable without parsing the instruction
        true_labels: list[str] = list(dict.fromkeys(row.surfaces()))
        return TrainingExample(
            text=row.text,
            classifications=[
                Classification(task="topic classification", labels=true_labels, true_label=true_labels, multi_label=len(true_labels) > 1)
            ],
        )


class ExtractionFamily(RowFamily):
    """instruction-prefixed span extraction: every span is a 'match' surface to pull out of the text"""

    PRIORITY: ClassVar[int] = 3

    def matches(self: Self, labels: tuple[str, ...]) -> bool:
        return bool(labels) and set(labels) <= EXTRACTION_LABELS

    def build(self: Self, row: ParsedRow) -> TrainingExample:
        return TrainingExample(
            text=row.text,
            structures=[Structure(name="extraction", fields=[StructureField(name="span", value=row.surfaces())])],
        )


class EntityFamily(RowFamily):
    """open-vocabulary NER: fullmap-first biolink resolution plus gazetteer distant-supervision relations"""

    PRIORITY: ClassVar[int] = 4

    def matches(self: Self, labels: tuple[str, ...]) -> bool:
        return bool(labels)

    def build(self: Self, row: ParsedRow) -> TrainingExample:
        ner: list[list[Any]] = [[start, end, label] for start, end, label in row.spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(ScriptUtils.mentions(row.tokens, ner), row.label_map)
        # spans and mentions filter identically, so zip pairs each span with its resolution positionally
        resolved_spans: list[tuple[int, int, str]] = [(start, end, item.category) for (start, end, _), item in zip(row.spans, resolved, strict=True)]
        return TrainingExample(
            text=row.text,
            entities=ScriptUtils.group_entities(resolved),
            relations=extract_relations(row.tokens, resolved_spans),
        )

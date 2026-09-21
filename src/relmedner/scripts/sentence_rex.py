from __future__ import annotations

from typing import ClassVar, Self

from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils

E1_OPEN: str = "<e1>"
E1_CLOSE: str = "</e1>"
E2_OPEN: str = "<e2>"
E2_CLOSE: str = "</e2>"
# exactly one of each literal marks a well-formed row (521 measured rows violate this)
TAG_LITERALS: tuple[str, ...] = (E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE)


def parse_tagged_sentence(sentence: str) -> tuple[str, str] | None:
    """pure (tagged sentence) -> (e1 surface, e2 surface); None means drop-the-row, never coerce.

    Extraction splits on the first '<eN>' and the sentence's single closer, so tag order (e2 before
    e1) cannot crash the parse and interleaved markup lands inside a surface where the '<' guard
    catches it. Every drop rule maps to a measured failure class on the real 44,115-row train split:
    550 null/blank sentences, 521 with tag counts != (1, 1, 1, 1), 18 with '<' inside a stripped
    surface (nested markup like 'H < sub>1</sub > receptor'), 0 empty surfaces, and 48
    case-insensitive self-loops -> 42,978 rows shipped.
    """
    if not sentence.strip():
        return None
    if any(sentence.count(tag) != 1 for tag in TAG_LITERALS):
        return None
    head: str = sentence.split(E1_OPEN, 1)[1].split(E1_CLOSE, 1)[0].strip()
    tail: str = sentence.split(E2_OPEN, 1)[1].split(E2_CLOSE, 1)[0].strip()
    if "<" in head or "<" in tail:
        return None
    if not head or not tail:
        return None
    if head.lower() == tail.lower():
        return None
    return head, tail


class SentenceRexScript(Script):
    """streams knowledgator/sentence_rex rows (tagged sentence, gold relation label) into
    relations-only gliner2 examples: the tags mark the two relation participants and carry no type
    labels, and predicates keep native snake_case names when no biolink member matches (17 of the
    measured 837 distinct labels are biolink members; the card advertises 847 unique relations)
    """

    NAME: ClassVar[str] = "SentenceRexScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out for this dataset will be [sentences, labels], so values = (sentence, label).

        The text strips ONLY the four tag literals -- no whitespace normalization -- because gliner2
        whitespace-tokenizes (the doubled spaces around '<e2> Venerable </e2>' are inert) and the
        measured 43,044/43,044 well-formed rows keep both surfaces verbatim in the tag-stripped text,
        which InputExample.validate() requires of every relation field value. No entity shapes exist
        for this corpus; bad rows return the empty example the pipeline filters downstream
        (skip-don't-coerce).
        """
        sentence_value, label_value = values
        sentence: str = sentence_value if isinstance(sentence_value, str) else ""
        label: str = label_value if isinstance(label_value, str) else ""
        parsed: tuple[str, str] | None = parse_tagged_sentence(sentence)
        if parsed is None or not label.strip():
            return TrainingExample(text="")
        head, tail = parsed
        predicate, _ = ScriptUtils.resolve_predicate(label)
        text: str = sentence
        for tag in TAG_LITERALS:
            text = text.replace(tag, "")
        return TrainingExample(
            text=text,
            relations=[
                Relation(
                    name=predicate,
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    description=ScriptUtils.predicate_description(predicate),
                    evidence="asserted",
                    negated=False,
                )
            ],
        )

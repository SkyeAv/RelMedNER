from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.models import Classification, TrainingExample
from relmedner.types import Script, ScriptValues


class SuperGlueMultiRCScript(Script):
    """turns SuperGlue MultiRC rows into answer-verification classification examples

    Every row is one (paragraph, question, answer, label) tuple in the columns_out order declared
    in ingests.yaml. MultiRC asks whether a candidate answer to a question about a passage is
    correct, so each mapped row emits exactly one Classification over the ['False', 'True']
    vocabulary with the row's own gold as true_label. The label arrives either as a ClassLabel
    int (0/1) or its decoded string ('False'/'True') depending on the streaming decode form, so
    both are accepted and neither decode form is assumed.

    Rows whose label is outside that set -- -1, None, bool, float, other strings -- ship as
    text-only examples: skip-don't-coerce, never guess a label. A row whose text coerces to
    empty ships empty with no classification, mirroring CtkpInterventionsScript's empty-record
    path. There is no LABEL_MAP (this corpus has no NER labels) and no dependency on fullmap,
    families, or the gazetteer.
    """

    NAME: ClassVar[str] = "SuperGlueMultiRCScript"

    def true_label_of(self: Self, label_value: Any) -> str | None:
        """maps the row's label to the classification vocabulary form, or None when unmappable

        bools are guarded BEFORE the int check: isinstance(True, int) is True in Python, so an
        unguarded bool would silently masquerade as 1/0 and guess a label the row never had.
        """
        if isinstance(label_value, bool):
            return None
        if isinstance(label_value, int):
            return {0: "False", 1: "True"}.get(label_value)
        if isinstance(label_value, str):
            return label_value if label_value in ("False", "True") else None
        return None

    def text_of(self: Self, paragraph_value: Any, question_value: Any, answer_value: Any) -> str:
        """paragraph first, then the question and the answer; None/non-string fields coerce to ""

        only real strings survive coercion (arbitrary objects never become text), and the
        all-empty row composes to "" so the empty-record path can drop it.
        """
        paragraph: str = paragraph_value.strip() if isinstance(paragraph_value, str) else ""
        question: str = question_value.strip() if isinstance(question_value, str) else ""
        answer: str = answer_value.strip() if isinstance(answer_value, str) else ""
        if not paragraph and not question and not answer:
            return ""
        return f"{paragraph} Question: {question} Answer: {answer}"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        paragraph_value, question_value, answer_value, label_value = values
        text: str = self.text_of(paragraph_value, question_value, answer_value)
        if not text:
            return TrainingExample(text="")
        true_label: str | None = self.true_label_of(label_value)
        if true_label is None:
            return TrainingExample(text=text)
        return TrainingExample(
            text=text,
            classifications=[Classification(task="answer verification", labels=["False", "True"], true_label=[true_label], multi_label=False)],
        )

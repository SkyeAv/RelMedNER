from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.models import Classification, TrainingExample
from relmedner.types import Script, ScriptValues


class PubmedQaScript(Script):
    """turns bigbio/pubmed_qa rows (PubMedQA, mit) into yes/no/maybe biomedical question
    answering classification examples.

    Every row is one (question, contexts, final_decision) tuple in the columns_out order
    declared in ingests.yaml. The question and its evidence-context sentences (CONTEXTS is a
    list of abstract snippets) compose the example text; the expert label `final_decision`
    (`yes` / `no` / `maybe`) is the true label over the fixed ['no', 'yes', 'maybe']
    vocabulary. Only the expert-labeled folds are declared (gold tier): the repo-id row key
    may carry one weight, and the 200k-row artificial split's weak heuristic labels would need
    silver, conflicting with the folds' gold. The artificial split contains no `maybe` rows
    (measured 185,703 yes / 14,297 no on 2026-09-24); the script's fixed 3-label vocabulary
    would still teach one consistent task if a future mirror ever declares it.

    Labels outside the yes/no/maybe set -- bools, ints, other strings, None -- ship as
    text-only examples: skip-don't-coerce, never guess a label (SuperGlueMultiRCScript
    precedent). A row whose question and contexts both coerce to empty ships empty with no
    classification. CONTEXTS entries that are not strings are dropped from the join; no
    dependency on fullmap, families, or the gazetteer.
    """

    NAME: ClassVar[str] = "PubmedQaScript"
    LABELS: ClassVar[list[str]] = ["no", "yes", "maybe"]

    def true_label_of(self: Self, label_value: Any) -> str | None:
        """maps the row's final_decision to the classification vocabulary form, or None when
        unmappable; bools are guarded before any str/int coercion (isinstance(True, int) is
        True in Python, so an unguarded bool would silently masquerade as a label)"""
        if isinstance(label_value, bool) or not isinstance(label_value, str):
            return None
        stripped: str = label_value.strip().lower()
        return stripped if stripped in self.LABELS else None

    def text_of(self: Self, question_value: Any, contexts_value: Any) -> str:
        """question first, then the context sentences joined with spaces; None/non-string
        fields coerce to "" (non-string CONTEXTS entries drop out of the join), and the
        all-empty row composes to "" so the empty-record path can drop it"""
        question: str = question_value.strip() if isinstance(question_value, str) else ""
        contexts: list[str] = []
        if isinstance(contexts_value, list):
            contexts = [part.strip() for part in contexts_value if isinstance(part, str) and part.strip()]
        return " ".join([question, *contexts])

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        question_value, contexts_value, label_value = values
        text: str = self.text_of(question_value, contexts_value)
        if not text:
            return TrainingExample(text="")
        true_label: str | None = self.true_label_of(label_value)
        if true_label is None:
            return TrainingExample(text=text)
        return TrainingExample(
            text=text,
            classifications=[
                Classification(
                    task="biomedical question answering",
                    labels=list(self.LABELS),
                    true_label=[true_label],
                    multi_label=False,
                )
            ],
        )

from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.models import Classification, TrainingExample
from relmedner.types import Script, ScriptValues


class MedQaScript(Script):
    """turns GBaker/MedQA-USMLE-4-options rows (MedQA, Jin et al. arXiv:2009.13081, cc-by-4.0)
    into gold four-option multiple-choice question answering classification examples.

    Every row is one (question, options, answer_idx) tuple in the columns_out order declared in
    ingests.yaml. The example text is the question stem followed by the four options rendered in
    the fixed A, B, C, D order as `A. <text>`; the gold label `answer_idx` (one of A|B|C|D) is
    the true label over the fixed ['A', 'B', 'C', 'D'] vocabulary. Rendering the options is what
    makes the letter label learnable, and it leaks nothing: the correct option text sits in the
    prompt alongside the three distractors, exactly as it does at exam time. The row's `answer`
    column (the correct option's text) is deliberately NOT declared in columns_out, so a label
    leak through the text is structurally impossible.

    Measured over all 10,178 train rows (laptop census, 2026-09-24): options is a dict with
    exactly the keys A, B, C, D and every value a non-empty string on 10,178/10,178 rows,
    answer_idx is one of A|B|C|D on 10,178/10,178 rows (A 2,584 / B 2,654 / C 2,557 / D 2,383),
    no question is empty (min 66 chars, mean 724, max 3,577), and options[answer_idx] equals
    answer on every row. The guards below therefore never fire on the declared split; they exist
    so a re-published or mirrored file degrades to text-only examples instead of teaching a
    guessed label.

    Drop rules, skip-don't-coerce (PubmedQaScript precedent):
    - a row whose question and options all coerce to empty ships as TrainingExample(text=""),
      the pipeline's empty-record path;
    - an unmappable answer_idx (None, bool, int, float, an out-of-vocabulary letter) ships a
      text-only example; bools are guarded before any str/int coercion because
      isinstance(True, int) is True in Python;
    - an unusable options value (not a dict, a key set other than exactly A, B, C, D, or any
      value that is not a non-empty string after strip()) ships a text-only example: without the
      rendered option set the letter label is unlearnable, and inventing or truncating option
      text would be a guess;
    - an empty question with usable options ships the options alone, never a fabricated stem.

    `metamap_phrases` and `meta_info` stay undeclared: the phrase list is a MetaMap mining
    artifact of the upstream release, not a gold annotation, and the exam-step tag is metadata
    with no bearing on the label. No dependency on fullmap, families, or the gazetteer.
    """

    NAME: ClassVar[str] = "MedQaScript"
    LABELS: ClassVar[list[str]] = ["A", "B", "C", "D"]

    def true_label_of(self: Self, label_value: Any) -> str | None:
        """maps the row's answer_idx to the classification vocabulary form, or None when
        unmappable; bools are rejected before any coercion so True never masquerades as a label"""
        if isinstance(label_value, bool) or not isinstance(label_value, str):
            return None
        letter: str = label_value.strip().upper()
        return letter if letter in self.LABELS else None

    def options_of(self: Self, options_value: Any) -> dict[str, str] | None:
        """returns the four rendered options keyed by letter, or None when the option set is
        unusable: not a dict, a key set that is not exactly A, B, C, D, or a value that is not a
        non-empty string after strip(). A superset (a five-option mirror row) is refused rather
        than truncated: dropping the extra option would silently misrepresent the source item"""
        if not isinstance(options_value, dict) or set(options_value) != set(self.LABELS):
            return None
        rendered: dict[str, str] = {}
        for letter in self.LABELS:
            value = options_value[letter]
            if not isinstance(value, str) or not value.strip():
                return None
            rendered[letter] = value.strip()
        return rendered

    def text_of(self: Self, question_value: Any, rendered: dict[str, str] | None) -> str:
        """question stem first, then the options in A, B, C, D order as `A. <text>`; a None or
        non-string question coerces to "" and an unusable option set contributes nothing, so the
        all-empty row composes to "" and the empty-record path can drop it"""
        question: str = question_value.strip() if isinstance(question_value, str) else ""
        if not rendered:
            return question
        return " ".join([question, *(f"{letter}. {rendered[letter]}" for letter in self.LABELS)]).strip()

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        question_value, options_value, label_value = values
        rendered: dict[str, str] | None = self.options_of(options_value)
        text: str = self.text_of(question_value, rendered)
        if not text:
            return TrainingExample(text="")
        true_label: str | None = self.true_label_of(label_value)
        # an unusable option set makes the letter label unlearnable, so the example ships
        # text-only even when answer_idx itself is in vocabulary
        if true_label is None or rendered is None:
            return TrainingExample(text=text)
        return TrainingExample(
            text=text,
            classifications=[
                Classification(
                    task="medical multiple-choice question answering",
                    labels=list(self.LABELS),
                    true_label=[true_label],
                    multi_label=False,
                )
            ],
        )

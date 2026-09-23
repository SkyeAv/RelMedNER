from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.models import Classification, TrainingExample
from relmedner.types import Script, ScriptValues


class GadBlurbScript(Script):
    """turns bigbio/gad gad_blurb_bigbio_text rows into gene-disease association classifications

    Every row is one (text, labels) tuple in the declared columns_out order. GAD sentences judge
    whether the sentence reports a gene-disease association: label 1 = associated, 0 = not. The
    hub parquet conversion carries the label as a one-element list of strings ("1"/"0"; measured
    over the full blurb census on wenceslaus 2026-09-23: train 2,227/2,034, validation 293/242,
    test 281/253, no third label, no nulls). The text carries literal @GENE$ / @DISEASE$
    anonymization placeholders and ships as-is by decision: the training signal is the
    association judgment, and the corpus has no entity spans at all, hence classifications-only
    output. Text is short (7-81 tokens, median 25) and passes through unmodified.

    Decode posture (skip, don't coerce): a one-element list is unwrapped first (the measured
    parquet shape), then bools are rejected BEFORE the int check (isinstance(True, int) is True
    in Python, so an unguarded bool would masquerade as 1/0 and guess a label the row never had),
    then int 1/0 and str "1"/"0" map to the vocabulary. Anything else - None, floats, empty or
    multi-element lists, other strings - ships text-only, which the declared-outputs filter
    drops. A row whose text is None/non-str/blank ships TrainingExample(text=""). There is no
    LABEL_MAP, no fullmap dependency, and no gazetteer.
    """

    NAME: ClassVar[str] = "GadBlurbScript"

    def true_label_of(self: Self, labels_value: Any) -> str | None:
        """maps the row's labels value to the classification vocabulary, or None when unmappable"""
        if isinstance(labels_value, list):
            if len(labels_value) != 1:
                return None
            labels_value = labels_value[0]
        if isinstance(labels_value, bool):
            return None
        if isinstance(labels_value, int):
            return {0: "not associated", 1: "associated"}.get(labels_value)
        if isinstance(labels_value, str):
            return {"0": "not associated", "1": "associated"}.get(labels_value)
        return None

    def text_of(self: Self, text_value: Any) -> str:
        """only real strings survive coercion (arbitrary objects never become text); a blank or
        whitespace-only text strips to "" so the empty-text path can drop the row"""
        return text_value.strip() if isinstance(text_value, str) else ""

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text_value, labels_value = values
        text: str = self.text_of(text_value)
        if not text:
            return TrainingExample(text="")
        true_label: str | None = self.true_label_of(labels_value)
        if true_label is None:
            return TrainingExample(text=text)
        return TrainingExample(
            text=text,
            classifications=[
                Classification(
                    task="gene-disease association",
                    labels=["associated", "not associated"],
                    true_label=[true_label],
                    multi_label=False,
                )
            ],
        )

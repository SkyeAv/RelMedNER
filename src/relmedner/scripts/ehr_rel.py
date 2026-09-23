from __future__ import annotations

from typing import ClassVar, Self

from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils

# mean-rating bar above which a pair ships as a related_to relation, measured on the FULL splits
# (wenceslaus census, 2026-09-22): 2,535 of 3,741 source/pairs rows rate >= 1.0 (a_source 21 of
# 111, b_source 2,514 of 3,630). The pipeline never emits negated=True, so a pair rated below the
# bar can neither ship as a positive nor as a negation; it drops.
SHIP_THRESHOLD: float = 1.0


def parse_rating(rating: object) -> float | None:
    """pure (rating cell) -> float; None means drop-the-row, never coerce.

    The hub parquet columns are declared all-string, so the measured decode form is a str float
    ("1.6", "0.3333333333333333"); int/float are accepted defensively in case a future parquet
    revision types the column numerically. bools are rejected before the numeric check
    (isinstance(True, int) is True) and unparseable strings return None.
    """
    if isinstance(rating, bool):
        return None
    if isinstance(rating, int | float):
        return float(rating)
    if isinstance(rating, str):
        try:
            return float(rating.strip())
        except ValueError:
            return None
    return None


def parse_concept_pair(head: object, tail: object) -> tuple[str, str] | None:
    """pure (head cell, tail cell) -> (head surface, tail surface); None means drop-the-row.

    Every drop rule maps to a measured failure class on the real splits (wenceslaus census of all
    four train splits, 2026-09-22): 0 blank surfaces, 0 case-insensitive self-loops. The guards
    stay because the corpus is human-curated in two independent batches and a detached punctuation
    token or a copy-paste duplicate would otherwise fabricate a training relation.
    """
    head_surface: str = head.strip() if isinstance(head, str) else ""
    tail_surface: str = tail.strip() if isinstance(tail, str) else ""
    if not head_surface or not tail_surface:
        return None
    if head_surface.lower() == tail_surface.lower():
        return None
    return head_surface, tail_surface


class EhrRelScript(Script):
    """streams bigbio/ehr_rel rows (concept pair + human relatedness rating) into relations-only
    gliner2 examples under the biolink predicate related_to.

    EHR-Rel (Schulz et al., COLING 2020) pairs SNOMED concepts sampled from real EHRs; five (a
    batch) or three (b batch) raters scored each pair's relatedness 0-3 and the hub ships the mean
    as a string. The corpus carries NO context text, so the emitted text is exactly the two
    surfaces joined by one space -- which keeps every relation field value a substring of text,
    gliner2's InputExample.validate() requirement. This is a trust-gold stance like
    SuperGlueRecordScript: the dataset's own concept labels ship as relation field values, no
    fullmap re-resolution (there is no prose to resolve against), and no LABEL_MAP (the predicate
    is fixed, and related_to is a verified tablassert.biolink.Predicates member). The source and
    bigbio_pairs subsets carry the same pairs as a+b under different column names, so the script
    is column-agnostic: it consumes (surface1, surface2, rating) positionally in columns_out
    order. Bad rows return the empty example the pipeline filters downstream (skip-don't-coerce).
    """

    NAME: ClassVar[str] = "EhrRelScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out for every declared entry is [surface1, surface2, rating]: the three source
        subsets project [snomed_label_1, snomed_label_2, mean_rating] and bigbio_pairs projects
        [text_1, text_2, label].
        """
        head_value, tail_value, rating_value = values
        rating: float | None = parse_rating(rating_value)
        parsed: tuple[str, str] | None = parse_concept_pair(head_value, tail_value)
        if parsed is None or rating is None or rating < SHIP_THRESHOLD:
            return TrainingExample(text="")
        head, tail = parsed
        return TrainingExample(
            text=f"{head} {tail}",
            relations=[
                Relation(
                    name="related_to",
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    description=ScriptUtils.predicate_description("related_to"),
                    evidence="asserted",
                    negated=False,
                )
            ],
        )

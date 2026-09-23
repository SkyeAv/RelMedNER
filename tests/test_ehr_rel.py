from __future__ import annotations

from relmedner.models import TrainingExample
from relmedner.scripts import EhrRelScript
from relmedner.types import Script

SCRIPT: EhrRelScript = EhrRelScript()


def row(
    surface1: object = "Abdominal pain",
    surface2: object = "Dysuria",
    rating: object = "1.6",
) -> tuple[object, ...]:
    """one EHR-Rel row in the columns_out order shared by all four declared entries: the two
    concept surfaces then the mean rating (the source subsets project snomed_label_1,
    snomed_label_2, mean_rating; bigbio_pairs projects text_1, text_2, label -- the script is
    column-agnostic by design)"""
    return (surface1, surface2, rating)


def test_the_script_self_registers_under_its_declared_name() -> None:
    assert isinstance(Script.REGISTRY["EhrRelScript"], EhrRelScript)


def test_a_well_formed_pair_ships_one_related_to_relation() -> None:
    """the happy path measured on 2,535 of 3,741 source/pairs rows: both surfaces ship verbatim,
    the predicate is the fixed biolink member related_to, and the relation is asserted-positive
    (this pipeline never emits negated=True)"""
    Example: TrainingExample = SCRIPT.run(row())

    assert Example.text == "Abdominal pain Dysuria"
    assert len(Example.relations) == 1
    Relation = Example.relations[0]
    assert Relation.name == "related_to"
    assert Relation.evidence == "asserted"
    assert Relation.negated is False
    assert [(field.name, field.value) for field in Relation.fields] == [("head", "Abdominal pain"), ("tail", "Dysuria")]


def test_every_emitted_relation_value_occurs_in_the_emitted_text() -> None:
    """gliner2's InputExample.validate() requires every relation field value to be a substring of
    text; the joined-pair text makes that hold by construction, and the test pins it"""
    Example: TrainingExample = SCRIPT.run(row("Cough", "Chronic cough", "2.6"))

    assert Example.relations[0].fields[0].value in Example.text
    assert Example.relations[0].fields[1].value in Example.text


def test_the_rating_threshold_drops_below_one_and_keeps_the_boundary() -> None:
    """mean rating < 1.0 drops (1,206 of 3,741 measured): the pipeline never emits negations, so a
    weak-mean pair can neither ship as a positive nor as a negation; 1.0 exactly ships"""
    Dropped02: TrainingExample = SCRIPT.run(row(rating="0.2"))
    Dropped08: TrainingExample = SCRIPT.run(row(rating="0.8"))
    DroppedThird: TrainingExample = SCRIPT.run(row(rating="0.6666666666666666"))
    Boundary: TrainingExample = SCRIPT.run(row(rating="1.0"))

    for dropped in (Dropped02, Dropped08, DroppedThird):
        assert dropped.relations == []
        assert dropped.populated() == frozenset()
    assert Boundary.relations[0].name == "related_to"


def test_a_b_batch_thirds_rating_ships_when_at_or_above_the_bar() -> None:
    """b batch ratings sit on a 1/3 grid (3 raters) instead of a's 0.2 grid (5 raters); both grid
    forms must parse and compare against the same threshold"""
    Kept: TrainingExample = SCRIPT.run(row(rating="2.3333333333333335"))
    Dropped: TrainingExample = SCRIPT.run(row(rating="0.3333333333333333"))

    assert Kept.relations[0].name == "related_to"
    assert Dropped.populated() == frozenset()


def test_an_unparseable_rating_drops_the_row() -> None:
    """0 unparseable ratings measured across all four train splits; the guard stays so a future
    revision cannot fabricate a relation from a coercion"""
    for bad in ("n/a", "", "  ", None, True):
        Example: TrainingExample = SCRIPT.run(row(rating=bad))
        assert Example.populated() == frozenset(), f"rating {bad!r} must drop the row"


def test_a_blank_surface_drops_the_row() -> None:
    """0 blank surfaces measured; a blank head or tail would fabricate an empty mention value the
    gliner2 validator rejects"""
    assert SCRIPT.run(row(surface1="")).populated() == frozenset()
    assert SCRIPT.run(row(surface2=None)).populated() == frozenset()


def test_a_case_insensitive_self_loop_drops_the_row() -> None:
    """0 self-loops measured (sentence_rex measured 48, so the failure class is real in the wild);
    a head equal to its own tail carries no relational signal"""
    Example: TrainingExample = SCRIPT.run(row(surface1="Cough", surface2="cough", rating="2.0"))

    assert Example.populated() == frozenset()


def test_a_distinct_pair_differing_only_by_case_still_ships() -> None:
    """the self-loop guard is equality, not distinctness: two different surfaces that merely share
    a case pattern ship, and whitespace is stripped before the comparison"""
    Example: TrainingExample = SCRIPT.run(row(surface1=" Abdominal pain ", surface2="Dysuria "))

    assert Example.relations[0].fields[0].value == "Abdominal pain"
    assert Example.relations[0].fields[1].value == "Dysuria"


def test_numeric_rating_decode_forms_are_accepted_defensively() -> None:
    """the measured hub decode form is a string (every column is declared string), but a future
    parquet revision may type the column numerically; int and float must behave identically"""
    ByFloat: TrainingExample = SCRIPT.run(row(rating=1.6))
    ByInt: TrainingExample = SCRIPT.run(row(rating=2))
    BelowByFloat: TrainingExample = SCRIPT.run(row(rating=0.8))

    assert ByFloat.relations[0].name == "related_to"
    assert ByInt.relations[0].name == "related_to"
    assert BelowByFloat.populated() == frozenset()


def test_a_realistic_batch_yields_at_the_measured_rate() -> None:
    """nonzero-yield guard over rows copied from the wenceslaus census (the silent-zero-yield bug
    shipped once in this repo's history): 5 of these 8 rows rate >= 1.0 and 3 drop (two zero-rated,
    one 0.8), mirroring the measured 2,535/3,741 ship rate's direction"""
    Batch: list[tuple[object, ...]] = [
        row("Abdominal pain", "Dysuria", "1.6"),
        row("Abdominal pain", "Sore mouth", "0.0"),
        row("Cough", "Chronic cough", "2.6"),
        row("Anxiety", "Backache", "0.8"),
        row("Impotence", "Complaining of erectile dysfunction", "2.6"),
        row("Knee pain", "Osteoarthritis", "2.0"),
        row("Conjunctivitis", "Counselling", "0.0"),
        row("Hearing loss", "Giddiness", "1.2"),
    ]

    Shipped: list[TrainingExample] = [SCRIPT.run(values) for values in Batch]

    assert sum(1 for example in Shipped if example.relations) == 5
    for example in Shipped:
        if example.relations:
            assert example.relations[0].fields[0].value in example.text
            assert example.relations[0].fields[1].value in example.text

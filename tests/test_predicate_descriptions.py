"""Style invariants for the curated predicate-description overlay.

These tests exist because the training data consumes description strings as gliner2
relation_descriptions label prompts: a regression in register (policy prose, schema
plumbing text) or in the contrastive boundaries between confusable predicates would
silently degrade every downstream training example. Each test guards one documented
invariant from specs/predicate-description-overlay.md.
"""

from relmedner.gazetteer import PREDICATE_TRIGGERS
from relmedner.utils import ScriptUtils

# curation-policy and schema-engineering prose has no place in a label prompt; these
# tokens only appear in biolink source text that the overlay must replace
POLICY_TOKENS = ("knowledge_level", "ABox", "grouping mixin", "association basis qualifier")

MIN_WORDS = 5
MAX_WORDS = 60


def descriptions() -> dict[str, str]:
    """final per-predicate prompt text as every script call site sees it"""
    return {predicate: ScriptUtils.predicate_description(predicate) or "" for predicate in PREDICATE_TRIGGERS}


def test_overlay_covers_every_declared_predicate() -> None:
    """a predicate shipping without a description loses its label prompt entirely, so
    coverage must be total over the gazetteer, not best-effort"""
    overlay = ScriptUtils._load_description_overlay()
    missing = [predicate for predicate in PREDICATE_TRIGGERS if not overlay.get(predicate)]
    assert missing == []


def test_overlay_file_is_a_non_empty_mapping() -> None:
    """the loader must never merge a silently-empty overlay over every biolink prompt"""
    overlay = ScriptUtils._load_description_overlay()
    assert overlay
    assert all(isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in overlay.items())


def test_no_policy_or_schema_prose_in_any_prompt() -> None:
    """policy prose (knowledge_level assertion rules) and schema plumbing (ABox, grouping
    mixin, qualifier plumbing) teach the model curation mechanics, not span semantics"""
    offenders = [(predicate, token) for predicate, text in descriptions().items() for token in POLICY_TOKENS if token in text]
    assert offenders == []


def test_prompt_length_bounds() -> None:
    """the gliner2 encoder packs label prompts inline with the text; runaway lengths
    (the biolink treats slot ran 158 words) shrink the usable context budget"""
    lengths = {predicate: len(text.split()) for predicate, text in descriptions().items()}
    assert all(MIN_WORDS <= count <= MAX_WORDS for count in lengths.values()), lengths


def test_inverse_pairs_cross_reference() -> None:
    """inverse pairs are the highest-confusion structure; each direction must name the
    other so the model learns the subject/object asymmetry from the prompt itself"""
    text = descriptions()
    assert "treated_by" in text["treats"] and "treats" in text["treated_by"]
    assert "caused_by" in text["causes"] and "causes" in text["caused_by"]
    assert "increases_amount_or_activity_of" in text["decreases_amount_or_activity_of"]
    assert "decreases_amount_or_activity_of" in text["increases_amount_or_activity_of"]


def test_near_synonym_pairs_cross_reference() -> None:
    """near-synonyms (binds/interacts_with, associated_with/correlated_with) need an
    explicit boundary naming the sibling, or the model collapses them into one class"""
    text = descriptions()
    assert "interacts_with" in text["binds"] and "binds" in text["interacts_with"]
    assert "correlated_with" in text["associated_with"] and "associated_with" in text["correlated_with"]
    assert "preventative_for_condition" in text["treats"]


def test_confusable_prompts_carry_inline_exemplars() -> None:
    """exemplars are the strongest single lever for guideline-following models (GoLLIE
    ablation); the most-confused predicates must show a concrete mention pair"""
    text = descriptions()
    assert "E.g." in text["binds"]
    assert "E.g." in text["treats"]

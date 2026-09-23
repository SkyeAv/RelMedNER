from __future__ import annotations

from typing import Any

import pytest

from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import TrainingExample
from relmedner.scripts import SyntheticNerAdeTweetsScript
from relmedner.scripts.synthetic_ner_ade_tweets import parse_ann_spans
from relmedner.types import Script
from relmedner.utils import ResolvedMention, ScriptUtils

SCRIPT: SyntheticNerAdeTweetsScript = SyntheticNerAdeTweetsScript()

# real corpus rows (verbatim from the probe samples); offsets are END-EXCLUSIVE, measured
# 10,341/10,341 on the full corpus
ROW_A_TEXT = (
    "@USER_________ Simpinicline and Teriflunomide are the ones I'm on. "
    "The only side effect I've had is nasopharyngitis which is a fancy word for a cold."
)
ROW_A_ANN = "T1\tADE 100 115\tnasopharyngitis\nN1\tReference T1 meddra_llt_id:10028810\tNasopharyngitis\n"
ROW_B_TEXT = "@USER_________ Rosuvastatin is a statin. I had a heart attack on it."
ROW_B_ANN = "T1\tADE 49 61\theart attack\nN1\tReference T1 meddra_llt_id:10028596\tMyocardial infarction\n"

ATTACK_START = ROW_B_TEXT.index("heart attack")
ATTACK_END = ATTACK_START + len("heart attack")
SIBLING = "Rosuvastatin"
SIBLING_START = ROW_B_TEXT.index(SIBLING)
SIBLING_END = SIBLING_START + len(SIBLING)
TEXT_LEN = len(ROW_B_TEXT)

SIBLING_LINE = f"T2\tADE {SIBLING_START} {SIBLING_END}\t{SIBLING}"


def record(text: str = ROW_B_TEXT, ann: str = ROW_B_ANN) -> dict[str, Any]:
    return {"text": text, "ann": ann}


def dispatch_row(ann: str, text: str = ROW_B_TEXT) -> TrainingExample:
    _, example = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), (record(text, ann),)))
    return example


def patch_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """stand-in for the shared chain's map/fallback + raw-tail tiers; fullmap itself is not
    mounted on the laptop and is exercised in the skip-gated test below. Mimics _resolve: a raw
    label the map covers resolves to the mapped biolink class (fallback origin), anything else
    stays raw so pascal_raw_labels cases it"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        label_map = label_map or {}
        return [
            ResolvedMention(
                mention=mention,
                category=label_map[raw.lower()] if raw.lower() in label_map else raw,
                origin="fallback" if raw.lower() in label_map else "raw",
            )
            for mention, raw in mentions
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))


def mentions_of(example: TrainingExample) -> list[str]:
    return [mention for entity in example.entities for mention in entity.mentions]


def test_the_script_self_registers_under_its_declared_name() -> None:
    assert isinstance(Script.REGISTRY["SyntheticNerAdeTweetsScript"], SyntheticNerAdeTweetsScript)


def test_parse_ann_spans_reads_end_exclusive_offsets_and_ignores_n_lines() -> None:
    """the parser is the ingest's whole annotation contract: T-lines become (start, end_exclusive,
    raw_label) triples, N-line provenance is never read, and the surface field is not trusted"""
    assert parse_ann_spans(ROW_A_ANN, len(ROW_A_TEXT)) == [(100, 115, "ADE")]
    assert parse_ann_spans(ROW_B_ANN, TEXT_LEN) == [(ATTACK_START, ATTACK_END, "ADE")]
    assert parse_ann_spans("N1\tReference T1 meddra_llt_id:10028810\tNasopharyngitis\n", len(ROW_A_TEXT)) == []
    assert parse_ann_spans("", TEXT_LEN) == []


def test_a_t_line_with_wrong_tab_arity_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """an arity != 3 tab-split line is not a well-formed T-line; skip-don't-coerce keeps the row's
    well-formed sibling instead of aborting the whole annotation"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE {ATTACK_START} {ATTACK_END}\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_a_t_line_with_non_integer_offsets_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """non-integer offsets can never index the text; the span drops rather than coerce via int()"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE {ATTACK_START} xx\theart attack\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_a_degenerate_start_equals_end_span_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """a zero-length span covers no characters; 0 exist in the corpus but a defensive row must not
    crash or ship an empty-surface mention"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE {ATTACK_START} {ATTACK_START}\theart attack\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_an_inverted_start_greater_than_end_span_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """start > end is a corrupt annotation; the slice would be empty, so the span drops"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE {ATTACK_END} {ATTACK_START}\theart attack\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_an_out_of_bounds_low_start_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """a negative start is outside the text; gliner2 could not locate the surface, so the span drops"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE -1 {ATTACK_END}\theart attack\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_an_out_of_bounds_high_end_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """an end past the text extent is the classic measured OOB defect; the span drops instead of
    slicing short and training a truncated surface"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE {ATTACK_START} {TEXT_LEN + 10}\theart attack\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_a_discontinuous_span_drops_while_its_sibling_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """a ';' in the offsets field marks a discontinuous BRAT span, which the single-token-interval
    bridge cannot represent; 0 exist in the corpus, and a defensive row drops it"""
    patch_resolve(monkeypatch)
    Example = dispatch_row(f"T1\tADE {ATTACK_START} {ATTACK_END};{ATTACK_END + 1} {ATTACK_END + 3}\theart attack on\n{SIBLING_LINE}\n")

    assert "heart attack" not in mentions_of(Example)
    assert SIBLING in mentions_of(Example)


def test_an_empty_ann_row_ships_a_text_only_example(monkeypatch: pytest.MonkeyPatch) -> None:
    """8,502 measured docs (8,500 negatives plus pos_5808 train and pos_341 val) carry an empty
    .ann; they must ship text-only so the declared-outputs filter drops them, never raise"""
    patch_resolve(monkeypatch)
    for Ann in ("", "   \n"):
        Example = dispatch_row(Ann)

        assert Example.text == ScriptUtils.join_tokens(FullmapMiner.tokenize(ROW_B_TEXT))
        assert Example.entities == []
        assert Example.populated() == frozenset()


def test_an_ann_of_only_n_lines_ships_a_text_only_example(monkeypatch: pytest.MonkeyPatch) -> None:
    """provenance N-lines without any T-line carry no spans; never reading them means they also
    cannot anchor a mention"""
    patch_resolve(monkeypatch)
    Example = dispatch_row("N1\tReference T1 meddra_llt_id:10028810\tNasopharyngitis\n")

    assert Example.entities == []
    assert Example.populated() == frozenset()


def test_realistic_rows_yield_entities_and_rejoin_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """nonzero yield over real-shaped rows is the silent-zero-yield guard: every positive row must
    emit entities, and the emitted text is the re-joined token stream, not the raw text (gliner2's
    containment rule)"""
    patch_resolve(monkeypatch)
    RowCText = "Metformin gives me terrible nausea every morning."
    Rows = [
        {"text": ROW_A_TEXT, "ann": ROW_A_ANN},
        {"text": ROW_B_TEXT, "ann": ROW_B_ANN},
        {"text": RowCText, "ann": f"T1\tADE {RowCText.index('nausea')} {RowCText.index('nausea') + 6}\tnausea\n"},
    ]

    for Row in Rows:
        _, Example = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), (Row,)))

        assert Example.entities != []
        assert Example.text == ScriptUtils.join_tokens(FullmapMiner.tokenize(Row["text"]))
    _, RowA = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), (Rows[0],)))
    assert {entity.label: entity.mentions for entity in RowA.entities} == {"DiseaseOrPhenotypicFeature": ["nasopharyngitis"]}
    _, RowB = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), (Rows[1],)))
    assert {entity.label: entity.mentions for entity in RowB.entities} == {"DiseaseOrPhenotypicFeature": ["heart attack"]}


def test_every_emitted_mention_surface_is_a_substring_of_the_emitted_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """gliner2's InputExample.validate() discards spans it cannot find in the text; because the
    emitted text is the re-joined token stream, every mention surface must be a substring of it"""
    patch_resolve(monkeypatch)
    RowCText = "Metformin gives me terrible nausea every morning."
    Rows = [
        {"text": ROW_A_TEXT, "ann": ROW_A_ANN},
        {"text": ROW_B_TEXT, "ann": ROW_B_ANN},
        {"text": RowCText, "ann": f"T1\tADE {RowCText.index('nausea')} {RowCText.index('nausea') + 6}\tnausea\n"},
    ]

    for Row in Rows:
        _, Example = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), (Row,)))
        for mention in mentions_of(Example):
            assert mention in Example.text


def test_the_ade_label_resolves_through_the_map_and_a_raw_label_stays_pascalcased(monkeypatch: pytest.MonkeyPatch) -> None:
    """label-map edges: 'ade' reaches the biolink fallback tier (DiseaseOrPhenotypicFeature), while
    an out-of-map raw label (a synthetic RASH line) must not be guessed into a bucket and ships as
    its PascalCased raw self"""
    patch_resolve(monkeypatch)
    Text = "I got a rash after starting the pill."
    RashStart = Text.index("rash")
    Ann = f"T1\tADE {RashStart} {RashStart + 4}\trash\nT2\tRASH {RashStart} {RashStart + 4}\trash\n"
    _, Example = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), (record(text=Text, ann=Ann),)))

    Labels = {entity.label: entity.mentions for entity in Example.entities}
    assert Labels["DiseaseOrPhenotypicFeature"] == ["rash"]
    assert Labels["Rash"] == ["rash"]
    Raw = next(entity for entity in Example.entities if entity.label == "Rash")
    assert Raw.description is None


def test_every_label_map_target_is_a_biolink_category() -> None:
    """tablassert Categories HAS DiseaseOrPhenotypicFeature and has NO SignOrSymptom; the import-time
    validate_label_map guard pins the value as a real biolink class"""
    assert all(ScriptUtils.is_biolink_category(category) for category in SyntheticNerAdeTweetsScript.LABEL_MAP.values())


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """the registry NAME is what ingests.yaml's task.name must equal; dispatch must return the
    declared outputs tuple alongside a populated example for a positive row"""
    patch_resolve(monkeypatch)
    Outputs, Example = Script.dispatch("SyntheticNerAdeTweetsScript", (("entities",), ({"text": ROW_A_TEXT, "ann": ROW_A_ANN},)))

    assert Outputs == ("entities",)
    assert Example.populated() == frozenset({"entities"})
    assert mentions_of(Example) == ["nasopharyngitis"]


@pytest.mark.skipif(not ScriptUtils.fullmap_available(), reason="fullmap database is not mounted")
def test_fullmap_resolves_the_ade_tail_through_the_map_fallback_tier() -> None:
    """real resolution chain: a mention fullmap cannot resolve still reaches the biolink class via
    LABEL_MAP's fallback tier, proving 'ade' lands on DiseaseOrPhenotypicFeature without fullmap"""
    Resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(
        [("totally-unresolvable-xyz", "ade")], label_map=SyntheticNerAdeTweetsScript.LABEL_MAP
    )

    assert Resolved[0].origin == "fallback"
    assert Resolved[0].category == "DiseaseOrPhenotypicFeature"
    assert ScriptUtils.is_biolink_category(Resolved[0].category)

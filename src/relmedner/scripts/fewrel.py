from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

from relmedner.families import validate_category
from relmedner.models import Entity, Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils

PARTICIPANT_SUFFIXES: tuple[str, ...] = ("(e1,e2)", "(e2,e1)")
"""SemEval label tails marking which participant fills which slot; val_semeval labels carry them
on 17 labels (9 + 8 rows) and the suffix is slot metadata, never part of the predicate name"""


def strip_participant_suffix(label: str) -> str:
    """pure (label) -> label without a SemEval participant suffix: "Component-Of(e1,e2)" and
    "Component-Of (e1,e2)" both become "Component-Of"; anything else returns unchanged"""
    for suffix in PARTICIPANT_SUFFIXES:
        if label.endswith(suffix):
            stripped: str = label[: -len(suffix)]
            return stripped[:-1] if stripped.endswith(" ") else stripped
    return label


def resolve_pid_label(label: str, pid2name: Mapping[str, Any]) -> str:
    """pure (label, pid2name) -> predicate name or "" when unresolvable; the converter passes the
    real pid2name at build time, the streaming run passes {} (no guessing downstream)

    The chain is strip participant suffix, then resolve a bare "P<digits>" id through
    pid2name[pid][0]; a P-id with no pid2name entry yields "" (skip-don't-coerce), and native
    names pass through untouched.
    """
    stripped: str = strip_participant_suffix(label)
    if stripped.startswith("P") and stripped[1:].isdigit():
        names: Any = pid2name.get(stripped)
        if isinstance(names, list) and names and isinstance(names[0], str):
            stripped = names[0]
        else:
            return ""
    return stripped


def entity_runs(tokens: list[str], index_groups: Any) -> list[str]:
    """surfaces of the surviving contiguous index runs, one surface per run; a malformed run
    drops, never coerces (wrong container shape, non-int or bool element, empty run, any index
    outside 0 <= i < len(tokens), or non-contiguous indices)

    The measured corpus has 0 out-of-bounds, 0 unsorted runs, and 0 bool/str elements across all
    70,851 rows, so every drop here is pure defense; a run of consecutive indices [13, 14]
    surfaces as " ".join(tokens[13..15]), a verbatim substring of the emitted text.
    """
    if not isinstance(index_groups, list):
        return []
    surfaces: list[str] = []
    for group in index_groups:
        if not isinstance(group, list) or not group:
            continue
        if any(isinstance(index, bool) or not isinstance(index, int) for index in group):
            continue
        if any(not 0 <= index < len(tokens) for index in group):
            continue
        if any(group[position + 1] != group[position] + 1 for position in range(len(group) - 1)):
            continue
        surfaces.append(" ".join(tokens[group[0] : group[-1] + 1]))
    return surfaces


class FewRelScript(Script):
    """turns one FewRel avro record into one training example: the dataset's own gold token-index
    spans under NamedThing plus the row's gold predicate resolved to biolink-shaped snake_case

    FewRel ships tokenized sentences annotated with gold head/tail token-index groups and a gold
    relation label (a Wikidata P-id on train_wiki, a native snake_case name on val_pubmed, a
    SemEval name with a participant suffix on val_semeval). The spans are dataset gold, so
    nothing is re-resolved through fullmap: every gold mention rides under the single catch-all
    biolink class NamedThing (the trust-gold philosophy of CtkpInterventionsScript and
    SuperGlueRecordScript). Surfaces are derived from tokens plus indices, never from the
    detokenized h_text/t_text fields (91.8% of those case-mismatch the token slice, 6.1% even
    case-insensitively), and the converter omits them from the container entirely.

    Measured census over the full six-split corpus (wenceslaus probes): 70,851 rows total --
    train_wiki 44,800 (64 relations x 700), val_wiki 11,200 (16 x 700), val_nyt 2,500 (25 x
    100), val_semeval 8,851 (17 labels), val_pubmed 1,000 (10 x 100), and pubmed_unsupervised
    2,500 (no relation label). Indices are lists of lists of python ints with 0 out-of-bounds,
    0 unsorted runs, and 0 bool/str elements, so every defensive drop below is never exercised
    by real data. Multi-run entity mentions exist: 1-run 87,903, 2-run 1,624, 3-run 68, and
    4-run 5 of 89,600 train_wiki mentions, and each contiguous run emits one mention. Self-loops
    (head and tail first surviving run equal case-insensitively) hit 9 rows, all val_semeval, and
    drop the relation only; head/tail token-index overlap is 0 rows. Resolution: 0 of 97 P-ids
    are missing from pid2name and 0 normalized-name collide; val_pubmed labels are already
    snake_case native (0 of 10 are biolink members, SentenceRex precedent) and SemEval labels
    carry "(e1,e2)"/"(e2,e1)" suffixes on 9 + 8 rows. The converter resolves P-ids at build time
    through resolve_pid_label with the real pid2name; a bare P-id that still reaches this script
    resolves against the empty mapping and ships entities only, never a guessed name.
    """

    NAME: ClassVar[str] = "FewRelScript"

    # FewRel annotates no entity types, so every gold span shares one catch-all biolink class;
    # validated at import by families.validate_category below
    CATEGORY: ClassVar[str] = "NamedThing"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """values = (record,) where record is the whole avro row {"tokens", "label", "h_indices",
        "t_indices"}; the converter writes one record per row and LocalAvroDataStream ships the
        entire dict, so the head/tail surfaces and the re-joined text all come from the same
        token list (InputExample.validate() requires every relation field value in the text)

        Drop rules: an empty token list returns the canonical empty example the declared-outputs
        filter drops downstream; a malformed index run drops that run alone while sibling runs
        still ship; a self-loop drops the relation but ships its entities; an empty or
        unresolvable label ships entities only. run never raises on a bad row (skip-don't-coerce).
        """
        (record,) = values
        if not isinstance(record, dict):
            record = {}
        raw_tokens: Any = record.get("tokens")
        tokens: list[str] = [str(token) for token in raw_tokens] if isinstance(raw_tokens, list) else []
        if not tokens:
            return TrainingExample(text="")
        text: str = ScriptUtils.join_tokens(tokens)
        head_runs: list[str] = entity_runs(tokens, record.get("h_indices"))
        tail_runs: list[str] = entity_runs(tokens, record.get("t_indices"))
        resolved: list[ResolvedMention] = [
            ResolvedMention(mention=surface, category=self.CATEGORY, origin="raw") for surface in [*head_runs, *tail_runs]
        ]
        entities: list[Entity] = ScriptUtils.group_entities(resolved) if resolved else []
        label: str = record.get("label") if isinstance(record.get("label"), str) else ""
        predicate: str = ScriptUtils.resolve_predicate(resolve_pid_label(label, {}))[0]
        head: str | None = head_runs[0] if head_runs else None
        tail: str | None = tail_runs[0] if tail_runs else None
        relations: list[Relation] = []
        if predicate and head is not None and tail is not None and head.lower() != tail.lower():
            relations = [
                Relation(
                    name=predicate,
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    description=ScriptUtils.predicate_description(predicate),
                    evidence="asserted",
                    negated=False,
                )
            ]
        return TrainingExample(text=text, entities=entities, relations=relations)


validate_category(FewRelScript.CATEGORY, FewRelScript.NAME)

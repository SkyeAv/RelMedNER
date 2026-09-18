from __future__ import annotations

import string
from collections.abc import Mapping

from tablassert.biolink import Predicates

from relmedner.models import Relation, RelationField

SENTENCE_BREAKS: frozenset[str] = frozenset({".", ";"})
MAX_TRIGGER_DISTANCE: int = 15

PREDICATE_TRIGGERS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "treats": (
        ("treats",),
        ("to", "treat"),
        ("is", "used", "to", "treat"),
        ("for", "the", "treatment", "of"),
        ("in", "the", "treatment", "of"),
    ),
    "associated_with": (("associated", "with"),),
    "interacts_with": (("interacts", "with"), ("interaction", "with")),
    "causes": (("causes",), ("caused", "by"), ("cause", "of")),
    "biomarker_for": (("biomarker", "for"),),
    "expressed_in": (("expressed", "in"),),
}


def validate_trigger_table(table: Mapping[str, tuple[tuple[str, ...], ...]]) -> None:
    """fail loudly on non-biolink predicates, malformed phrases, or one phrase claimed by two predicates"""
    valid_predicates: frozenset[str] = frozenset(predicate.value for predicate in Predicates)
    owners: dict[tuple[str, ...], str] = {}
    for predicate, phrases in table.items():
        if predicate not in valid_predicates:
            raise ValueError(f"predicate {predicate!r} is not a tablassert.biolink.Predicates member")
        for phrase in phrases:
            if not phrase:
                raise ValueError(f"predicate {predicate!r} has an empty phrase")
            for token in phrase:
                if not token:
                    raise ValueError(f"predicate {predicate!r} phrase {phrase!r} contains an empty token")
                if token != token.lower():
                    raise ValueError(f"predicate {predicate!r} phrase {phrase!r} contains uppercase token {token!r}")
            owner = owners.setdefault(phrase, predicate)
            if owner != predicate:
                raise ValueError(f"phrase {phrase!r} is claimed by both {owner!r} and {predicate!r}")


def find_triggers(tokens: list[str]) -> list[tuple[int, int, str]]:
    """left-to-right greedy scan; the longest phrase matching at a start position wins and consumes its span"""
    lowered: list[str] = [token.lower() for token in tokens]
    triggers: list[tuple[int, int, str]] = []
    start = 0
    while start < len(lowered):
        best_length: int = 0
        best_predicate: str | None = None
        for predicate, phrases in PREDICATE_TRIGGERS.items():
            for phrase in phrases:
                length = len(phrase)
                if length > best_length and lowered[start : start + length] == list(phrase):
                    best_length, best_predicate = length, predicate
        if best_predicate is None:
            start += 1
            continue
        triggers.append((start, start + best_length - 1, best_predicate))
        start += best_length
    return triggers


def _nearest_before(mention_spans: list[tuple[int, int, str]], trigger_start: int) -> tuple[int, int, str] | None:
    """the candidate with the greatest end still ending before the trigger is the head"""
    best: tuple[int, int, str] | None = None
    for span in mention_spans:
        if span[1] < trigger_start and (best is None or span[1] > best[1]):
            best = span
    return best


def _nearest_after(mention_spans: list[tuple[int, int, str]], trigger_end: int) -> tuple[int, int, str] | None:
    """the candidate with the smallest start still starting after the trigger is the tail"""
    best: tuple[int, int, str] | None = None
    for span in mention_spans:
        if span[0] > trigger_end and (best is None or span[0] < best[0]):
            best = span
    return best


def _surface(tokens: list[str], span: tuple[int, int, str]) -> str:
    """rejoin the mention's tokens the way they appeared in the token stream"""
    return " ".join(tokens[span[0] : span[1] + 1])


def _is_punctuation_only(surface: str) -> bool:
    """a surface stripped of punctuation and whitespace guards against span artifacts like a lone period"""
    return not surface.strip(string.punctuation).strip()


def extract_relations(tokens: list[str], mention_spans: list[tuple[int, int, str]]) -> list[Relation]:
    """pair each trigger with its nearest bracketing mentions under sentence-break, window, and surface guards"""
    relations: list[Relation] = []
    seen: set[tuple[str, tuple[int, int, str], tuple[int, int, str]]] = set()
    for trigger_start, trigger_end, predicate in find_triggers(tokens):
        head = _nearest_before(mention_spans, trigger_start)
        tail = _nearest_after(mention_spans, trigger_end)
        if head is None or tail is None:
            continue
        if any(tokens[index] in SENTENCE_BREAKS for index in range(head[1] + 1, trigger_start)):
            continue
        if any(tokens[index] in SENTENCE_BREAKS for index in range(trigger_end + 1, tail[0])):
            continue
        if trigger_start - head[0] > MAX_TRIGGER_DISTANCE or tail[0] - (trigger_end + 1) > MAX_TRIGGER_DISTANCE:
            continue
        head_surface = _surface(tokens, head)
        tail_surface = _surface(tokens, tail)
        if _is_punctuation_only(head_surface) or _is_punctuation_only(tail_surface):
            continue
        key = (predicate, head, tail)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            Relation(
                name=predicate,
                fields=[RelationField(name="head", value=head_surface), RelationField(name="tail", value=tail_surface)],
            )
        )
    return relations


validate_trigger_table(PREDICATE_TRIGGERS)

from __future__ import annotations

from collections.abc import Mapping

from tablassert.biolink import Predicates

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


validate_trigger_table(PREDICATE_TRIGGERS)

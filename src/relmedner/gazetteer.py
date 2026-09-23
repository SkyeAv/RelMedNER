from __future__ import annotations

import logging
import string
from collections.abc import Mapping

from pydantic import ValidationError
from tablassert.biolink import Predicates, Qualifiers

from relmedner.constants import JUNKY_CATEGORIES, NEGATIVE_NAME_PREFIX
from relmedner.models import GazetteerSpec, Relation, RelationField
from relmedner.utils import PredicateRangeGate, ScriptUtils

SENTENCE_BREAKS: frozenset[str] = frozenset({".", ";"})
MAX_TRIGGER_DISTANCE: int = 15

# 23 predicates / 129 phrases, mined from the Pile-NER-biomed-IOB corpus itself: every
# inter-entity gap n-gram (1-6 tokens, no sentence break) ranked by frequency, filtered to
# stopword-only gaps, and kept when a clear tablassert.biolink.Predicates member owned it.
# Direction matters: "such as"/"including" put the supertype first, so they map to superclass_of
# (not subclass_of); "caused by" maps to caused_by (not causes) because the bracketing heuristic
# puts the patient/condition head first. High-noise phrases ("in patients with", bare "during",
# bare "before", "within the", "levels in", "related to") were dropped after sampling showed
# they mostly emit co-occurrence rather than the predicate they claim.
# WHY a separate BUILTIN constant: configure_gazetteer rebuilds the mutable PREDICATE_TRIGGERS
# table from this constant plus the YAML-declared predicates on every parse, so the builtin
# arm must survive reconfiguration untouched (US-010)
BUILTIN_PREDICATE_TRIGGERS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "treats": (
        ("treats",),
        ("to", "treat"),
        ("is", "used", "to", "treat"),
        ("used", "to", "treat"),
        ("for", "the", "treatment", "of"),
        ("in", "the", "treatment", "of"),
        ("therapy", "for"),
        ("effective", "against"),
    ),
    "treated_by": (
        ("treated", "with"),
        ("were", "treated", "with"),
        ("was", "treated", "with"),
        ("undergoing",),
        ("receiving",),
        ("who", "received"),
        ("managed", "with"),
    ),
    "preventative_for_condition": (
        ("prevents",),
        ("to", "prevent"),
        ("prevention", "of"),
        ("protects", "against"),
        ("protective", "against"),
    ),
    "causes": (
        ("causes",),
        ("cause", "of"),
        ("induces",),
        ("leads", "to"),
        ("leading", "to"),
        ("results", "in"),
        ("resulting", "in"),
        ("triggers",),
    ),
    "caused_by": (
        ("caused", "by"),
        ("induced", "by"),
        ("due", "to"),
        ("secondary", "to"),
        ("resulting", "from"),
        ("attributable", "to"),
    ),
    "associated_with": (
        ("associated", "with"),
        ("is", "associated", "with"),
        ("are", "associated", "with"),
        ("was", "associated", "with"),
        ("were", "associated", "with"),
        ("linked", "to"),
        ("in", "association", "with"),
    ),
    "correlated_with": (
        ("correlated", "with"),
        ("correlates", "with"),
        ("in", "correlation", "with"),
    ),
    "interacts_with": (
        ("interacts", "with"),
        ("interaction", "with"),
        ("interacting", "with"),
        ("interactions", "with"),
    ),
    "binds": (
        ("binds",),
        ("binds", "to"),
        ("bound", "to"),
        ("binding", "to"),
    ),
    "biomarker_for": (
        ("biomarker", "for"),
        ("biomarkers", "for"),
        ("marker", "for"),
        ("predictor", "of"),
        ("indicative", "of"),
    ),
    "expressed_in": (
        ("expressed", "in"),
        ("expression", "in"),
        ("is", "expressed", "in"),
        ("are", "expressed", "in"),
        ("overexpressed", "in"),
    ),
    "located_in": (
        ("located", "in"),
        ("localized", "to"),
        ("localized", "in"),
        ("found", "in"),
        ("present", "in"),
    ),
    "decreases_amount_or_activity_of": (
        ("inhibits",),
        ("inhibited",),
        ("suppresses",),
        ("suppressed",),
        ("reduces",),
        ("reduced",),
        ("decreases",),
        ("blocks",),
        ("downregulates",),
        ("antagonizes",),
        ("to", "reduce"),
    ),
    "increases_amount_or_activity_of": (
        ("increases",),
        ("enhances",),
        ("stimulates",),
        ("upregulates",),
        ("activates",),
        ("augments",),
        ("potentiates",),
        ("promotes",),
    ),
    "has_adverse_event": (
        ("side", "effects", "of"),
        ("adverse", "events", "of"),
        ("adverse", "effects", "of"),
        ("toxicity", "of"),
    ),
    "diagnoses": (
        ("diagnosis", "of"),
        ("used", "to", "diagnose"),
        ("to", "diagnose"),
        ("diagnostic", "for"),
        ("diagnosed", "with"),
    ),
    "has_phenotype": (
        ("presenting", "with"),
        ("suffering", "from"),
        ("characterized", "by"),
        ("manifests", "as"),
    ),
    "part_of": (
        ("part", "of"),
        ("component", "of"),
        ("subunit", "of"),
    ),
    "in_taxon": (
        ("isolated", "from"),
        ("derived", "from"),
        ("obtained", "from"),
        ("in", "the", "genome", "of"),
    ),
    "superclass_of": (
        ("such", "as"),
        ("including",),
        ("includes",),
        ("include",),
        ("a", "type", "of"),
        ("a", "form", "of"),
        ("classified", "as"),
    ),
    "participates_in": (
        ("involved", "in"),
        ("participates", "in"),
        ("plays", "a", "role", "in"),
        ("implicated", "in"),
        ("mediates",),
        ("regulates",),
        ("modulates",),
        ("contributes", "to"),
    ),
    "precedes": (
        ("followed", "by"),
        ("prior", "to"),
        ("preceded", "by"),
    ),
    "occurs_in": (
        ("occurs", "in"),
        ("occurring", "in"),
        ("observed", "in"),
        ("detected", "in"),
        ("seen", "in"),
    ),
}

PREDICATE_TRIGGERS: dict[str, tuple[tuple[str, ...], ...]] = dict(BUILTIN_PREDICATE_TRIGGERS)
"""the live trigger table find_triggers scans; rebuilt from BUILTIN_PREDICATE_TRIGGERS plus the
last-parsed YAML section by configure_gazetteer (last-parse-wins), never mutated in place"""

_EMISSIONS: dict[str, int] = {}
"""per-YAML-declared-predicate emission counts, reset by every configure_gazetteer call; builtin
predicates are not tracked (their yield is already pinned by test_gazetteer.py's phrase locks)"""

_REPORTED_ZERO: set[str] = set()
"""declared predicates already warned about, so a corpus emits ONE WARNING per zero-yielding key,
not one per reporting call"""

_GAZETTEER_LOG = logging.getLogger("relmedner.gazetteer")


def configure_gazetteer(spec: GazetteerSpec | None) -> None:
    """rebuild the module trigger tables from BUILTIN_PREDICATE_TRIGGERS plus the YAML-declared
    predicates. Call sites: YamlIngestsParser.parse_ingests (driver, once per parse, so the
    tables always reflect the last parsed ingests.yaml) and ResolveMinedBatches.setup (every
    Beam worker, because driver-side module state does not survive serialization -- the spec
    ships through the pipeline as pickled DoFn data). Pure rebuild and idempotent: spec=None
    (the default when ingests.yaml declares no gazetteer section) leaves the tables equal to
    the builtins, so the driver/worker double configuration is harmless. The merge is ADDITIVE -- YAML phrases join the predicate they name and new
    predicate keys are added -- and every phrase claimed by two owners (builtin-vs-YAML or
    YAML-vs-YAML) raises a ValidationError naming both owners. WHY module tables instead of an
    injected parameter: find_triggers reads the module global, so a rebuild keeps every existing
    call site (fullmap_mine, tests) correct without threading state through the pipeline.
    Qualifiers and negation_cues are structurally validated by GazetteerSpec, but their scanner
    machinery lands with PR #22 (add-qualifiers-to-relationship-pipelines): declaring them here
    raises NotImplementedError naming that PR rather than silently accepting and ignoring them.
    """
    global PREDICATE_TRIGGERS, _EMISSIONS, _REPORTED_ZERO
    if spec is not None and (spec.qualifiers or spec.negation_cues):
        raise NotImplementedError(
            "gazetteer qualifiers/negation_cues are declared but the qualifier/negation scanner "
            "machinery is not in this tree -- it lands with PR #22 "
            "(add-qualifiers-to-relationship-pipelines). Structural validation only until then."
        )
    merged: dict[str, tuple[tuple[str, ...], ...]] = {predicate: tuple(phrases) for predicate, phrases in BUILTIN_PREDICATE_TRIGGERS.items()}
    declared: list[str] = []
    if spec is not None and spec.predicates is not None:
        owners: dict[tuple[str, ...], str] = {phrase: predicate for predicate, phrases in BUILTIN_PREDICATE_TRIGGERS.items() for phrase in phrases}
        for entry in spec.predicates:
            declared.append(entry.name)
            existing: tuple[tuple[str, ...], ...] = merged.get(entry.name, ())
            additions: list[tuple[str, ...]] = []
            for phrase in entry.triggers:
                key = tuple(phrase)
                owner = owners.setdefault(key, entry.name)
                if owner != entry.name:
                    raise _spec_validation_error(
                        f"phrase {key!r} is claimed by both {owner!r} (builtin or earlier YAML entry) and YAML predicate {entry.name!r}"
                    )
                if key not in existing and key not in additions:
                    additions.append(key)
            merged[entry.name] = existing + tuple(additions)
    validate_trigger_table(merged)
    PREDICATE_TRIGGERS = merged
    _EMISSIONS = {name: 0 for name in declared}
    _REPORTED_ZERO = set()


def _spec_validation_error(message: str) -> ValidationError:
    """configure-time ownership conflicts surface with the same pydantic ValidationError shape
    model parsing raises, so callers catch one error type for every gazetteer-section bug"""
    return ValidationError.from_exception_data(
        "GazetteerSpec",
        [{"type": "value_error", "loc": ("gazetteer", "predicates"), "input": None, "ctx": {"error": ValueError(message)}}],
    )


def report_zero_emission_triggers() -> None:
    """log ONE WARNING per YAML-declared predicate that has emitted zero relations so far, on the
    relmedner.gazetteer logger. Called from FullmapMiner.resolve_batch (fullmap mining's
    completion point): batches report incrementally and _REPORTED_ZERO guarantees exactly one
    WARNING per key per process, so a zero-yielding declared trigger can never ship a silent
    empty relation arm (the US-009 zero-yield lesson applied to the gazetteer)"""
    for name, count in _EMISSIONS.items():
        if count == 0 and name not in _REPORTED_ZERO:
            _REPORTED_ZERO.add(name)
            _GAZETTEER_LOG.warning("gazetteer predicate %r declared in ingests.yaml emitted zero relations", name)


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


PhraseIndex = dict[str, tuple[tuple[tuple[str, ...], str], ...]]
"""first token -> (phrase, key) candidates ordered longest first, then by table order"""

_PHRASE_INDEXES: dict[int, tuple[Mapping[str, tuple[tuple[str, ...], ...]], PhraseIndex]] = {}
"""id(table) -> (table, its index). The table object is held alongside its index so an id can
never be recycled onto a different table while cached; configure_gazetteer swaps the global
for a NEW dict rather than mutating one, so identity is the right cache key"""


def _phrase_index(table: Mapping[str, tuple[tuple[str, ...], ...]]) -> PhraseIndex:
    """bucket every phrase under its first token, longest first and table order within a length,
    so the first bucket entry that matches is exactly the phrase the full scan would pick"""
    cached = _PHRASE_INDEXES.get(id(table))
    if cached is not None and cached[0] is table:
        return cached[1]
    ordered: list[tuple[int, int, tuple[str, ...], str]] = []
    for key, phrases in table.items():
        for phrase in phrases:
            if phrase:
                ordered.append((-len(phrase), len(ordered), tuple(phrase), key))
    buckets: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for _length, _order, phrase, key in sorted(ordered):
        buckets.setdefault(phrase[0], []).append((phrase, key))
    index: PhraseIndex = {token: tuple(entries) for token, entries in buckets.items()}
    _PHRASE_INDEXES[id(table)] = (table, index)
    return index


def _scan(table: Mapping[str, tuple[tuple[str, ...], ...]], tokens: list[str]) -> list[tuple[int, int, str]]:
    """left-to-right greedy scan over a phrase table; the longest phrase matching at a start
    position wins and consumes its span (empty phrase tuples attach by type and never match);
    between equal-length matches the one earlier in table order wins.

    Only phrases whose first token equals the token at the start position are tried (first-token
    index), instead of every phrase in the table at every position: same result, and the
    per-token cost drops from O(total phrases) to one dict probe for the vast majority of tokens
    that begin no phrase at all."""
    index: PhraseIndex = _phrase_index(table)
    lowered: list[str] = [token.lower() for token in tokens]
    triggers: list[tuple[int, int, str]] = []
    start = 0
    total: int = len(lowered)
    while start < total:
        match: tuple[tuple[str, ...], str] | None = None
        for phrase, key in index.get(lowered[start], ()):
            length = len(phrase)
            if start + length <= total and tuple(lowered[start : start + length]) == phrase:
                match = (phrase, key)
                break
        if match is None:
            start += 1
            continue
        triggers.append((start, start + len(match[0]) - 1, match[1]))
        start += len(match[0])
    return triggers


def find_triggers(tokens: list[str]) -> list[tuple[int, int, str]]:
    """predicate-trigger scan; the longest phrase matching at a start position wins and consumes its span"""
    return _scan(PREDICATE_TRIGGERS, tokens)


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


def _has_sentence_break(tokens: list[str], start: int, end_exclusive: int) -> bool:
    return any(tokens[index] in SENTENCE_BREAKS for index in range(start, end_exclusive))


def _cue_covers(lowered: list[str], tokens: list[str], trigger_start: int, trigger_end: int) -> bool:
    """a negation cue left-scopes a trigger when the cue starts before it and ends no later
    than the trigger's last token, in the same sentence ('failed to prevent': the cue shares
    the auxiliary 'to' with the trigger); greedy longest-match mirrors the trigger scan so
    'did not' beats bare 'not'"""
    start = 0
    while start < trigger_start:
        best_length: int = 0
        for phrase in NEGATION_CUES:
            length = len(phrase)
            if length > best_length and lowered[start : start + length] == list(phrase):
                best_length = length
        if best_length:
            cue_end = start + best_length
            if cue_end <= trigger_end and not _has_sentence_break(tokens, cue_end, trigger_start):
                return True
            start += max(best_length, 1)
        else:
            start += 1
    return False


def _overlap(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _gap(a: tuple[int, int, str], b: tuple[int, int, str]) -> int:
    """token distance between two spans (0 when they touch/overlap, which callers reject)"""
    return max(a[0] - b[1], b[0] - a[1], 0)


def _restates_endpoint(tokens: list[str], span: tuple[int, int, str], endpoints: list[tuple[int, int, str]]) -> bool:
    """DAKP's qualifier_restarts_object guard: a context that restates a statement endpoint carries
    no new information. Positional overlap catches the endpoint itself; the surface check catches
    the same surface repeated elsewhere in the sentence ("treats diabetes in patients with
    diabetes"), which would otherwise emit a head==tail qualifier -- no signal, and gliner2 cannot
    represent it distinctly (the same drop the predicate path applies to self-loops)."""
    surface: str = _surface(tokens, span).lower()
    return any(_overlap(span, endpoint) or _surface(tokens, endpoint).lower() == surface for endpoint in endpoints)


def _qualifiers(
    tokens: list[str],
    mention_spans: list[tuple[int, int, str]],
    endpoints: list[tuple[int, int, str]],
    tails: frozenset[tuple[int, int, str]],
) -> list[Relation]:
    """attach DAKP's qualifier subset to statement endpoints. A qualifier relation only fires
    in a sentence where a predicate relation already fired (biolink: a qualifier is a
    *statement* qualifier); the host is the endpoint nearest the context span with tail
    preference on ties (DAKP hosts are the object/disease mentions). Context mentions are
    introduced by the slot's trigger phrases (patient templates, frequency/temporal cues) or
    identified by the slot's type gazetteer (anatomy/sex/population, DAKP's type-driven field
    map), must clear the slot's range gate, and never restate a statement endpoint (DAKP
    qualifier_restarts_object guard)."""
    relations: list[Relation] = []
    if not endpoints:
        # a qualifier qualifies a STATEMENT: with no fired predicate relation there is no host to
        # attach a context to, so nothing below could emit
        return relations
    seen: set[tuple[str, int, int, int, int]] = set()
    phrase_contexts: dict[str, list[tuple[int, int, str]]] = {}
    for _phrase_start, phrase_end, slot in _scan(QUALIFIER_TRIGGERS, tokens):
        context = _nearest_after(mention_spans, phrase_end)
        if (
            context is not None
            and context[0] - (phrase_end + 1) <= MAX_TRIGGER_DISTANCE
            and not _has_sentence_break(tokens, phrase_end + 1, context[0])
            and PredicateRangeGate.category_ok(QUALIFIER_RANGES[slot], context[2])
            and context[2] not in JUNKY_CATEGORIES
            and not _restates_endpoint(tokens, context, endpoints)
            and not _is_punctuation_only(_surface(tokens, context))
        ):
            phrase_contexts.setdefault(slot, []).append(context)
        elif (
            phrase_end - _phrase_start >= 1
            and QUALIFIER_RANGES[slot] is None
            and not _is_punctuation_only(_surface(tokens, (_phrase_start, phrase_end, slot)))
        ):
            # value-style slots (frequency/temporal): DAKP emits them as the cell's literal
            # text, so when no entity mention follows the phrase the multi-token phrase
            # surface itself is the qualifier value ("twice daily"), hosted by the nearest
            # statement endpoint (the shared emit loop picks it; endpoint overlap rejection
            # drops mid-statement values). Single-token cue words carry no value and never
            # fall back ("during" alone is not a temporal context).
            pseudo: tuple[int, int, str] = (_phrase_start, phrase_end, slot)
            if not _restates_endpoint(tokens, pseudo, endpoints):
                phrase_contexts.setdefault(slot, []).append(pseudo)
    for slot, allowed in QUALIFIER_RANGES.items():
        contexts: list[tuple[int, int, str]] = list(phrase_contexts.get(slot, ()))
        if not QUALIFIER_TRIGGERS[slot]:
            # type-gazetteer attachment (DAKP's field map): any in-sentence mention whose
            # category clears the slot's range gate is a candidate context
            contexts = [
                span
                for span in mention_spans
                if PredicateRangeGate.category_ok_strict(allowed, span[2])
                and span[2] not in JUNKY_CATEGORIES
                and not _restates_endpoint(tokens, span, endpoints)
                and not _is_punctuation_only(_surface(tokens, span))
            ]
        # biolink qualifiers are single-valued per statement and DAKP buckets one value per
        # slot, so a sentence emits at most one qualifier relation per slot (nearest context
        # wins) instead of drowning the predicate signal in per-mention attachments
        emitted: list[tuple[int, int, int, int, tuple[int, int, str], tuple[int, int, str]]] = []
        for context_span in contexts:
            # every context already cleared _restates_endpoint, which subsumes endpoint overlap,
            # so the host pick only needs the nearest-with-tail-preference rule
            host = min(endpoints, key=lambda endpoint: (_gap(endpoint, context_span), endpoint not in tails))
            key = (slot, host[0], host[1], context_span[0], context_span[1])
            if key in seen:
                continue
            emitted.append((_gap(host, context_span), context_span[0], host[0], host[1], host, context_span))
        for _gap_value, _start, _host_start, _host_end, host, context_span in sorted(emitted, key=lambda entry: entry[:4])[:1]:
            seen.add((slot, host[0], host[1], context_span[0], context_span[1]))
            relations.append(
                Relation(
                    name=slot,
                    fields=[
                        RelationField(name="head", value=_surface(tokens, host)),
                        RelationField(name="tail", value=_surface(tokens, context_span)),
                    ],
                    description=ScriptUtils.predicate_description(slot),
                    negated=False,
                    evidence="asserted",
                )
            )
    return relations


def extract_relations(tokens: list[str], mention_spans: list[tuple[int, int, str]]) -> list[Relation]:
    """pair each trigger with its nearest bracketing mentions under sentence-break, window,
    surface, and biolink domain/range guards; span[2] is the mention's biolink category (or its
    raw label -- non-biolink categories impose no range constraint, see PredicateRangeGate).
    A negation cue left-scoping the trigger re-encodes the statement as not_<predicate> with
    negated=True (RelationFamily's gliner2-safe negative encoding); qualifier context
    attachments ride along as extra relations over the fired statements."""
    relations: list[Relation] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    lowered: list[str] = [token.lower() for token in tokens]
    endpoints: list[tuple[int, int, str]] = []
    tails: set[tuple[int, int, str]] = set()
    for trigger_start, trigger_end, predicate in find_triggers(tokens):
        head = _nearest_before(mention_spans, trigger_start)
        tail = _nearest_after(mention_spans, trigger_end)
        if head is None or tail is None:
            continue
        if not PredicateRangeGate.accept(predicate, head[2], tail[2]):
            continue
        if _has_sentence_break(tokens, head[1] + 1, trigger_start):
            continue
        if _has_sentence_break(tokens, trigger_end + 1, tail[0]):
            continue
        if trigger_start - head[0] > MAX_TRIGGER_DISTANCE or tail[0] - (trigger_end + 1) > MAX_TRIGGER_DISTANCE:
            continue
        head_surface = _surface(tokens, head)
        tail_surface = _surface(tokens, tail)
        if _is_punctuation_only(head_surface) or _is_punctuation_only(tail_surface):
            continue
        # self-loops carry no signal and gliner2 cannot represent them distinctly; they arise when one
        # surface occurs on both sides of a trigger ("wrecks ... such as ... wrecks"), which the
        # surface-keyed pile-ner-type spans make common. Mirrors RelationFamily._negatives' head==tail drop
        if head_surface.lower() == tail_surface.lower():
            continue
        key = (predicate, head[0], head[1], tail[0], tail[1])
        if key in seen:
            continue
        seen.add(key)
        negated = _cue_covers(lowered, tokens, trigger_start, trigger_end)
        relations.append(
            Relation(
                name=f"{NEGATIVE_NAME_PREFIX}{predicate}" if negated else predicate,
                fields=[RelationField(name="head", value=head_surface), RelationField(name="tail", value=tail_surface)],
                negated=negated,
                description=ScriptUtils.predicate_description(predicate),
                evidence="asserted",
            )
        )
        if predicate in _EMISSIONS:
            _EMISSIONS[predicate] += 1
        endpoints.extend((head, tail))
        tails.add(tail)
    relations.extend(_qualifiers(tokens, mention_spans, endpoints, frozenset(tails)))
    return relations


validate_trigger_table(PREDICATE_TRIGGERS)

# --------------------------------------------------------------------------- qualifiers --
# DAKP's declared subset (tables/*.yaml, all nullable): six context qualifiers. Validation is
# against the installed tablassert.biolink.Qualifiers enum (the same enum DAKP's tests pin),
# with species_context_qualifier deliberately excluded -- tablassert marks it
# DISABLED_EDGE_FIELDS (never emittable; v12 disabled its derivation).
DISABLED_QUALIFIERS: frozenset[str] = frozenset({"species_context_qualifier"})

# Phrase-introduced qualifier contexts, mirroring DAKP's patient-template cue semantics:
# the phrase introduces the context mention, which must follow within the usual window and
# clear the slot's range gate. An empty phrase tuple means the slot attaches by DAKP's own
# type gazetteer (a typed mention in the sentence, see QUALIFIER_TYPE_GROUPS) -- anatomy,
# sex, and population contexts carry no reliable introductory phrase, exactly as in DAKP,
# where the field map is type-driven with no cue regex.
QUALIFIER_TRIGGERS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "disease_context_qualifier": (
        ("in", "patients", "with"),
        ("among", "patients", "with"),
        ("in", "those", "with"),
        ("in", "people", "with"),
        ("in", "subjects", "with"),
        ("in", "individuals", "with"),
        ("in", "patients", "who", "have"),
        ("in", "patients", "having"),
        ("in", "patients", "diagnosed", "with"),
        ("in", "patients", "suffering", "from"),
    ),
    "anatomical_context_qualifier": (),
    "sex_qualifier": (),
    "population_context_qualifier": (),
    "frequency_qualifier": (
        ("twice", "daily"),
        ("once", "daily"),
        ("three", "times", "daily"),
        ("once", "a", "week"),
    ),
    "temporal_context_qualifier": (
        ("during",),
        ("after", "surgery"),
        ("following", "surgery"),
    ),
}

# Slot -> the ancestor group a context mention's category must hit (DAKP's field map,
# range-gated the same way PredicateRangeGate gates predicates). None = any mention.
QUALIFIER_RANGES: Mapping[str, str | None] = {
    "disease_context_qualifier": "DIS",
    "anatomical_context_qualifier": "ANAT",
    "sex_qualifier": "SEX",
    "population_context_qualifier": "POP",
    "frequency_qualifier": None,
    "temporal_context_qualifier": None,
}

# Word-bounded negation cues, DAKP PREVENTION_CUE style (closed list, longest phrase wins).
# A cue left-scopes a fired predicate trigger in the same sentence: the emitted encoding
# reuses RelationFamily's gliner2-safe not_<predicate> name with negated=True (biolink's
# negated slot is boolean: "if set to true, then the association is negated i.e. is not true").
NEGATION_CUES: tuple[tuple[str, ...], ...] = (
    ("no", "evidence", "that"),
    ("not", "shown", "to"),
    ("not", "been", "shown", "to"),
    ("has", "not", "been", "shown", "to"),
    ("have", "not", "been", "shown", "to"),
    ("did", "not"),
    ("does", "not"),
    ("do", "not"),
    ("failed", "to"),
    ("failure", "to"),
    ("lack", "of"),
    ("absence", "of"),
    ("without",),
    ("not",),
    ("never",),
)


def validate_qualifier_table(table: Mapping[str, tuple[tuple[str, ...], ...]]) -> None:
    """fail loudly on non-biolink qualifier slots, the tablassert-disabled species slot,
    malformed phrases, or one phrase claimed by two slots"""
    valid_slots: frozenset[str] = frozenset(slot.value for slot in Qualifiers)
    owners: dict[tuple[str, ...], str] = {}
    for slot, phrases in table.items():
        if slot in DISABLED_QUALIFIERS:
            raise ValueError(f"qualifier slot {slot!r} is tablassert-disabled (never emittable)")
        if slot not in valid_slots:
            raise ValueError(f"qualifier slot {slot!r} is not a tablassert.biolink.Qualifiers member")
        for phrase in phrases:
            if not phrase:
                raise ValueError(f"qualifier slot {slot!r} has an empty phrase")
            for token in phrase:
                if not token:
                    raise ValueError(f"qualifier slot {slot!r} phrase {phrase!r} contains an empty token")
                if token != token.lower():
                    raise ValueError(f"qualifier slot {slot!r} phrase {phrase!r} contains uppercase token {token!r}")
            owner = owners.setdefault(phrase, slot)
            if owner != slot:
                raise ValueError(f"phrase {phrase!r} is claimed by both {owner!r} and {slot!r}")

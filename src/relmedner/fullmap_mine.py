"""Distant-supervision entity/relation mining over unlabeled text via the fullmap redb.

Pipeline per batch of documents (all knobs measured against the live fullmap, see PLAN.md):

1. tokenize each text with gliner2's own ``WhitespaceTokenSplitter`` (imported by file path
   because the ``gliner2`` package root pulls torch) so mention surfaces agree with the
   token boundaries gliner2 training itself uses;
2. enumerate contiguous n-grams 1..max_ngram as candidates, dropping per-token-cleaned
   empties (which also makes sentence-crossing impossible -- the splitter emits ``.`` as its
   own token), all-numeric grams, and all-function-word grams; add hyphen/slash folds,
   unicode/Greek folds, and one-directional in-document acronym expansions
   (``expansion (ACRO)`` -> resolve the expansion, keep the ACRO span);
3. one ``rs.normalize_terms`` + one ``lookup_rows`` + one ``filter_and_rank`` for the whole
   batch (distinct surfaces grow ~54k/100 docs with no cross-document saturation, so
   per-batch dedup is the only and sufficient lever);
4. annotate every fullmap row with EXACT = normalize(PREFERRED_NAME) == term -- the true
   exact-name-match signal. The published PR tiers are NOT a precision dial: PR=50 merely
   means a preferred name that already happens to be in sorted-stem order, and PR=250/500
   are dominated by stopword collisions ("of" -> CHEBI:30241 fluorosyl group);
5. per document: exclude NONHUMAN_PREFIXES (human-centric genes), prefer EXACT rows,
   apply the junk-category gate, the unigram gates (minimum length, GENELIKE casing rule,
   strict name agreement), reject spans that begin or end with a function word, then
   greedy longest-match non-overlap selection;
6. group entities by biolink category with class-definition + fullmap-CURIE descriptions,
   and run the shared predicate gazetteer over the mined spans (self-loops dropped,
   evidence="distant").

Deferred by decision (keep this comment as the slot): a teacher-distillation pass running
``Ihor/gliner-biomed-bi-large-v1.0`` over each document and keeping only mined spans that
agree with teacher spans. ``resolve_batch`` is the natural interception point.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from importlib.util import spec_from_file_location
from pathlib import Path
from typing import Any, ClassVar, Self

import polars as pl
from tablassert import rs
from tablassert.fullmap import filter_and_rank, lookup_rows

from relmedner.constants import (
    FOLD_CACHE_TOKENS,
    FUNCTION_WORDS,
    GENELIKE_CATEGORIES,
    JUNKY_CATEGORIES,
    MIN_UNIGRAM_LENGTH,
    NONHUMAN_PREFIXES,
)
from relmedner.gazetteer import extract_relations, report_zero_emission_triggers
from relmedner.models import Entity, FullmapTask, Relation, TrainingExample
from relmedner.utils import ResolvedMention, ScriptUtils


def _splitter_path() -> Path:
    """locate gliner2/processing/word_splitter.py WITHOUT importing the package root (torch)"""
    located = importlib.util.find_spec("gliner2")
    if located is None or not located.submodule_search_locations:
        raise ImportError("gliner2 is not installed; its word splitter is required for mining")
    return Path(next(iter(located.submodule_search_locations))) / "processing" / "word_splitter.py"


EDGE_NONWORD: re.Pattern[str] = re.compile(r"^[^\w]+|[^\w]+$")
ALL_NUMERIC: re.Pattern[str] = re.compile(r"^\d[\d.,:%/]*$")
ACRONYM: re.Pattern[str] = re.compile(r"^([A-Z][A-Z0-9]{1,7})$")
EXPANSION_ACRO: re.Pattern[str] = re.compile(r"([A-Za-z][A-Za-z0-9\- ]{6,90}?)\s*\(\s*([A-Z][A-Z0-9]{1,7})\s*\)")
GREEK: dict[str, str] = {"β": "beta", "α": "alpha", "γ": "gamma", "δ": "delta", "ω": "omega", "μ": "micro"}

_FOLD_CACHE: OrderedDict[str, tuple[str, ...]] = OrderedDict()
"""per-token fold results, bounded at FOLD_CACHE_TOKENS. Tokens recur across every n-gram of every
document, so memoizing the per-character work is what makes fold_variant cheap. Safe because folding
is per-character (greek map, NFKD, ascii-ignore, hyphen/slash to space, edge strip) and a space has
combining class 0, so folding the joined string and folding each token give the same word list."""


def _fold_text(text: str) -> tuple[str, ...]:
    """the raw fold of one string: greek letters spelled out (before NFKD, which would strip them),
    NFKD -> ascii, hyphen/slash -> space, non-word characters stripped at each part's edges"""
    for glyph, name in GREEK.items():
        text = text.replace(glyph, name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.replace("-", " ").replace("/", " ")
    return tuple(word for word in (EDGE_NONWORD.sub("", part) for part in text.split()) if word)


def _fold_token(token: str) -> tuple[str, ...]:
    """one token's fold, cached (see _FOLD_CACHE)"""
    cached = _FOLD_CACHE.get(token)
    if cached is not None:
        _FOLD_CACHE.move_to_end(token)
        return cached
    folded = _fold_text(token)
    while len(_FOLD_CACHE) >= FOLD_CACHE_TOKENS:
        _FOLD_CACHE.popitem(last=False)
    _FOLD_CACHE[token] = folded
    return folded


ACRO_PAREN: re.Pattern[str] = re.compile(r"\(\s*[A-Z][A-Z0-9]{1,7}\s*\)")
"""cheap linear pre-filter: any EXPANSION_ACRO match implies a '(ACRO)' substring, so texts without
one skip the lazy full-text scan in expansion_map"""


def load_splitter(path: Path) -> Any:
    """instantiate gliner2's WhitespaceTokenSplitter without importing the torch-pulling package root"""
    spec = spec_from_file_location("relmedner_word_splitter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gliner2 word splitter from {path}")
    module = types.ModuleType("relmedner_word_splitter")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WhitespaceTokenSplitter()


@dataclass(frozen=True, slots=True)
class Candidate:
    """one n-gram (or augmentation variant) with the document span it attributes to"""

    doc: int
    start: int
    end: int  # inclusive token index
    n: int  # gram length of the ATTRIBUTED span, not the variant
    surface: str  # cleaned attributed surface (" ".join of per-token cleaned raw tokens)
    key: str  # the exact lookup string for this variant
    bridged: bool = False  # acronym-bridge variant: key is the in-document expansion


@dataclass(frozen=True)
class MinedSpan:
    """one accepted mention span after all gates and longest-match selection"""

    start: int
    end: int
    surface: str
    curie: str
    preferred_name: str
    category: str


class FullmapMiner:
    """batched n-gram -> fullmap resolution producing gliner2 TrainingExamples"""

    _splitter: ClassVar[Any] = None

    def __init__(self: Self, task: FullmapTask) -> None:
        self.task: FullmapTask = task

    @classmethod
    def splitter(cls) -> Any:
        if cls._splitter is None:
            cls._splitter = load_splitter(_splitter_path())
        return cls._splitter

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return [token for token, _start, _end in cls.splitter()(text, lower=False)]

    @classmethod
    def db(cls) -> Path:
        """one shared mount-path resolution with the script path (cached in ScriptUtils)"""
        return ScriptUtils.fullmap_db()

    @classmethod
    def available(cls) -> bool:
        return ScriptUtils.fullmap_available()

    # ------------------------------------------------------------ candidate generation --

    @staticmethod
    def clean_token(token: str) -> str:
        return EDGE_NONWORD.sub("", token)

    @classmethod
    def fold_variant(cls, words: list[str]) -> list[str]:
        """greek letters spelled out, hyphen/slash -> space, unicode NFKD -> ascii.
        Greek replacement MUST precede ascii folding: NFKD decomposes 'β' to nothing.
        Folded per token and concatenated: equivalent to folding the joined string (folding is
        per-character and a space has combining class 0, see _FOLD_CACHE), and the per-token memo
        is what keeps the per-candidate call cheap."""
        folded: list[str] = []
        for word in words:
            folded.extend(_fold_token(word))
        return folded

    @classmethod
    def expansion_map(cls, text: str) -> dict[str, str]:
        """in-document acronym definitions: 'body mass index (BMI)' -> {'BMI': 'body mass index'}"""
        if "(" not in text or not ACRO_PAREN.search(text):
            return {}  # the pattern cannot match without a paren + '(ACRO)'; skip the full-text scan
        out: dict[str, str] = {}
        for match in EXPANSION_ACRO.finditer(text):
            out.setdefault(match.group(2), match.group(1).strip())
        return out

    @classmethod
    def ngram_candidates(cls, doc: int, text: str, max_ngram: int, tokens: list[str] | None = None) -> list[Candidate]:
        """contiguous n-grams plus measured-safe augmentation variants.

        `tokens` lets the caller pass the already-split document (resolve_batch needs the raw tokens
        for offsets anyway) instead of paying the splitter twice per document.
        """
        words = [cls.clean_token(token) for token in (cls.tokenize(text) if tokens is None else tokens)]
        expansions = cls.expansion_map(text)
        candidates: list[Candidate] = []
        # Per-token flags with prefix counts turn the three per-span scans from O(n) each into O(1):
        # a span drops when it holds an empty (cleaned-away) word, when every word is numeric, or
        # when every word is a function word. Same predicates, same drops, same order.
        total = len(words)
        prefix_empty = [0] * (total + 1)
        prefix_numeric = [0] * (total + 1)
        prefix_function = [0] * (total + 1)
        for index, word in enumerate(words):
            prefix_empty[index + 1] = prefix_empty[index] + (not word)
            prefix_numeric[index + 1] = prefix_numeric[index] + bool(ALL_NUMERIC.match(word))
            prefix_function[index + 1] = prefix_function[index] + (word.lower() in FUNCTION_WORDS)
        for n in range(1, max_ngram + 1):
            for i in range(total - n + 1):
                end = i + n
                if prefix_empty[end] > prefix_empty[i]:
                    continue  # covers sentence-crossing: "." cleans to empty
                if prefix_numeric[end] - prefix_numeric[i] == n:
                    continue
                if prefix_function[end] - prefix_function[i] == n:
                    continue
                span_words = words[i:end]
                surface = " ".join(span_words)
                keys = [surface]
                # A pure-ascii span with no hyphen or slash folds to itself (no greek, nothing for
                # NFKD to decompose, no separator to split, edges already clean), so the fold key
                # would fail the != surface test anyway: skip the fold for the vast majority.
                if not surface.isascii() or "-" in surface or "/" in surface:
                    folded = cls.fold_variant(span_words)
                    if len(folded) >= n and " ".join(folded) != surface:
                        keys.append(" ".join(folded))
                if n == 1:
                    acro = ACRONYM.match(span_words[0])
                    if acro and acro.group(1) in expansions:
                        # agreement is by construction (the key IS the in-document expansion),
                        # so the bridge bypasses the strict unigram gate in span_accepted
                        candidates.append(Candidate(doc, i, i + n - 1, n, surface, expansions[acro.group(1)], bridged=True))
                candidates.extend([Candidate(doc, i, i + n - 1, n, surface, key) for key in keys])
        return candidates

    # ---------------------------------------------------------------- batch resolution --

    @classmethod
    def resolve_batch(cls, rows: list[tuple[str, FullmapTask]], db: Path | None = None) -> list[TrainingExample]:
        """rows of (text, task) -> one shared redb round trip -> one TrainingExample each"""
        if not rows:
            report_zero_emission_triggers()
            return []
        db = db or cls.db()
        tokens_per_doc = [cls.tokenize(text) for text, _task in rows]  # once: candidates + offsets
        candidate_groups = [cls.ngram_candidates(doc, text, task.max_ngram, tokens_per_doc[doc]) for doc, (text, task) in enumerate(rows)]
        flat = [candidate for group in candidate_groups for candidate in group]
        if not flat:
            report_zero_emission_triggers()
            return [TrainingExample(text=text) for text, _task in rows]
        norm = rs.normalize_terms([candidate.key for candidate in flat])
        distinct = sorted({key for key in norm if key})
        best = cls._best_rows(db, distinct, rows[0][1].taxon)
        mined: list[TrainingExample] = []
        offset = 0
        for (_text, task), group, tokens in zip(rows, candidate_groups, tokens_per_doc, strict=True):
            keys = norm[offset : offset + len(group)]
            offset += len(group)
            spans = cls._select_spans(group, keys, best)
            entities = cls._entities(spans)
            relations = cls._relations(tokens, spans) if task.relations else []
            mined.append(TrainingExample(text=" ".join(tokens), entities=entities, relations=relations))
        report_zero_emission_triggers()
        return mined

    @classmethod
    def _best_rows(cls, db: Path, distinct: list[str], taxon: str) -> dict[str, dict[str, Any]]:
        """term -> one best fullmap row, EXACT-normalized-name rows winning ties"""
        rows = lookup_rows(db, distinct)
        if not rows:
            return {}
        frame = pl.DataFrame(rows)
        frame = frame.filter(~pl.col("CURIE").str.split(":").list.first().is_in(sorted(NONHUMAN_PREFIXES)))
        if frame.height == 0:
            return {}
        names = frame.select("PREFERRED_NAME").unique().get_column("PREFERRED_NAME").to_list()
        normalized = dict(zip(names, rs.normalize_terms(names), strict=True))
        frame = frame.with_columns(
            pl.col("PREFERRED_NAME").replace_strict(normalized, default=None).alias("PN_NORM"),
        ).with_columns((pl.col("PN_NORM") == pl.col("term")).alias("EXACT"))
        ranked = filter_and_rank(frame, pl.DataFrame({"term": distinct, "nlp_level": [1] * len(distinct)}), taxon, None, None, False)
        best: dict[str, dict[str, Any]] = {}
        for row in ranked.iter_rows(named=True):
            previous = best.get(row["term"])
            if previous is None or (row["EXACT"] and not previous["EXACT"]):
                best[row["term"]] = row
        return best

    # --------------------------------------------------------------------- span gates --

    @staticmethod
    def strict_unigram_agrees(surface: str, preferred_name: str) -> bool:
        """case-insensitive equality or simple plural agreement; rejects Porter2 derivational
        collisions ('oxidative'->oxide, 'enters'->enteritis, 'formation'->formate) while
        keeping inflection ('cancers'->cancer, 'gliomas'->glioma)"""
        low, name = surface.lower(), preferred_name.lower()
        return low == name or low.rstrip("s") == name or low == name.rstrip("s") or low + "s" == name

    @classmethod
    def span_accepted(cls, candidate: Candidate, row: dict[str, Any]) -> bool:
        """measured gates; see PLAN.md for the keep/reject evidence on every branch"""
        category = str(row["CATEGORY_NAME"])
        if category in JUNKY_CATEGORIES:
            return False
        if candidate.n == 1:
            surface = candidate.surface
            if len(surface) < MIN_UNIGRAM_LENGTH and not any(ch.isdigit() for ch in surface):
                return False
            if category in GENELIKE_CATEGORIES and surface.islower() and not any(ch.isdigit() for ch in surface):
                return False
            if not candidate.bridged and not cls.strict_unigram_agrees(surface, str(row["PREFERRED_NAME"])):
                return False
        return True

    @staticmethod
    def boundary_function_word(candidate: Candidate) -> bool:
        """entity mentions never begin or end with a closed-class word; their cores are
        separately enumerated n-grams, so rejecting the padded span loses nothing measured"""
        words = candidate.surface.split()
        return words[0].lower() in FUNCTION_WORDS or words[-1].lower() in FUNCTION_WORDS

    @classmethod
    def _select_spans(
        cls,
        group: list[Candidate],
        keys: list[str],
        best: dict[str, dict[str, Any]],
    ) -> list[MinedSpan]:
        accepted: list[tuple[Candidate, dict[str, Any]]] = []
        for candidate, key in zip(group, keys, strict=True):
            row = best.get(key)
            if row is None or not row["EXACT"]:
                continue
            if cls.boundary_function_word(candidate):
                continue
            if cls.span_accepted(candidate, row):
                accepted.append((candidate, row))
        # greedy longest-match: longest span first, then earliest start; overlaps lose
        accepted.sort(key=lambda item: (-(item[0].end - item[0].start + 1), item[0].start))
        used: set[int] = set()
        spans: list[MinedSpan] = []
        for candidate, row in accepted:
            token_range = range(candidate.start, candidate.end + 1)
            if any(index in used for index in token_range):
                continue
            used.update(token_range)
            spans.append(
                MinedSpan(
                    start=candidate.start,
                    end=candidate.end,
                    surface=candidate.surface,
                    curie=str(row["CURIE"]),
                    preferred_name=str(row["PREFERRED_NAME"]),
                    category=str(row["CATEGORY_NAME"]),
                )
            )
        return sorted(spans, key=lambda span: span.start)

    # ------------------------------------------------------------------ example shape --

    @classmethod
    def _entities(cls, spans: list[MinedSpan]) -> list[Entity]:
        # grouping, per-label first-evidence, and description building live in the shared
        # implementation (the same one the script path uses); mined spans always carry curie evidence
        return ScriptUtils.group_entities(
            [
                ResolvedMention(mention=span.surface, category=span.category, curie=span.curie, preferred_name=span.preferred_name, origin="fullmap")
                for span in spans
            ]
        )

    @classmethod
    def _relations(cls, tokens: list[str], spans: list[MinedSpan]) -> list[Relation]:
        mention_spans = [(span.start, span.end, span.category) for span in spans]
        relations: list[Relation] = []
        for relation in extract_relations(tokens, mention_spans):
            fields = {field.name: field.value for field in relation.fields}
            if fields["head"] == fields["tail"]:
                continue  # mined self-loop, e.g. ME3 -> ME3
            relations.append(Relation(name=relation.name, fields=list(relation.fields), negated=relation.negated, evidence="distant"))
        return relations

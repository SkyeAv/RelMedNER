from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils

# dataset relation types that map onto honest biolink predicates; everything else rides
# ScriptUtils.resolve_predicate (biolink member when one matches, biolink-shaped native
# snake_case otherwise), and the AND/OR integrators drop outright
RELATION_MAP: dict[str, str] = {"subsumes": "superclass_of", "has_temporal": "temporally_related_to"}
INTEGRATORS: frozenset[str] = frozenset({"and", "or"})


class ChiaScript(Script):
    """streams bigbio/chia rows (clinical-trial eligibility criteria with gold entity spans and
    id-linked relations) into biolink-labeled entity and relation examples, over two row shapes:
    the four flat subsets project a str `text`, while chia_bigbio_kb projects a one-passage
    `passages` list (measured 2000/2000 single passage, gapless tiling; `events` and
    `coreferences` are always empty and are ignored).

    Why the bridge: entity offsets are char offsets into raw text, end-EXCLUSIVE (verified on
    47,091/47,091 spans of both fixed variants), so the gliner2 whitespace splitter tokenizes each
    row once and char spans snap to token spans before the shared resolution chain runs. Entities
    can be multi-part (1,752 with 2 offsets, 53 with 3, 1 with 4, 1 with 10 on the scope subsets):
    each offset part becomes its own token span carrying the entity's label, and an entity's
    relation surface is the re-joined token slice of its first surviving part. The two `*_source`
    variants ship KNOWN-BAD offsets (the end-convention neither-rate is 58% and out-of-bounds is
    1.97% on chia_source); they are declared because all five subsets are wanted, the bridge's
    drop rules (out-of-bounds 0.02% on the fixed variants, whitespace slop, degenerate spans)
    handle them, and their lower yield is a documented measured fact, not something to fix.
    Relations are emitted from the gold arg1_id/arg2_id links: Subsumes maps to the biolink
    superclass_of (arg1 is the superclass), Has_temporal to the symmetric temporally_related_to,
    the has_* family keeps native snake_case names, and the logical integrators AND (2,631 scope /
    3,677 without_scope) and OR (7) drop because they are sentence combinatorics, not relation
    semantics. Dropped relations per measured rule: dangling arg id 0, id self-loop 0, surface
    self-loop 33 scope / 48 without_scope. The 16-type entity vocabulary maps where an honest
    biolink target exists (CONDITION, DRUG, PROCEDURE, PERSON, DEVICE, MEASUREMENT, OBSERVATION,
    QUALIFIER; 30,744 spans) and leaves the criterion-structure types deliberately unmapped
    (SCOPE, VALUE, TEMPORAL, REFERENCE_POINT, NEGATION, MULTIPLIER, MOOD, VISIT; 16,356 spans)
    so they surface as raw PascalCase tails.
    """

    NAME: ClassVar[str] = "ChiaScript"

    # lowercase dataset classes -> biolink classes; span counts measured on the full train split of
    # the scope subsets (census harness, all five subsets, 2,000 rows each). Values are validated
    # against tablassert Categories at import below.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "condition": "DiseaseOrPhenotypicFeature",
        "drug": "Drug",
        "procedure": "Procedure",
        "person": "Human",
        "device": "Device",
        "measurement": "ClinicalMeasurement",
        "observation": "ClinicalFinding",
        "qualifier": "ClinicalModifier",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        first, entities_value, relations_value = values
        if isinstance(first, str):
            text: str = first
        elif isinstance(first, list):
            text = self.passage_text(first)
        else:
            return TrainingExample(text="")
        entities: list[Any] = entities_value if isinstance(entities_value, list) else []
        relations: list[Any] = relations_value if isinstance(relations_value, list) else []
        if not text:
            return TrainingExample(text="")
        triples: list[tuple[str, int, int]] = list(FullmapMiner.splitter()(text, lower=False))
        tokens: list[str] = [token for token, _start, _end in triples]
        emitted: str = ScriptUtils.join_tokens(tokens)
        char_spans, char_ids = self.char_parts(entities)
        spans: list[tuple[int, int, str]] = ScriptUtils.char_spans_to_token_spans(triples, char_spans)
        # the bridge is order-preserving and never merges spans, so spans[i] is the i-th SURVIVING
        # char part; the parallel id list turns each surviving span back into its entity
        surfaces: dict[str, str] = {}
        for entity_id, (start, end, _label) in zip(char_ids, spans, strict=False):
            if entity_id not in surfaces:
                surfaces[entity_id] = ScriptUtils.join_tokens(tokens[start : end + 1])
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # raw labels surface PascalCased (biolink-style casing) while mapped/fallback entries already
        # name a biolink class and stay untouched
        labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
        return TrainingExample(
            text=emitted,
            entities=ScriptUtils.group_entities(labeled),
            relations=self.gold_relations(relations, surfaces, emitted),
        )

    @staticmethod
    def passage_text(passages: list[Any]) -> str:
        """concatenate the passages' text lists in offset order; the measured bigbio_kb rows carry
        exactly one gapless passage, so this is a reconstruction guard, not a guess"""
        ordered: list[tuple[int, str]] = []
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            offsets = passage.get("offsets")
            texts = passage.get("text")
            start = offsets[0][0] if isinstance(offsets, list) and offsets and isinstance(offsets[0], list) else 0
            if isinstance(texts, list):
                ordered.append((start, "".join(str(part) for part in texts)))
        return "".join(part for _start, part in sorted(ordered))

    @staticmethod
    def char_parts(entities: list[Any]) -> tuple[list[tuple[int, int, str]], list[str]]:
        """extract (start, end_exclusive, class) char triples plus the parallel entity id per part;
        malformed entries are skipped, never coerced (mirrors ScriptUtils.mention_spans) -- bounds,
        degeneracy, and mid-token snapping live in the bridge, not here. Multi-part entities emit
        one part per offset; a text/offsets length mismatch (measured 0 on the fixed subsets) drops
        the whole entity because the part-to-surface alignment is no longer trustworthy."""
        spans: list[tuple[int, int, str]] = []
        ids: list[str] = []
        for entry in entities:
            if not isinstance(entry, dict):
                continue
            label = entry.get("type")
            entity_id = entry.get("id")
            offsets = entry.get("offsets")
            texts = entry.get("text")
            if not isinstance(label, str) or not isinstance(entity_id, str) or not entity_id:
                continue
            if not isinstance(offsets, list) or not isinstance(texts, list) or len(texts) != len(offsets):
                continue
            for offset in offsets:
                if not isinstance(offset, list) or len(offset) != 2:
                    continue
                start, end = offset
                if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                    continue
                spans.append((start, end, label))
                ids.append(entity_id)
        return spans, ids

    @staticmethod
    def gold_relations(
        relations: list[Any],
        surfaces: dict[str, str],
        emitted: str,
    ) -> list[Relation]:
        """gold arg1_id/arg2_id links -> Relation records; every drop rule drops the relation, not
        the row (measured rates in the class docstring). The in-text backstop re-checks both
        surfaces against the emitted token stream."""
        out: list[Relation] = []
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            raw_type = relation.get("type")
            arg1 = relation.get("arg1_id")
            arg2 = relation.get("arg2_id")
            if not isinstance(raw_type, str) or not isinstance(arg1, str) or not isinstance(arg2, str):
                continue
            lowered = raw_type.strip().lower()
            if lowered in INTEGRATORS:
                continue
            head = surfaces.get(arg1)
            tail = surfaces.get(arg2)
            # dangling arg id and id self-loop guards (both measured 0); the surface self-loop
            # covers the measured 33/48 cases where two ids carry the same surface
            if head is None or tail is None or arg1 == arg2 or head.lower() == tail.lower():
                continue
            if head not in emitted or tail not in emitted:
                continue
            predicate = RELATION_MAP.get(lowered) or ScriptUtils.resolve_predicate(lowered)[0]
            out.append(
                Relation(
                    name=predicate,
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    description=ScriptUtils.predicate_description(predicate),
                    evidence="asserted",
                )
            )
        return out


validate_label_map(ChiaScript.LABEL_MAP, ChiaScript.NAME)

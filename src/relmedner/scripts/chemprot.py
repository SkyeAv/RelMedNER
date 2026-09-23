from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class ChemprotScript(Script):
    """streams bigbio/chemprot rows (chemprot_full_source flat source tables: pmid, title +
    abstract text, gold char-offset entities, gold CPR relations) into biolink-labeled entity and
    relation examples.

    Measured facts that shaped the mapping (full-split census over train/validation/test, do not
    re-measure): the hub parquet auto-conversion delivers entities and relations as dicts of
    parallel lists (the super_glue_record entity_spans shape), so both transpose before use, and
    a ragged or non-dict table yields zero rows of that shape, never a crash. Char offsets are
    END-EXCLUSIVE on 62,147/62,147 spans with 0 out-of-bounds, 0 whitespace boundary slop, and 0
    discontinuous spans, so the KnowledgatorBiomed char-offset bridge is reused verbatim and every
    annotation defect that does appear (malformed entries, 0 measured) drops the span only. The
    entity label census is exactly three classes: CHEMICAL 31,831, GENE-Y 20,157, GENE-N 10,159;
    GENE-N falls back to GeneFamily, not Gene, because -N marks nonspecific mentions ("kinase",
    "tumor necrosis factor") that name gene families/classes, and mislabeling them Gene would
    train "kinase" as a specific gene. Relations: arg1 is CHEMICAL in 100% of 15,739 relations
    and arg2 never is, so direction is always chemical -> gene; 0 self-loops, 0 dangling args.
    CPR:0 (Undefined, 3 relations) and CPR:10 ("Not", the corpus's 683 asserted negatives) drop
    because this pipeline never asserts negations (Relation.negated is always False by landed
    decision); the rows' entities still ship. CPR:7 (73 relations) and CPR:8 (61) keep native
    snake_case modulator/cofactor because biolink has no slot and regulates/has_catalyst would
    overcommit a direction or level the corpus leaves unspecified. 602 rows (253 train / 169
    validation / 180 test) carry entities with zero relations and ship under the permitted-shapes
    contract with no code path special-casing them. Gold relations are trusted as-is (native-gold
    stance, the sentence_rex pattern): no gazetteer overlay re-derives what the corpus already
    gold-labels, so evidence is "asserted", not "distant".
    """

    NAME: ClassVar[str] = "ChemprotScript"

    # the complete measured entity vocabulary, lowercased corpus labels -> biolink classes
    # (resolve_mentions probes the lowercased raw label, then the IOB-normalized form, so
    # whitespace/underscore drift needs no extra entries). Values are validated against tablassert
    # Categories at import below.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "chemical": "ChemicalEntity",
        "gene-y": "Gene",
        "gene-n": "GeneFamily",
    }

    # gold CPR labels -> predicates, keyed by the corpus's own "CPR:N" spelling (arg1 is the
    # chemical, arg2 the gene, in 100% of measured relations). Mapped values must be tablassert
    # Predicates members or one of the two documented native snake_case omissions below; the
    # module-scope guard fails at import on anything else so a typo never trains garbage
    # predicates. CPR:0 (Undefined) and CPR:10 ("Not", asserted negatives) are deliberately
    # absent: both drop, and the rows' entities still ship.
    PREDICATE_MAP: ClassVar[dict[str, str]] = {
        "CPR:1": "part_of",
        "CPR:2": "regulates",
        "CPR:3": "increases_amount_or_activity_of",
        "CPR:4": "decreases_amount_or_activity_of",
        "CPR:5": "increases_amount_or_activity_of",
        "CPR:6": "decreases_amount_or_activity_of",
        "CPR:7": "modulator",
        "CPR:8": "cofactor",
        "CPR:9": "is_substrate_of",
    }

    # native snake_case predicates biolink has no honest slot for (CPR:7 Modulator, CPR:8
    # Cofactor); kept per the resolve_predicate convention, documented omission in the README
    NATIVE_PREDICATES: ClassVar[frozenset[str]] = frozenset({"modulator", "cofactor"})

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out is [text, entities, relations], so values = (text, entities, relations).

        The entities/relations dict-of-parallel-lists transpose is skip-don't-coerce: a non-dict
        or ragged table yields zero rows of that shape (the row still ships what survived), and a
        malformed individual entry drops only itself. Relations referencing an entity whose span
        did not survive the bridge drop with it. Gold relations emit evidence="asserted",
        negated=False; CPR:0/CPR:10 and unknown labels drop the relation, never the row.
        """
        text_value, entities_value, relations_value = values
        text: str = text_value if isinstance(text_value, str) else ""
        if not text:
            return TrainingExample(text="")
        triples: list[tuple[str, int, int]] = list(FullmapMiner.splitter()(text, lower=False))
        tokens: list[str] = [token for token, _start, _end in triples]
        structs: list[dict[str, Any]] = self.entity_structs(entities_value) or []
        # the bridge carries each entity's id in the label slot so surviving token spans stay
        # aligned with their structs (char_spans_to_token_spans is order-preserving but silently
        # drops, and this must not re-derive its drop decisions)
        char_spans: list[tuple[int, int, str]] = [(struct["start"], struct["end"], struct["id"]) for struct in structs]
        spans: list[tuple[int, int, str]] = ScriptUtils.char_spans_to_token_spans(triples, char_spans)
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        structs_by_id: dict[str, dict[str, Any]] = {struct["id"]: struct for struct in structs}
        mentions: list[tuple[str, str]] = [
            (ScriptUtils.join_tokens(tokens[start : end + 1]), structs_by_id[entity_id]["type"]) for start, end, entity_id in spans
        ]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # raw labels surface PascalCased (biolink-style casing) while mapped and fallback entries
        # already name a biolink class and stay untouched
        labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
        # spans and mentions filter identically, so zip pairs each span with its resolution
        # positionally, and strict=True turns a dropped mention into an error instead of a silent
        # misalignment of every later span
        surfaces: dict[str, str] = {}
        for (_start, _end, entity_id), item in zip(spans, labeled, strict=True):
            surfaces[entity_id] = item.mention
        relations: list[Relation] = []
        for relation in self.relation_structs(relations_value) or []:
            predicate: str | None = self.gold_predicate(relation, surfaces)
            if predicate is None:
                continue
            head: str = surfaces[relation["arg1"]]
            tail: str = surfaces[relation["arg2"]]
            relations.append(
                Relation(
                    name=predicate,
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    description=ScriptUtils.predicate_description(predicate),
                    evidence="asserted",
                    negated=False,
                )
            )
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=relations,
        )

    @staticmethod
    def gold_predicate(relation: dict[str, Any], surfaces: dict[str, str]) -> str | None:
        """one gold CPR relation -> its mapped predicate, or None to drop (skip-don't-coerce).

        Drop rules, in order: non-str args or a dangling arg id (the entity's span did not
        survive the bridge; 0 measured), case-insensitive self-loop (0 measured, the sentence_rex
        rule), CPR:0 (Undefined) and CPR:10 ("Not", the corpus's asserted negatives; this
        pipeline never asserts negations), and any label outside CPR:1..CPR:9 (defensive drift
        guard, never guess).
        """
        rel_type: Any = relation["type"]
        arg1: Any = relation["arg1"]
        arg2: Any = relation["arg2"]
        if not isinstance(rel_type, str) or not isinstance(arg1, str) or not isinstance(arg2, str):
            return None
        if arg1 not in surfaces or arg2 not in surfaces:
            return None
        if surfaces[arg1].lower() == surfaces[arg2].lower():
            return None
        return ChemprotScript.PREDICATE_MAP.get(rel_type)

    @staticmethod
    def entity_structs(value: Any) -> list[dict[str, Any]] | None:
        """transpose the entities dict-of-parallel-lists into per-entity structs keyed by id; None
        means zero entities (non-dict table, a missing required key, or ragged list lengths), and
        a malformed individual entry (non-str id/type, offsets not a 2-list of non-bool ints)
        drops only that entity, never coerced"""
        if not isinstance(value, dict):
            return None
        columns: dict[str, list[Any]] = {}
        for key in ("id", "type", "text", "offsets"):
            column: Any = value.get(key)
            if not isinstance(column, list):
                return None
            columns[key] = column
        lengths: set[int] = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            return None
        structs: list[dict[str, Any]] = []
        for index in range(lengths.pop()):
            entity_id: Any = columns["id"][index]
            entity_type: Any = columns["type"][index]
            offsets: Any = columns["offsets"][index]
            if not isinstance(entity_id, str) or not isinstance(entity_type, str):
                continue
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                continue
            start, end = offsets
            if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                continue
            structs.append({"id": entity_id, "type": entity_type, "start": start, "end": end})
        return structs

    @staticmethod
    def relation_structs(value: Any) -> list[dict[str, Any]] | None:
        """transpose the relations dict-of-parallel-lists (type/arg1/arg2); None means no
        relations (non-dict table or ragged lengths), matching the entity transpose discipline"""
        if not isinstance(value, dict):
            return None
        columns: dict[str, list[Any]] = {}
        for key in ("type", "arg1", "arg2"):
            column: Any = value.get(key)
            if not isinstance(column, list):
                return None
            columns[key] = column
        lengths: set[int] = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            return None
        return [{"type": columns["type"][i], "arg1": columns["arg1"][i], "arg2": columns["arg2"][i]} for i in range(lengths.pop())]


def _validate_predicate_map() -> None:
    """import-time guard: every mapped predicate is a tablassert Predicates member or a documented
    native snake_case omission, so a typo fails at import rather than mislabeling relations"""
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label, predicate in ChemprotScript.PREDICATE_MAP.items():
        if predicate not in members and predicate not in ChemprotScript.NATIVE_PREDICATES:
            raise ValueError(
                f"{ChemprotScript.NAME} predicate {label!r} -> {predicate!r} is neither "
                "a tablassert Predicates member nor a documented native predicate"
            )


validate_label_map(ChemprotScript.LABEL_MAP, ChemprotScript.NAME)
_validate_predicate_map()

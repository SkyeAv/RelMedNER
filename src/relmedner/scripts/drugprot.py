from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class DrugprotScript(Script):
    """streams OpenMed/drugprot-parquet rows (DrugProt gold: pre-joined title + abstract text,
    flat char-offset entity dicts, gold mention-level chemical-gene relations) into biolink-labeled
    entity and relation examples.

    Provenance: bigbio/drugprot is script-only (its hub loading script is unsupported on the
    repo's datasets version, the datasets-server errors, and refs/convert/parquet is empty), so the
    declared source is OpenMed/drugprot-parquet, the cc-by-4.0 parquet mirror whose rows carry
    exactly the DrugProt tables (pmid, title, abstract, text, entities, relations); the `text`
    column is the pre-joined title + abstract the gold offsets are document-relative against, so
    no passages join is re-derived. Measured full-split census (wenceslaus, 2026-09-23; do not
    re-measure): 3,500 train rows with 89,529 entities and 17,274 relations, 750 validation rows
    with 18,858 entities and 3,761 relations. Char offsets are END-EXCLUSIVE with exact surface
    matches on 108,387/108,387 spans, 0 out-of-bounds, 0 mismatched surfaces, so the
    KnowledgatorBiomed char-offset bridge is reused verbatim and a malformed entry (non-dict,
    non-str id/type, non-int or bool start/end: 0 measured, defensive) drops only its span. Entity
    label census: train CHEMICAL 46,274 / GENE-Y 28,421 / GENE-N 14,834, validation CHEMICAL
    9,853 / GENE 9,005; GENE-Y and the plain GENE twin both map to Gene, and GENE-N falls back to
    GeneFamily per the ChemprotScript precedent (the -N mark names nonspecific mentions that would
    otherwise train "kinase" as a specific gene). Relations: 13 gold labels (census: INHIBITOR
    6,538, DIRECT-REGULATOR 2,705, SUBSTRATE 2,497, ACTIVATOR 1,674, INDIRECT-UPREGULATOR 1,680,
    INDIRECT-DOWNREGULATOR 1,661, ANTAGONIST 1,190, AGONIST 789, PRODUCT-OF 1,078, PART-OF 1,142,
    AGONIST-ACTIVATOR 39, SUBSTRATE_PRODUCT-OF 27, AGONIST-INHIBITOR 15); arg1 is always the
    chemical and arg2 the gene, 0 self-loops, 0 dangling args measured, so the corpus-native
    DrugProt labels are the Chemprot CPR codes unabbreviated and the mapped predicates reuse the
    landed ChemprotScript choices. Labels biolink has no honest slot for (AGONIST,
    AGONIST-ACTIVATOR, AGONIST-INHIBITOR, ANTAGONIST, PRODUCT-OF, SUBSTRATE_PRODUCT-OF) keep
    native snake_case per the resolve_predicate convention, documented omission in the README.
    1,275 of 4,250 rows (1,067 train / 208 validation) carry entities with zero relations and ship
    under the permitted-shapes contract with no code path special-casing them. Gold relations are
    trusted as-is (native-gold stance, the ChemprotScript pattern): evidence="asserted",
    negated always False.
    """

    NAME: ClassVar[str] = "DrugprotScript"

    # the complete measured entity vocabulary, lowercased corpus labels -> biolink classes
    # (resolve_mentions probes the lowercased raw label, then the IOB-normalized form). train
    # ships GENE-Y/GENE-N and validation ships the plain GENE twin; all three spellings land
    # here. Values are validated against tablassert Categories at import below.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "chemical": "ChemicalEntity",
        "gene": "Gene",
        "gene-y": "Gene",
        "gene-n": "GeneFamily",
    }

    # gold DrugProt labels -> predicates, keyed by the corpus's own spelling (arg1 is the
    # chemical, arg2 the gene, in 100% of measured relations). The five directional families reuse
    # the landed ChemprotScript predicate choices (PART-OF->CPR:1, DIRECT-REGULATOR->CPR:2,
    # INDIRECT-UP/DOWNREGULATOR->CPR:3/4, ACTIVATOR/INHIBITOR->CPR:5/6, SUBSTRATE->CPR:9); the
    # DrugProt-only labels keep native snake_case because biolink has no honest slot. Mapped
    # values must be tablassert Predicates members or documented native snake_case omissions, and
    # the module-scope guard fails at import on anything else so a typo never trains garbage
    # predicates.
    PREDICATE_MAP: ClassVar[dict[str, str]] = {
        "ACTIVATOR": "increases_amount_or_activity_of",
        "AGONIST": "agonist",
        "AGONIST-ACTIVATOR": "agonist_activator",
        "AGONIST-INHIBITOR": "agonist_inhibitor",
        "ANTAGONIST": "antagonist",
        "DIRECT-REGULATOR": "regulates",
        "INDIRECT-DOWNREGULATOR": "decreases_amount_or_activity_of",
        "INDIRECT-UPREGULATOR": "increases_amount_or_activity_of",
        "INHIBITOR": "decreases_amount_or_activity_of",
        "PART-OF": "part_of",
        "PRODUCT-OF": "product_of",
        "SUBSTRATE": "is_substrate_of",
        "SUBSTRATE_PRODUCT-OF": "substrate_product_of",
    }

    # native snake_case predicates biolink has no honest slot for; kept per the
    # resolve_predicate convention, documented omission in the README
    NATIVE_PREDICATES: ClassVar[frozenset[str]] = frozenset(
        {"agonist", "agonist_activator", "agonist_inhibitor", "antagonist", "product_of", "substrate_product_of"}
    )

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out is [text, entities, relations], so values = (text, entities, relations).

        The entity table is a flat dict-per-entry list (id/type/text/start/end), the relation
        table a flat dict-per-entry list (type/arg1/arg2): a non-list table yields zero rows of
        that shape and a malformed entry drops only itself (skip-don't-coerce). Relations
        referencing an entity whose span did not survive the bridge drop with it. run never
        raises on a bad row.
        """
        text_value, entities_value, relations_value = values
        text: str = text_value if isinstance(text_value, str) else ""
        if not text:
            return TrainingExample(text="")
        triples: list[tuple[str, int, int]] = list(FullmapMiner.splitter()(text, lower=False))
        tokens: list[str] = [token for token, _start, _end in triples]
        structs: list[dict[str, Any]] = self.entity_structs(entities_value)
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
        # one batched resolution round trip per row, never per mention
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
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=self.relations(relations_value, structs_by_id, surfaces),
        )

    @staticmethod
    def entity_structs(value: Any) -> list[dict[str, Any]]:
        """the flat entity list (dicts with id/type/text/start/end) -> well-formed per-entity
        structs; a non-list table yields zero entities, and a malformed entry (non-dict, non-str
        id/type, non-int or bool start/end) drops only that entity, never coerced. Char ends are
        exclusive per the census, matching the bridge's convention; bounds and degeneracy live in
        the bridge, not here."""
        entries: list[Any] = value if isinstance(value, list) else []
        structs: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entity_id: Any = entry.get("id")
            entity_type: Any = entry.get("type")
            start: Any = entry.get("start")
            end: Any = entry.get("end")
            if not isinstance(entity_id, str) or not isinstance(entity_type, str):
                continue
            if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                continue
            structs.append({"id": entity_id, "type": entity_type, "start": start, "end": end})
        return structs

    @classmethod
    def relations(
        cls,
        value: Any,
        structs_by_id: dict[str, dict[str, Any]],
        surfaces: dict[str, str],
    ) -> list[Relation]:
        """the flat relation list (dicts with type/arg1/arg2) -> gold Relations; drop rules in
        this exact order (mirrors the measured census): non-dict entry or non-str
        type/arg1/arg2; a self-loop arg1 == arg2 (0 measured, defensive); a dangling arg id
        absent from the well-formed entity table (0 measured, defensive); an arg whose entity did
        not survive the char-offset bridge (0 measured, a relation must not outlive its span); and
        a type outside PREDICATE_MAP (defensive drift guard, never guess). Gold relations emit
        evidence="asserted", negated=False; head/tail values are the mention surfaces, which is
        what keeps them substrings of the emitted text."""
        entries: list[Any] = value if isinstance(value, list) else []
        relations: list[Relation] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rel_type: Any = entry.get("type")
            arg1: Any = entry.get("arg1")
            arg2: Any = entry.get("arg2")
            if not isinstance(rel_type, str) or not isinstance(arg1, str) or not isinstance(arg2, str):
                continue
            if arg1 == arg2:
                continue
            if arg1 not in structs_by_id or arg2 not in structs_by_id:
                continue
            if arg1 not in surfaces or arg2 not in surfaces:
                continue
            predicate: str | None = cls.PREDICATE_MAP.get(rel_type)
            if predicate is None:
                continue
            relations.append(
                Relation(
                    name=predicate,
                    fields=[RelationField(name="head", value=surfaces[arg1]), RelationField(name="tail", value=surfaces[arg2])],
                    description=ScriptUtils.predicate_description(predicate),
                    evidence="asserted",
                    negated=False,
                )
            )
        return relations


def _validate_predicate_map() -> None:
    """import-time guard: every mapped predicate is a tablassert Predicates member or a documented
    native snake_case omission, so a typo fails at import rather than mislabeling relations"""
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label, predicate in DrugprotScript.PREDICATE_MAP.items():
        if predicate not in members and predicate not in DrugprotScript.NATIVE_PREDICATES:
            raise ValueError(
                f"{DrugprotScript.NAME} predicate {label!r} -> {predicate!r} is neither "
                "a tablassert Predicates member nor a documented native predicate"
            )


validate_label_map(DrugprotScript.LABEL_MAP, DrugprotScript.NAME)
_validate_predicate_map()

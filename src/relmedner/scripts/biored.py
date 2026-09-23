from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class BioredScript(Script):
    """streams wcole3/biored-parquet rows (BioRED gold as bigbio_kb-shaped rows: title + abstract
    passages, gold char-offset entities, gold relations) into biolink-labeled entity and relation
    examples.

    Provenance: bigbio/biored is script-only (its hub loading script is unsupported on datasets
    5.x, the datasets-server errors, and refs/convert/parquet is empty), so the declared source is
    wcole3/biored-parquet, the verified parquet conversion of the same bigbio_kb data; provenance
    checked against the upstream NLM zip (20,419 entity lines, 6,503 concept REL lines). Measured
    full-split census (wenceslaus, 2026-09-22; do not re-measure): 600 docs (400 train / 100
    validation / 100 test), exactly 2 passages per row (title, abstract), 0 empty texts. 20,419
    entities, all with end-exclusive document-relative char offsets against title + " " + abstract
    and 20,419/20,419 surface matches (1,964 title-anchored, 18,455 abstract-anchored, exactly one
    offset pair per entity, 0 discontinuous), so the KnowledgatorBiomed char-offset bridge is
    reused verbatim and an annotation defect drops only its span. Label census behind the map:
    GeneOrGeneProduct 6,697, DiseaseOrPhenotypicFeature 5,545, ChemicalEntity 4,429,
    OrganismTaxon 2,192, SequenceVariant 1,381, CellLine 175; GeneOrGeneProduct maps to Gene, the
    other five identity-map. Relations: the upstream gold is concept-pair REL lines that the bigbio
    conversion expanded into ALL mention-pair combinations (128,460 raw mention pairs, ~20x
    inflation), so the script collapses back to concept level: mention-level self-loops drop first
    (121 measured), dangling args and bridge casualties drop next (0 measured, defensive), then
    concept-signature dedupe keeps the FIRST mention pair in row order keyed by (type, frozenset
    of arg1 normalized db_ids, frozenset of arg2 normalized db_ids): 128,460 -> 6,767 relations
    (4,390 train / 1,243 validation / 1,134 test). The native snake_case omissions comparison/
    cotreatment/conversion stay per the resolve_predicate convention (biolink has no honest slot
    for them). Gold relations are trusted as-is (native-gold stance, the chemprot pattern):
    evidence="asserted", negated always False. Entity normalization db_ids feed the dedupe key but
    are deliberately NOT shipped on the emitted entities (per-dataset payload fields stay off;
    tuple-lock blast radius). 8 rows end with zero relations and ship entities-only under the
    permitted-shapes contract, with no code path special-casing them.
    """

    NAME: ClassVar[str] = "BioredScript"

    # the complete measured entity vocabulary, lowercased corpus labels -> biolink classes
    # (resolve_mentions probes the lowercased raw label, then the IOB-normalized form). Gene ->
    # GeneOrGeneProduct has no honest biolink slot as a class value, so it lands on Gene; every
    # other value identity-maps. Values are validated against tablassert Categories at import
    # below.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "cellline": "CellLine",
        "chemicalentity": "ChemicalEntity",
        "diseaseorphenotypicfeature": "DiseaseOrPhenotypicFeature",
        "geneorgeneproduct": "Gene",
        "organismtaxon": "OrganismTaxon",
        "sequencevariant": "SequenceVariant",
    }

    # gold relation labels -> predicates, keyed by the corpus's own spelling; mapped values must
    # be tablassert Predicates members or documented native snake_case omissions, and the
    # module-scope guard fails at import on anything else so a typo never trains garbage
    # predicates. Deduped counts: Association 3,510; Positive_Correlation 1,854;
    # Negative_Correlation 1,172; Bind 120; Drug_Interaction 13; Comparison 39; Cotreatment 55;
    # Conversion 4. No NA and no Cause label exists in this corpus; any unknown label drops the
    # relation (defensive drift guard, never guess) and the row's entities still ship.
    PREDICATE_MAP: ClassVar[dict[str, str]] = {
        "Association": "associated_with",
        "Positive_Correlation": "positively_correlated_with",
        "Negative_Correlation": "negatively_correlated_with",
        "Bind": "physically_interacts_with",
        "Drug_Interaction": "pharmacologically_interacts_with",
        "Comparison": "comparison",
        "Cotreatment": "cotreatment",
        "Conversion": "conversion",
    }

    # native snake_case predicates biolink has no honest slot for (Comparison, Cotreatment,
    # Conversion); kept per the resolve_predicate convention, documented omission in the README
    NATIVE_PREDICATES: ClassVar[frozenset[str]] = frozenset({"comparison", "cotreatment", "conversion"})

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out is [passages, entities, relations], so values = (passages, entities,
        relations); unpack positionally and defensively.

        Passages must be exactly two well-formed entries (dict with a str type and a non-empty
        list of str text) joined as title + " " + abstract: entity offsets are document-relative
        against exactly this join, so any other passage layout would re-derive every offset
        against a shifted text and instead returns TrainingExample(text="") for the
        declared-outputs filter to drop. A malformed entity or relation entry drops only itself
        (skip-don't-coerce); relations referencing an entity whose span did not survive the
        bridge drop with it. run never raises on a bad row.
        """
        passages_value, entities_value, relations_value = values
        text: str = self.join_passages(passages_value)
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
        # the concept-signature dedupe key reads the normalized db_ids off the well-formed entity
        # table (independent of bridge survival), so build the id -> normalized-ids map up front
        ids_of: dict[str, frozenset[str]] = {struct["id"]: self.normalized_ids(struct["normalized"]) for struct in structs}
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
            relations=self.relations(relations_value, ids_of, surfaces),
        )

    @staticmethod
    def join_passages(value: Any) -> str:
        """the passages column -> title + " " + abstract, or "" to drop the row (skip-don't-coerce).

        Entity offsets are document-relative against exactly this two-passage join, so any other
        passage layout (a non-list, 1 or 3+ passages, a non-dict entry, a missing/empty or
        non-str text part) must not ship re-derived offsets: it returns "" (0 measured on the
        full corpus; the guard is defensive).
        """
        if not isinstance(value, list) or len(value) != 2:
            return ""
        parts: list[str] = []
        for passage in value:
            if not isinstance(passage, dict):
                return ""
            ptype: Any = passage.get("type")
            ptext: Any = passage.get("text")
            if not isinstance(ptype, str) or not isinstance(ptext, list) or not ptext:
                return ""
            if any(not isinstance(part, str) for part in ptext):
                return ""
            parts.append(" ".join(ptext))
        return parts[0] + " " + parts[1]

    @staticmethod
    def entity_structs(value: Any) -> list[dict[str, Any]]:
        """the bigbio_kb entities list-of-dicts -> well-formed per-entity structs; a non-list table
        yields zero entities, and a malformed entry (non-dict, non-str id/type, offsets not a list
        with exactly one [start, end] pair of non-bool ints) drops only that entity, never coerced.
        The raw normalized list rides along for the dedupe key; bounds and degeneracy live in the
        bridge, not here."""
        entries: list[Any] = value if isinstance(value, list) else []
        structs: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entity_id: Any = entry.get("id")
            entity_type: Any = entry.get("type")
            offsets: Any = entry.get("offsets")
            if not isinstance(entity_id, str) or not isinstance(entity_type, str):
                continue
            if not isinstance(offsets, list) or len(offsets) != 1:
                continue
            pair: Any = offsets[0]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            start, end = pair
            if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                continue
            structs.append({"id": entity_id, "type": entity_type, "start": start, "end": end, "normalized": entry.get("normalized")})
        return structs

    @staticmethod
    def normalized_ids(value: Any) -> frozenset[str]:
        """one entity's normalized db_ids as a frozenset for the concept-signature dedupe key; a
        missing or malformed normalized list contributes nothing, and a dict entry without a
        db_id key is skipped rather than raised (run never raises on a bad row)"""
        if not isinstance(value, list):
            return frozenset()
        return frozenset(str(entry["db_id"]) for entry in value if isinstance(entry, dict) and "db_id" in entry)

    @classmethod
    def relations(
        cls,
        value: Any,
        ids_of: dict[str, frozenset[str]],
        surfaces: dict[str, str],
    ) -> list[Relation]:
        """the bigbio_kb relations list-of-dicts -> deduped gold Relations; drop rules in this
        exact order (mirrors the measured dedupe census): non-dict entry or non-str
        type/arg1_id/arg2_id; mention-level self-loop arg1_id == arg2_id (121 measured); a
        dangling arg id absent from the well-formed entity table (0 measured, defensive); an arg
        whose entity did not survive the char-offset bridge (0 measured, a relation must not
        outlive its span); a repeat concept signature (type, frozenset arg1 db_ids, frozenset
        arg2 db_ids) whose earlier occurrence already kept a mention pair in row order; and a
        type outside PREDICATE_MAP (defensive drift guard, never guess). Gold relations emit
        evidence="asserted", negated=False; head/tail values are the mention surfaces, which is
        what keeps them substrings of the emitted text."""
        entries: list[Any] = value if isinstance(value, list) else []
        relations: list[Relation] = []
        seen: set[tuple[str, frozenset[str], frozenset[str]]] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rel_type: Any = entry.get("type")
            arg1: Any = entry.get("arg1_id")
            arg2: Any = entry.get("arg2_id")
            if not isinstance(rel_type, str) or not isinstance(arg1, str) or not isinstance(arg2, str):
                continue
            if arg1 == arg2:
                continue
            if arg1 not in ids_of or arg2 not in ids_of:
                continue
            if arg1 not in surfaces or arg2 not in surfaces:
                continue
            signature: tuple[str, frozenset[str], frozenset[str]] = (rel_type, ids_of[arg1], ids_of[arg2])
            if signature in seen:
                continue
            seen.add(signature)
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
    for label, predicate in BioredScript.PREDICATE_MAP.items():
        if predicate not in members and predicate not in BioredScript.NATIVE_PREDICATES:
            raise ValueError(
                f"{BioredScript.NAME} predicate {label!r} -> {predicate!r} is neither "
                "a tablassert Predicates member nor a documented native predicate"
            )


validate_label_map(BioredScript.LABEL_MAP, BioredScript.NAME)
_validate_predicate_map()

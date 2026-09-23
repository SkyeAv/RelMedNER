from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils

UNNORMALIZED_MESH: str = "-1"
"""the builder keeps the corpus's raw MESH infon; '-1' is the unnormalized sentinel (measured
76 / 60 / 91 occurrences across train / dev / test) and never becomes a CURIE"""

PREDICATE: str = "causes"


class Bc5CdrScript(Script):
    """turns one BC5CDR avro document into one entity-plus-relation training example

    Every record is one PubMed document (title + abstract) annotated with gold
    chemical/disease spans and chemical-induced-disease (CID) relation pairs. The spans here are
    corpus gold, not model predictions, so this script does NOT re-resolve through fullmap --
    it trusts the annotation's own MESH id exactly the way CtkpInterventionsScript trusts the
    KP (trust-gold philosophy): a normalized mesh rides as a MESH: curie, the unnormalized
    '-1' sentinel rides raw, and the entity carries origin 'fullmap' only when a curie exists.

    Spans must satisfy gliner2's occurs-in-text contract verbatim: the builder's census measured
    74 / 43 / 37 document-absolute slice mismatches (train / dev / test) that the containers
    keep as corpus noise, and this script drops exactly those instead of shipping spans the
    downstream sanitizer would silently discard; out-of-bounds offsets drop through the same
    slice comparison. Unknown annotation types and malformed fields drop too, and a document
    whose annotations all drop still ships its text for the declared-outputs filter.

    Relations resolve each record relation's chemical_mesh / disease_mesh against the SURVIVING
    spans' surfaces (never the dropped ones), emit one asserted, non-negated 'causes' relation
    per distinct surface pair, cross-product multiple same-mesh annotations with dedupe, and
    drop unresolved meshes and self-loops.
    """

    NAME: ClassVar[str] = "Bc5CdrScript"

    # dataset gold annotation type -> biolink class; validated at import by validate_label_map below
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "Chemical": "ChemicalEntity",
        "Disease": "Disease",
    }

    def text_of(self: Self, record: dict[str, Any]) -> str:
        """the builder's document text contract: title, one newline, abstract"""
        title: str = record["title"] if isinstance(record.get("title"), str) else ""
        abstract: str = record["abstract"] if isinstance(record.get("abstract"), str) else ""
        return f"{title}\n{abstract}"

    def spans_of(self: Self, record: dict[str, Any], text: str) -> list[tuple[str, str, str]]:
        """(surface, biolink label, raw mesh) triples for gold annotations that survive every
        contract; a malformed table yields zero spans rather than a crash (skip-don't-coerce)

        A span survives only when its type maps through LABEL_MAP, mesh/text are non-empty
        strings, offset/length are non-bool non-negative ints, and the document-absolute slice
        equals the declared text exactly -- the measured 74 / 43 / 37 mismatches and every
        out-of-bounds annotation drop right here.
        """
        entities: Any = record.get("entities")
        if not isinstance(entities, list):
            return []
        spans: list[tuple[str, str, str]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            ann_type: Any = entity.get("type")
            label: str | None = self.LABEL_MAP.get(ann_type) if isinstance(ann_type, str) else None
            mesh: Any = entity.get("mesh")
            surface: Any = entity.get("text")
            offset: Any = entity.get("offset")
            length: Any = entity.get("length")
            if label is None or not isinstance(mesh, str) or not mesh or not isinstance(surface, str) or not surface:
                continue
            if isinstance(offset, bool) or isinstance(length, bool) or not isinstance(offset, int) or not isinstance(length, int):
                continue
            if offset < 0 or length < 0 or text[offset : offset + length] != surface:
                continue
            spans.append((surface, label, mesh))
        return spans

    def mentions_of(self: Self, record: dict[str, Any], text: str) -> list[ResolvedMention]:
        """one resolved mention per surviving gold span, carrying its own mesh as MESH curie when
        normalized ('-1' stays raw); gold is never re-resolved through fullmap"""
        return [
            ResolvedMention(
                mention=surface,
                category=label,
                curie=f"MESH:{mesh}" if mesh != UNNORMALIZED_MESH else None,
                origin="fullmap" if mesh != UNNORMALIZED_MESH else "raw",
            )
            for surface, label, mesh in self.spans_of(record, text)
        ]

    def relations_of(self: Self, record: dict[str, Any], spans: list[tuple[str, str, str]]) -> list[Relation]:
        """one asserted, non-negated 'causes' relation per distinct (head, tail) surface pair

        Each record relation names its participants by raw mesh id; meshes resolve against the
        surviving spans only, so a relation whose participant dropped (slice mismatch,
        out-of-bounds, unknown type) is unresolved and drops. Multiple same-mesh annotations
        cross-product their surfaces and identical pairs dedupe; head == tail self-loops drop.
        Every emitted surface is char-exact gold text, so it occurs in the output text.
        """
        relations: Any = record.get("relations")
        if not isinstance(relations, list) or not spans:
            return []
        surfaces_by_mesh: dict[str, list[str]] = {}
        for surface, _label, mesh in spans:
            surfaces = surfaces_by_mesh.setdefault(mesh, [])
            if surface not in surfaces:
                surfaces.append(surface)
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            chemical_mesh: Any = relation.get("chemical_mesh")
            disease_mesh: Any = relation.get("disease_mesh")
            heads: list[str] = surfaces_by_mesh.get(chemical_mesh, []) if isinstance(chemical_mesh, str) else []
            tails: list[str] = surfaces_by_mesh.get(disease_mesh, []) if isinstance(disease_mesh, str) else []
            if not heads or not tails:
                continue
            for head in heads:
                for tail in tails:
                    pair: tuple[str, str] = (head, tail)
                    if head == tail or pair in seen:
                        continue
                    seen.add(pair)
                    pairs.append(pair)
        return [
            Relation(
                name=PREDICATE,
                fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                description=ScriptUtils.predicate_description(PREDICATE),
            )
            for head, tail in pairs
        ]

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """LocalAvroDataStream ships the whole avro record as one value; a malformed record
        degrades to the empty example the pipeline filters downstream (skip-don't-coerce)"""
        (record,) = values
        if not isinstance(record, dict):
            return TrainingExample(text="")
        text: str = self.text_of(record)
        if not text.strip():
            return TrainingExample(text="")
        spans: list[tuple[str, str, str]] = self.spans_of(record, text)
        resolved: list[ResolvedMention] = self.mentions_of(record, text)
        return TrainingExample(
            text=text,
            entities=ScriptUtils.group_entities(resolved) if resolved else [],
            relations=self.relations_of(record, spans),
        )


validate_label_map(Bc5CdrScript.LABEL_MAP, Bc5CdrScript.NAME)

from __future__ import annotations

import re
from typing import Any, ClassVar, Self

from relmedner.gazetteer import DISABLED_QUALIFIERS
from relmedner.models import Classification, Description, Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class DakpNerExportScript(Script):
    """turns one DAKP ``dakp.ner.export.v1`` avro record into one gliner2-valid training example

    DAKP's exporter (``dakp_pipeline/ner_export.py``) already writes records shaped like this
    repo's ``TrainingExample`` (same ``relmedner.ingests`` namespace, minus ``weight`` and
    ``Relation.description``): DailyMed contraindication / warning / indication sections, EMA
    therapeutic indications, and FAERS indication strings, mined by DAKP's composite GLiNER
    backend into entities, drug -> object assertion relations, qualifier relations, and one
    indication-context classification per row. So this script is a sanitizer, not a decoder:
    it enforces the contracts the exporter does not.

    Measured on the full 187,267-record bundle (wenceslaus 2026-09-24): 0 entity mentions fall
    outside their text, but 2,136,998 relation values do -- all but 6 are the asserted relation
    HEAD (the row's subject drug) differing only by case, because DAKP's subject guard is
    case-insensitive (head 'FLUCONAZOLE', text 'Fluconazole') while gliner2 validates relation
    values as exact in-text surfaces. Such a value is REALIGNED to the text's own first
    case-insensitive occurrence rather than dropped, which would discard most of the corpus's
    relation signal; a value absent even case-insensitively drops its relation. The same bundle
    carries 30,236 exact duplicate relations and 96 head == tail self-loops, which drop here.

    ``species_context_qualifier`` relations drop: tablassert marks the slot DISABLED_EDGE_FIELDS
    and the repo's gazetteer never emits it (``gazetteer.DISABLED_QUALIFIERS``). Entity labels
    that are biolink classes pass through; DAKP's three qualifier-channel labels
    (temporal_interval_qualifier, temporal_context_qualifier, frequency_qualifier) have no
    honest biolink class -- their measured surfaces are 'dosage', 'history of', '14 days' --
    so they ride as PascalCase raw tails like every other unmapped label in the repo.
    ``negated`` and ``evidence`` (asserted for the drug -> object relation, mined for qualifier
    attachments) ride through verbatim, and each relation gains its biolink slot description.
    Classifications ride through unchanged; the exporter emits no structures.

    Malformed sub-structures drop without raising (skip-don't-coerce): a non-dict record or a
    blank text degrades to the empty example the declared-outputs filter removes.
    """

    NAME: ClassVar[str] = "DakpNerExportScript"

    def entities_of(self: Self, record: dict[str, Any], text: str) -> list[ResolvedMention]:
        """one resolved mention per in-text surface; biolink-class labels keep their name and the
        rest PascalCase, so no invented class ever reaches training"""
        entities: Any = record.get("entities")
        if not isinstance(entities, list):
            return []
        resolved: list[ResolvedMention] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            label: Any = entity.get("label")
            mentions: Any = entity.get("mentions")
            if not isinstance(label, str) or not label.strip() or not isinstance(mentions, list):
                continue
            category: str = label if ScriptUtils.is_biolink_category(label) else ScriptUtils.pascal_label(label)
            resolved.extend(
                ResolvedMention(mention=mention, category=category)
                for mention in mentions
                if isinstance(mention, str) and mention.strip() and mention in text
            )
        return resolved

    @staticmethod
    def realign(value: str, text: str) -> str | None:
        """the text's own surface for a relation value: the value itself when it occurs verbatim,
        else its first case-insensitive occurrence, else None (the relation drops)"""
        if value in text:
            return value
        match: re.Match[str] | None = re.search(re.escape(value), text, flags=re.IGNORECASE)
        return match.group(0) if match is not None else None

    def aligned_fields(self: Self, fields: Any, text: str) -> list[tuple[str, str]] | None:
        """(name, in-text surface) per relation field, or None when any field is malformed or its
        value is absent from the text even case-insensitively (the whole relation drops)"""
        if not isinstance(fields, list) or not fields:
            return None
        aligned: list[tuple[str, str]] = []
        for field in fields:
            name: Any = field.get("name") if isinstance(field, dict) else None
            value: Any = field.get("value") if isinstance(field, dict) else None
            if not isinstance(name, str) or not isinstance(value, str) or not value.strip():
                return None
            surface: str | None = self.realign(value, text)
            if surface is None:
                return None
            aligned.append((name, surface))
        return aligned

    def relations_of(self: Self, record: dict[str, Any], text: str) -> list[Relation]:
        """in-text, deduplicated, loop-free relations with their biolink slot descriptions"""
        relations: Any = record.get("relations")
        if not isinstance(relations, list):
            return []
        emitted: list[Relation] = []
        seen: set[tuple[str, tuple[tuple[str, str], ...], bool]] = set()
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            name: Any = relation.get("name")
            negated: Any = relation.get("negated", False)
            evidence: Any = relation.get("evidence", "asserted")
            if not isinstance(name, str) or not name.strip() or name in DISABLED_QUALIFIERS:
                continue
            if not isinstance(negated, bool) or not isinstance(evidence, str):
                continue
            aligned: list[tuple[str, str]] | None = self.aligned_fields(relation.get("fields"), text)
            if aligned is None:
                continue
            values: dict[str, str] = dict(aligned)
            head: str | None = values.get("head")
            tail: str | None = values.get("tail")
            if head is not None and tail is not None and head.casefold() == tail.casefold():
                continue
            key: tuple[str, tuple[tuple[str, str], ...], bool] = (name, tuple(aligned), negated)
            if key in seen:
                continue
            seen.add(key)
            emitted.append(
                Relation(
                    name=name,
                    fields=[RelationField(name=field_name, value=value) for field_name, value in aligned],
                    description=ScriptUtils.predicate_description(name),
                    negated=negated,
                    evidence=evidence,
                )
            )
        return emitted

    @staticmethod
    def descriptions_of(value: Any) -> list[Description] | None:
        """label_descriptions ride through only as a well-formed list of key/description pairs"""
        if not isinstance(value, list):
            return None
        descriptions: list[Description] = [
            Description(key=entry["key"], description=entry["description"])
            for entry in value
            if isinstance(entry, dict) and isinstance(entry.get("key"), str) and isinstance(entry.get("description"), str)
        ]
        return descriptions or None

    def classifications_of(self: Self, record: dict[str, Any]) -> list[Classification]:
        """the exporter's indication-context classification, kept only when its true label is
        one of its own labels (a label outside the choice set is not a supervisable example)"""
        classifications: Any = record.get("classifications")
        if not isinstance(classifications, list):
            return []
        kept: list[Classification] = []
        for classification in classifications:
            if not isinstance(classification, dict):
                continue
            task: Any = classification.get("task")
            labels: Any = classification.get("labels")
            true_label: Any = classification.get("true_label")
            multi_label: Any = classification.get("multi_label", False)
            prompt: Any = classification.get("prompt")
            if not isinstance(task, str) or not task.strip() or not isinstance(multi_label, bool):
                continue
            if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
                continue
            if not isinstance(true_label, list) or not true_label or not all(label in labels for label in true_label):
                continue
            kept.append(
                Classification(
                    task=task,
                    labels=labels,
                    true_label=true_label,
                    multi_label=multi_label,
                    prompt=prompt if isinstance(prompt, str) and prompt.strip() else None,
                    label_descriptions=self.descriptions_of(classification.get("label_descriptions")),
                )
            )
        return kept

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """LocalAvroDataStream ships the whole avro record as one value; a malformed record
        degrades to the empty example the pipeline filters downstream (skip-don't-coerce)"""
        (record,) = values
        if not isinstance(record, dict):
            return TrainingExample(text="")
        text: Any = record.get("text")
        if not isinstance(text, str) or not text.strip():
            return TrainingExample(text="")
        resolved: list[ResolvedMention] = self.entities_of(record, text)
        return TrainingExample(
            text=text,
            entities=ScriptUtils.group_entities(resolved) if resolved else [],
            classifications=self.classifications_of(record),
            relations=self.relations_of(record, text),
        )

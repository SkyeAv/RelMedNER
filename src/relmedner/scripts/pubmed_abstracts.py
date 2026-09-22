from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class PubmedAbstractsScript(Script):
    """streams knowledgator/PubMedAbstractsNER rows (token list + end-inclusive
    [start, end, "MeSH heading - definition"] spans) into biolink-labeled entity examples;
    unresolved headings surface as PascalCased raw categories for zero-shot breadth"""

    NAME: ClassVar[str] = "PubmedAbstractsScript"

    # MeSH heading -> biolink class; resolve_mentions looks keys up on label.lower(), so keys
    # are the lowercased headings. Seeded from the measured 35,000-row corpus (383,721 spans,
    # 5,850 headings): the raw-origin anchors plus only clearly-faithful headings. Headings
    # with no faithful biolink class stay UNMAPPED -- a wrong bucket silently mislabels
    # training data (measured unmapped examples: "Diagnosis" spans are the abstract concepts
    # 'diagnosis'/'prognosis', not diagnostic aids; "Population Characteristics" spans include
    # 'health'/'Healthcare', which are not populations; "Blood" is 'plasma'/'Serum' with no
    # exact class; "Investigative Techniques"; "Chemical Phenomena"). The >=90% coverage push
    # is US-004; "age groups" maps only because its mentions ('adult', 'aged') are genuinely
    # populations of humans, matching the shared person bucket's own allowance.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "pathologic processes": "PathologicalProcess",
        "disease": "Disease",
        "persons": "Human",
        "age groups": "PopulationOfIndividualOrganisms",
        "publication formats": "Publication",
        "cells": "Cell",
        "tissues": "AnatomicalEntity",
        "body regions": "GrossAnatomicalStructure",
        "digestive system": "GrossAnatomicalStructure",
        "cardiovascular system": "GrossAnatomicalStructure",
        "nervous system": "GrossAnatomicalStructure",
        "central nervous system": "GrossAnatomicalStructure",
        "integumentary system": "GrossAnatomicalStructure",
        "bacteria": "Bacterium",
        "viruses": "Virus",
        "eukaryota": "OrganismTaxon",
        "murinae": "OrganismTaxon",
        "metals, heavy": "ChemicalEntity",
        "nucleic acids": "NucleicAcidEntity",
    }

    @staticmethod
    def heading(label: str) -> str:
        """bare MeSH heading: the " - " definition tail is corpus metadata, not resolution
        signal -- the shared gate keys buckets on label words, so definition words ('region',
        'leg', 'process') would flip fullmap outcomes (measured 10.6% false rejections)"""
        return label.split(" - ", 1)[0]

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, ner_value = values
        tokens: list[str] = [str(token) for token in tokens_value] if isinstance(tokens_value, list) else []
        ner: list[Any] = ner_value if isinstance(ner_value, list) else []
        if not tokens or not ner:
            # the 13 measured empty-ner rows exit here as text-only examples; an empty
            # populated() is dropped by pipeline.matches_declared_outputs downstream
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        # mention_spans validates shape and bounds first (end inclusive), THEN the label is
        # split so the definition tail never reaches resolution, gates, or output
        spans: list[tuple[int, int, str]] = [(start, end, self.heading(label)) for start, end, label in ScriptUtils.mention_spans(tokens, ner)]
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), heading) for start, end, heading in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # mentions derives from spans, so zip pairs each span with its resolution positionally;
        # raw-tail headings surface PascalCased (biolink-style casing) while label_map/fallback
        # entries already name a biolink class and stay untouched
        labeled: list[ResolvedMention] = [
            item if item.origin != "raw" else ResolvedMention(mention=item.mention, category=ScriptUtils.pascal_label(heading), origin=item.origin)
            for item, (_, heading) in zip(resolved, mentions, strict=True)
        ]
        resolved_spans: list[tuple[int, int, str]] = [(start, end, item.category) for (start, end, _), item in zip(spans, labeled, strict=True)]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )


validate_label_map(PubmedAbstractsScript.LABEL_MAP, PubmedAbstractsScript.NAME)

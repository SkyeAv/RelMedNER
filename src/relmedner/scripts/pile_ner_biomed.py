from __future__ import annotations

from typing import ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class PileNerBiomedScript(Script):
    """streams Pile-NER-biomed-IOB rows (python-repr token/tag string columns) into biolink-labeled
    entity examples; unresolved labels are kept as PascalCased raw categories for zero-shot breadth"""

    NAME: ClassVar[str] = "PileNerBiomedScript"

    # this corpus's head labels (mined from its 3,896-type vocabulary) merged over
    # ScriptUtils.FALLBACK_LABEL_MAP at resolution time; it lives with the dataset that needs it so
    # sibling dataset worktrees extend their own vocabulary without colliding on the shared base
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "medical condition": "Disease",
        "disorder": "Disease",
        "health condition": "Disease",
        "chemical": "ChemicalEntity",
        "compound": "ChemicalEntity",
        "substance": "ChemicalEntity",
        "chemical compound": "ChemicalEntity",
        "chemical substance": "ChemicalEntity",
        "medication": "Drug",
        "gene/protein": "Gene",
        "enzyme": "Protein",
        "cell": "Cell",
        "cell line": "CellLine",
        "anatomical structure": "GrossAnatomicalStructure",
        "anatomical entity": "AnatomicalEntity",
        "body part": "GrossAnatomicalStructure",
        "organ": "GrossAnatomicalStructure",
        "anatomy": "GrossAnatomicalStructure",
        "tissue": "AnatomicalEntity",
        "organism": "OrganismTaxon",
        "species": "OrganismTaxon",
        "animal": "OrganismTaxon",
        "bacteria": "Bacterium",
        "virus": "Virus",
        "process": "BiologicalProcess",
        "sign or symptom": "PhenotypicFeature",
        "treatment": "Treatment",
        "medical treatment": "Treatment",
        "therapy": "Treatment",
        "medical procedure": "Procedure",
        "procedure": "Procedure",
        "medical test": "DiagnosticAid",
        "test": "DiagnosticAid",
        "medical device": "Device",
        "device": "Device",
        "instrument": "Device",
        "medical equipment": "Device",
        "measurement": "ClinicalMeasurement",
        "medical measurement": "ClinicalMeasurement",
        "quantity": "ClinicalMeasurement",
        "food": "Food",
        "nutrient": "Food",
        "publication": "Publication",
        "study": "Study",
        "location": "GeographicLocation",
        "country": "GeographicLocation",
        "city": "GeographicLocation",
        "organization": "Agent",
        "person": "Human",
        "patient": "Human",
        "group": "PopulationOfIndividualOrganisms",
        "population": "PopulationOfIndividualOrganisms",
        "cohort": "Cohort",
        "mutation": "SequenceVariant",
        "genetic variation": "SequenceVariant",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, tags_value = values
        tokens: list[str] = ScriptUtils.parse_literal_list(tokens_value)
        tags: list[str] = ScriptUtils.parse_literal_list(tags_value)
        if not tokens or len(tokens) != len(tags):
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans: list[tuple[int, int, str]] = ScriptUtils.iob_spans(tags)
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # fullmap hits the shared gate rejects fall through to fallback/raw, so no mention is dropped;
        # raw labels surface PascalCased (biolink-style casing) while fallback entries already name a
        # biolink class and stay untouched
        labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
        # spans and mentions filter identically, so zip pairs each span with its resolution
        # positionally, and strict=True turns a dropped mention into an error instead of a silent
        # misalignment of every later span
        resolved_spans: list[tuple[int, int, str]] = [(start, end, item.category) for (start, end, _), item in zip(spans, labeled, strict=True)]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )


validate_label_map(PileNerBiomedScript.LABEL_MAP, PileNerBiomedScript.NAME)

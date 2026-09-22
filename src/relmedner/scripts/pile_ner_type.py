from __future__ import annotations

from typing import ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class PileNerTypeScript(Script):
    """streams Pile-NER-type rows (conversation-QA columns: one 'Text: ' turn plus one
    'What describes <type> in the text?' question per entity type, answered with a JSON list of
    surface mentions) into biolink-labeled entity examples; unresolved labels are kept as
    PascalCased raw categories for zero-shot breadth, exactly like its biomed sibling"""

    NAME: ClassVar[str] = "PileNerTypeScript"

    # this corpus's head labels (mined from its open-ended GPT-generated type vocabulary --
    # 1,666 distinct types in a 1k-row probe, with casing variants of the same heads) merged over
    # ScriptUtils.FALLBACK_LABEL_MAP at resolution time; it lives with the dataset that needs it so
    # sibling dataset worktrees extend their own vocabulary without colliding on the shared base.
    # Keys are lowercase because fallback lookup lowercases, which collapses person/Person/PERSON.
    # Organization, Product and CreativeWork are NOT biolink classes, so 'organization' lands on
    # Agent and 'product' stays a raw PascalCased tail rather than being forced into a wrong class
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "person": "Human",
        "organization": "Agent",
        "company": "Agent",
        "group": "PopulationOfIndividualOrganisms",
        "location": "GeographicLocation",
        "country": "GeographicLocation",
        "city": "GeographicLocation",
        "state": "GeographicLocation",
        "date": "Attribute",
        "time": "Attribute",
        "quantity": "Attribute",
        "event": "Event",
        "software": "InformationContentEntity",
        "technology": "InformationContentEntity",
        "website": "InformationContentEntity",
        "programming language": "InformationContentEntity",
        # the Pile carries plenty of biomedical documents, so the biomed heads stay mapped
        "disease": "Disease",
        "medical condition": "Disease",
        "condition": "Disease",
        "symptom": "PhenotypicFeature",
        "drug": "Drug",
        "medication": "Drug",
        "treatment": "Treatment",
        "medical treatment": "Treatment",
        "procedure": "Procedure",
        "medical procedure": "Procedure",
        "measurement": "ClinicalMeasurement",
        "chemical": "ChemicalEntity",
        "compound": "ChemicalEntity",
        "chemical compound": "ChemicalEntity",
        "substance": "ChemicalEntity",
        "protein": "Protein",
        "enzyme": "Protein",
        "gene": "Gene",
        "cell": "Cell",
        "cell type": "Cell",
        "cell line": "CellLine",
        "organism": "OrganismTaxon",
        "species": "OrganismTaxon",
        "animal": "OrganismTaxon",
        "anatomical structure": "GrossAnatomicalStructure",
        "body part": "GrossAnatomicalStructure",
        "organ": "GrossAnatomicalStructure",
        "food": "Food",
        "publication": "Publication",
        "book": "Publication",
        "study": "Study",
        "mutation": "SequenceVariant",
        "genetic variation": "SequenceVariant",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        (conversations_value,) = values
        text, answered = ScriptUtils.parse_conversations(conversations_value)
        if not text:
            return TrainingExample(text=text)
        # the corpus answers with surfaces, not offsets, so spans are recovered against the
        # document's whitespace tokens; a mention nobody can locate is dropped from BOTH entities
        # and relation spans (one rule keeps the two positionally aligned, as in every sibling script)
        tokens: list[str] = text.split()
        # folded once per document: every answered mention below matches against this same haystack
        lowered: list[str] = ScriptUtils.lowered_tokens(tokens)
        # (start, end_inclusive, index into mentions); the mention surface is the rejoined token
        # slice rather than the answer string, so it is always literally present in the emitted text
        # (gliner2 sanitizes away any surface it cannot find) and carries the document's own casing
        spans: list[tuple[int, int, int]] = []
        mentions: list[tuple[str, str]] = []
        indexed: dict[tuple[str, str], int] = {}
        for raw_label, answers in answered:
            for answer in dict.fromkeys(answers):
                occurrences: list[tuple[int, int]] = ScriptUtils.token_occurrences(tokens, answer, lowered)
                if not occurrences:
                    continue
                surface: str = ScriptUtils.join_tokens(tokens[occurrences[0][0] : occurrences[0][1] + 1])
                index: int = indexed.setdefault((surface, raw_label), len(mentions))
                if index == len(mentions):
                    mentions.append((surface, raw_label))
                spans.extend((start, end, index) for start, end in occurrences)
        if not mentions:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # fullmap hits the shared gate rejects fall through to fallback/raw, so no mention is dropped --
        # raw labels surface PascalCased (biolink-style casing) while fallback entries already name a
        # biolink class and stay untouched
        labeled: list[ResolvedMention] = [
            item if item.origin != "raw" else ResolvedMention(mention=item.mention, category=ScriptUtils.pascal_label(item.category))
            for item in resolved
        ]
        # one mention can occur many times, so each occurrence carries its mention's resolved
        # category by index (mentions and their resolutions are positionally aligned)
        resolved_spans: list[tuple[int, int, str]] = [(start, end, labeled[index].category) for start, end, index in spans]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )


validate_label_map(PileNerTypeScript.LABEL_MAP, PileNerTypeScript.NAME)

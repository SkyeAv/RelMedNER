from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class KnowledgatorBiomedScript(Script):
    """streams knowledgator/biomed_NER rows (raw untokenized text + character-offset entity structs)
    into biolink-labeled entity examples.

    Why the bridge: the dataset's entities are char offsets into raw text, so the gliner2 whitespace
    splitter tokenizes each row once and char spans snap to token spans before the shared resolution
    chain runs. Char ends are EXCLUSIVE (verified on 18,685 sampled spans: {start: 0, end: 4} slices
    text[0:4] == "Weed"), and the splitter detaches punctuation ("seeds," -> tokens "seeds", ","), so
    raw-text containment fails for 26.7% of char-slice surfaces; TrainingExample.text is therefore the
    re-joined token stream, which makes every mention surface a substring by construction. Annotation
    defects are rare but real (out-of-bounds ends 0.04-0.24%, whitespace boundary slop ~0.4%) and the
    bridge drops/normalizes them instead of crashing. The full 4,840-row train split carries 35
    distinct raw class strings, not the 24 documented on the HF card; LABEL_MAP covers the 21
    canonical classes that have an honest biolink target plus the 8 measured plural/legacy variants
    (29 entries), and leaves the five catch-alls (LANGUAGE, REGULATION OR LAW, MONEY, Unlabelled and
    the bare INTELLECTUAL variant) unmapped so they surface as raw PascalCase tails. The
    trailing-space 'ORGANISMS ' variant needs no entry of its own: the shared fallback lookup
    retries through normalize_iob_label, which strips it.
    """

    NAME: ClassVar[str] = "KnowledgatorBiomedScript"

    # lowercase dataset classes -> biolink classes (resolve_mentions looks up raw_class.lower(), then
    # the underscore/whitespace-normalized form). Covers the 21 canonical classes with an honest
    # biolink target plus the 8 plural/legacy variants; span counts measured on the full 4,840-row
    # train split: PRODUCTS 1,941, GENES 1,105, DISORDERS 962, FINDINGS/PHENOTYPES 935, ORGANISMS 502
    # (plus 253 more as the trailing-space variant), ORGANIZATIONS 117, LOCATIONS 107, EVENTS 23.
    # Values are validated against tablassert Categories at import below.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "chemicals": "ChemicalEntity",
        "clinical drug": "Drug",
        "body substance": "AnatomicalEntity",
        "anatomical structure": "GrossAnatomicalStructure",
        "cells and their components": "Cell",
        "gene and gene products": "Gene",
        "genes": "Gene",
        "intellectual property": "Patent",
        "geographical areas": "GeographicLocation",
        "location": "GeographicLocation",
        "locations": "GeographicLocation",
        "organism": "OrganismTaxon",
        "organisms": "OrganismTaxon",
        "group": "PopulationOfIndividualOrganisms",
        "person": "Human",
        "organization": "Agent",
        "organizations": "Agent",
        "product": "Device",
        "products": "Device",
        "phenotype": "PhenotypicFeature",
        "findings/phenotypes": "PhenotypicFeature",
        "disorder": "Disease",
        "disorders": "Disease",
        "signaling molecules": "MolecularEntity",
        "event": "Event",
        "events": "Event",
        "medical procedure": "Procedure",
        "activity": "Activity",
        "function": "BiologicalProcess",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text_value, entities_value = values
        text: str = text_value if isinstance(text_value, str) else ""
        entities: list[Any] = entities_value if isinstance(entities_value, list) else []
        if not text or not entities:
            return TrainingExample(text="")
        triples: list[tuple[str, int, int]] = list(FullmapMiner.splitter()(text, lower=False))
        tokens: list[str] = [token for token, _start, _end in triples]
        spans: list[tuple[int, int, str]] = ScriptUtils.char_spans_to_token_spans(triples, self.char_spans(entities))
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # raw labels surface PascalCased (biolink-style casing) while mapped/fallback entries already
        # name a biolink class and stay untouched
        labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
        # the multi-class fan-out makes resolved longer than spans, so pair_spans re-pairs by
        # span_index (items[0] is the primary; a misaligned shape raises instead of silently
        # misaligning later spans) and every fan-out row extends resolved_spans with its category
        resolved_spans: list[tuple[int, int, str]] = [
            (start, end, item.category) for (start, end, _), items in ScriptUtils.pair_spans(spans, labeled) for item in items
        ]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )

    @staticmethod
    def char_spans(entities: list[Any]) -> list[tuple[int, int, str]]:
        """extract (start, end_exclusive, class) triples from the HF entity structs; malformed entries
        are skipped, never coerced (mirrors ScriptUtils.mention_spans) -- bounds, degeneracy, and
        mid-token snapping live in the bridge, not here"""
        spans: list[tuple[int, int, str]] = []
        for entry in entities:
            start = entry.get("start") if isinstance(entry, dict) else None
            end = entry.get("end") if isinstance(entry, dict) else None
            label = entry.get("class") if isinstance(entry, dict) else None
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not isinstance(label, str)
            ):
                continue
            spans.append((start, end, label))
        return spans


validate_label_map(KnowledgatorBiomedScript.LABEL_MAP, KnowledgatorBiomedScript.NAME)

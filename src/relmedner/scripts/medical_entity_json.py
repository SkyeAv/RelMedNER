from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Self

from relmedner.models import Entity, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class MedicalEntityJsonScript(Script):
    """turns one Pennlaine/Medical-Entity-JSON-Extraction row into one entity example carrying the
    dataset's own consumer-health attribute labels

    Every row is one consumer-health biography: an [INST] turn wrapping a passage with an embedded
    question and an extraction instruction, then a fenced ```json block holding one object with
    Question, Answer, and Entities (a list of single-key dicts mapping an attribute slot to a
    surface). Measured on all 50 rows: the fenced block ends at the JSON object and every row
    ships WITHOUT a closing fence, so the JSON is decoded with raw_decode from the first brace and
    never needs the trailing fence (a fence present parses identically).

    Measured over the full 50-row test split (2026-09-22, wenceslaus): 259 entity entries, 0 JSON
    errors, 0 [INST] non-matches, 0 malformed entries. 102 surfaces (39.4%) occur verbatim in the
    passage, 130 (50.2%) occur casefold-only, and 27 (10.4%) are no-substring paraphrases that
    drop (skip-don't-coerce). Labels are consumer-health attribute slots, NOT biomedical types --
    50 distinct keys, 30 hapaxes, top: Name 50, Age 50, Profession 34, Condition 22, Specialty 11,
    Treatment 9, Management 9 -- so nothing is re-resolved through fullmap and there is no
    LABEL_MAP: every emitted entity rides its raw PascalCase label (ScriptUtils.pascal_label) with
    origin "raw", the same trust-gold stance as SuperGlueRecordScript.

    Surfaces are located by case-insensitive substring search over the once-folded re-joined
    passage, NOT by token-level matching: the corpus systematically attaches punctuation to
    tokens ("Reynolds,"), hyphenates ages ("39-year-old"), and writes "Dr." where a punctuation-
    splitting tokenizer yields "Dr" + ".". Measured over the full split, token equality finds
    7/259 entries on a naive whitespace split and 0/259 on the gliner2 splitter; the best token-
    level variant (the surface as a substring of some token) still finds only 77/259 (29.7%),
    while char-level containment reproduces the census exactly (232/259 ship, 27 drop). The emitted text is the
    [INST] inner content whitespace-tokenized and re-joined via ScriptUtils.join_tokens, and every
    emitted mention is sliced out of that text in passage casing, so each surface is literally a
    substring of the emitted text (the gliner2 validator rule).

    Malformed input never raises: a row without an [INST] pair ships text-only; unparseable JSON,
    a non-dict object, or an Entities value that is not a list ships the passage text-only; an
    entity entry that is not a single-key dict, a non-string label or surface, or a surface with no
    case-insensitive occurrence drops that entry while its well-formed siblings still ship. All
    text-only outcomes are dropped by the declared-outputs filter (this ingest declares entities
    only), which is the intended outcome for a row with no trustworthy labels.
    """

    NAME: ClassVar[str] = "MedicalEntityJsonScript"

    # one [INST] pair per row; DOTALL because the passage carries its own punctuation and the
    # embedded question, all inside the turn
    INST_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\[INST\](.*?)\[/INST\]", re.DOTALL)
    JSON_DECODER: ClassVar[json.JSONDecoder] = json.JSONDecoder()

    def entities_payload(self: Self, tail: str) -> list[Any] | None:
        """the Entities list of the fenced JSON block after [/INST], or None when the payload is
        untrustworthy (no brace, unparseable JSON, a non-dict object, or Entities not a list)

        raw_decode stops at the end of the first JSON object, so a missing closing fence -- which
        the sibling instruction column shows occurs in this corpus -- parses fine
        """
        brace: int = tail.find("{")
        if brace < 0:
            return None
        try:
            obj: Any = self.JSON_DECODER.raw_decode(tail[brace:])[0]
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        entities: Any = obj.get("Entities")
        return entities if isinstance(entities, list) else None

    def resolved_mentions(self: Self, text: str, entries: list[Any]) -> list[ResolvedMention]:
        """one raw PascalCase resolved mention per locatable surface, sliced out of the re-joined
        passage in passage casing; unlocatable paraphrases and malformed entries drop individually
        (skip-don't-coerce), the haystack is folded once per row"""
        folded: str = text.casefold()
        resolved: list[ResolvedMention] = []
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                continue
            label, surface = next(iter(entry.items()))
            if not isinstance(label, str) or not isinstance(surface, str) or not surface.strip():
                continue
            start: int = folded.find(surface.casefold())
            if start < 0:
                continue
            mention: str = text[start : start + len(surface)]
            category: str = ScriptUtils.pascal_label(label)
            if (mention, category) in seen:
                continue
            seen.add((mention, category))
            resolved.append(ResolvedMention(mention=mention, category=category, origin="raw"))
        return resolved

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        (text_value,) = values
        if not isinstance(text_value, str) or not text_value:
            return TrainingExample(text="")
        match: re.Match[str] | None = self.INST_PATTERN.search(text_value)
        if match is None:
            # no passage to anchor surfaces in; ship the row text-only and let the
            # declared-outputs filter drop it
            return TrainingExample(text=text_value)
        # the emitted text: the [INST] inner content tokenized and re-joined, so downstream
        # tokenization sees a whitespace-normalized passage and every mention stays a substring
        text: str = ScriptUtils.join_tokens(match.group(1).split())
        entries: list[Any] | None = self.entities_payload(text_value[match.end() :])
        if entries is None:
            return TrainingExample(text=text)
        resolved: list[ResolvedMention] = self.resolved_mentions(text, entries)
        entities: list[Entity] = ScriptUtils.group_entities(resolved) if resolved else []
        return TrainingExample(text=text, entities=entities)

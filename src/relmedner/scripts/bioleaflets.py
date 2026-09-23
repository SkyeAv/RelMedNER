from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import Relation, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class BioleafletsScript(Script):
    """streams Bioleaflets rows (six Section_1..Section_6 python-repr dict cells, positional) into
    biolink-labeled entity examples; unmapped types surface as PascalCased raw categories for
    zero-shot breadth, exactly like the sibling scripts.

    Measured row shape (468-row probe): each Section_N cell decodes (ast.literal_eval) to a dict with
    exactly Title / Section_Content / Entity_Recognition; Section_Content is lower-cased and
    pre-tokenized (special chars already stand alone as their own tokens), so content.split() IS the
    tokenizer and tokens get accumulated char offsets that the shared char->token bridge consumes.
    Entity_Recognition is a list (92.8%) or None (7.2%); list entries always carry
    Text/Type/BeginOffset/EndOffset (char offsets into Section_Content), where Comprehend entries add
    Id/Score/Category/Traits and sometimes Attributes, and Stanza entries carry only the 4 base keys.

    Each section is an independent token stream/document: sections are never merged into one token
    stream, because extract_relations would bracket a trigger sitting at a section boundary and
    fabricate cross-section relation pairs no section asserts. Spans, resolution, and relation
    extraction run per section; only the grouped entities and the example text aggregate the row, and
    every mention surface stays a substring of the joined text by construction. A section that is
    None/malformed, has a None/empty Section_Content, or a None/non-list Entity_Recognition
    contributes nothing; a row where nothing yields emits a text-only example (never raises).
    """

    NAME: ClassVar[str] = "BioleafletsScript"

    # lowercased Comprehend/Stanza Type -> biolink class; resolve_mentions looks keys up on
    # raw_label.lower(). The measured 30-type vocabulary splits into these 10 faithful mappings
    # (Comprehend and Stanza name one concept two ways: dx_name/problem, generic_name vs brand_name,
    # test_name vs test, treatment_name vs treatment) and a PHI/noise tail (AGE, ADDRESS, DATE, ID,
    # NAME, PHONE_OR_FAX, PROFESSION, NUMBER, PRODUCT_NAME, TIME_TO_*) that stays UNMAPPED on
    # purpose: those types are not biomedical concepts, and forcing them into a class would silently
    # mislabel training data -- they surface as PascalCased raw tails instead.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "dx_name": "Disease",
        "problem": "Disease",
        "generic_name": "Drug",
        "brand_name": "Drug",
        "procedure_name": "Procedure",
        "test_name": "ClinicalMeasurement",
        "test": "ClinicalMeasurement",
        "treatment_name": "Treatment",
        "treatment": "Treatment",
        "system_organ_site": "AnatomicalEntity",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """one streamed row (the Section_1..Section_6 cells, positional) -> one training example"""
        text_parts: list[str] = []
        resolved_all: list[ResolvedMention] = []
        relations: list[Relation] = []
        for section_value in values:
            section: tuple[str, list[Any]] | None = self.decode_section(section_value)
            if section is None:
                continue
            content, entries = section
            tokens, triples = self.tokenize(content)
            # entry validation lives in entity_spans (shape, Text-mismatch, duplicate collapse);
            # bounds, boundary slop, and mid-token snapping live in the shared bridge
            spans: list[tuple[int, int, str]] = ScriptUtils.char_spans_to_token_spans(triples, self.entity_spans(content, entries))
            text_parts.append(ScriptUtils.join_tokens(tokens))
            if not spans:
                continue
            mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
            resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
            # raw labels surface PascalCased (biolink-style casing) while mapped/fallback entries
            # already name a biolink class and stay untouched
            labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
            # spans and mentions filter identically, so zip pairs each span with its resolution
            # positionally (strict=True turns a dropped mention into an error, not a misalignment)
            resolved_spans = [(start, end, item.category) for (start, end, _), item in zip(spans, labeled, strict=True)]
            resolved_all.extend(labeled)
            # per-section extraction only: the section's own token stream is the whole haystack
            relations.extend(extract_relations(tokens, resolved_spans))
        return TrainingExample(text=" ".join(text_parts), entities=ScriptUtils.group_entities(resolved_all), relations=relations)

    @staticmethod
    def decode_section(value: Any) -> tuple[str, list[Any]] | None:
        """decode one Section_N cell into (content, entity entries), or None when the section cannot
        yield text: not a dict after the shared literal decode (None/junk/list/malformed -> {}) or a
        None/empty Section_Content. Skip-don't-coerce: the measured cells are dict reprs whose
        Entity_Recognition is a list or None, so a None Entity_Recognition -- or one that decodes to
        a non-list -- yields an empty entry list (the section keeps its text but contributes no
        entities) instead of vanishing or crashing."""
        section: dict[str, Any] = ScriptUtils.parse_literal_dict(value)
        content: Any = section.get("Section_Content")
        if not isinstance(content, str) or not content:
            return None
        recognized: Any = section.get("Entity_Recognition")
        if not isinstance(recognized, list):
            recognized = ScriptUtils._decode_container(recognized)
        return (content, recognized if isinstance(recognized, list) else [])

    @staticmethod
    def tokenize(content: str) -> tuple[list[str], list[tuple[str, int, int]]]:
        """(tokens, (token, start_char, end_char) triples) over the pre-tokenized content: the
        corpus's own tokenizer IS content.split(), so accumulated single-space offsets tile the text
        exactly the way the shared bridge's bounds check expects (char ends exclusive)"""
        tokens: list[str] = content.split()
        triples: list[tuple[str, int, int]] = []
        position: int = 0
        for token in tokens:
            triples.append((token, position, position + len(token)))
            position += len(token) + 1
        return tokens, triples

    @staticmethod
    def entity_spans(content: str, entries: list[Any]) -> list[tuple[int, int, str]]:
        """validate Entity_Recognition entries into (begin, end_exclusive, type) char spans;
        malformed entries are skipped, never coerced (mirrors KnowledgatorBiomedScript.char_spans).

        An entry must be a dict carrying int (bool is not int) BeginOffset/EndOffset and a str Type;
        Id/Score/Category/Traits/Attributes are ignored. Offsets are authoritative: the surface is
        content[begin:end], and an entry whose Text differs from that slice drops (a stale
        annotation); the Text check is also what catches inverted and negative offsets, whose slice
        can never equal a real Text. Exact (begin, end, type) duplicates collapse (Comprehend
        re-annotates spans the Stanza pass already annotated), while the same span under two types
        survives -- that measured Comprehend+Stanza overlap is real signal, not duplication.
        """
        spans: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int, str]] = set()
        for entry in entries:
            begin: Any = entry.get("BeginOffset") if isinstance(entry, dict) else None
            end: Any = entry.get("EndOffset") if isinstance(entry, dict) else None
            label: Any = entry.get("Type") if isinstance(entry, dict) else None
            if (
                isinstance(begin, bool)
                or not isinstance(begin, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not isinstance(label, str)
            ):
                continue
            if entry.get("Text") != content[begin:end]:
                continue
            key: tuple[int, int, str] = (begin, end, label)
            if key not in seen:
                seen.add(key)
                spans.append(key)
        return spans


validate_label_map(BioleafletsScript.LABEL_MAP, BioleafletsScript.NAME)

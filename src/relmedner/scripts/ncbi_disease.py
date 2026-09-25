from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class NcbiDiseaseScript(Script):
    """streams ncbi/ncbi_disease rows (tokens + integer ClassLabel ner_tags over the
    refs/convert/parquet branch, the bigbio/ehr_rel read path) into biolink-labeled entity and
    gazetteer-relation examples.

    Provenance: the hub repo ships an ncbi_disease.py loader script (unsupported on this repo's
    datasets version), so the declared source is the ehr_rel pattern: one hf_parquet entry per
    split file. The corpus is the NCBI disease corpus (Dogan et al. 2014, PUBLIC DOMAIN US
    government work): 793 PubMed abstracts, sentence-split, dual-annotator consensus with MeSH /
    OMIM concept normalization. Measured over the full parquet splits on wenceslaus
    2026-09-24: train 5,433 / validation 924 / test 941 sentences; tag census (single closed
    Disease class) O 169,361 / B-DISEASE 6,892 / I-DISEASE 8,299; max sentence 123 tokens.
    The raw parquet loses the ClassLabel name table, so the script carries the fixed 3-name
    vocabulary (O, B-DISEASE, I-DISEASE) and decodes int or str-int tags itself; a tag index
    outside the vocabulary is not a span and not a guess, so it decodes as O
    (skip-don't-coerce; the IOB decoder already treats non-B/I tags as background). LABEL_MAP
    covers the corpus's complete measured vocabulary (the ClassLabel list is closed):
    disease -> Disease. Tags may also arrive as their B-/I- strings from a future schema
    revision; those pass the decoder through untouched. The corpus ships no relations; the
    gazetteer relations emitted here are derived signal, the gliner_biomed/jnlpba convention.
    """

    NAME: ClassVar[str] = "NcbiDiseaseScript"

    # the fixed ClassLabel vocabulary of the parquet refs, index -> tag string (O first)
    TAG_NAMES: ClassVar[tuple[str, ...]] = ("O", "B-DISEASE", "I-DISEASE")

    # the complete measured corpus vocabulary, lowercased -> biolink classes; values validated
    # against tablassert Categories at import below
    LABEL_MAP: ClassVar[dict[str, str]] = {"disease": "Disease"}

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out is [tokens, ner_tags], so values = (tokens, ner_tags); unpack positionally
        and defensively. A non-list or ragged pair yields the rejoined text with zero spans (the
        entities shape simply never fills), and a malformed tag decodes as background rather than
        fabricating a span. run never raises on a bad row."""
        tokens_value, tags_value = values
        tokens: list[str] = self.token_list(tokens_value)
        tags: list[str] = self.tag_list(tags_value)
        if not tokens or len(tokens) != len(tags):
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans: list[tuple[int, int, str]] = ScriptUtils.iob_spans(tags)
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        # one batched resolution round trip per row, never per mention
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
        # the resolution chain can return MORE resolutions than spans (the fullmap multi-class
        # fan-out adds a secondary class per mention), so a strict positional zip of spans with
        # resolutions raised 'zip() argument 2 is longer than argument 1' on real hub rows and
        # crashed the smoke pipeline at dispatch (wenceslaus 2026-09-24). pair_spans re-pairs by
        # span_index instead, the JnlpbaScript / GlinerBiomedScript pattern: every fan-out item
        # extends its span, and group_entities dedups the mention surfaces.
        resolved_spans: list[tuple[int, int, str]] = [
            (start, end, item.category) for (start, end, _), items in ScriptUtils.pair_spans(spans, resolved) for item in items
        ]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )

    @classmethod
    def token_list(cls, value: Any) -> list[str]:
        """the tokens column -> list[str]; a real list of str passes, a python-repr string decodes
        (ast.literal_eval, no code execution), anything else yields [] (skip-don't-coerce)"""
        if isinstance(value, list):
            return [token for token in value if isinstance(token, str)]
        return ScriptUtils.parse_literal_list(value)

    @classmethod
    def tag_list(cls, value: Any) -> list[str]:
        """the ner_tags column -> IOB tag strings; parquet delivers ClassLabel ints, a future
        schema revision may deliver the tag strings themselves, and a drift to another int
        vocabulary must not silently renumber spans. An int (or str-int) inside the vocabulary
        decodes through TAG_NAMES; anything out of range decodes as O, a missing background tag
        rather than a guessed span (skip-don't-coerce)."""
        entries: list[Any] = value if isinstance(value, list) else []
        tags: list[str] = []
        for entry in entries:
            tag: str = "O"
            if isinstance(entry, bool):
                pass
            elif isinstance(entry, int) and 0 <= entry < len(cls.TAG_NAMES):
                tag = cls.TAG_NAMES[entry]
            elif isinstance(entry, str):
                if entry.isdigit() and int(entry) < len(cls.TAG_NAMES):
                    tag = cls.TAG_NAMES[int(entry)]
                elif entry == "O" or (len(entry) > 2 and entry[1] == "-" and entry[:1] in ("B", "I")):
                    tag = entry
            tags.append(tag)
        return tags


validate_label_map(NcbiDiseaseScript.LABEL_MAP, NcbiDiseaseScript.NAME)

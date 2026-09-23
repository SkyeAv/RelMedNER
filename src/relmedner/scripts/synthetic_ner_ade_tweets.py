from __future__ import annotations

from typing import ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


def parse_ann_spans(ann: str, text_len: int) -> list[tuple[int, int, str]]:
    """parse BRAT standoff T-lines into (start, end_exclusive, raw_label) char spans.

    Skip-don't-coerce: a T-line drops on any defect (wrong tab arity, non-integer offsets,
    start >= end, out-of-bounds vs text_len, a ';' in the offsets field marking a discontinuous
    span) and never becomes a coerced mention. N-lines (Reference / meddra_llt_id provenance)
    are never read: provenance is neither shipped nor used. The surface after the second tab is
    not trusted for offsets; the token bridge re-joins surfaces from whole tokens, so
    containment holds by construction.
    """
    spans: list[tuple[int, int, str]] = []
    for line in ann.splitlines():
        if not line.startswith("T"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        _token_id, offsets, _surface = fields
        if ";" in offsets:
            continue
        parts = offsets.split()
        if len(parts) != 3:
            continue
        raw_label, start_text, end_text = parts
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            continue
        if start >= end or start < 0 or end > text_len:
            continue
        spans.append((start, end, raw_label))
    return spans


class SyntheticNerAdeTweetsScript(Script):
    """streams synthetic-ner-ade-tweets rows (a BRAT-annotated tweet: raw text plus a standoff
    .ann column) into biolink-labeled entity examples.

    A row is one tweet: {text, ann}. The 17,000-doc corpus carries 10,341 T-lines, all with the
    single raw label ADE, and char offsets are END-EXCLUSIVE (measured 10,341/10,341 exclusive, 0
    inclusive, on the full corpus; 0 malformed, 0 out-of-bounds, 0 degenerate, 0 discontinuous, 0
    containment failures on text[start:end] == surface). 8,502 docs (8,500 negatives plus
    pos_5808 train and pos_341 val) ship an empty .ann and become text-only examples, which the
    declared-outputs filter drops. The 10,341 N-lines (10,314 numeric meddra_llt_id plus 27 None)
    are never read: provenance is neither shipped nor used. Resolution is the shared chain
    (fullmap first, then the LABEL_MAP fallback tier), so a mention fullmap resolves lands on its
    specific biolink class and the map's DiseaseOrPhenotypicFeature is the generic tier for the
    ADE tail.
    """

    NAME: ClassVar[str] = "SyntheticNerAdeTweetsScript"

    # the corpus's single raw label, lowercased, to its honest biolink class (tablassert
    # Categories HAS DiseaseOrPhenotypicFeature and has NO SignOrSymptom, which is why the map
    # points here; validated at import below)
    LABEL_MAP: ClassVar[dict[str, str]] = {"ade": "DiseaseOrPhenotypicFeature"}

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        (record,) = values
        text: str = record.get("text") if isinstance(record, dict) else None
        ann: str = record.get("ann") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text:
            return TrainingExample(text="")
        triples: list[tuple[str, int, int]] = list(FullmapMiner.splitter()(text, lower=False))
        tokens: list[str] = [token for token, _start, _end in triples]
        char_spans: list[tuple[int, int, str]] = parse_ann_spans(ann, len(text)) if isinstance(ann, str) and ann.strip() else []
        spans: list[tuple[int, int, str]] = ScriptUtils.char_spans_to_token_spans(triples, char_spans)
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # raw labels surface PascalCased (biolink-style casing) while mapped and fallback entries
        # already name a biolink class and stay untouched
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


validate_label_map(SyntheticNerAdeTweetsScript.LABEL_MAP, SyntheticNerAdeTweetsScript.NAME)

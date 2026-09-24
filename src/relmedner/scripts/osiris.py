from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class OsirisScript(Script):
    """streams bigbio/osiris rows (OSIRIS corpus of genetic-variant mentions in PubMed
    abstracts, cc-by-3.0) into biolink-labeled entity examples.

    Provenance: the hub repo is script-only (unsupported on this repo's datasets version), so
    the declared source reads the `osiris_bigbio_kb` config off refs/convert/parquet, the
    bigbio/ehr_rel read path. Measured over the full single parquet file on the laptop
    2026-09-24: 105 documents, 1,230 gold entities across exactly two passages per document
    (title + abstract, inter-passage gap 1 char on 105/105), types `gene` (689) and `variant`
    (541), every entity single-part (1,230/1,230), 1,228/1,230 spans verify end-EXCLUSIVE
    against an offset-aware passage reconstruction (text[start:end] == surface); the remaining
    2 are sloppy-source trailing-space offsets that the char bridge clamps or drops. Each
    entity also carries `normalized` dbSNP/Entrez ids; those are provenance, not training
    signal (the pipeline emits no entity normalization), so they are read and dropped. Offsets
    snap to token spans through the shared char bridge (out-of-bounds and degenerate spans
    drop, whitespace slop clamps). Snap-to-token spans go through NO re-resolution: the
    `gene`/`variant` labels are gold, the trust-gold stance of LinnaeusScript /
    CtkpInterventionsScript / SuperGlueRecordScript: each surviving span ships under its own
    mapped biolink class, and an entity type with no honest biolink target drops instead of
    surfacing as a guessed label (skip-don't-coerce). The emitted text is the re-joined token
    stream so every mention surface stays a substring. The corpus ships no relations
    (relations column empty on every row) and none are fabricated: the declared outputs are
    [entities] only. The `_source` config re-renders the same documents and is deliberately
    not declared.
    """

    NAME: ClassVar[str] = "OsirisScript"

    # lowercase dataset classes -> biolink classes; the corpus's complete measured vocabulary is
    # `gene` + `variant`. Values are validated against tablassert Categories at import below.
    # Trust-gold: a type with no entry drops rather than guessing.
    LABEL_MAP: ClassVar[dict[str, str]] = {"gene": "Gene", "variant": "SequenceVariant"}

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out is [passages, entities], so values = (passages, entities); unpack
        positionally and defensively. A malformed passage list yields the empty example, and a
        malformed entity dict is skipped, never coerced (skip-don't-coerce). run never raises on
        a bad row."""
        passages_value, entities_value = values
        text: str = self.passage_text(passages_value if isinstance(passages_value, list) else [])
        entities: list[Any] = entities_value if isinstance(entities_value, list) else []
        if not text:
            return TrainingExample(text="")
        triples: list[tuple[str, int, int]] = list(FullmapMiner.splitter()(text, lower=False))
        tokens: list[str] = [token for token, _start, _end in triples]
        emitted: str = ScriptUtils.join_tokens(tokens)
        char_spans: list[tuple[int, int, str]] = self.char_spans(entities)
        spans: list[tuple[int, int, str]] = ScriptUtils.char_spans_to_token_spans(triples, char_spans)
        if not spans:
            return TrainingExample(text=emitted)
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), label) for start, end, label in spans]
        # one batched resolution round trip per row, never per mention
        # trust-gold: ship each surviving span under its own mapped biolink class; no fullmap
        # round trip, which measurably remaps gold species surfaces (LinnaeusScript docstring)
        resolved: list[ResolvedMention] = []
        for surface, label in mentions:
            category: str | None = self.LABEL_MAP.get(label.lower())
            if category is None or not surface:
                continue
            resolved.append(ResolvedMention(mention=surface, category=category, origin="raw"))
        if not resolved:
            return TrainingExample(text=emitted)
        return TrainingExample(text=emitted, entities=ScriptUtils.group_entities(resolved))

    @staticmethod
    def passage_text(passages: list[Any]) -> str:
        """reconstruct the document text by placing each passage at its declared char offset
        and gap-filling with spaces; the measured corpus carries exactly two passages per row
        (title + abstract, gap 1 char on 105/105) with global entity offsets, so the gap fill
        is what keeps title-relative and abstract-relative offsets both addressable"""
        placed: list[tuple[int, str]] = []
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            offsets = passage.get("offsets")
            texts = passage.get("text")
            if not isinstance(offsets, list) or not offsets or not isinstance(offsets[0], list):
                continue
            start = offsets[0][0]
            if isinstance(start, bool) or not isinstance(start, int):
                continue
            body = "".join(str(part) for part in texts) if isinstance(texts, list) else ""
            placed.append((start, body))
        if not placed:
            return ""
        ordered = sorted(placed)
        chunks: list[str] = []
        cursor = 0
        for start, body in ordered:
            if start > cursor:
                chunks.append(" " * (start - cursor))
            chunks.append(body)
            cursor = max(cursor, start + len(body))
        return "".join(chunks)

    @staticmethod
    def char_spans(entities: list[Any]) -> list[tuple[int, int, str]]:
        """extract (start, end_exclusive, class) char triples; malformed entries are skipped,
        never coerced (mirrors ScriptUtils.mention_spans) -- bounds, degeneracy, and mid-token
        snapping live in the bridge, not here. The measured corpus is single-part (1,230/1,230
        one offset each); a text/offsets length mismatch drops the whole entity because the
        offset-to-surface alignment is no longer trustworthy."""
        spans: list[tuple[int, int, str]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            texts = entity.get("text")
            offsets = entity.get("offsets")
            entity_type = entity.get("type")
            if not isinstance(texts, list) or not isinstance(offsets, list) or not isinstance(entity_type, str):
                continue
            if len(texts) != len(offsets):
                continue
            for surface, span in zip(texts, offsets, strict=True):
                if not isinstance(surface, str) or not isinstance(span, list) or len(span) != 2:
                    continue
                start, end = span
                if isinstance(start, bool) or isinstance(end, bool):
                    continue
                if isinstance(start, int) and isinstance(end, int):
                    spans.append((start, end, entity_type))
        return spans


validate_label_map(OsirisScript.LABEL_MAP, OsirisScript.NAME)

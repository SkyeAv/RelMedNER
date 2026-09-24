from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class JnlpbaScript(Script):
    """streams commanderstrife/jnlpba rows (tokens + integer ClassLabel ner_tags over the
    datasets-server parquet refs, the bigbio/ehr_rel read path) into biolink-labeled entity and
    gazetteer-relation examples.

    Provenance: the hub repo ships a jnlpba.py loader script (unsupported on this repo's datasets
    version), but its refs/convert/parquet branch carries the three splits as parquet files, so the
    declared source is the ehr_rel pattern: one hf_parquet entry per split file. The raw parquet
    loses the ClassLabel name table, so the script carries the fixed 19-name vocabulary
    (O, B/I-GENE, CHEMICAL, DISEASE, DNA, RNA, CELL_LINE, CELL_TYPE, PROTEIN, SPECIES) and decodes
    int or str-int tags itself; a tag index outside the vocabulary is not a span and not a guess,
    so it decodes as O (skip-don't-coerce; the IOB decoder already treats non-B/I tags as
    background). LABEL_MAP covers the corpus's complete measured vocabulary (the ClassLabel list is
    closed): chemical -> ChemicalEntity, gene -> Gene, protein -> Protein, disease -> Disease,
    dna/rna -> NucleicAcidEntity (the pubmed_abstracts convention), cell_line -> CellLine,
    cell_type -> Cell (the pile_ner_type convention), species -> OrganismTaxon. Tags may also
    arrive as their B-/I- strings from a future schema revision; those pass the decoder through
    untouched. 9 of the 10 keys are direct lowercase hits and cell_line/cell_type additionally
    resolve through normalize_iob_label's underscore folding.
    """

    NAME: ClassVar[str] = "JnlpbaScript"

    # the fixed ClassLabel vocabulary of the parquet refs, index -> tag string (O first)
    TAG_NAMES: ClassVar[tuple[str, ...]] = (
        "O",
        "B-GENE",
        "I-GENE",
        "B-CHEMICAL",
        "I-CHEMICAL",
        "B-DISEASE",
        "I-DISEASE",
        "B-DNA",
        "I-DNA",
        "B-RNA",
        "I-RNA",
        "B-CELL_LINE",
        "I-CELL_LINE",
        "B-CELL_TYPE",
        "I-CELL_TYPE",
        "B-PROTEIN",
        "I-PROTEIN",
        "B-SPECIES",
        "I-SPECIES",
    )

    # the complete measured corpus vocabulary, lowercased -> biolink classes; values validated
    # against tablassert Categories at import below
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "chemical": "ChemicalEntity",
        "gene": "Gene",
        "protein": "Protein",
        "disease": "Disease",
        "dna": "NucleicAcidEntity",
        "rna": "NucleicAcidEntity",
        "cell_line": "CellLine",
        "cell_type": "Cell",
        "species": "OrganismTaxon",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out is [tokens, ner_tags], so values = (tokens, ner_tags); unpack positionally
        and defensively. A non-list or ragged pair yields the rejoined text with zero spans (the
        declared-outputs filter drops nothing here: entities shape simply never fills), and a
        malformed tag decodes as background rather than fabricating a span. run never raises on a
        bad row."""
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
        # spans and mentions filter identically, so zip pairs each span with its resolution
        # positionally, and strict=True turns a dropped mention into an error instead of a silent
        # misalignment of every later span
        resolved_spans: list[tuple[int, int, str]] = [(start, end, item.category) for (start, end, _), item in zip(spans, labeled, strict=True)]
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


validate_label_map(JnlpbaScript.LABEL_MAP, JnlpbaScript.NAME)

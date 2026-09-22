from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


def coerced_span(span: Any, token_count: int) -> tuple[int, int, str] | None:
    """coerce one gliner-multilingual NER span into (start, end_inclusive, label) or None to drop it.

    The corpus ships two span shapes: real ints [18, 21, "organization"] and stringified
    ["18", "21", "\\"organization\\""] (string indices, label wrapped in literal quote chars). The
    shim is strict (mirror ScriptUtils.mention_spans semantics, skip-don't-coerce): a span that is
    not a 3-element sequence, whose indices are neither real ints nor strings int() parses (decimal
    digits, though int() is lenient about padding and underscore separators: int(" 18 ") and
    int("1_8") are both 18), or whose label
    is not a string once the quote chars are stripped, is dropped -- malformed rows must never
    crash and never become silently coerced garbage. Bounds are end-inclusive against the row's
    tokens: 0 <= start <= end < token_count.
    """
    if not isinstance(span, (list, tuple)) or len(span) != 3:
        return None
    raw_start, raw_end, raw_label = span
    indices: list[int] = []
    for raw_index in (raw_start, raw_end):
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, str)):
            return None
        try:
            indices.append(int(raw_index))
        except ValueError:
            return None
    start, end = indices
    if not isinstance(raw_label, str):
        return None
    label: str = raw_label.strip("\"'")
    if not label:
        return None
    if start < 0 or end < start or end >= token_count:
        return None
    return start, end, label


class GlinerMultilingualScript(Script):
    """streams gliner-multilingual-synthetic rows (tokenized_text + ner span lists) into biolink-labeled
    entity examples; only the corpus's cross-lingual head labels are mapped to biolink classes, and
    everything else rides pascal_label so the zero-shot breadth survives

    The labeling chain is deliberately DIRECT: no ScriptUtils.resolve_mentions and no fullmap --
    multilingual synthetic surfaces (disease names in eight languages) have no fullmap term entries
    to begin with, and a LABEL_MAP hit plus PascalCased raw tail labels keep the whole vocabulary
    in biolink-style casing without paying a per-mention database lookup.
    """

    NAME: ClassVar[str] = "GlinerMultilingualScript"

    # cross-lingual head labels -> biolink classes; keys are lowercase and looked up on
    # raw_label.lower() (mirror the fallback-map lookup convention); values must be biolink classes
    # (checked loudly by the module-bottom validate_label_map call)
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "person": "Human",
        "personne": "Human",
        "persona": "Human",
        "pessoa": "Human",
        "osoba": "Human",
        "osobnik": "Human",
        "location": "GeographicLocation",
        "lieu": "GeographicLocation",
        "luogo": "GeographicLocation",
        "ort": "GeographicLocation",
        "lugar": "GeographicLocation",
        "local": "GeographicLocation",
        "locatie": "GeographicLocation",
        "miejsce": "GeographicLocation",
        "país": "GeographicLocation",
        "organization": "Agent",
        "organisation": "Agent",
        "organización": "Agent",
        "organização": "Agent",
        "disease": "Disease",
        "enfermedad": "Disease",
        "malattia": "Disease",
        "maladie": "Disease",
        "krankheit": "Disease",
        "choroba": "Disease",
        "drug": "Drug",
        "medication": "Drug",
        "medicamento": "Drug",
        "medicament": "Drug",
        "arzneimittel": "Drug",
        "lekarstwo": "Drug",
        "food": "Food",
        "alimento": "Food",
        "lebensmittel": "Food",
        "jedzenie": "Food",
        "nourriture": "Food",
        "plant": "Plant",
        "plante": "Plant",
        "pflanze": "Plant",
        "roślina": "Plant",
        "planta": "Plant",
        "animal": "OrganismTaxon",
        "animale": "OrganismTaxon",
        "tier": "OrganismTaxon",
        "zwierzę": "OrganismTaxon",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, ner_value = values
        tokens: list[str] = [str(token) for token in tokens_value] if isinstance(tokens_value, list) else []
        ner: list[Any] = ner_value if isinstance(ner_value, list) else []
        if not tokens or not ner:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans: list[tuple[int, int, str]] = [span for span in (coerced_span(entry, len(tokens)) for entry in ner) if span is not None]
        if not spans:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        # direct labeling chain: LABEL_MAP hit (lowercased lookup, mirroring the fallback-map
        # convention) first, PascalCased raw label second; mapped entries keep the "fallback"
        # origin, unmapped ones "raw" -- mirroring the resolution chain's provenance vocabulary
        # without ever consulting the fullmap
        resolved: list[ResolvedMention] = []
        for start, end, raw_label in spans:
            mapped: str | None = self.LABEL_MAP.get(raw_label.lower()) or self.LABEL_MAP.get(ScriptUtils.normalize_iob_label(raw_label))
            resolved.append(
                ResolvedMention(
                    mention=ScriptUtils.join_tokens(tokens[start : end + 1]),
                    category=mapped if mapped else ScriptUtils.pascal_label(raw_label),
                    origin="fallback" if mapped else "raw",
                )
            )
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(resolved),
        )


validate_label_map(GlinerMultilingualScript.LABEL_MAP, GlinerMultilingualScript.NAME)

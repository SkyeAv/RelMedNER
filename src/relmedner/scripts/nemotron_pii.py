from __future__ import annotations

import ast
from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


def parse_literal_spans(value: Any) -> list[dict[str, Any]]:
    """safely decode the dataset's spans column (a python-repr string of span dicts as datasets-server
    serves it, or an already-parsed list); anything that is not a list yields [] so callers skip the row
    instead of crashing or coercing (skip-don't-coerce, mirrors parse_literal_list). Non-dict entries
    inside the list drop individually so one junk entry never kills the row's good spans."""
    try:
        decoded = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return []
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, dict)]


def is_valid_span(span: dict[str, Any], text: str) -> bool:
    """one span dict passes only with int (not bool) start/end, a non-empty str label, and
    0 <= start < end <= len(text); malformed spans drop individually, never the whole row"""
    start: Any = span.get("start")
    end: Any = span.get("end")
    label: Any = span.get("label")
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and isinstance(label, str)
        and bool(label)
        and 0 <= start < end <= len(text)
    )


class NemotronPiiScript(Script):
    """streams nvidia/Nemotron-PII rows (text + python-repr span-dict column) into entity examples for
    the PII/PHI families the biomedical corpora never see

    Two measured dataset quirks shape this script. First, span dicts carry their own `text` field that
    drifts from the real surface (int-typed for age/cvv, case-drifted like 'spanish' against 'Spanish'),
    so mention surfaces come from text[start:end] end-exclusive slices ONLY and the field is never read.
    Second, the label vocabulary is PII-shaped, not biomedical: a small head maps onto biolink classes
    (people, places, employers) while the long tail (Ssn, Ipv4, Password, ...) stays unmapped and
    surfaces raw PascalCased, the Pile-NER precedent, so no PII label is ever dropped for lack of a
    biolink mapping.

    Entity-only by design: the gazetteer's predicates are biomedical-mined, so running relation
    extraction here would fabricate edges between PII mentions under unbounded predicates.
    document_format needs no special handling -- both formats decode identically as plain text.
    """

    NAME: ClassVar[str] = "NemotronPiiScript"

    # PII labels with a defensible biolink class, merged over ScriptUtils.FALLBACK_LABEL_MAP at
    # resolution time; everything else stays unmapped and surfaces PascalCased via the raw tail
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "first_name": "Human",
        "last_name": "Human",
        "occupation": "SocioeconomicAttribute",
        "education_level": "SocioeconomicAttribute",
        "employment_status": "SocioeconomicAttribute",
        "gender": "BiologicalSex",
        "country": "GeographicLocation",
        "city": "GeographicLocation",
        "state": "GeographicLocation",
        "county": "GeographicLocation",
        "street_address": "GeographicLocation",
        "postcode": "GeographicLocation",
        "coordinate": "GeographicLocation",
        "company_name": "Agent",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text_value, spans_value = values
        text: str = text_value if isinstance(text_value, str) else ""
        if not text.strip():
            return TrainingExample(text="")
        spans: list[dict[str, Any]] = parse_literal_spans(spans_value)
        # surfaces are end-exclusive slices of text only; the span dict's own `text` field is
        # deliberately never read (measured int-typed and case-drifted values)
        mentions: list[tuple[str, str]] = [(text[span["start"] : span["end"]], span["label"]) for span in spans if is_valid_span(span, text)]
        if not mentions:
            return TrainingExample(text=text)
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # mentions and resolutions pair positionally; raw-tail labels surface PascalCased while
        # fullmap/fallback hits already name a biolink class and stay untouched (Pile-NER pattern)
        labeled: list[ResolvedMention] = [
            item if item.origin != "raw" else ResolvedMention(mention=item.mention, category=ScriptUtils.pascal_label(label), origin=item.origin)
            for item, (_, label) in zip(resolved, mentions, strict=True)
        ]
        return TrainingExample(text=text, entities=ScriptUtils.group_entities(labeled))


validate_label_map(NemotronPiiScript.LABEL_MAP, NemotronPiiScript.NAME)

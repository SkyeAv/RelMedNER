# Script template and `ScriptUtils` map

A `script` task needs one class. A `fullmap` task needs none (mining is shared). This page is the
skeleton, the four landed variants, the utility map, and the drop-rule-to-test mapping.

## The contract (`src/relmedner/types.py`)

```python
class Script(ABC):
    NAME: ClassVar[str]
    REGISTRY: ClassVar[dict[str, Script]] = {}

    def __init_subclass__(cls, **kwargs): ...      # self-registers an INSTANCE under cls.NAME
    @classmethod
    def dispatch(cls, name, payload): ...          # payload = (outputs_tuple, values_tuple)
    @abstractmethod
    def run(self, values: ScriptValues) -> TrainingExample: ...
```

- `NAME` must equal the `task.name` declared in `ingests.yaml`; `parse_ingests` rejects an undeclared
  name and prints the declared list.
- `values` is the row projected onto `columns_out`, in declaration order. Unpack it positionally and
  defensively: hub streaming hands back `ClassLabel` ints, python-repr strings, stringified indices,
  and `None` depending on the builder and the cache state.
- `run` returns ONE `TrainingExample`. It never raises on a bad row: malformed input yields a
  text-only example (which the declared-outputs filter drops) or drops the offending span.

## Minimal skeleton (token list + span list, entities and gazetteer relations)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils

if TYPE_CHECKING:
    from relmedner.utils import ResolvedMention

class ExampleScript(Script):
    """one sentence on what a row is, then the measured facts that shaped the mapping: span
    encoding, bounds convention, label-census size, drop rates, and any deliberate omission"""

    NAME: ClassVar[str] = "ExampleScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, ner_value = values
        tokens: list[str] = [str(token) for token in tokens_value] if isinstance(tokens_value, list) else []
        ner: list[Any] = ner_value if isinstance(ner_value, list) else []
        if not tokens or not ner:
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        spans: list[tuple[int, int, str]] = ScriptUtils.mention_spans(tokens, ner)
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(ScriptUtils.mentions(tokens, ner))
        # spans and mentions filter identically, so zip pairs them positionally; strict=True turns a
        # dropped mention into an error instead of a silent misalignment of every later span
        resolved_spans: list[tuple[int, int, str]] = [
            (start, end, item.category) for (start, end, _), item in zip(spans, resolved, strict=True)
        ]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(resolved),
            relations=extract_relations(tokens, resolved_spans),
        )
```

Then `validate_label_map(ExampleScript.LABEL_MAP, ExampleScript.NAME)` at module scope if the class
carries a `LABEL_MAP`, so a non-biolink value fails at import rather than mislabeling training data.

## The four landed variants

| row shape | model to copy | the one rule that matters |
| --- | --- | --- |
| token list + `[start, end_inclusive, label]` spans | `scripts/gliner_biomed.py` | `mention_spans` + `resolve_mentions` + `group_entities`; `extract_relations` over the resolved spans |
| raw text + char-offset structs (`end` exclusive) | `scripts/knowledgator_biomed.py` | `FullmapMiner.splitter()(text, lower=False)` once per row, then `char_spans_to_token_spans`; emit the RE-JOINED token stream as `text` |
| python-repr strings / IOB tag lists | `scripts/pile_ner_biomed.py` | `parse_literal_list` then `iob_spans`; orphan `I-` tags promote to single-token spans instead of dropping |
| inline `<e1>`/`<e2>` tags plus a gold predicate | `scripts/sentence_rex.py` | a pure `(tagged sentence) -> (head, tail)` parser returning `None` to drop; strip only the four tag literals, no whitespace normalization |
| chat-shaped QA with surfaces, not offsets | `scripts/pile_ner_type.py` | recover spans with `token_occurrences` over `lowered_tokens` (folded ONCE per row); an unlocatable mention drops from entities AND relation spans so the two stay aligned |
| gold spans that must not be re-resolved | `scripts/super_glue_record.py`, `scripts/ctkp_interventions.py` | trust-gold: ship the dataset's own labels under a biolink class, verify `text[start:end] == surface`, no fullmap |
| labels only, no spans | `scripts/super_glue_multirc.py` | one `Classification`; guard `bool` BEFORE `int` (`isinstance(True, int)` is True) and never guess an out-of-vocabulary label |
| mixed multi-task rows | `scripts/gliner_biomed_post.py` + `families.py` | `RowFamily` dispatch on the label set alone; sampled negatives train under `not_<predicate>` |
| stringified indices / quote-wrapped labels | `scripts/gliner_multilingual.py` | a `coerced_span` that collapses both encodings and returns `None` for wrong arity, non-numeric, fractional, or boolean indices |

## `ScriptUtils` map (`src/relmedner/utils.py`)

| call | use it for |
| --- | --- |
| `join_tokens(tokens)` | the emitted `text`; keeps every mention surface a substring |
| `mention_spans(tokens, ner)` | `[start, end_inclusive, label]` spans out of a hub `ner` list, malformed entries skipped |
| `mentions(tokens, ner)` | the `(surface, raw_label)` pairs `resolve_mentions` wants |
| `char_spans_to_token_spans(triples, char_spans)` | char offsets -> token spans, with drop-then-snap (out-of-bounds drop, whitespace slop clamped, mid-token snapped out) |
| `iob_spans(tags)` | BIO/BIOES tag lists -> spans |
| `parse_literal_list(value)` | a python-repr string column (`"['a', 'b']"`); the single decode posture for external columns |
| `parse_conversations(value)` | chat-shaped columns -> `(text, [(role, mentions)])` |
| `parse_mention_list(value)` | a JSON list of surfaces inside one chat answer |
| `lowered_tokens(tokens)` / `token_occurrences(tokens, mention, lowered=...)` | surface -> token spans; fold once per row, not once per mention |
| `normalize_iob_label(label)` | strips `B-`/`I-`, underscores, whitespace (this is what makes an `'ORGANISMS '` variant need no map entry) |
| `pascal_label(label)` | the raw-label tail: `Abdominal Core` -> `AbdominalCore` |
| `lookup_label(label_map, raw)` | dataset-local map, then the normalizing retry |
| `pascal_raw_labels(resolved)` | PascalCase only the still-raw labels; mapped and fallback hits keep their biolink class |
| `resolve_mentions(spans, label_map=None)` | the shared chain: fullmap -> lowercased `FALLBACK_LABEL_MAP` -> raw. One batched redb round trip per call, so call it ONCE per row, never per mention |
| `group_entities(resolved)` | `ResolvedMention` list -> `Entity` list with class definitions and `[fullmap: CURIE | name]` descriptions |
| `is_biolink_category(label)` / `biolink_categories()` | membership checks for a trust-gold script |
| `resolve_predicate(raw)` | `(predicate, is_biolink_member)`; non-members keep a biolink-shaped `snake_case` native form |
| `fullmap_available()` | the mount check every fullmap-dependent test skips on |
| `ResolutionGate.accept(...)` / `PredicateRangeGate.accept(...)` | already wired into the chain; call them directly only when a script resolves outside `resolve_mentions` |

`ResolvedMention` carries `mention`, `category`, `curie`, `preferred_name`, and `origin`
(`fullmap` / `fallback` / `raw`), which is what the README's origin shares are counted from.

## LABEL_MAP rules

- Keys are the corpus's own label strings, lowercased; values MUST be biolink classes.
  `families.validate_label_map` enforces that at import.
- Map only what has an honest biolink target. A wrong bucket silently mislabels training data, so an
  unmappable label stays out and surfaces as a PascalCased raw tail. `LANGUAGE`, `MONEY`,
  `REGULATION OR LAW`, `Investigative Techniques`, `Group Processes`, and `OTHER` are landed examples
  of deliberate omissions.
- Do not add plural/whitespace variants that `normalize_iob_label` already folds; do add variants it
  cannot reach, and pin both halves of that contrast in a test.
- Census the FULL split before writing the map. The coverage rule that landed: every label whose
  actual mention surfaces fit one real biolink class and whose raw-origin span count is material gets
  an entry; the residual tail stays unmapped on purpose. Chasing a 90% share is how wrong buckets get
  invented.

## Drop rules and the tests they require

| drop rule | required test |
| --- | --- |
| malformed span (wrong arity, non-numeric, fractional, bool index) | one test per malformed shape, asserting the span is absent and its well-formed sibling survives |
| out-of-bounds or degenerate span | boundary test at both ends, plus a zero-length span |
| unlocatable mention (surface not in the text) | a hallucinated-surface row: the mention drops from entities AND relation spans |
| null / blank row | the row ships text-only (or empty) and the declared-outputs filter drops it |
| out-of-vocabulary label | the row ships with no guessed label |
| ragged parallel tables | unequal-length lists yield zero spans, not a crash |
| self-loop (head surface == tail surface) | the relation drops |
| tokenization tore a surface from the text (`CC-chemokines` vs `CC`, `-`, `chemokines`) | the span is filtered at extraction |

Plus, always: a nonzero-yield test over realistic rows, and a mention-in-text assertion over every
emitted entity and relation surface.

## Do not

- import torch or gliner2 at module scope: `FullmapMiner.splitter()` loads gliner2's
  `WhitespaceTokenSplitter` by file path precisely so torch never loads. Keep it that way.
- call `resolve_mentions` per mention, or re-tokenize per mention.
- normalize whitespace in a way that breaks `text` containment for a mention surface.
- retune anything in `constants.py`.
- swallow an exception to keep a row: drop the row or the span, and say so in the docstring with the
  measured rate.
- leave a placeholder or a `TODO` body; a stub that returns `TrainingExample(text="")` passes the
  suite and ships nothing.

# Statement qualifiers and negation

Gazetteer relations also carry the six DAKP-declared nullable qualifier slots
(`disease_context_qualifier`, `anatomical_context_qualifier`, `sex_qualifier`,
`population_context_qualifier`, `frequency_qualifier`, `temporal_context_qualifier`;
`species_context_qualifier` is deliberately absent per tablassert
`DISABLED_EDGE_FIELDS`). Attachment mirrors DAKP's `attach_qualifiers_with_scores`:
qualifiers fire only where a predicate relation fired (they qualify statements, not
entity pairs), the host is the nearest statement endpoint with the tail preferred, a
typed context never restates an endpoint, one value per slot per sentence, and
value-style slots (`frequency`/`temporal`) fall back to the multi-token phrase surface
("twice daily") when no mention follows the trigger. Word-bounded negation cues ("not",
"failed to", "without", ...) sharing a sentence with a fired predicate re-encode the
statement as `not_<predicate>` with `negated=true`.

Dataset-format notes (`PileNerBiomedScript`): `tokens`/`ner_tags` are python-repr strings
(`ast.literal_eval`, malformed rows skip); orphan `I-` tags promote to single-token spans
rather than dropping; raw labels PascalCase so the full 3,896-type tail stays
biolink-shaped. Measured full corpus: 100% of rows emit, ~188k entity mentions, and 6,058
gazetteer relations across 5,501 rows (9.3% relation-bearing, across 23 biolink predicates,
each carrying its biolink slot description as `relation_descriptions`).

Dataset-format notes (`SentenceRexScript`): each `sentences` row wraps its two relation
participants in `<e1>`/`<e2>` tags and `labels` carries the gold predicate; the text ships
with ONLY the four tag literals stripped (no whitespace normalization), measured true on all
43,044 well-formed rows. Of the 44,115 train rows, 550 are null and drop; 521 violate the
one-pair-per-tag invariant, 18 well-formed-count rows carry nested angle-bracket markup
inside a surface (`< sub>`), and 48 rows have case-insensitively identical head and tail
surfaces (self-loops), so 42,978 rows ship. Labels: 846 distinct raw strings, 837 after
snake_case normalization, of which 17 are biolink `Predicates` members covering 1,263 rows
(2.9%); only 4.2% of rows carry a biomedical-marker label, and general-domain rows are kept.
Formatting the strip-only rule deliberately tolerates: ~99% of rows wrap tag surfaces in
internal whitespace (`<e1> Myristica fragrans </e1>`) and 95.8% use spaced punctuation
(` ,`, ` .`).

Dataset-format notes (`PileNerTypeScript`): the general-domain pile-ner sibling packages NER
as chat rather than IOB columns. One human turn carries the document behind a `Text: `
prefix, then each entity type is asked for with the templated question
`What describes <type> in the text?` and answered by the next gpt turn with a JSON list of
surface mentions (`[]` for the negative-sampled types, whose rows fall out on the outputs
gate). Answers are surfaces, not offsets, so spans are recovered by matching the mention's
token subsequence case-insensitively against the document's whitespace tokens: that single
rule is both the hallucination guard (gpt occasionally answers with surfaces the document
never contains) and the tokenization guard, and it keeps entities and relation spans
aligned. Emitted mentions are the document's own token slice, so gliner2's sanitizer can
always find them, and repeated mentions contribute every occurrence as a relation span
while appearing once in the entity list. The type vocabulary is open-ended GPT output
(1,666 distinct types in a 1k-row probe, with `person`/`Person`/`PERSON` casing variants
collapsing on the lowercased fallback lookup); head labels ride the dataset-local
`LABEL_MAP` and the tail stays PascalCased raw. `Organization`, `Product` and `CreativeWork`
are not biolink classes, so `organization` maps to `Agent` and `product` stays a raw tail.

Dataset-format notes (`KnowledgatorBiomedScript`, dataset `knowledgator/biomed_NER`): rows are raw untokenized text plus
character-offset entity structs (`{start, end, class}`, end exclusive); char spans bridge to token
spans through `ScriptUtils.char_spans_to_token_spans` with drop-then-snap (measured: 0.04-0.24%
out-of-bounds dropped, ~0.4% whitespace slop normalized, 3.43% mid-token snaps); emitted `text` is
the re-joined token stream so every mention surface stays findable (26.7% of raw-text surfaces
would fail gliner2's validator); the 29-entry `LABEL_MAP` covers the 35 distinct raw class strings
measured on the full 4,840-row train split (21 canonical classes with an honest biolink target plus
8 plural/legacy variants; LANGUAGE, REGULATION OR LAW, MONEY, Unlabelled,
and the bare INTELLECTUAL variant stay unmapped -> raw PascalCase tails; gazetteer relations may
still fire as free signal beyond the declared shapes).

Dataset-format notes (`GlinerMultilingualScript`): the streaming loader delivers this
corpus's `ner` spans as stringified indices with quote-wrapped labels
(`[["18", "21", "\"organization\""]]`) while `tokenized_text` arrives as a real list, so the
script coerces defensively and skips malformed spans. The label tail is long and
multilingual: 21,640 distinct labels with the top-60 covering only 43.6% of spans, and the
labels themselves are multilingual (`person`/`Person`/`personne`/`Persona`/`Pessoa`/`Osoba`),
so the cross-lingual head labels map onto biolink classes (Human, GeographicLocation, Agent,
Disease, Drug, Food, Plant, OrganismTaxon) and everything else rides PascalCase for
zero-shot breadth. Fullmap is bypassed by design: its keys are byte-sorted bags of Porter2
English stems over a biomedical vocabulary, so general-domain non-English surfaces never
match it and label directly off `LABEL_MAP` plus PascalCase. Only `entities` is declared:
the gazetteer's 129 triggers are English biomedical phrases, so relations are not declared.

Dataset-format notes (`SuperGlueMultiRCScript`): general-domain English true/false reading-comprehension QA
(SuperGLUE MultiRC), mapped classification-only -- the corpus carries no entity or relation annotations. The
`label` column arrives as a ClassLabel index or its decoded string; the test split also carries unlabeled rows
(`label` -1), which this pipeline never ingests (train split only, per repo convention) and would skip rather
than coerce. The nested `idx` column is deliberately excluded: row provenance, not training signal.

Dataset-format notes (`PubmedAbstractsScript`): the hub file is one 35,000-object JSON
array (151,668,527 bytes); `tokenized_text` is a token list and `ner` carries end-inclusive
`[start, end, "<MeSH heading> - <definition>"]` spans -- 383,721 of them over 5,850
distinct labels, every label shaped `heading - definition`. Rows are long (median 233 /
max 1,558 tokens, none truncated) and 13 rows have empty `ner`, exiting as text-only
examples that the permitted-shapes contract drops. The repo ships a broken old-style
`dataset_infos.json`: under `datasets` 5.0.1 the hub path raises `KeyError: 'feature'`
inside `Features.from_dict`, with and without `data_files=`, so the ingest reads the raw
file non-streaming through the `hf_json` source (`HuggingFaceJsonDataStream`). Never read
it streaming: on a cold cache the streaming JSON builder promotes every `ner` cell to utf8
straight from the raw JSON text, `mention_spans` would then drop every span, and the
ingest would silently yield nothing. Each label is split on its first `" - "` and only the
bare heading reaches resolution: with the definition riding along, the shared gate's
label-word buckets false-reject 45,975 of 383,721 full-corpus spans (12.0%: gate accepts
the bare heading, rejects the full label -- 14.5% of the 316,600 fullmap-hit spans; e.g.
'ankle' loses a correct Disease hit because the definition contains 'region'/'leg').
Unresolved headings surface PascalCased (`Abdominal Core` -> `AbdominalCore`). Measured on
the FULL 35,000-row corpus with one batched fullmap round trip and the shared gates:
fullmap 316,600 spans (82.5%), fallback 15,437 (4.0%), raw 51,684 (13.5%) under the
US-002 seed map. The `LABEL_MAP` coverage rule maps every heading whose actual mention
surfaces fit one real biolink class -- every clean heading with >=30 raw-origin spans is
mapped, and the residual >=30 tail is mixed/junk or borderline -- rather than chasing a
>=90% share that is unreachable without mislabeling data; the US-004 map holds 211 entries
(19 seed + 192 measured), and the honest measured coverage is 44.2% (origin shares after:
82.5% / 10.0% / 7.5%) because the residual raw tail is dominated by headings with no
faithful class -- `Investigative Techniques` is 94% the surfaces 'methods'/'METHODS',
`Group Processes` is 99.7% 'role' and biolink has no Role class, `Chemical Phenomena` and
`Genetic Phenomena` mix categories -- which stay unmapped on purpose (a wrong bucket
silently mislabels training data). Re-measuring never re-downloads the 152MB file: cached
full file + one batched fullmap round trip (`rs.normalize_terms` then `_fullmap_best`) +
the shared gates make every candidate map pure set arithmetic over headings, so the harness
stays throwaway and is deliberately not committed. Gazetteer relations over a 5,000-row
sample through the production `run()` path with the expanded map: 18.4% of rows carry >=1
relation (1,107 relations, 22 distinct predicates); 100% of rows emit entities.

## Related

- [README index](../README.md) -- the full docs index
- [fullmap mining](fullmap-mining.md) -- the mined spans the gazetteer matches
- [ingests](ingests.md) -- the PredicateRangeGate rules qualifier contexts must satisfy

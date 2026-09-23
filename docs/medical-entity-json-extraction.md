# Medical Entity JSON Extraction

The smallest declared ingest: `Pennlaine/Medical-Entity-JSON-Extraction` via
`MedicalEntityJsonScript`, a 50-row consumer-health vignette corpus whose gold spans arrive
as JSON in the assistant turn.

Dataset-format notes (`MedicalEntityJsonScript`, dataset `Pennlaine/Medical-Entity-JSON-Extraction`):
50 instruction-tuned consumer-health vignettes (hub metadata declares apache-2.0; the README body
is empty), single `test` split, 50 rows (source: datasets-server `size` endpoint). Each `text` is an
`[INST]` turn wrapping the passage, then a fenced ```json block holding `{"Question", "Answer",
"Entities": [{<Label>: <surface>}, ...]}`; no row carries a closing fence, so the JSON is decoded
with `raw_decode` from the first brace. The sibling `instruction` column is a dead dangling variant
(a truncated instruction ending in an open ```json fence with no body) and is deliberately not
declared. Full-split census (2026-09-22, wenceslaus): 259 entity entries, 0 JSON errors, 0
`[INST]` non-matches, 0 malformed entries; surfaces 102 verbatim (39.4%), 130 casefold-only
(50.2%), 27 no-substring paraphrases (10.4%) that drop, skip-don't-coerce. Labels are
consumer-health attribute slots, not biomedical types (50 distinct keys, 30 hapaxes; top Name 50,
Age 50, Profession 34, Condition 22, Specialty 11, Treatment 9, Management 9), so the ingest trusts
gold: no fullmap re-resolution and no `LABEL_MAP`, every emitted mention rides its raw PascalCase
label (the `SuperGlueRecordScript` stance). Surfaces locate by case-insensitive char-level
containment over the once-folded re-joined passage and ship in passage casing (232/259); token
equality finds only 7/259 (naive whitespace split) and 0/259 (gliner2 splitter), the best
token-level variant (substring of some token) still only 77/259 (29.7%), which is why the locate
is char-level. Declared-probe verification: `rows_in=50 rows_out=50`, 100% of rows emit entities,
232 entity mentions. Rows are short (median 113 tokens, max 158), no filter block needed.

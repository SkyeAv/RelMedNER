# agentlans/json-extraction

The structured-extraction corpus: six source configs, one `train` split each, 24,768 rows
of `text` + JSON schema + extracted JSON, harvested from six different extraction tasks.
It is the pipeline's first `structures`-shape corpus: each row's decoded JSON flattens into
one `json_structures` block, with honest-biolink entities where a value's surface survives
the containment guard.

Dataset-format notes (`JsonExtractionScript`): six source configs of `agentlans/json-extraction`, one
`train` split each; the hub `all` config is the exact union of the six (24,768 rows) and is
deliberately NOT declared, which would double-stream every row. A row is `text` + `schema` + `json`
(all strings) + `source`, where `source` carries the original hub repo id with slashes, not the
dashed config name; `columns_out` projects `[text, json, source]` and the schema column is not
projected (descriptions would add no measured signal). Per-source rows: ProfessorBob 6,920,
roborovski 5,945, sandeeppanem 4,879, Jiraya 3,035, HenriqueGodoy 2,606, owkin 1,383. Every number
below names its measurement once here: the full-split census (`~/json_extraction_census.py`, 24,768
rows), the US-002 qualifier pass (`~/json_extraction_measure.py`), or the declared probe. The census
measured 0 json parse failures and 0 empty texts; decoded json: 14,813 dicts / 9,955 lists, 443,021
string leaves, median 15 scalar fields per row (max 392), max nesting depth 8, text median 1,788
chars (max 315,692). Each row ships one Structure named by its source id: flattening drops nulls
(5,617) and blank strings, stringifies scalars, collapses scalar lists into one terminal list
field, and flattens list-of-dict children to `parent[].child` paths; the 37,366 distinct
corpus-wide key paths collapse to a small closed set per source schema. Entities trust gold with no
fullmap re-resolution (general-domain text against a biomedical vocabulary is noise; same
philosophy as `SuperGlueRecordScript`): a string leaf becomes a mention only when its (source,
path) has an honest biolink map AND the surface occurs verbatim in the text. Corpus-wide verbatim
containment is 57.4% (owkin 11.8%, ProfessorBob 45.2%, sandeeppanem 55.0%, HenriqueGodoy 52.0%,
roborovski 71.8%, Jiraya 83.3%), so containment is a per-leaf guard, never a row guard. The maps
hold only honest biolink targets: owkin `conditions[]` -> Disease and `interventions[].name` ->
ChemicalEntity only when the sibling type is Drug/Biological/Dietary Supplement/Genetic (contained
678/2,530; conditions 141/1,963, a corpus property of normalized abstract extractions); roborovski
name paths -> Agent only behind a sibling Person marker (33 names corpus-wide, 32 contained);
sandeeppanem `current_company`/`previous_companies[]` -> Agent with a measured "Company Name"
placeholder guard dropping 5,650 of 13,992 leaves (40.4%); Jiraya `[].Job Location` ->
GeographicLocation; HenriqueGodoy ships structures only (`entity_name[]` mixes drugs, paper titles
and finance concepts with no type qualifier). Everything without an honest biolink target stays
unmapped and out (sandeeppanem location is omitted on placeholder noise even though 2,894/3,787
locate). Relations are ProfessorBob only: the census of all 222 distinct predicate values found
exactly one honest biolink target, "member of" -> member_of (158 triples, 85 with both endpoints
verbatim in text); 31,303/31,740 triples carry non-biolink predicates and stay structures-only, "no
relation" (14,333 triples) drops, and self-loops plus endpoints absent from the text drop.
Declared-probe yields over the first 200 rows of each entry (`probe.py --declared <entry> --limit
200 --script JsonExtractionScript --outputs <per entry>`): structures emit 100% of rows on all six
entries (200/200 each, rows_in=rows_out=200, no drops); entity rows Jiraya 193, owkin 68,
sandeeppanem 27, roborovski 0 (the Person marker is rare by design); ProfessorBob and
HenriqueGodoy entities are not declared.

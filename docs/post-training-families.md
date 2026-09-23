# How post-training row families work

`src/relmedner/families.py` splits the multi-task corpus into disjoint families on the NER
label set alone (`RowFamily` ABC, self-registering, consulted in priority order):

| family | share | output shape |
| --- | --- | --- |
| native relations (`head <> predicate <> tail` span labels) | 2.4% | `relations` |
| classification option lists (`label`/`category`/`class`/`tag`) | 9.0% | `classifications` |
| `match` span extraction | 16.0% | `structures` |
| open-vocabulary NER (anything else) | 68.1% | `entities` + gazetteer `relations` |
| empty `ner` | 4.4% | dropped |

Zero-shot breadth rules on this corpus:

- predicates map onto `tablassert.biolink.Predicates` when one matches (`associated with`
  -> `associated_with`); everything else keeps a biolink-shaped `snake_case` native form;
- sampled negatives from the dataset's `negatives` column train under `not_<predicate>`
  names after guards: malformed, self-loop, duplicate, positive-colliding, and
  not-in-text triples drop, capped at 2x the row's positive count;
- relation surfaces that tokenization tore away from the text (`CC-chemokines` vs tokens
  `CC`, `-`, `chemokines`) are filtered at extraction -- gliner2 would drop them anyway;
- `GlinerBiomedPostScript.LABEL_MAP` extends the shared `FALLBACK_LABEL_MAP` with 25
  validated biolink classes; values are validated loudly at import.

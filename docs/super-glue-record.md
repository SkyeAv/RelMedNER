# SuperGLUE ReCoRD

The one ingest whose script ships general-domain text: `aps/super_glue` subset `record` is a
CNN news passage per row (100,730 training rows), annotated with gold entity spans
(`entity_spans` is a dict of parallel `text`/`start`/`end` lists) and a cloze-style query whose
answers name which spans fill its placeholder.

`SuperGlueRecordScript` trusts the dataset's gold and re-resolves nothing: spans ship verbatim
under the single catch-all biolink class `NamedThing`, the same trust-gold stance as
`CtkpInterventionsScript`, because the passages are news text unrelated to the biomedical
vocabularies fullmap resolves against -- re-categorizing "Dallas" or "the Rams" through those
would be noise, and ReCoRD declares no gold relations either. A span survives only when its
table is well-formed (equal-length lists) and `passage[start:end]` matches `text` exactly;
malformed or ragged span tables yield zero spans rather than a crash.

The cloze task ships as one `Classification(task='cloze entity resolution')`: labels are the
surviving span surfaces, `true_label` the gold answers that occur in that label set, and
`prompt` the raw query. Answers whose surface never survives span validation drop rather than
corrupting the label set; when no gold answer survives, the row ships entities only (subset
semantics of the declared-outputs filter).

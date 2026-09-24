# NVIDIA Nemotron-PII

The one general-domain ingest: NVIDIA's synthetic, persona-grounded PII/PHI corpus
(CC-BY-4.0, generated with NeMo Data Designer over personas grounded in U.S. Census data),
100,000 train + 100,000 test rows, 55+ span labels, `document_format` structured/unstructured,
`locale` us/intl, 30 domains. It earns its place next to the biomedical corpora two ways: the
PHI labels the biomedical sources never carry (`medical_record_number`,
`health_plan_beneficiary_number`, `date_of_birth`, `blood_type`) ride healthcare-domain
documents, and the person/place/employer labels widen the de-identification repertoire the
gliner2 model can name on clinical text.

Format notes: `spans` arrives as a python-repr string of span dicts (`ast.literal_eval`,
skip-don't-coerce, malformed entries drop individually). The offsets are end-exclusive and
slice-authoritative: the span dict's own `text` field drifts in case and type (int-typed for
`age`/`cvv`, lowercased surfaces like `spanish` against the real `Spanish`), so mention
surfaces come from `text[start:end]` and the field is never read.

Label policy: 14 labels map onto exact biolink classes (`first_name`/`last_name` -> `Human`,
seven location labels -> `GeographicLocation`, `company_name` -> `Agent`, `gender` ->
`BiologicalSex`, `occupation`/`education_level`/`employment_status` -> `SocioeconomicAttribute`);
the ~40-label PII tail stays raw PascalCase (`Ssn`, `Ipv4`, `MedicalRecordNumber`, ...), the
Pile-NER precedent, for zero-shot de-identification breadth. Entity-only: the gazetteer's
predicates are biomedical-mined, so emitting relations here would fabricate edges between PII
mentions on general-domain documents.

Gate note: `ResolutionGate` gained a `pii-person-name` bucket plus person/place/measure key
extensions, so a surname that fullmap-hits as a gene is rejected (only
`Human`/`IndividualOrganism` ancestors survive); rejections fall through to fallback/raw like
every other gate rejection.

## Related

- [README index](../README.md) -- the ingest-table row and the full docs index
- [ingests](ingests.md) -- the resolution chain its rows enter
- [weighting](weighting.md) -- the tier and trust prior for its row key

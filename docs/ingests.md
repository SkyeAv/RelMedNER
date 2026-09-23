# Ingests

[ingest table](../README.md#ingests)

All script tasks except the multilingual ingest (which labels directly, see its
notes under [Statement qualifiers and negation](qualifiers.md)) share one resolution chain -- fullmap first, a shared lowercased
`FALLBACK_LABEL_MAP` second (dataset vocabularies ride on top via
`resolve_mentions(label_map=...)`), raw labels last -- and two shared quality gates:

- `ResolutionGate` rejects fullmap hits contradicting the source corpus's own label
  (label<->category buckets, model-organism CURIE guard, acronym-over-catch-all guard);
  rejections fall through to fallback/raw, never dropping the mention. ~20% of fullmap
  hits rejected on the pile-ner corpus, nearly all true false positives.
- `PredicateRangeGate` rejects gazetteer relations whose head/tail biolink categories
  contradict the predicate's domain/range (raw labels impose no constraint). ~21% of
  candidate relations rejected, all sampled rejects genuinely wrong. Qualifier contexts are
  stricter: typed slots demand resolvable biolink ancestors and reject
  `JUNKY_CATEGORIES` (UMLS qualifier/indexing concepts).

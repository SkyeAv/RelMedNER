# Deduplication

Before Avro write, every merged `TrainingExample` passes through a dedup stage so
repeated text does not reach training data twice. Exact dedup keeps one record per
whitespace-normalized text (a blake2b fingerprint of the normalized form is the key);
near dedup then groups records by LSH band keys over MinHash signatures and collapses
each group to a single winner. Signatures use 128 fixed-seed permutations over
lowercased word 5-gram shingles, banded 8 x 16, so the estimated-Jaccard bar for a merge
sits near 0.88 (the S-curve 50% point at s ~= 0.878): only "same abstract re-ingested"
texts merge, half-overlapping ones never do. Texts under 10 tokens bypass near-dedup
entirely, because below that a 5-gram shingle set has too few members for a stable
Jaccard estimate, but they still get exact dedup.

Where a duplicate group collapses, the survivor is the highest-weight record, ties broken
by `content_id` (a 128-bit blake2b fingerprint of the record's canonical JSON) so the
winner is deterministic across runners, shard counts, and arrival order. Weights are not
summed; the export step owns weighted duplication.

Near dedup ships a compact element through its band shuffle: each record emits
`(band_key, priority)` per band, where `priority` is `(-weight, content_id)`, plus its
full payload exactly once on a separate stream that the join step keys by `content_id`.
A record is dropped when it carries at least one loser marker from any band bucket, so
winning one band never resurrects a record that lost another. The earlier shape attached
the whole `TrainingExample` to every band emission and again on the collapse step, 16 full
copies per record (~23 KiB measured on a 200-token record versus ~2.2 KiB now).

`build-dataset --dedup-mode` controls the stage: `near` (the default) runs exact then
near, `exact` runs exact only, and `off` adds no dedup transforms at all. Drop counts
land in Beam counters under the `relmedner.dedup` namespace; on DirectRunner runs the
pipeline logs one summary line afterwards, `dedup: exact -<n> near -<n> of <total>
records`. On the Flink cluster the same counters surface in the Flink UI / REST job
metrics.

## Related

- [README index](../README.md) -- the full docs index
- [output](output.md) -- the records the dedup stage merges before the Avro write
- [weighting](weighting.md) -- how weight decides which duplicate survives a collision

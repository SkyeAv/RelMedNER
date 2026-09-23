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

Where a duplicate group collapses, the survivor is the highest-weight record, ties
broken by canonical JSON so the winner is deterministic. Weights are not summed; the
export step owns weighted duplication.

`build-dataset --dedup-mode` controls the stage: `near` (the default) runs exact then
near, `exact` runs exact only, and `off` adds no dedup transforms at all. Drop counts
land in Beam counters under the `relmedner.dedup` namespace; on DirectRunner runs the
pipeline logs one summary line afterwards, `dedup: exact -<n> near -<n> of <total>
records`. On the Flink cluster the same counters surface in the Flink UI / REST job
metrics.

# How fullmap mining works

Per batch of documents (Beam `BatchElements`, one redb round trip per batch):

1. tokenize with gliner2's own `WhitespaceTokenSplitter` (imported by file path so torch
   never loads), strip per-token edge punctuation -- which also makes sentence-crossing
   n-grams impossible because `.` tokenizes alone and cleans to empty;
2. enumerate contiguous n-grams up to `max_ngram`, dropping all-numeric and
   all-function-word grams; add hyphen/slash folds, Greek/unicode folds, and one-way
   in-document acronym bridges (`body mass index (BMI)` resolves the expansion, keeps the
   `BMI` span);
3. one `rs.normalize_terms` + `lookup_rows` + `filter_and_rank` per batch -- fullmap keys
   are **byte-sorted bags of Porter2 stems**, so word order is already irrelevant and
   permutation-style augmentation is a proven no-op;
4. accept a row only when `normalize(PREFERRED_NAME) == term` (EXACT). The published PR
   tiers are *not* a quality dial: PR=50 just means a preferred name already in
   sorted-stem order; PR=250/500 are stopword collisions;
5. gates (constants in `relmedner/constants.py`, each fixed by measurement -- see
   `PLAN.md`): exclude model-organism CURIE prefixes (`FB`, `ZFIN`, `MGI`, ... -- human
   genes only), junk-category gate (UMLS qualifier/indexing concepts), unigram minimum
   length (digit-bearing exempt), gene/protein casing rule (rejects `in`->`NCBIGene:3630
   INS` collisions), strict unigram name agreement (case-insensitive equality or simple
   plural -- kills Porter2 derivational collisions like `oxidative`->`oxide`), and
   rejection of spans that begin or end with a function word;
6. greedy longest-match non-overlap selection, then group by biolink category with
   class-definition + `[fullmap: CURIE | name]` descriptions.

Measured expectations on the curated corpus: **~20 mentions/doc** (~8.5M projected over
418,381 docs), unigram precision ~ 78%, multi-token precision ~ 90%. Yield by gram
length per 200 docs: n=1 2,962; n=2 918; n=3 137; n=4 29; n>=5 4. Distinct candidate
surfaces grow ~54k per 100 docs with no cross-document saturation, so per-batch dedup is
the only and sufficient lever (~11.5 us/key lookup).

A future teacher-distillation pass (gliner-biomed-large agreeing with mined spans) is
deliberately deferred; `fullmap_mine.resolve_batch` is the interception point.

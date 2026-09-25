# MedQA (USMLE, 4 options)

`GBaker/MedQA-USMLE-4-options` (cc-by-4.0): 10,178 USMLE-style four-option multiple-choice
questions from Jin et al., "What Disease does this Patient Have?" (arXiv:2009.13081), gold
`answer_idx` labels, declared as one `hf_json` ingest at the gold tier.

The repo publishes jsonl on its main branch (`phrases_no_exclude_train.jsonl`,
`phrases_no_exclude_test.jsonl`) and has NO `refs/convert/parquet` revision (404 measured on
2026-09-24), so the `hf_parquet` source kind cannot read it; this ingest goes through
`hf_json`, the same route as `thunlp/docred` and `knowledgator/PubMedAbstractsNER`. Despite
the file name, every line is a full QA record with the keys `question`, `answer`, `options`,
`meta_info`, `answer_idx`, and `metamap_phrases`.

`MedQaScript` composes the example text as the question stem followed by the four options
rendered in the fixed `A`, `B`, `C`, `D` order as `A. <text>`, and emits exactly one
`Classification` per row over the fixed `['A', 'B', 'C', 'D']` vocabulary. Rendering the
options is what makes a letter label learnable, and it leaks nothing: the correct option text
sits in the prompt beside the three distractors, exactly as at exam time. The row's `answer`
column (the correct option's text) is deliberately kept out of `columns_out`, so a label leak
through the text is structurally impossible, and a fourth value reaching the script fails
loudly in the tests rather than being ignored.

Drop rules are skip-don't-coerce (`PubmedQaScript` precedent): an unmappable `answer_idx`
(None, bool, int, float, an out-of-vocabulary letter) ships a text-only example; an unusable
`options` value (not a dict, a missing letter, a value that is not a non-empty string) also
ships text-only, because without the rendered option set the letter label is unlearnable and
inventing option text would be a guess; a row whose question and options all coerce to empty
ships as `TrainingExample(text="")` for the pipeline's empty-record path; an empty question
with usable options ships the options alone, never a fabricated stem.

Measured over all 10,178 declared rows (laptop census 2026-09-24): `options` is a dict with
exactly the keys `A`, `B`, `C`, `D` and non-empty string values on 10,178/10,178 rows;
`answer_idx` is in `A|B|C|D` on 10,178/10,178 rows (A 2,584 / B 2,654 / C 2,557 / D 2,383);
no question is empty (min 66 chars, mean 724, max 3,577); `options[answer_idx] == answer` on
10,178/10,178 rows; `meta_info` is `step1` (5,629) or `step2&3` (4,549). A 500-row declared
probe on wenceslaus (2026-09-24) dispatched 500/500 rows emitting, all with the
`classifications` shape, question text median 692 chars (min 177, max 1,689):

```
uv run python .pi/skills/add-dataset/scripts/probe.py \
  --declared 'GBaker/MedQA-USMLE-4-options:phrases_no_exclude_train.jsonl' \
  --script MedQaScript --outputs classifications --text-column question --limit 500
```

The 1,273-row test jsonl stays undeclared (repo convention: test rows are not training data),
and `metamap_phrases` stays out of `columns_out` because it is a MetaMap mining artifact of
the upstream release, not a gold annotation. Declared outputs are [classifications] only; no
entities or relations are mined here.

Screened alternatives, all rejected: `bigbio/med_qa` and `openlifescienceai/medqa` carry
`license: unknown` or none; `GBaker/MedQA-USMLE-4-options-hf` is `cc-by-sa-4.0` (share-alike)
and republishes the same items as json; the `michaelw94` ACI-Bench mirror is `agpl-3.0`,
whose network copyleft clause conflicts with the commercial bar. `openlifescienceai/medmcqa`
is already declared silver over its `exp` column, and its repo-id row key may carry only one
weight (`weights_by_source`, models.py), so it cannot also host a gold MCQA entry.

## Related

- [README index](../README.md) -- the ingest-table row for this corpus and the full docs index.
- [Ingests](ingests.md) -- the resolution chain and gates these rows flow through.
- [Weighting](weighting.md) -- the tier prior this corpus is declared at.
- [PubMedQA](pubmed-qa.md) -- the other gold question-answering classification ingest.

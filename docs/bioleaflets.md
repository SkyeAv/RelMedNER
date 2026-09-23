# Bioleaflets

The one EMA-sourced ingest: `ruslan/bioleaflets-biomedical-ner` via `BioleafletsScript`,
declared as train (1,068 rows) + test (134 rows) over the six `Section_N` columns
with `outputs: [entities, relations]`.

Dataset-format notes (`BioleafletsScript`, dataset `ruslan/bioleaflets-biomedical-ner`): 1,336 EMA
package leaflets scraped from ema.europa.eu (dataset card
https://huggingface.co/datasets/ruslan/bioleaflets-biomedical-ner, paper
https://aclanthology.org/2021.inlg-1.40/), ingested as train (1,068 rows) + test (134 rows); the upstream validation split (134 rows) is
deliberately not ingested, because all three splits share one generation process and train already
covers that distribution. Each row carries six `Section_1`..`Section_6` cells that decode (`ast.literal_eval`) to
python-repr dicts with the keys Title / Section_Content / Entity_Recognition. `Section_Content` is
lower-cased, pre-tokenized text (special characters already stand alone as their own tokens), so
`content.split()` is the tokenizer and character offsets tile the text exactly the way the shared
char-to-token bridge expects. `Entity_Recognition` is an Amazon Comprehend Medical + Stanza ensemble
of character-offset entries (Text/Type/BeginOffset/EndOffset, with Comprehend entries adding
Id/Score/Category/Traits and sometimes Attributes); 202 of 2,808 probed sections carry a `None`
value and exit as text-only examples the permitted-shapes contract drops. Sections are independent
documents -- spans and relations extract per section, never over one merged token stream -- so the
gazetteer cannot fabricate pairs across a section boundary. The seed `LABEL_MAP` covers the measured
medical heads (dx_name/problem -> Disease, generic_name/brand_name -> Drug, procedure_name ->
Procedure, test_name/test -> ClinicalMeasurement, treatment_name/treatment -> Treatment,
system_organ_site -> AnatomicalEntity); PHI/noise types (AGE, ADDRESS, DATE, ID, NAME, PHONE_OR_FAX,
PROFESSION, NUMBER, PRODUCT_NAME, TIME_TO_*) stay PascalCased raw. One measured `ResolutionGate`
change serves this corpus: the label/category buckets gained a dedicated drugname bucket (generic
and brand labels demand chemical compatibility before a fullmap hit is accepted) and dx/problem
keywords joined the disease bucket.


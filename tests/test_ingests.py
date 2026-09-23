from __future__ import annotations

import pathlib
import re
from collections import Counter

import pytest

from relmedner.ingests import YamlIngestsParser
from relmedner.models import (
    Cluster,
    FullmapTask,
    GazetteerPredicate,
    GazetteerQualifier,
    GazetteerSpec,
    HuggingFaceDataset,
    HuggingFaceJsonDataset,
    HuggingFaceParquetDataset,
    LocalAvroDataset,
    LocalDelimitedDataset,
    MatchOn,
    ScriptTask,
    WorkerNode,
    YamlIngests,
)

# per-ingest locks: each declared ingest's entry is asserted independently so adding a dataset is an
# additive block here rather than a rewritten literal (and a guaranteed merge conflict) across the
# parallel dataset worktrees
# the single health-community allowlist match_on lock, shared by all seven tensorshield reddit
# entries (values match the corpus EXACTLY: apply_match is exact membership, so casing must be
# the Reddit canonical form or the row is silently dropped)
EXPECTED_REDDIT_MATCH: tuple[object, ...] = (
    (
        "communityName",
        (
            "r/AskDocs",
            "r/medical",
            "r/medicine",
            "r/Health",
            "r/diabetes",
            "r/ADHD",
            "r/autism",
            "r/Anxiety",
            "r/depression",
            "r/SuicideWatch",
            "r/cancer",
            "r/Celiac",
            "r/ibs",
            "r/IBD",
            "r/CrohnsDisease",
            "r/UlcerativeColitis",
            "r/eczema",
            "r/Psoriasis",
            "r/acne",
            "r/migraine",
            "r/ChronicPain",
            "r/Menopause",
            "r/endometriosis",
            "r/PCOS",
            "r/infertility",
            "r/birthcontrol",
            "r/lupus",
            "r/MultipleSclerosis",
            "r/hypothyroidism",
            "r/Hashimotos",
            "r/asthma",
            "r/COPD",
            "r/epilepsy",
            "r/schizophrenia",
            "r/bipolar",
            "r/BPD",
            "r/OCD",
            "r/ptsd",
            "r/EatingDisorders",
            "r/Dentistry",
            "r/pharmacy",
            "r/nursing",
        ),
    ),
)

EXPECTED: dict[str, tuple[object, ...]] = {
    "anthonyyazdaniml/gliner-biomed-pre-training": (
        "hf",
        (
            ("script", "GlinerBiomedScript", ("entities", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-pre-training",
            None,
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    ),
    "anthonyyazdaniml/gliner-biomed-curated-corpus": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-curated-corpus",
            None,
            "train",
            None,
            ("text",),
        ),
    ),
    "anthonyyazdaniml/gliner-biomed-balanced-curated-corpus": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-balanced-curated-corpus",
            None,
            "train",
            None,
            ("text",),
        ),
    ),
    "anthonyyazdaniml/gliner-biomed-post-training": (
        "hf",
        (
            ("script", "GlinerBiomedPostScript", ("entities", "classifications", "structures", "relations")),
            1.0,
            "anthonyyazdaniml/gliner-biomed-post-training",
            None,
            "train",
            None,
            ("tokenized_text", "ner", "negatives"),
        ),
    ),
    "disi-unibo-nlp/Pile-NER-biomed-IOB": (
        "hf",
        (
            ("script", "PileNerBiomedScript", ("entities",)),
            1.0,
            "disi-unibo-nlp/Pile-NER-biomed-IOB",
            None,
            "train",
            None,
            ("tokens", "ner_tags"),
        ),
    ),
    "Universal-NER/Pile-NER-type": (
        "hf",
        (
            ("script", "PileNerTypeScript", ("entities",)),
            1.0,
            "Universal-NER/Pile-NER-type",
            None,
            "train",
            None,
            ("conversations",),
        ),
    ),
    "TrialPanorama/TrialPanorama-database:studies": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "TrialPanorama/TrialPanorama-database",
            "studies",
            "all",
            None,
            ("abstract",),
        ),
    ),
    "aps/super_glue:multirc": (
        "hf",
        (
            ("script", "SuperGlueMultiRCScript", ("classifications",)),
            1.0,
            "aps/super_glue",
            "multirc",
            "train",
            None,
            ("paragraph", "question", "answer", "label"),
        ),
    ),
    # the one local-source entry: no subset/split/match_on/columns_out, just the avro path, so its
    # tuple is deliberately shorter than the hf ones above (it still carries the weight field, which
    # DatasetBase declares for every source)
    "interventions/interventions.avro": (
        "local",
        (
            ("script", "CtkpInterventionsScript", ("entities",)),
            1.0,
            "interventions/interventions.avro",
        ),
    ),
    # bc5cdr (US-001): one local avro container per split, each with its own row_key slot
    "bc5cdr/train.avro": (
        "local",
        (
            ("script", "Bc5CdrScript", ("entities", "relations")),
            1.0,
            "bc5cdr/train.avro",
        ),
    ),
    "bc5cdr/dev.avro": (
        "local",
        (
            ("script", "Bc5CdrScript", ("entities", "relations")),
            1.0,
            "bc5cdr/dev.avro",
        ),
    ),
    "bc5cdr/test.avro": (
        "local",
        (
            ("script", "Bc5CdrScript", ("entities", "relations")),
            1.0,
            "bc5cdr/test.avro",
        ),
    ),
    "knowledgator/sentence_rex": (
        "hf",
        (
            ("script", "SentenceRexScript", ("relations",)),
            1.0,
            "knowledgator/sentence_rex",
            None,
            "train",
            None,
            ("sentences", "labels"),
        ),
    ),
    "knowledgator/biomed_NER": (
        "hf",
        (
            ("script", "KnowledgatorBiomedScript", ("entities",)),
            1.0,
            "knowledgator/biomed_NER",
            None,
            "train",
            None,
            ("text", "entities"),
        ),
    ),
    "knowledgator/gliner-multilingual-synthetic": (
        "hf",
        (
            ("script", "GlinerMultilingualScript", ("entities",)),
            1.0,
            "knowledgator/gliner-multilingual-synthetic",
            None,
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    ),
    "knowledgator/PubMedAbstractsNER:train.json": (
        "hf_json",
        (
            ("script", "PubmedAbstractsScript", ("entities", "relations")),
            1.0,
            "knowledgator/PubMedAbstractsNER",
            "train.json",
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    ),
    "thunlp/docred:data/train_annotated.json.gz": (
        "hf_json",
        (
            ("script", "DocredScript", ("entities", "relations")),
            1.0,
            "thunlp/docred",
            "data/train_annotated.json.gz",
            "train",
            None,
            ("sents", "vertexSet", "labels"),
        ),
    ),
    "thunlp/docred:data/train_distant.json.gz": (
        "hf_json",
        (
            ("script", "DocredScript", ("entities", "relations")),
            1.0,
            "thunlp/docred",
            "data/train_distant.json.gz",
            "train",
            None,
            ("sents", "vertexSet", "labels"),
        ),
    ),
    "thunlp/docred:data/dev.json.gz": (
        "hf_json",
        (
            ("script", "DocredScript", ("entities", "relations")),
            1.0,
            "thunlp/docred",
            "data/dev.json.gz",
            "train",
            None,
            ("sents", "vertexSet", "labels"),
        ),
    ),
    "thunlp/docred:data/test.json.gz": (
        "hf_json",
        (
            ("script", "DocredScript", ("entities",)),
            1.0,
            "thunlp/docred",
            "data/test.json.gz",
            "train",
            None,
            ("sents", "vertexSet", "labels"),
        ),
    ),
    "aps/super_glue:record": (
        "hf",
        (
            ("script", "SuperGlueRecordScript", ("entities", "classifications")),
            1.0,
            "aps/super_glue",
            "record",
            "train",
            None,
            ("passage", "query", "entities", "entity_spans", "answers"),
        ),
    ),
    "qualifiers/qualifier_corpus.tsv": (
        "local_delimited",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "qualifiers/qualifier_corpus.tsv",
            ("text",),
            None,
        ),
    ),
    # local avro script ingest: same short-tuple shape as interventions above (no subset/split,
    # no columns_out; DatasetBase still carries the weight)
    "synthetic-ner-ade-tweets/ade_tweets.avro": (
        "local",
        (("script", "SyntheticNerAdeTweetsScript", ("entities",)), 1.0, "synthetic-ner-ade-tweets/ade_tweets.avro"),
    ),
    # local delimited fullmap ingest: same shape as qualifiers above (columns_out joins the tuple,
    # match_on stays None)
    "synthetic-ner-ade-tweets/ade_tweets_unannotated.tsv": (
        "local_delimited",
        (("fullmap", 6, "9606", True, ("entities", "relations")), 1.0, "synthetic-ner-ade-tweets/ade_tweets_unannotated.tsv", ("text",), None),
    ),
    # nvidia/Nemotron-PII declares one ingest PER SPLIT off one repo id: entry_key qualifies on the
    # split when no subset is declared, so the two locks coexist instead of silently overwriting
    "nvidia/Nemotron-PII:train": (
        "hf",
        (
            ("script", "NemotronPiiScript", ("entities",)),
            1.0,
            "nvidia/Nemotron-PII",
            None,
            "train",
            None,
            ("text", "spans"),
        ),
    ),
    "nvidia/Nemotron-PII:test": (
        "hf",
        (
            ("script", "NemotronPiiScript", ("entities",)),
            1.0,
            "nvidia/Nemotron-PII",
            None,
            "test",
            None,
            ("text", "spans"),
        ),
    ),
    # wcole3/biored-parquet declares one ingest PER SPLIT off one repo id with no subset (the
    # config-less load streams the default biored_bigbio_kb config), so entry_key split-qualifies
    # the three locks exactly like nvidia/Nemotron-PII above
    "wcole3/biored-parquet:train": (
        "hf",
        (
            ("script", "BioredScript", ("entities", "relations")),
            1.0,
            "wcole3/biored-parquet",
            None,
            "train",
            None,
            ("passages", "entities", "relations"),
        ),
    ),
    "wcole3/biored-parquet:validation": (
        "hf",
        (
            ("script", "BioredScript", ("entities", "relations")),
            1.0,
            "wcole3/biored-parquet",
            None,
            "validation",
            None,
            ("passages", "entities", "relations"),
        ),
    ),
    "wcole3/biored-parquet:test": (
        "hf",
        (
            ("script", "BioredScript", ("entities", "relations")),
            1.0,
            "wcole3/biored-parquet",
            None,
            "test",
            None,
            ("passages", "entities", "relations"),
        ),
    ),
    # the seven tensorshield reddit ingests share one allowlist via EXPECTED_REDDIT_MATCH: the
    # parsed match_on tuple must equal the declaration, so a yaml-side fork fails here
    "tensorshield/reddit_dataset_157": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_157",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_171": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_171",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_217": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_217",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_237": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_237",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_30": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_30",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_84": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_84",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    "tensorshield/reddit_dataset_85": (
        "hf",
        (
            ("fullmap", 6, "9606", True, ("entities", "relations")),
            1.0,
            "tensorshield/reddit_dataset_85",
            None,
            "train",
            (("communityName", EXPECTED_REDDIT_MATCH[0][1]),),
            ("text",),
        ),
    ),
    # ruslan/bioleaflets-biomedical-ner: one repo id, two ingests distinguished ONLY by split (no
    # subset), so entry_key qualifies BOTH keys with the split (same convention as nvidia/Nemotron-PII)
    "ruslan/bioleaflets-biomedical-ner:train": (
        "hf",
        (
            ("script", "BioleafletsScript", ("entities", "relations")),
            1.0,
            "ruslan/bioleaflets-biomedical-ner",
            None,
            "train",
            None,
            ("Section_1", "Section_2", "Section_3", "Section_4", "Section_5", "Section_6"),
        ),
    ),
    "ruslan/bioleaflets-biomedical-ner:test": (
        "hf",
        (
            ("script", "BioleafletsScript", ("entities", "relations")),
            1.0,
            "ruslan/bioleaflets-biomedical-ner",
            None,
            "test",
            None,
            ("Section_1", "Section_2", "Section_3", "Section_4", "Section_5", "Section_6"),
        ),
    ),
    # agentlans/json-extraction: one lock per source config; the six entries share the repo-id row
    # key, so entry_key qualifies on the subset (the hub config is the real filter)
    "agentlans/json-extraction:owkin-medical_knowledge_from_extracts": (
        "hf",
        (
            ("script", "JsonExtractionScript", ("structures", "entities")),
            1.0,
            "agentlans/json-extraction",
            "owkin-medical_knowledge_from_extracts",
            "train",
            None,
            ("text", "json", "source"),
        ),
    ),
    "agentlans/json-extraction:ProfessorBob-relation_extraction": (
        "hf",
        (
            ("script", "JsonExtractionScript", ("structures", "relations")),
            1.0,
            "agentlans/json-extraction",
            "ProfessorBob-relation_extraction",
            "train",
            None,
            ("text", "json", "source"),
        ),
    ),
    "agentlans/json-extraction:roborovski-dolly-entity-extraction": (
        "hf",
        (
            ("script", "JsonExtractionScript", ("structures", "entities")),
            1.0,
            "agentlans/json-extraction",
            "roborovski-dolly-entity-extraction",
            "train",
            None,
            ("text", "json", "source"),
        ),
    ),
    "agentlans/json-extraction:sandeeppanem-resume-json-extraction-5k": (
        "hf",
        (
            ("script", "JsonExtractionScript", ("structures", "entities")),
            1.0,
            "agentlans/json-extraction",
            "sandeeppanem-resume-json-extraction-5k",
            "train",
            None,
            ("text", "json", "source"),
        ),
    ),
    "agentlans/json-extraction:Jiraya-html_to_json_information_extraction_dataset": (
        "hf",
        (
            ("script", "JsonExtractionScript", ("structures", "entities")),
            1.0,
            "agentlans/json-extraction",
            "Jiraya-html_to_json_information_extraction_dataset",
            "train",
            None,
            ("text", "json", "source"),
        ),
    ),
    "agentlans/json-extraction:HenriqueGodoy-extract-0": (
        "hf",
        (
            ("script", "JsonExtractionScript", ("structures",)),
            1.0,
            "agentlans/json-extraction",
            "HenriqueGodoy-extract-0",
            "train",
            None,
            ("text", "json", "source"),
        ),
    ),
    "Pennlaine/Medical-Entity-JSON-Extraction": (
        "hf",
        (("script", "MedicalEntityJsonScript", ("entities",)), 1.0, "Pennlaine/Medical-Entity-JSON-Extraction", None, "test", None, ("text",)),
    ),
    "bigbio/chemprot:chemprot_full_source:train": (
        "hf",
        (
            ("script", "ChemprotScript", ("entities", "relations")),
            1.0,
            "bigbio/chemprot",
            "chemprot_full_source",
            "train",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chemprot:chemprot_full_source:validation": (
        "hf",
        (
            ("script", "ChemprotScript", ("entities", "relations")),
            1.0,
            "bigbio/chemprot",
            "chemprot_full_source",
            "validation",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chemprot:chemprot_full_source:test": (
        "hf",
        (
            ("script", "ChemprotScript", ("entities", "relations")),
            1.0,
            "bigbio/chemprot",
            "chemprot_full_source",
            "test",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    # the four bigbio/ehr_rel entries share one repo-id row key and one weight (0.5); the file is
    # the scalar discriminator, exactly like the hf_json train.json key above
    "bigbio/ehr_rel:ehr_rel_a_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "EhrRelScript", ("relations",)),
            0.5,
            "bigbio/ehr_rel",
            "ehr_rel_a_source/train/0000.parquet",
            "train",
            None,
            ("snomed_label_1", "snomed_label_2", "mean_rating"),
        ),
    ),
    "bigbio/ehr_rel:ehr_rel_b_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "EhrRelScript", ("relations",)),
            0.5,
            "bigbio/ehr_rel",
            "ehr_rel_b_source/train/0000.parquet",
            "train",
            None,
            ("snomed_label_1", "snomed_label_2", "mean_rating"),
        ),
    ),
    "bigbio/ehr_rel:ehr_rel_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "EhrRelScript", ("relations",)),
            0.5,
            "bigbio/ehr_rel",
            "ehr_rel_source/train/0000.parquet",
            "train",
            None,
            ("snomed_label_1", "snomed_label_2", "mean_rating"),
        ),
    ),
    "bigbio/ehr_rel:ehr_rel_bigbio_pairs/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "EhrRelScript", ("relations",)),
            0.5,
            "bigbio/ehr_rel",
            "ehr_rel_bigbio_pairs/train/0000.parquet",
            "train",
            None,
            ("text_1", "text_2", "label"),
        ),
    ),
    # the five bigbio/chia subset entries frozen from probe.py --freeze on wenceslaus (2026-09-23);
    # the discriminator is the declared file, so the key carries the subset-relative parquet path
    "bigbio/chia:chia_bigbio_kb/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "ChiaScript", ("entities", "relations")),
            1.0,
            "bigbio/chia",
            "chia_bigbio_kb/train/0000.parquet",
            "train",
            None,
            ("passages", "entities", "relations"),
        ),
    ),
    "bigbio/chia:chia_fixed_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "ChiaScript", ("entities", "relations")),
            1.0,
            "bigbio/chia",
            "chia_fixed_source/train/0000.parquet",
            "train",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chia:chia_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "ChiaScript", ("entities", "relations")),
            1.0,
            "bigbio/chia",
            "chia_source/train/0000.parquet",
            "train",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chia:chia_without_scope_fixed_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "ChiaScript", ("entities", "relations")),
            1.0,
            "bigbio/chia",
            "chia_without_scope_fixed_source/train/0000.parquet",
            "train",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "bigbio/chia:chia_without_scope_source/train/0000.parquet": (
        "hf_parquet",
        (
            ("script", "ChiaScript", ("entities", "relations")),
            1.0,
            "bigbio/chia",
            "chia_without_scope_source/train/0000.parquet",
            "train",
            None,
            ("text", "entities", "relations"),
        ),
    ),
    "fewrel/train_wiki.avro": ("local", (("script", "FewRelScript", ("entities", "relations")), 1.0, "fewrel/train_wiki.avro")),
    "fewrel/val_wiki.avro": ("local", (("script", "FewRelScript", ("entities", "relations")), 1.0, "fewrel/val_wiki.avro")),
    "fewrel/val_nyt.avro": ("local", (("script", "FewRelScript", ("entities", "relations")), 1.0, "fewrel/val_nyt.avro")),
    "fewrel/val_semeval.avro": ("local", (("script", "FewRelScript", ("entities", "relations")), 1.0, "fewrel/val_semeval.avro")),
    "fewrel/val_pubmed.avro": ("local", (("script", "FewRelScript", ("entities", "relations")), 1.0, "fewrel/val_pubmed.avro")),
    "fewrel/pubmed_unsupervised.avro": ("local", (("script", "FewRelScript", ("entities", "relations")), 1.0, "fewrel/pubmed_unsupervised.avro")),
}


def entry_base_key(payload: tuple[object, ...]) -> str:
    """repo id plus a scalar discriminator (subset for "hf", file for "hf_json"): two ingests
    reading different subsets of aps/super_glue already collide at this base, and a bare repo id
    would silently lose one lock in the EXPECTED literal"""
    dataset = str(payload[2])
    discriminator = payload[3] if len(payload) > 3 else None
    # only a SCALAR discriminator qualifies the key: for the local sources payload position 3 is
    # columns_out (a tuple), and there the declared path is already unique per file
    return f"{dataset}:{discriminator}" if isinstance(discriminator, str) else dataset


def entry_key(payload: tuple[object, ...], colliding_bases: set[str]) -> str:
    """one key per DECLARED INGEST, not per repo: a base key declared on more than one ingest
    (nvidia/Nemotron-PII once PER SPLIT, bigbio/chemprot sharing dataset AND subset) gets the
    split appended, so the locks coexist instead of one silently overwriting the other in
    tuples_by_ingest -- the loser would vanish from both locks and the failure would be silent.
    Kept identical to .pi/skills/add-dataset/scripts/probe.py (report_freeze, iter_declared),
    which freezes and matches on these keys"""
    base = entry_base_key(payload)
    if base not in colliding_bases:
        return base
    split = payload[4] if len(payload) > 4 else None
    if not isinstance(split, str):
        raise ValueError(f"entry base {base!r} is declared on more than one ingest but its split {split!r} is not a str")
    return f"{base}:{split}"


def tuples_by_ingest() -> dict[str, tuple[object, ...]]:
    # the repo id moved to payload position 2 when the weight field joined DatasetBase; a base
    # key appearing on more than one declared ingest gets split-qualified keys (see entry_key)
    entries: list[tuple[str, tuple[object, ...]]] = list(YamlIngestsParser().generate_tuples())
    base_counts: Counter[str] = Counter(entry_base_key(payload) for _, payload in entries)
    colliding: set[str] = {base for base, count in base_counts.items() if count > 1}
    keyed: dict[str, tuple[object, ...]] = {}
    for source, payload in entries:
        key = entry_key(payload, colliding)
        if key in keyed:
            raise ValueError(f"entry key {key!r} is produced by more than one declared ingest")
        keyed[key] = (source, payload)
    return keyed


@pytest.mark.parametrize("ingest", sorted(EXPECTED))
def test_the_declared_tuple_shape_is_locked_per_ingest(ingest: str) -> None:
    assert tuples_by_ingest()[ingest] == EXPECTED[ingest]


def test_every_declared_ingest_is_accounted_for() -> None:
    assert set(tuples_by_ingest()) == set(EXPECTED)


# doc drift guard: docs/yaml-config.md is the agent-facing schema reference, and it rots silently
# when a model field is added or renamed; parametrizing over the live model_fields (not a copied
# list) means the guard itself cannot go stale. Every name below must appear in the doc.
# Every concrete dataset source is listed: main split the single LocalDataset into the avro,
# delimited, and hf_json kinds, and a doc naming only one of them is exactly the silent rot
# this guard exists to catch.
_DOC_MODELS = (
    YamlIngests,
    ScriptTask,
    FullmapTask,
    HuggingFaceDataset,
    HuggingFaceJsonDataset,
    HuggingFaceParquetDataset,
    LocalAvroDataset,
    LocalDelimitedDataset,
    MatchOn,
    Cluster,
    WorkerNode,
    GazetteerPredicate,
    GazetteerQualifier,
    GazetteerSpec,
)

_DOC_FIELD_NAMES: tuple[str, ...] = tuple(sorted({name for model in _DOC_MODELS for name in model.model_fields}))


@pytest.mark.parametrize("field_name", _DOC_FIELD_NAMES)
def test_every_yaml_model_field_is_named_in_docs_yaml_config(field_name: str) -> None:
    text = pathlib.Path("docs/yaml-config.md").read_text(encoding="utf-8")
    # backtick-anchored match, not a bare substring: prose words like "name" or "range" must
    # not satisfy the guard -- only a code span (table cell or inline) naming the field counts
    assert re.search(rf"`{re.escape(field_name)}`", text), f"field {field_name!r} is missing from docs/yaml-config.md"

from __future__ import annotations

import re
from typing import Any

import pytest

from relmedner.families import validate_label_map
from relmedner.models import TrainingExample
from relmedner.scripts import BioredScript  # registry must stay populated alongside the new script
from relmedner.scripts.biored import _validate_predicate_map
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: BioredScript = BioredScript()


def run(values: tuple[Any, Any, Any]) -> TrainingExample:
    """the script consumes exactly the declared columns_out projection (passages, entities,
    relations); every test calls the same entry point the registry dispatch would"""
    return SCRIPT.run(values)


# verbatim wcole3/biored-parquet train row captured from the stream (document_id 10491763;
# 32 entities, 107 relations). Every negative test below mutates THIS row or a real-shaped trim
# of it, so no test trains on an invented row shape.
ROW_10491763 = {
    "document_id": "10491763",
    "passages": [
        {
            "id": "0_title",
            "type": "title",
            "text": [
                "Hepatocyte nuclear factor-6: associations between genetic variability "
                "and type II diabetes and between genetic variability and estimates of "
                "insulin secretion."
            ],
            "offsets": [[0, 158]],
        },
        {
            "id": "0_abstract",
            "type": "abstract",
            "text": [
                "The transcription factor hepatocyte nuclear factor (HNF)-6 is an "
                "upstream regulator of several genes involved in the pathogenesis of "
                "maturity-onset diabetes of the young. We therefore tested the "
                "hypothesis that variability in the HNF-6 gene is associated with "
                "subsets of Type II (non-insulin-dependent) diabetes mellitus and "
                "estimates of insulin secretion in glucose tolerant subjects.   We "
                "cloned the coding region as well as the intron-exon boundaries of the "
                "HNF-6 gene. We then examined them on genomic DNA in six MODY probands "
                "without mutations in the MODY1, MODY3 and MODY4 genes and in 54 "
                "patients with late-onset Type II diabetes by combined single strand "
                "conformational polymorphism-heteroduplex analysis followed by direct "
                "sequencing of identified variants. An identified missense variant was "
                "examined in association studies and genotype-phenotype studies.   We "
                "identified two silent and one missense (Pro75 Ala) variant. In an "
                "association study the allelic frequency of the Pro75Ala polymorphism "
                "was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type "
                "II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched "
                "glucose tolerant control subjects. Moreover, in studies of 238 "
                "middle-aged glucose tolerant subjects, of 226 glucose tolerant "
                "offspring of Type II diabetic patients and of 367 young healthy "
                "subjects, the carriers of the polymorphism did not differ from "
                "non-carriers in glucose induced serum insulin or C-peptide responses.   "
                "Mutations in the coding region of the HNF-6 gene are not associated "
                "with Type II diabetes or with changes in insulin responses to glucose "
                "among the Caucasians examined."
            ],
            "offsets": [[159, 1797]],
        },
    ],
    "entities": [
        {
            "id": "10491763_0",
            "type": "GeneOrGeneProduct",
            "text": ["Hepatocyte nuclear factor-6"],
            "offsets": [[0, 27]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        },
        {
            "id": "10491763_1",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["type II diabetes"],
            "offsets": [[74, 90]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_2",
            "type": "GeneOrGeneProduct",
            "text": ["insulin"],
            "offsets": [[140, 147]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3630"}],
        },
        {
            "id": "10491763_3",
            "type": "GeneOrGeneProduct",
            "text": ["hepatocyte nuclear factor (HNF)-6"],
            "offsets": [[184, 217]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        },
        {
            "id": "10491763_4",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["maturity-onset diabetes"],
            "offsets": [[292, 315]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_5",
            "type": "GeneOrGeneProduct",
            "text": ["HNF-6"],
            "offsets": [[389, 394]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        },
        {
            "id": "10491763_6",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["Type II (non-insulin-dependent) diabetes mellitus"],
            "offsets": [[430, 479]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_7",
            "type": "GeneOrGeneProduct",
            "text": ["insulin"],
            "offsets": [[497, 504]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3630"}],
        },
        {
            "id": "10491763_8",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[518, 525]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
        {
            "id": "10491763_9",
            "type": "GeneOrGeneProduct",
            "text": ["HNF-6"],
            "offsets": [[620, 625]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        },
        {
            "id": "10491763_10",
            "type": "GeneOrGeneProduct",
            "text": ["MODY"],
            "offsets": [[676, 680]],
            "normalized": [
                {"db_name": "NCBIGene", "db_id": "3172"},
                {"db_name": "NCBIGene", "db_id": "3651"},
                {"db_name": "NCBIGene", "db_id": "6927"},
            ],
        },
        {
            "id": "10491763_11",
            "type": "GeneOrGeneProduct",
            "text": ["MODY1"],
            "offsets": [[715, 720]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3172"}],
        },
        {
            "id": "10491763_12",
            "type": "GeneOrGeneProduct",
            "text": ["MODY3"],
            "offsets": [[722, 727]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "6927"}],
        },
        {
            "id": "10491763_13",
            "type": "GeneOrGeneProduct",
            "text": ["MODY4"],
            "offsets": [[732, 737]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3651"}],
        },
        {
            "id": "10491763_14",
            "type": "OrganismTaxon",
            "text": ["patients"],
            "offsets": [[754, 762]],
            "normalized": [{"db_name": "NCBITaxon", "db_id": "9606"}],
        },
        {
            "id": "10491763_15",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["Type II diabetes"],
            "offsets": [[779, 795]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_16",
            "type": "SequenceVariant",
            "text": ["Pro75 Ala"],
            "offsets": [[1070, 1079]],
            "normalized": [{"db_name": "dbSNP", "db_id": "rs74805019"}],
        },
        {
            "id": "10491763_17",
            "type": "SequenceVariant",
            "text": ["Pro75Ala"],
            "offsets": [[1143, 1151]],
            "normalized": [{"db_name": "dbSNP", "db_id": "rs74805019"}],
        },
        {
            "id": "10491763_18",
            "type": "OrganismTaxon",
            "text": ["patients"],
            "offsets": [[1216, 1224]],
            "normalized": [{"db_name": "NCBITaxon", "db_id": "9606"}],
        },
        {
            "id": "10491763_19",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["Type II diabetes mellitus"],
            "offsets": [[1230, 1255]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_20",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[1304, 1311]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
        {
            "id": "10491763_21",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[1379, 1386]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
        {
            "id": "10491763_22",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[1413, 1420]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
        {
            "id": "10491763_23",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["Type II diabetic"],
            "offsets": [[1443, 1459]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_24",
            "type": "OrganismTaxon",
            "text": ["patients"],
            "offsets": [[1460, 1468]],
            "normalized": [{"db_name": "NCBITaxon", "db_id": "9606"}],
        },
        {
            "id": "10491763_25",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[1573, 1580]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
        {
            "id": "10491763_26",
            "type": "GeneOrGeneProduct",
            "text": ["insulin"],
            "offsets": [[1595, 1602]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3630"}],
        },
        {
            "id": "10491763_27",
            "type": "GeneOrGeneProduct",
            "text": ["C-peptide"],
            "offsets": [[1606, 1615]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3630"}],
        },
        {
            "id": "10491763_28",
            "type": "GeneOrGeneProduct",
            "text": ["HNF-6"],
            "offsets": [[1667, 1672]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        },
        {
            "id": "10491763_29",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["Type II diabetes"],
            "offsets": [[1702, 1718]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_30",
            "type": "GeneOrGeneProduct",
            "text": ["insulin"],
            "offsets": [[1738, 1745]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3630"}],
        },
        {
            "id": "10491763_31",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[1759, 1766]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
    ],
    "relations": [
        {"id": "0_relation_0", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_1", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_2", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_3", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_4", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_5", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_6", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_7", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_8", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_9", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_10", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_11", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_12", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_13", "type": "Association", "arg1_id": "10491763_3", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_14", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_15", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_16", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_17", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_18", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_19", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_20", "type": "Association", "arg1_id": "10491763_5", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_21", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_22", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_23", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_24", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_25", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_26", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_27", "type": "Association", "arg1_id": "10491763_9", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_28", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_29", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_30", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_31", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_32", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_33", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_34", "type": "Association", "arg1_id": "10491763_28", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_35", "type": "Positive_Correlation", "arg1_id": "10491763_8", "arg2_id": "10491763_2", "normalized": []},
        {"id": "0_relation_36", "type": "Positive_Correlation", "arg1_id": "10491763_8", "arg2_id": "10491763_7", "normalized": []},
        {"id": "0_relation_37", "type": "Positive_Correlation", "arg1_id": "10491763_8", "arg2_id": "10491763_26", "normalized": []},
        {"id": "0_relation_38", "type": "Positive_Correlation", "arg1_id": "10491763_8", "arg2_id": "10491763_27", "normalized": []},
        {"id": "0_relation_39", "type": "Positive_Correlation", "arg1_id": "10491763_8", "arg2_id": "10491763_30", "normalized": []},
        {"id": "0_relation_40", "type": "Positive_Correlation", "arg1_id": "10491763_20", "arg2_id": "10491763_2", "normalized": []},
        {"id": "0_relation_41", "type": "Positive_Correlation", "arg1_id": "10491763_20", "arg2_id": "10491763_7", "normalized": []},
        {"id": "0_relation_42", "type": "Positive_Correlation", "arg1_id": "10491763_20", "arg2_id": "10491763_26", "normalized": []},
        {"id": "0_relation_43", "type": "Positive_Correlation", "arg1_id": "10491763_20", "arg2_id": "10491763_27", "normalized": []},
        {"id": "0_relation_44", "type": "Positive_Correlation", "arg1_id": "10491763_20", "arg2_id": "10491763_30", "normalized": []},
        {"id": "0_relation_45", "type": "Positive_Correlation", "arg1_id": "10491763_21", "arg2_id": "10491763_2", "normalized": []},
        {"id": "0_relation_46", "type": "Positive_Correlation", "arg1_id": "10491763_21", "arg2_id": "10491763_7", "normalized": []},
        {"id": "0_relation_47", "type": "Positive_Correlation", "arg1_id": "10491763_21", "arg2_id": "10491763_26", "normalized": []},
        {"id": "0_relation_48", "type": "Positive_Correlation", "arg1_id": "10491763_21", "arg2_id": "10491763_27", "normalized": []},
        {"id": "0_relation_49", "type": "Positive_Correlation", "arg1_id": "10491763_21", "arg2_id": "10491763_30", "normalized": []},
        {"id": "0_relation_50", "type": "Positive_Correlation", "arg1_id": "10491763_22", "arg2_id": "10491763_2", "normalized": []},
        {"id": "0_relation_51", "type": "Positive_Correlation", "arg1_id": "10491763_22", "arg2_id": "10491763_7", "normalized": []},
        {"id": "0_relation_52", "type": "Positive_Correlation", "arg1_id": "10491763_22", "arg2_id": "10491763_26", "normalized": []},
        {"id": "0_relation_53", "type": "Positive_Correlation", "arg1_id": "10491763_22", "arg2_id": "10491763_27", "normalized": []},
        {"id": "0_relation_54", "type": "Positive_Correlation", "arg1_id": "10491763_22", "arg2_id": "10491763_30", "normalized": []},
        {"id": "0_relation_55", "type": "Positive_Correlation", "arg1_id": "10491763_25", "arg2_id": "10491763_2", "normalized": []},
        {"id": "0_relation_56", "type": "Positive_Correlation", "arg1_id": "10491763_25", "arg2_id": "10491763_7", "normalized": []},
        {"id": "0_relation_57", "type": "Positive_Correlation", "arg1_id": "10491763_25", "arg2_id": "10491763_26", "normalized": []},
        {"id": "0_relation_58", "type": "Positive_Correlation", "arg1_id": "10491763_25", "arg2_id": "10491763_27", "normalized": []},
        {"id": "0_relation_59", "type": "Positive_Correlation", "arg1_id": "10491763_25", "arg2_id": "10491763_30", "normalized": []},
        {"id": "0_relation_60", "type": "Positive_Correlation", "arg1_id": "10491763_31", "arg2_id": "10491763_2", "normalized": []},
        {"id": "0_relation_61", "type": "Positive_Correlation", "arg1_id": "10491763_31", "arg2_id": "10491763_7", "normalized": []},
        {"id": "0_relation_62", "type": "Positive_Correlation", "arg1_id": "10491763_31", "arg2_id": "10491763_26", "normalized": []},
        {"id": "0_relation_63", "type": "Positive_Correlation", "arg1_id": "10491763_31", "arg2_id": "10491763_27", "normalized": []},
        {"id": "0_relation_64", "type": "Positive_Correlation", "arg1_id": "10491763_31", "arg2_id": "10491763_30", "normalized": []},
        {"id": "0_relation_65", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_66", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_67", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_68", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_69", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_70", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_71", "type": "Association", "arg1_id": "10491763_8", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_72", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_73", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_74", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_75", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_76", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_77", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_78", "type": "Association", "arg1_id": "10491763_20", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_79", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_80", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_81", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_82", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_83", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_84", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_85", "type": "Association", "arg1_id": "10491763_21", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_86", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_87", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_88", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_89", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_90", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_91", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_92", "type": "Association", "arg1_id": "10491763_22", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_93", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_94", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_95", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_96", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_97", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_98", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_99", "type": "Association", "arg1_id": "10491763_25", "arg2_id": "10491763_29", "normalized": []},
        {"id": "0_relation_100", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_101", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_4", "normalized": []},
        {"id": "0_relation_102", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_6", "normalized": []},
        {"id": "0_relation_103", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_15", "normalized": []},
        {"id": "0_relation_104", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_19", "normalized": []},
        {"id": "0_relation_105", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_23", "normalized": []},
        {"id": "0_relation_106", "type": "Association", "arg1_id": "10491763_31", "arg2_id": "10491763_29", "normalized": []},
    ],
}

MINI_ROW = {
    "document_id": "10491763-mini",
    "passages": [
        {
            "id": "0_title",
            "type": "title",
            "text": [
                "Hepatocyte nuclear factor-6: associations between genetic variability "
                "and type II diabetes and between genetic variability and estimates of "
                "insulin secretion."
            ],
            "offsets": [[0, 158]],
        },
        {
            "id": "0_abstract",
            "type": "abstract",
            "text": [
                "The transcription factor hepatocyte nuclear factor (HNF)-6 is an "
                "upstream regulator of several genes involved in the pathogenesis of "
                "maturity-onset diabetes of the young. We therefore tested the "
                "hypothesis that variability in the HNF-6 gene is associated with "
                "subsets of Type II (non-insulin-dependent) diabetes mellitus and "
                "estimates of insulin secretion in glucose tolerant subjects.   We "
                "cloned the coding region as well as the intron-exon boundaries of the "
                "HNF-6 gene. We then examined them on genomic DNA in six MODY probands "
                "without mutations in the MODY1, MODY3 and MODY4 genes and in 54 "
                "patients with late-onset Type II diabetes by combined single strand "
                "conformational polymorphism-heteroduplex analysis followed by direct "
                "sequencing of identified variants. An identified missense variant was "
                "examined in association studies and genotype-phenotype studies.   We "
                "identified two silent and one missense (Pro75 Ala) variant. In an "
                "association study the allelic frequency of the Pro75Ala polymorphism "
                "was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type "
                "II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched "
                "glucose tolerant control subjects. Moreover, in studies of 238 "
                "middle-aged glucose tolerant subjects, of 226 glucose tolerant "
                "offspring of Type II diabetic patients and of 367 young healthy "
                "subjects, the carriers of the polymorphism did not differ from "
                "non-carriers in glucose induced serum insulin or C-peptide responses.   "
                "Mutations in the coding region of the HNF-6 gene are not associated "
                "with Type II diabetes or with changes in insulin responses to glucose "
                "among the Caucasians examined."
            ],
            "offsets": [[159, 1797]],
        },
    ],
    "entities": [
        {
            "id": "10491763_0",
            "type": "GeneOrGeneProduct",
            "text": ["Hepatocyte nuclear factor-6"],
            "offsets": [[0, 27]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        },
        {
            "id": "10491763_1",
            "type": "DiseaseOrPhenotypicFeature",
            "text": ["type II diabetes"],
            "offsets": [[74, 90]],
            "normalized": [{"db_name": "MESH", "db_id": "D003924"}],
        },
        {
            "id": "10491763_2",
            "type": "GeneOrGeneProduct",
            "text": ["insulin"],
            "offsets": [[140, 147]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3630"}],
        },
        {
            "id": "10491763_8",
            "type": "ChemicalEntity",
            "text": ["glucose"],
            "offsets": [[518, 525]],
            "normalized": [{"db_name": "MESH", "db_id": "D005947"}],
        },
    ],
    "relations": [
        {"id": "0_relation_0", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_1", "normalized": []},
        {"id": "0_relation_1", "type": "Positive_Correlation", "arg1_id": "10491763_8", "arg2_id": "10491763_2", "normalized": []},
    ],
}

ENTITY_ONLY_ROW = {
    "document_id": "10491763-entonly",
    "passages": [
        {
            "id": "0_title",
            "type": "title",
            "text": [
                "Hepatocyte nuclear factor-6: associations between genetic variability "
                "and type II diabetes and between genetic variability and estimates of "
                "insulin secretion."
            ],
            "offsets": [[0, 158]],
        },
        {
            "id": "0_abstract",
            "type": "abstract",
            "text": [
                "The transcription factor hepatocyte nuclear factor (HNF)-6 is an "
                "upstream regulator of several genes involved in the pathogenesis of "
                "maturity-onset diabetes of the young. We therefore tested the "
                "hypothesis that variability in the HNF-6 gene is associated with "
                "subsets of Type II (non-insulin-dependent) diabetes mellitus and "
                "estimates of insulin secretion in glucose tolerant subjects.   We "
                "cloned the coding region as well as the intron-exon boundaries of the "
                "HNF-6 gene. We then examined them on genomic DNA in six MODY probands "
                "without mutations in the MODY1, MODY3 and MODY4 genes and in 54 "
                "patients with late-onset Type II diabetes by combined single strand "
                "conformational polymorphism-heteroduplex analysis followed by direct "
                "sequencing of identified variants. An identified missense variant was "
                "examined in association studies and genotype-phenotype studies.   We "
                "identified two silent and one missense (Pro75 Ala) variant. In an "
                "association study the allelic frequency of the Pro75Ala polymorphism "
                "was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type "
                "II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched "
                "glucose tolerant control subjects. Moreover, in studies of 238 "
                "middle-aged glucose tolerant subjects, of 226 glucose tolerant "
                "offspring of Type II diabetic patients and of 367 young healthy "
                "subjects, the carriers of the polymorphism did not differ from "
                "non-carriers in glucose induced serum insulin or C-peptide responses.   "
                "Mutations in the coding region of the HNF-6 gene are not associated "
                "with Type II diabetes or with changes in insulin responses to glucose "
                "among the Caucasians examined."
            ],
            "offsets": [[159, 1797]],
        },
    ],
    "entities": [
        {
            "id": "10491763_0",
            "type": "GeneOrGeneProduct",
            "text": ["Hepatocyte nuclear factor-6"],
            "offsets": [[0, 27]],
            "normalized": [{"db_name": "NCBIGene", "db_id": "3175"}],
        }
    ],
    "relations": [],
}


# ---------------------------------------------------------------------------
# registration + import-time guards
# ---------------------------------------------------------------------------


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (the ingest wires
    Script.dispatch, which resolves by this exact NAME key)"""
    assert isinstance(Script.REGISTRY["BioredScript"], BioredScript)


def test_every_mapped_label_and_predicate_is_accounted_for() -> None:
    """the import-time guards only protect the module's own constants, so the contract is
    re-asserted here for drift: label-map values stay biolink classes and every predicate-map
    value is either a tablassert Predicates member or a documented native snake_case omission"""
    for raw_label, category in BioredScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label, predicate in BioredScript.PREDICATE_MAP.items():
        assert predicate in members or predicate in BioredScript.NATIVE_PREDICATES, f"{label!r} -> {predicate!r} is unguarded"


def test_the_import_time_label_guard_rejects_a_non_biolink_class() -> None:
    """a typo'd label-map value must fail loudly at import, not silently train garbage labels;
    families.validate_label_map is the shared guard the module calls on its own constant"""
    with pytest.raises(ValueError, match="NotAClass"):
        validate_label_map({"drift": "NotAClass"}, "BrokenBioredScript")


def test_the_import_time_predicate_guard_rejects_an_unmapped_predicate() -> None:
    """a typo'd predicate-map value must fail loudly at import rather than training relations
    under a garbage name; the module-scope guard re-runs over the (monkeypatched) map here"""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(BioredScript, "PREDICATE_MAP", {"Association": "not_a_predicate"})
    with pytest.raises(ValueError, match="not_a_predicate"):
        _validate_predicate_map()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# passages join: the drop rule that protects the document-relative offset reading
# ---------------------------------------------------------------------------


def test_a_non_list_passages_value_produces_the_empty_example() -> None:
    """hub-side schema drift can deliver a non-list passages column; the join must not guess a
    text because every entity offset is document-relative against the exact two-passage join"""
    example = run((None, ROW_10491763["entities"], ROW_10491763["relations"]))
    assert example == TrainingExample(text="")


@pytest.mark.parametrize("count", [1, 3])
def test_a_wrong_passage_count_produces_the_empty_example(count: int) -> None:
    """1 or 3 passages would re-derive every offset against a shifted text; measured corpus is
    always exactly 2 (title + abstract), so anything else drops the row"""
    passages = ROW_10491763["passages"][:count] if count < 2 else ROW_10491763["passages"] + [ROW_10491763["passages"][0]]
    example = run((passages, ROW_10491763["entities"], ROW_10491763["relations"]))
    assert example == TrainingExample(text="")


def test_a_non_dict_passage_entry_produces_the_empty_example() -> None:
    """a malformed passage entry must drop the row: joining the surviving passage would shift
    every abstract offset"""
    passages = [ROW_10491763["passages"][0], "abstract"]
    example = run((passages, ROW_10491763["entities"], ROW_10491763["relations"]))
    assert example == TrainingExample(text="")


@pytest.mark.parametrize("mutation", ["missing_text", "empty_text", "non_str_part"])
def test_a_passage_without_usable_text_produces_the_empty_example(mutation: str) -> None:
    """a passage dict without a non-empty list of str text parts cannot be joined; shipping it
    would train offsets measured against a different text"""
    passage = {"type": "abstract"}
    if mutation == "empty_text":
        passage["text"] = []
    elif mutation == "non_str_part":
        passage["text"] = ["ok", None]
    else:
        passage["text"] = "plain string"
    example = run(([ROW_10491763["passages"][0], passage], ROW_10491763["entities"], ROW_10491763["relations"]))
    assert example == TrainingExample(text="")


def test_the_verbatim_two_passage_join_feeds_the_offset_bridge() -> None:
    """the well-formed sibling: title + " " + abstract is the measured join, and the title-anchored
    entity [0, 27] must slice it exactly, proving the end-exclusive document-relative reading
    survives the bridge on the real row"""
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], ROW_10491763["relations"]))
    assert example.text.startswith("Hepatocyte nuclear factor-6")
    assert any("Hepatocyte nuclear factor-6" in entity.mentions for entity in example.entities)


# ---------------------------------------------------------------------------
# entity table: skip-don't-coerce per entry
# ---------------------------------------------------------------------------


def test_a_malformed_entity_entry_drops_only_itself() -> None:
    """a non-dict entry in the entities list drops only itself; the well-formed siblings still
    ship (skip-don't-coerce, never a whole-table drop for one bad row member). Surfaces, not
    labels, are asserted: the one-resolution chain may categorize a surface via fullmap instead
    of the dataset label map, and that choice belongs to the chain, not to this test."""
    entities = [*ROW_10491763["entities"][:2], "garbage", ROW_10491763["entities"][8]]
    example = run((ROW_10491763["passages"], entities, []))
    surfaces = [mention for entity in example.entities for mention in entity.mentions]
    assert surfaces == ["Hepatocyte nuclear factor-6", "type II diabetes", "glucose"]


def test_a_non_string_entity_id_or_type_drops_the_entry() -> None:
    """ids key the relation join and types key the label map, so a non-str field cannot be
    coerced; the entry drops and its siblings ship"""
    bad = dict(ROW_10491763["entities"][0])
    bad["type"] = 7
    example = run((ROW_10491763["passages"], [bad, ROW_10491763["entities"][1]], []))
    assert sum(len(entity.mentions) for entity in example.entities) == 1


def test_offsets_must_be_exactly_one_pair() -> None:
    """this conversion measured exactly one offset pair on 20,419/20,419 entities; zero pairs or
    two pairs (discontinuous) is schema drift and drops the entry"""
    zero = dict(ROW_10491763["entities"][0])
    zero["offsets"] = []
    two = dict(ROW_10491763["entities"][0])
    two["offsets"] = [[0, 5], [10, 15]]
    example = run((ROW_10491763["passages"], [zero, two, ROW_10491763["entities"][1]], []))
    assert sum(len(entity.mentions) for entity in example.entities) == 1


def test_bool_and_non_int_offsets_drop_the_entry() -> None:
    """bool is an int in python; an offset of True is schema poison, not a span"""
    truthy = dict(ROW_10491763["entities"][0])
    truthy["offsets"] = [[True, 27]]
    floaty = dict(ROW_10491763["entities"][0])
    floaty["offsets"] = [[0.0, 27.0]]
    example = run((ROW_10491763["passages"], [truthy, floaty, ROW_10491763["entities"][1]], []))
    assert sum(len(entity.mentions) for entity in example.entities) == 1


def test_a_non_list_entities_value_yields_zero_entities() -> None:
    """a corrupted table (stringified or None) must yield zero entities -- the silent zero-yield
    failure mode this repo shipped once -- and the row ships text-only for the declared-outputs
    filter to judge"""
    example = run((ROW_10491763["passages"], "not a list", ROW_10491763["relations"]))
    assert example.entities == []
    assert example.text


# ---------------------------------------------------------------------------
# char-offset bridge: per-span drops
# ---------------------------------------------------------------------------


def test_out_of_bounds_and_reversed_offsets_drop_per_span() -> None:
    """the bridge drops an out-of-bounds or reversed span and keeps the rest; 0 measured on the
    full corpus, so the guard is defensive drift armor, not a known defect path"""
    oob = dict(ROW_10491763["entities"][0])
    oob["offsets"] = [[999_999, 1_000_050]]
    reversed_span = dict(ROW_10491763["entities"][1])
    reversed_span["offsets"] = [[90, 74]]
    example = run((ROW_10491763["passages"], [oob, reversed_span, ROW_10491763["entities"][2]], []))
    assert sum(len(entity.mentions) for entity in example.entities) == 1
    assert any("insulin" in entity.mentions for entity in example.entities)


def test_zero_surviving_spans_ship_text_only() -> None:
    """when every span dies in the bridge the row still ships its text: text-only rows are a
    permitted under-shape, and the declared-outputs filter owns the verdict"""
    oob = dict(ROW_10491763["entities"][0])
    oob["offsets"] = [[999_999, 1_000_050]]
    example = run((ROW_10491763["passages"], [oob], ROW_10491763["relations"]))
    assert example.entities == []
    assert example.relations == []
    assert example.text


# ---------------------------------------------------------------------------
# relations: drop rules and the concept-signature dedupe
# ---------------------------------------------------------------------------


def test_a_non_dict_relation_entry_drops_only_itself() -> None:
    """a non-dict entry in the relations list drops only itself; the row's entities and the
    well-formed relations still ship"""
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], ["garbage", ROW_10491763["relations"][0]]))
    assert len(example.relations) == 1


def test_a_non_string_relation_field_drops_the_relation() -> None:
    """type and both arg ids key the dedupe and the join; a non-str field cannot be coerced"""
    bad = {"id": "x", "type": 9, "arg1_id": "10491763_0", "arg2_id": "10491763_1", "normalized": []}
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], [bad, ROW_10491763["relations"][0]]))
    assert len(example.relations) == 1


def test_a_mention_self_loop_relation_drops() -> None:
    """arg1_id == arg2_id is 121 measured drops upstream; a relation of a mention with itself
    trains nothing and drops (the sentence_rex rule)"""
    loop = {"id": "loop", "type": "Association", "arg1_id": "10491763_0", "arg2_id": "10491763_0", "normalized": []}
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], [loop, ROW_10491763["relations"][0]]))
    assert len(example.relations) == 1
    assert example.relations[0].fields[0].value != example.relations[0].fields[1].value


def test_a_dangling_arg_id_drops_only_that_relation() -> None:
    """an arg id absent from the entity table has no span to name (0 measured, defensive)"""
    dangling = {"id": "d", "type": "Association", "arg1_id": "10491763_9999", "arg2_id": "10491763_1", "normalized": []}
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], [dangling, ROW_10491763["relations"][0]]))
    assert len(example.relations) == 1


def test_a_relation_whose_arg_missed_the_bridge_drops() -> None:
    """a relation must not outlive its span: when one arg's entity died in the char-offset bridge
    the relation drops with it (an arg id that parses but has no surviving surface)"""
    doomed = dict(ROW_10491763["entities"][1])
    doomed["offsets"] = [[999_999, 1_000_050]]
    example = run((ROW_10491763["passages"], ROW_10491763["entities"][:3] + [doomed], ROW_10491763["relations"][:2]))
    assert all(relation.fields[1].value for relation in example.relations)


def test_a_repeat_concept_signature_dedupes_to_the_first_occurrence() -> None:
    """the bigbio conversion expanded 6,503 upstream concept-pair REL lines into 128,460 mention
    pairs; entities 1, 4, 6, 15, 19, 23, 29 of the verbatim row are all D003924 diseases, so the
    seven HNF-6 Association pairs collapse to ONE signature and the FIRST mention pair wins"""
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], ROW_10491763["relations"]))
    disease_mentions = [
        relation.fields[1].value
        for relation in example.relations
        if relation.name == "associated_with" and relation.fields[0].value == "Hepatocyte nuclear factor-6"
    ]
    assert len(disease_mentions) == 1, "all D003924 pairs must collapse to one kept mention pair"


def test_the_verbatim_row_dedupes_107_raw_relations_to_three_assertions() -> None:
    """the measured contract on the captured row: 107 raw mention pairs, 0 self-loops, exactly
    three concept signatures, kept in first-occurrence order with the mapped predicate names"""
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], ROW_10491763["relations"]))
    assert [relation.name for relation in example.relations] == [
        "associated_with",
        "positively_correlated_with",
        "associated_with",
    ]
    for relation in example.relations:
        assert relation.evidence == "asserted"
        assert relation.negated is False
        assert relation.fields[0].value in example.text
        assert relation.fields[1].value in example.text


def test_an_unknown_relation_type_drops_but_entities_still_ship() -> None:
    """no NA and no Cause label exists in this corpus (measured); an unknown type is drift and
    drops the relation, never guesses, while the row's entities still ship"""
    unknown = {"id": "u", "type": "Cause", "arg1_id": "10491763_0", "arg2_id": "10491763_1", "normalized": []}
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], [unknown]))
    assert example.relations == []
    assert example.entities


# ---------------------------------------------------------------------------
# label map and predicate map edges
# ---------------------------------------------------------------------------


def test_geneorgeneproduct_resolves_to_gene_through_the_label_map() -> None:
    """GeneOrGeneProduct maps to Gene (the corpus's gene mentions carry NCBIGene ids, unlike the
    Pile-NER nonspecific -N tail); the mapped value must surface as the emitted label"""
    example = run((ROW_10491763["passages"], ROW_10491763["entities"][:1], []))
    assert [entity.label for entity in example.entities] == ["Gene"]
    assert example.entities[0].mentions == ["Hepatocyte nuclear factor-6"]


def test_the_five_mapped_predicates_are_biolink_members() -> None:
    """Association, both correlations, Bind, and Drug_Interaction map onto real tablassert
    Predicates members, so their triples train under the canonical biolink name"""
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label in ("Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Drug_Interaction"):
        assert BioredScript.PREDICATE_MAP[label] in members


def test_the_three_native_predicates_stay_snake_case() -> None:
    """Comparison, Cotreatment, and Conversion have no honest biolink slot; they ride the
    resolve_predicate convention as native snake_case (documented omission, zero-shot breadth)
    and must NOT be biolink members"""
    members: frozenset[str] = ScriptUtils.biolink_predicates()
    for label in ("Comparison", "Cotreatment", "Conversion"):
        predicate = BioredScript.PREDICATE_MAP[label]
        assert predicate in BioredScript.NATIVE_PREDICATES
        assert predicate not in members


def test_a_native_predicate_relation_ships_its_snake_case_name() -> None:
    """the well-formed sibling for the native path: a Comparison relation ships under its native
    snake_case name with the same asserted evidence"""
    comparison = {"id": "c", "type": "Comparison", "arg1_id": "10491763_0", "arg2_id": "10491763_1", "normalized": []}
    example = run((ROW_10491763["passages"], ROW_10491763["entities"][:2], [comparison]))
    assert [relation.name for relation in example.relations] == ["comparison"]
    assert example.relations[0].evidence == "asserted"


# ---------------------------------------------------------------------------
# yield: the silent zero-yield guard, on real rows
# ---------------------------------------------------------------------------


def test_the_mini_real_shaped_row_ships_both_shapes() -> None:
    """four verbatim entities and two real relations must ship entities AND relations: this is
    the nonzero-yield assertion over a hand-checkable row"""
    example = run((MINI_ROW["passages"], MINI_ROW["entities"], MINI_ROW["relations"]))
    assert example.populated() == frozenset({"entities", "relations"})
    assert [relation.name for relation in example.relations] == ["associated_with", "positively_correlated_with"]
    for entity in example.entities:
        for mention in entity.mentions:
            assert mention in example.text


def test_a_row_with_zero_relations_ships_entities_only() -> None:
    """8 of 600 measured rows carry entities and no relations; the permitted-shapes contract
    ships them with no code path special-casing them"""
    example = run((ENTITY_ONLY_ROW["passages"], ENTITY_ONLY_ROW["entities"], ENTITY_ONLY_ROW["relations"]))
    assert example.populated() == frozenset({"entities"})
    assert example.relations == []


def test_the_verbatim_row_ships_every_mention_surface_in_text() -> None:
    """gliner2's validator rejects a mention outside the shipped text; every surface of the
    32-entity verbatim row must occur in the emitted (token-rejoined) text. Labels are not
    asserted per surface: the one-resolution chain may categorize through fullmap, the dataset
    label map, or the raw PascalCase tail, and all three are correct here."""
    example = run((ROW_10491763["passages"], ROW_10491763["entities"], ROW_10491763["relations"]))
    assert example.populated() == frozenset({"entities", "relations"})

    def key(surface: str) -> str:
        """punctuation-and-whitespace-insensitive comparison key: the emitted mention is the
        token-rejoined surface (join_tokens detaches punctuation), so '(HNF)-6' ships as
        '( HNF ) - 6' and only the squashed form is comparable"""
        return re.sub(r"[\s()\[\]-]+", "", surface)

    raw_keys = {key(entity["text"][0]) for entity in ROW_10491763["entities"]}
    emitted_keys = {key(mention) for entity in example.entities for mention in entity.mentions}
    assert emitted_keys == raw_keys, "every distinct gold surface must ship under some resolved label"
    for entity in example.entities:
        for mention in entity.mentions:
            assert mention in example.text, f"mention {mention!r} does not occur in the emitted text"


def test_the_script_never_touches_the_mixing_weight() -> None:
    """weight stamping is the stream's job (the declared weight rides the tuple), so the script
    must leave TrainingExample.weight at its default"""
    example = run((MINI_ROW["passages"], MINI_ROW["entities"], MINI_ROW["relations"]))
    assert example.weight == 1.0

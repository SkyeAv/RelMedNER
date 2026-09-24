from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.gazetteer import extract_relations
from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class PubmedAbstractsScript(Script):
    """streams knowledgator/PubMedAbstractsNER rows (token list + end-inclusive
    [start, end, "MeSH heading - definition"] spans) into biolink-labeled entity examples;
    unresolved headings surface as PascalCased raw categories for zero-shot breadth"""

    NAME: ClassVar[str] = "PubmedAbstractsScript"

    # MeSH heading -> biolink class; resolve_mentions looks keys up on label.lower(), so keys
    # are the lowercased headings. Seeded from the measured 35,000-row corpus (383,721 spans,
    # 5,850 headings) and extended (US-004) to 211 entries by full-corpus surface inspection:
    # with the seed map 51,684 spans land in raw origin over 1,637 headings; every heading
    # with >=30 raw-origin spans whose surfaces fit one real biolink class is mapped (floor
    # reached at 30 raw spans -- the residual >=30 tail is mixed/junk or borderline, incl.
    # three ~30-span Disease headings left unmapped on borderline surfaces; re-measure
    # recipe: cached full file + ONE batched fullmap round trip + the shared gates -- the
    # fullmap pass is independent of this map, so candidate maps are pure set arithmetic
    # over headings). Raw origin is NOT loss -- those spans train under
    # PascalCased raw headings -- and headings without a faithful class stay UNMAPPED: a wrong
    # bucket silently mislabels training data (measured unmapped monsters: "Investigative
    # Techniques" is 94% the bare surfaces 'methods'/'METHODS' with no fitting class; "Group
    # Processes" is 99.7% 'role'/'roles' and biolink has no Role class; "Chemical Phenomena"
    # mixes processes (hydrolysis, diffusion) with quantities (molecular weight, solubility);
    # "Genetic Phenomena" mixes phenotypes, processes, and genotypes; "Reproductive
    # Physiological Phenomena" mixes 'sex' (an attribute) with fertility/gestational age;
    # "Diagnosis" spans are the abstract concepts 'diagnosis'/'prognosis'; "Population
    # Characteristics" spans include 'health'/'Healthcare'; junk-surface headings like "Genome
    # Components" whose mentions are punctuation). Quirky-but-faithful calls, keyed to the
    # surfaces rather than the heading names: "drug resistance" -> PhenotypicFeature (88% of
    # surfaces are 'insulin resistance', an HPO phenotype, not drug properties); "brain" ->
    # AnatomicalEntity (surfaces are 'blood-brain barrier'); "epithelial cells" -> CellLine
    # (surfaces are Caco-2/PC-3/LLC-PK1 lines); "virus physiological phenomena" ->
    # ClinicalMeasurement (85% 'viral load'); "time" -> ClinicalMeasurement (all 'half-life');
    # "research design" -> Cohort (all 'control groups'); "age groups"/"white people" ->
    # PopulationOfIndividualOrganisms only because their mentions ('adult', 'white') are
    # genuinely human populations, matching the shared person bucket's own allowance.
    LABEL_MAP: ClassVar[dict[str, str]] = {
        "pathologic processes": "PathologicalProcess",
        "disease": "Disease",
        "persons": "Human",
        "age groups": "PopulationOfIndividualOrganisms",
        "publication formats": "Publication",
        "cells": "Cell",
        "tissues": "AnatomicalEntity",
        "body regions": "GrossAnatomicalStructure",
        "digestive system": "GrossAnatomicalStructure",
        "cardiovascular system": "GrossAnatomicalStructure",
        "nervous system": "GrossAnatomicalStructure",
        "central nervous system": "GrossAnatomicalStructure",
        "integumentary system": "GrossAnatomicalStructure",
        "bacteria": "Bacterium",
        "brucella": "Bacterium",
        "viruses": "Virus",
        "eukaryota": "OrganismTaxon",
        "murinae": "OrganismTaxon",
        "metals, heavy": "ChemicalEntity",
        "nucleic acids": "NucleicAcidEntity",
        "united nations": "Agent",
        "brain": "AnatomicalEntity",
        "cartilage": "AnatomicalEntity",
        "connective tissue": "AnatomicalEntity",
        "neural pathways": "AnatomicalEntity",
        "brucellaceae": "Bacterium",
        "staphylococcus aureus": "Bacterium",
        "sports": "Behavior",
        "antigen-antibody reactions": "BiologicalProcess",
        "bone development": "BiologicalProcess",
        "bone remodeling": "BiologicalProcess",
        "cell movement": "BiologicalProcess",
        "cell nucleus division": "BiologicalProcess",
        "cell proliferation": "BiologicalProcess",
        "dna damage": "BiologicalProcess",
        "dna repair": "BiologicalProcess",
        "dna replication": "BiologicalProcess",
        "epigenesis, genetic": "BiologicalProcess",
        "gametogenesis": "BiologicalProcess",
        "gastrointestinal motility": "BiologicalProcess",
        "gene silencing": "BiologicalProcess",
        "metabolism": "BiologicalProcess",
        "pharmacokinetics": "BiologicalProcess",
        "pharmacological and toxicological phenomena": "BiologicalProcess",
        "pharmacological phenomena": "BiologicalProcess",
        "postpartum period": "BiologicalProcess",
        "recombination, genetic": "BiologicalProcess",
        "regeneration": "BiologicalProcess",
        "regulated cell death": "BiologicalProcess",
        "stress, physiological": "BiologicalProcess",
        "transcription, genetic": "BiologicalProcess",
        "macrophages": "Cell",
        "cell line, tumor": "CellLine",
        "epithelial cells": "CellLine",
        "adenine nucleotides": "ChemicalEntity",
        "adenosine": "ChemicalEntity",
        "aminobutyrates": "ChemicalEntity",
        "butyrates": "ChemicalEntity",
        "carbohydrates": "ChemicalEntity",
        "citrates": "ChemicalEntity",
        "fatty acids, monounsaturated": "ChemicalEntity",
        "fatty acids, unsaturated": "ChemicalEntity",
        "glucosephosphates": "ChemicalEntity",
        "heterocyclic compounds, 2-ring": "ChemicalEntity",
        "nanoparticles": "ChemicalEntity",
        "organothiophosphates": "ChemicalEntity",
        "particulate matter": "ChemicalEntity",
        "protective agents": "ChemicalEntity",
        "tocopherols": "ChemicalEntity",
        "blood cell count": "ClinicalMeasurement",
        "blood physiological phenomena": "ClinicalMeasurement",
        "diagnostic techniques and procedures": "ClinicalMeasurement",
        "ocular physiological phenomena": "ClinicalMeasurement",
        "survival analysis": "ClinicalMeasurement",
        "time": "ClinicalMeasurement",
        "urinary tract physiological phenomena": "ClinicalMeasurement",
        "virus physiological phenomena": "ClinicalMeasurement",
        "research design": "Cohort",
        "electrical equipment and supplies": "Device",
        "optical devices": "Device",
        "prostheses and implants": "Device",
        "surgical fixation devices": "Device",
        "anemia, sickle cell": "Disease",
        "anti-neutrophil cytoplasmic antibody-associated vasculitis": "Disease",
        "carcinoma, islet cell": "Disease",
        "chromosome disorders": "Disease",
        "drug eruptions": "Disease",
        "endocrine gland neoplasms": "Disease",
        "gangliosidoses, gm2": "Disease",
        "gonadal dysgenesis, 46,xy": "Disease",
        "hepatitis, viral, animal": "Disease",
        "intestinal obstruction": "Disease",
        "malformations of cortical development, group i": "Disease",
        "metal metabolism, inborn errors": "Disease",
        "neoplasms, connective and soft tissue": "Disease",
        "neoplasms, germ cell and embryonal": "Disease",
        "neoplasms, glandular and epithelial": "Disease",
        "neoplasms, vascular tissue": "Disease",
        "paraproteinemias": "Disease",
        "pneumonia, viral": "Disease",
        "primary immunodeficiency diseases": "Disease",
        "purine-pyrimidine metabolism, inborn errors": "Disease",
        "retinal degeneration": "Disease",
        "retinal diseases": "Disease",
        "sexual dysfunctions, psychological": "Disease",
        "sphingolipidoses": "Disease",
        "tdp-43 proteinopathies": "Disease",
        "thalassemia": "Disease",
        "vascular diseases": "Disease",
        "anti-infective agents": "Drug",
        "barbiturates": "Drug",
        "central nervous system agents": "Drug",
        "dosage forms": "Drug",
        "hematologic agents": "Drug",
        "natriuretic agents": "Drug",
        "atmosphere": "EnvironmentalFeature",
        "beverages": "Food",
        "dietary fats, unsaturated": "Food",
        "fermented beverages": "Food",
        "food and beverages": "Food",
        "nutritional physiological phenomena": "Food",
        "gene components": "GenomicEntity",
        "genetic code": "GenomicEntity",
        "genetic structures": "GenomicEntity",
        "genome, bacterial": "GenomicEntity",
        "geographic locations": "GeographicLocation",
        "mid-atlantic region": "GeographicLocation",
        "midwestern united states": "GeographicLocation",
        "southeastern united states": "GeographicLocation",
        "southwestern united states": "GeographicLocation",
        "united kingdom": "GeographicLocation",
        "ear, external": "GrossAnatomicalStructure",
        "exocrine glands": "GrossAnatomicalStructure",
        "facial bones": "GrossAnatomicalStructure",
        "fibrocartilage": "GrossAnatomicalStructure",
        "gastrointestinal tract": "GrossAnatomicalStructure",
        "intestine, large": "GrossAnatomicalStructure",
        "laryngeal cartilages": "GrossAnatomicalStructure",
        "lower gastrointestinal tract": "GrossAnatomicalStructure",
        "rib cage": "GrossAnatomicalStructure",
        "stomatognathic system": "GrossAnatomicalStructure",
        "upper gastrointestinal tract": "GrossAnatomicalStructure",
        "health personnel": "Human",
        "invertebrates": "Invertebrate",
        "life cycle stages": "LifeStage",
        "canidae": "Mammal",
        "body size": "OrganismAttribute",
        "body weight": "OrganismAttribute",
        "hemorrhage": "PathologicalProcess",
        "neoplastic processes": "PathologicalProcess",
        "body temperature changes": "PhenotypicFeature",
        "body weight changes": "PhenotypicFeature",
        "drug resistance": "PhenotypicFeature",
        "drug resistance, bacterial": "PhenotypicFeature",
        "signs and symptoms, digestive": "PhenotypicFeature",
        "signs and symptoms, respiratory": "PhenotypicFeature",
        "unconsciousness": "PhenotypicFeature",
        "vision disorders": "PhenotypicFeature",
        "white people": "PopulationOfIndividualOrganisms",
        "cell transplantation": "Procedure",
        "chemical fractionation": "Procedure",
        "diagnostic imaging": "Procedure",
        "immunosorbent techniques": "Procedure",
        "mass spectrometry": "Procedure",
        "monitoring, physiologic": "Procedure",
        "pathology, clinical": "Procedure",
        "polymerase chain reaction": "Procedure",
        "sequence analysis, dna": "Procedure",
        "sequence analysis, rna": "Procedure",
        "stem cell transplantation": "Procedure",
        "surgical procedures, operative": "Procedure",
        "tissue transplantation": "Procedure",
        "amidohydrolases": "Protein",
        "aspartic acid endopeptidases": "Protein",
        "blood proteins": "Protein",
        "cell adhesion molecules": "Protein",
        "cellulases": "Protein",
        "colony-stimulating factors": "Protein",
        "endorphins": "Protein",
        "enzymes": "Protein",
        "flavoproteins": "Protein",
        "galactosidases": "Protein",
        "immunoproteins": "Protein",
        "intercellular signaling peptides and proteins": "Protein",
        "interleukin-1": "Protein",
        "interleukins": "Protein",
        "macrophage-activating factors": "Protein",
        "melanocyte-stimulating hormones": "Protein",
        "neuropeptides": "Protein",
        "oxidoreductases acting on ch-nh2 group donors": "Protein",
        "oxidoreductases acting on sulfur group donors": "Protein",
        "phosphotransferases (alcohol group acceptor)": "Protein",
        "phosphotransferases (nitrogenous group acceptor)": "Protein",
        "pituitary hormone release inhibiting hormones": "Protein",
        "pituitary hormone-releasing hormones": "Protein",
        "proprotein convertases": "Protein",
        "protein serine-threonine kinases": "Protein",
        "selectins": "Protein",
        "syndecans": "Protein",
        "tumor necrosis factors": "Protein",
        "genetic variation": "SequenceVariant",
        "mutation": "SequenceVariant",
        "clinical study": "Study",
        "cohort studies": "Study",
        "study characteristics": "Study",
        "salmonidae": "Vertebrate",
        "vertebrates": "Vertebrate",
        "hiv": "Virus",
        "lentiviruses, primate": "Virus",
        "picornaviridae": "Virus",
        "severe acute respiratory syndrome-related coronavirus": "Virus",
    }

    @staticmethod
    def heading(label: str) -> str:
        """bare MeSH heading: the " - " definition tail is corpus metadata, not resolution
        signal -- the shared gate keys buckets on label words, so definition words ('region',
        'leg', 'process') would flip fullmap outcomes (full corpus: 45,975 of 383,721 spans,
        12.0%, lose their fullmap hit when the definition rides along -- gate accepts the
        bare heading and rejects the full label)"""
        return label.split(" - ", 1)[0]

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        tokens_value, ner_value = values
        tokens: list[str] = [str(token) for token in tokens_value] if isinstance(tokens_value, list) else []
        ner: list[Any] = ner_value if isinstance(ner_value, list) else []
        if not tokens or not ner:
            # the 13 measured empty-ner rows exit here as text-only examples; an empty
            # populated() is dropped by pipeline.matches_declared_outputs downstream
            return TrainingExample(text=ScriptUtils.join_tokens(tokens))
        # mention_spans validates shape and bounds first (end inclusive), THEN the label is
        # split so the definition tail never reaches resolution, gates, or output
        spans: list[tuple[int, int, str]] = [(start, end, self.heading(label)) for start, end, label in ScriptUtils.mention_spans(tokens, ner)]
        mentions: list[tuple[str, str]] = [(ScriptUtils.join_tokens(tokens[start : end + 1]), heading) for start, end, heading in spans]
        resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions(mentions, label_map=self.LABEL_MAP)
        # raw-tail headings surface PascalCased (biolink-style casing) while label_map/fallback
        # entries already name a biolink class and stay untouched
        labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(resolved)
        # the multi-class fan-out makes resolved longer than spans, so pair_spans re-pairs by
        # span_index (items[0] is the primary; a misaligned shape raises instead of silently
        # misaligning later spans) and every fan-out row extends resolved_spans with its category
        resolved_spans: list[tuple[int, int, str]] = [
            (start, end, item.category) for (start, end, _), items in ScriptUtils.pair_spans(spans, labeled) for item in items
        ]
        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(labeled),
            relations=extract_relations(tokens, resolved_spans),
        )


validate_label_map(PubmedAbstractsScript.LABEL_MAP, PubmedAbstractsScript.NAME)

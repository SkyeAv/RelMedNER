from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar, Self

from relmedner.families import validate_label_map
from relmedner.models import Relation, RelationField, Structure, StructureField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils

# the `source` column carries the ORIGINAL hub repo ids (slashes), not the dashed config names the
# yaml subsets declare; a measured first pass keyed on the config names matched zero rows (the
# silent-zero-yield failure mode), so every map key below uses the hub-column value verbatim
HENRIQUE_SOURCE: str = "HenriqueGodoy/extract-0"
JIRAYA_SOURCE: str = "Jiraya/html_to_json_information_extraction_dataset"
OWKIN_SOURCE: str = "owkin/medical_knowledge_from_extracts"
PROFESSOR_BOB_SOURCE: str = "ProfessorBob/relation_extraction"
ROBOROVSKI_SOURCE: str = "roborovski/dolly-entity-extraction"
SANDEEPPANEM_SOURCE: str = "sandeeppanem/resume-json-extraction-5k"

# sandeeppanem resume corpora carry mojibake placeholder companies whose values all start with
# this prefix (measured forms: "Company Name", "Company Name <mojibake> City , State",
# "Company Name - City , State"); they stay structure fields but never become mentions
COMPANY_PLACEHOLDER_PREFIX: str = "Company Name"

# owkin interventions[].name maps to ChemicalEntity ONLY when the same object's
# interventions[].type (9 measured values: Drug, Biological, Device, Procedure, Dietary
# Supplement, Other, Behavioral, Genetic, Radiation) lowercased lands in this set; devices,
# procedures, behavioral, radiation and other interventions stay unmapped rather than mislabeled.
# Genetic rides along deliberately: a gene-therapy arm name is a gene symbol, and the 2-triple
# census slice is too small to separate from the chemical arms
CHEMICAL_INTERVENTION_TYPES: frozenset[str] = frozenset({"drug", "biological", "dietary supplement", "genetic"})

# roborovski person names qualify only when the containing object carries a sibling type leaf
# whose value is exactly this marker (measured marker paths: entity, result.entity, results[].entity)
PERSON_TYPE_MARKER: str = "Person"

# ProfessorBob's 222 distinct predicate values are almost all outside the biolink Predicates
# vocabulary (250 snake_case members): 31,303/31,740 triples carry a non-biolink predicate.
# The US-002 full-split census found exactly ONE honest target, "member of" -> member_of
# (158 triples, 85 with both endpoints verbatim in the text); the corpus's top predicates
# ("no relation" 14,333, "instance of" 1,598, "occupation" 1,387) stay unmapped
NO_RELATION_PREDICATE: str = "no relation"


def _flatten_child(path: str, child: Any) -> list[tuple[str, Any]]:
    """one flattened entry set for a single child value; empty containers and null/blank leaves
    produce no field, containers recurse, scalar lists collapse into one terminal list field"""
    if isinstance(child, dict):
        if not child:
            return []
        return _flatten_dict(child, path)
    if isinstance(child, list):
        if not child:
            return []
        if any(isinstance(item, (dict, list)) for item in child):
            # a container anywhere in the list puts every item behind the [] convention; scalar
            # members of a mixed list ride along as one terminal field at the same path
            entries: list[tuple[str, Any]] = []
            scalars: list[str] = []
            for item in child:
                if isinstance(item, (dict, list)):
                    entries.extend(_flatten_child(f"{path}[]", item))
                elif item is not None and item != "":
                    scalars.append(str(item))
            if scalars:
                entries.append((f"{path}[]", scalars))
            return entries
        members = [str(item) for item in child if item is not None and item != ""]
        return [(f"{path}[]", members)] if members else []
    if child is None:
        return []
    if isinstance(child, str):
        return [(path, child)] if child.strip() else []
    if isinstance(child, bool):
        return [(path, "true" if child else "false")]
    return [(path, str(child))]


def _flatten_dict(mapping: Mapping[str, Any], prefix: str) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for key, child in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        entries.extend(_flatten_child(path, child))
    return entries


def flatten_json(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """flatten one decoded JSON document into an ordered (path, value) list, path convention
    matching the measured census paths verbatim: dict children dot-join (result.properties.name),
    list-of-dict children take parent[].child (interventions[].name, [].Job Location at root),
    scalar lists take one terminal parent[] field with the whole list as a list[str] value
    (null and empty-string members dropped, non-strings stringified)"""
    if isinstance(value, dict):
        return _flatten_dict(value, prefix)
    if isinstance(value, list):
        return _flatten_child(prefix, value)
    return []


def _owkin_chemical_names(node: Any) -> set[str]:
    """every interventions[].name whose sibling interventions[].type qualifies as chemical"""
    names: set[str] = set()
    if isinstance(node, dict):
        interventions = node.get("interventions")
        if isinstance(interventions, list):
            for item in interventions:
                if not isinstance(item, dict):
                    continue
                type_value = item.get("type")
                if isinstance(type_value, str) and type_value.lower() in CHEMICAL_INTERVENTION_TYPES:
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        names.add(name)
        for child in node.values():
            names |= _owkin_chemical_names(child)
    elif isinstance(node, list):
        for item in node:
            names |= _owkin_chemical_names(item)
    return names


def _roborovski_person_names(node: Any) -> set[str]:
    """every name leaf in an object whose sibling type leaf is exactly the Person marker; covers
    the measured marker shapes (entity, result.entity, results[].entity) because the marker is a
    direct child of the same containing object as the name"""
    names: set[str] = set()
    if isinstance(node, dict):
        if node.get("entity") == PERSON_TYPE_MARKER:
            direct = node.get("name")
            if isinstance(direct, str) and direct:
                names.add(direct)
            properties = node.get("properties")
            if isinstance(properties, dict):
                nested = properties.get("name")
                if isinstance(nested, str) and nested:
                    names.add(nested)
        for child in node.values():
            names |= _roborovski_person_names(child)
    elif isinstance(node, list):
        for item in node:
            names |= _roborovski_person_names(item)
    return names


class JsonExtractionScript(Script):
    """turns one agentlans/json-extraction row into one training example: text plus the decoded
    JSON document flattened into a single structure, honest-biolink entities, and (ProfessorBob
    only) asserted relations

    Measured facts that shaped this mapping (full-split census, 24,768 rows across the six source
    configs; the hub `all` config is their exact union and is deliberately NOT declared, which
    would double-stream every row): per-source rows ProfessorBob 6,920, roborovski 5,945,
    sandeeppanem 4,879, Jiraya 3,035, HenriqueGodoy 2,606, owkin 1,383. Decoded json: 14,813
    dicts / 9,955 lists, 443,021 string leaves, median 15 scalar fields per row (max 392), max
    nesting depth 8 (roborovski). 0 json parse failures, 0 empty texts. Verbatim case-sensitive
    containment of a string leaf in its text is 57.4% corpus-wide (owkin 11.8%, ProfessorBob
    45.2%, sandeeppanem 55.0%, HenriqueGodoy 52.0%, roborovski 71.8%, Jiraya 83.3%), so
    containment is a per-leaf entity guard, never a row guard.

    Trust gold, same philosophy as SuperGlueRecordScript: the extracted JSON values are the gold
    output of the source task and general-domain text against a biomedical fullmap is noise, so
    no resolve_mentions/fullmap round trip happens; every entity ships with origin="fallback"
    under the dataset-local LABEL_MAPS class. Maps hold only honest biolink targets: HenriqueGodoy
    ships structures only (entity_name[] mixes drugs, paper titles and finance concepts with no
    type qualifier), ProfessorBob ships no entity map (open vocabulary) and a one-entry
    PREDICATE_MAP (the census-landed "member of" -> member_of), sandeeppanem companies map to
    Agent because Organization is not
    a tablassert Categories member (Agent's own definition covers organizations; landed precedent
    organization -> Agent), sandeeppanem location and Jiraya job titles/IDs/links stay unmapped,
    and owkin/roborovski names ride behind their measured sibling-marker qualifiers. No whitespace
    or case rewriting anywhere: containment is verbatim and gliner2 requires every mention surface
    in the emitted text.

    US-002 census of the qualifier-gated paths (full split): roborovski's Person marker qualifies only
    33 name leaves (32 contained verbatim; tiny but honest), sandeeppanem carries 13,992 company leaves
    of which 5,650 (40.4%) start with the "Company Name" mojibake placeholder prefix and never become
    mentions, and its 3,787 location leaves (2,894 contained verbatim) stay unmapped because the
    "City, State" placeholder noise has no honest biolink target; owkin chem-qualified intervention
    names are contained verbatim 678/2,530 (26.8%), other intervention types 54/424, and conditions
    only 141/1,963 (7.2%), a corpus property (values are normalized extractions from abstracts), not
    a bug. ProfessorBob predicate census: 222 distinct values, 31,303/31,740 triples non-biolink,
    exactly one honest relation target ("member of" -> member_of, 158 triples, 85 with both endpoints
    verbatim in the text).
    """

    NAME: ClassVar[str] = "JsonExtractionScript"

    # honest-biolink label maps keyed (source, path) against the flattened census paths; every
    # value is a tablassert Categories member validated at import below. Entries marked by the
    # qualifier rules (owkin interventions[].name, roborovski name paths) are additionally gated
    # by the measured sibling-marker sets computed per row
    LABEL_MAPS: ClassVar[dict[tuple[str, str], str]] = {
        (OWKIN_SOURCE, "conditions[]"): "Disease",
        (OWKIN_SOURCE, "interventions[].name"): "ChemicalEntity",
        (ROBOROVSKI_SOURCE, "name"): "Agent",
        (ROBOROVSKI_SOURCE, "result.name"): "Agent",
        (ROBOROVSKI_SOURCE, "result.properties.name"): "Agent",
        (ROBOROVSKI_SOURCE, "results[].name"): "Agent",
        (ROBOROVSKI_SOURCE, "results[].properties.name"): "Agent",
        (SANDEEPPANEM_SOURCE, "current_company"): "Agent",
        (SANDEEPPANEM_SOURCE, "previous_companies[]"): "Agent",
        (JIRAYA_SOURCE, "[].Job Location"): "GeographicLocation",
    }

    # paths whose map entry only applies when the leaf also survives the measured sibling-marker
    # qualifier for its source (the marker rule, not the map, carries the honest-bucket decision)
    QUALIFIED_PATHS: ClassVar[frozenset[tuple[str, str]]] = frozenset(
        {
            (OWKIN_SOURCE, "interventions[].name"),
            (ROBOROVSKI_SOURCE, "name"),
            (ROBOROVSKI_SOURCE, "result.name"),
            (ROBOROVSKI_SOURCE, "result.properties.name"),
            (ROBOROVSKI_SOURCE, "results[].name"),
            (ROBOROVSKI_SOURCE, "results[].properties.name"),
        }
    )

    # ProfessorBob predicate string -> biolink Predicates member, validated at import. The
    # US-002 full-split census of the 222 distinct predicate values found exactly one honest
    # target: "member of" is a biolink member (member_of, 158 triples, 85 with both endpoints
    # verbatim in the text). Every other predicate (top: "no relation" 14,333, "instance of"
    # 1,598, "occupation" 1,387) is outside the Predicates vocabulary and stays unmapped
    PREDICATE_MAP: ClassVar[dict[str, str]] = {"member of": "member_of"}

    def _qualifying_values(self: Self, source: str, decoded: Any) -> dict[tuple[str, str], set[str]]:
        """the surviving sibling-marker value sets for this row's qualified paths only"""
        qualified = {key for key in self.QUALIFIED_PATHS if key[0] == source}
        if not qualified:
            return {}
        if source == OWKIN_SOURCE:
            names = _owkin_chemical_names(decoded)
        elif source == ROBOROVSKI_SOURCE:
            names = _roborovski_person_names(decoded)
        else:
            names = set()
        return {key: names for key in qualified}

    def _entities(
        self: Self,
        source: str,
        text: str,
        flattened: list[tuple[str, Any]],
        qualifiers: dict[tuple[str, str], set[str]],
    ) -> list[ResolvedMention]:
        """one fallback-origin mention per surviving string leaf at a mapped path; a leaf drops on
        the placeholder guard, the sibling-marker qualifier, or verbatim containment failure --
        the leaf drops, never the row (skip-don't-coerce)"""
        resolved: list[ResolvedMention] = []
        for path, value in flattened:
            category = self.LABEL_MAPS.get((source, path))
            if category is None:
                continue
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if not isinstance(candidate, str) or not candidate:
                    continue
                if source == SANDEEPPANEM_SOURCE and candidate.startswith(COMPANY_PLACEHOLDER_PREFIX):
                    continue
                if (source, path) in qualifiers and candidate not in qualifiers[(source, path)]:
                    continue
                if candidate not in text:
                    continue
                resolved.append(ResolvedMention(mention=candidate, category=category, origin="fallback"))
        return resolved

    def _relations(self: Self, decoded: Any, text: str) -> list[Relation]:
        """ProfessorBob asserted relations: one per decoded root-list object whose predicate maps,
        whose endpoints differ, and both of which occur verbatim in the text; everything else
        drops the relation, never the row. The census-landed PREDICATE_MAP carries exactly one
        entry (member of -> member_of); every other triple drops at the map, the no-relation
        guard, the self-loop guard, or the endpoint-containment guard"""
        relations: list[Relation] = []
        if not isinstance(decoded, list):
            return relations
        for item in decoded:
            if not isinstance(item, dict):
                continue
            predicate = item.get("predicate")
            if not isinstance(predicate, str) or predicate == NO_RELATION_PREDICATE:
                continue
            mapped = self.PREDICATE_MAP.get(predicate)
            if mapped is None:
                continue
            head, tail = item.get("subject"), item.get("object")
            if not isinstance(head, str) or not isinstance(tail, str) or head == tail:
                continue
            if head not in text or tail not in text:
                continue
            relations.append(
                Relation(
                    name=mapped,
                    fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
                    description=ScriptUtils.predicate_description(mapped),
                    evidence="asserted",
                )
            )
        return relations

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        if len(values) != 3:
            return TrainingExample(text="")
        text_value, json_value, source_value = values
        if not isinstance(text_value, str) or not text_value:
            return TrainingExample(text="")
        source = source_value if isinstance(source_value, str) else ""
        try:
            decoded: Any = json.loads(str(json_value))
        except (ValueError, TypeError, RecursionError):
            return TrainingExample(text="")
        if not isinstance(decoded, (dict, list)):
            return TrainingExample(text="")
        try:
            flattened = flatten_json(decoded)
        except RecursionError:
            # json.loads can succeed on documents deeper than this module's recursive flattening
            # can walk (measured: a 2,000-level list parses, then _flatten_child blows the stack);
            # the row falls out as the text-only example, never crashing the stream
            return TrainingExample(text="")
        structure = Structure(name=source, fields=[StructureField(name=path, value=value) for path, value in flattened])
        qualifiers = self._qualifying_values(source, decoded)
        resolved = self._entities(source, text_value, flattened, qualifiers)
        relations = self._relations(decoded, text_value) if source == PROFESSOR_BOB_SOURCE else []
        return TrainingExample(
            text=text_value,
            structures=[structure],
            entities=ScriptUtils.group_entities(resolved) if resolved else [],
            relations=relations,
        )


validate_label_map(
    {f"{source} | {path}": category for (source, path), category in JsonExtractionScript.LABEL_MAPS.items()},
    JsonExtractionScript.NAME,
)
for _predicate, _member in JsonExtractionScript.PREDICATE_MAP.items():
    if _member not in ScriptUtils.biolink_predicates():
        raise ValueError(f"{JsonExtractionScript.NAME} PREDICATE_MAP {_predicate!r} -> {_member!r} is not a biolink predicate")

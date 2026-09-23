from __future__ import annotations

from typing import Any, ClassVar, Self

from relmedner.families import validate_category
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.types import Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils


class DocredScript(Script):
    """turns one thunlp/docred document (tokenized sentences, gold entity clusters, gold Wikidata
    relations) into one trust-gold entities-plus-relations training example

    Every span here is dataset gold, so nothing is re-resolved through fullmap: the mentions ride
    under biolink classes derived from the corpus's own six-type vocabulary (the same trust-gold
    philosophy as SuperGlueRecordScript). General-domain Wikipedia text means fullmap categories
    would be noise: the mentions are company and place names, not biomedical concepts.

    Measured over the full hub files on wenceslaus (2026-09-22, probes ~/docred_probe.log and
    ~/docred_probe2.log; route: load_dataset("json", data_files=hf://datasets/thunlp/docred/
    data/<file>.json.gz), which transparently decompresses .gz):
    - rows: train_annotated 3,053; train_distant 101,873; dev 998; test 1,000. test rows carry NO
      labels key; 26/293/13 labeled-split docs carry none either, and those ship entities only.
    - mention pos is SENTENCE-RELATIVE and END-EXCLUSIVE: 77,515 exclusive vs 145 inclusive
      surface matches on train_annotated (the 145 are single-token mentions). Surfaces are
      therefore the token-span joins, never the annotator `name` string: 2.3-5.3% of names differ
      from the token join by punctuation spacing ("Worker-Peasant" vs ['Worker', '-', 'Peasant']),
      which is annotation style, not corruption, so a name mismatch never drops a mention.
    - out-of-bounds pos dropped: 149 of 79,481 annotated mentions (0.19%), 14,319 of 2,558,350
      distant (0.56%), 55 of 26,141 dev (0.21%), 36 of 26,704 test (0.13%).
    - entity types are a closed set of six: PER, ORG, LOC, MISC map to biolink classes; TIME and
      NUM have no honest biolink class and stay raw TitleCase ("Time", "Num"), the same
      deliberate-omission stance as PubmedAbstractsScript's unmappable MeSH headings.
    - labels carry keys h/t/r (NOT head/tail/relation): zero self-loops, zero bad keys, zero
      head-out-of-bounds, zero missing evidence; 96 distinct r values, every one covered by the
      dataset's own data/rel_info.json.gz, whose 96 P-id -> English name entries are frozen into
      RELATION_MAP below (census: uncovered=[] unused=[]).
    """

    NAME: ClassVar[str] = "DocredScript"

    # corpus type vocabulary -> biolink class (keys lowercased); the four honest classes are
    # import-validated so a typo fails loudly instead of mislabeling training data
    TYPE_MAP: ClassVar[dict[str, str]] = {
        "per": "Human",
        "org": "Agent",
        "loc": "GeographicLocation",
        "misc": "NamedThing",
    }

    # the dataset's own relation-name table, frozen verbatim from thunlp/docred
    # data/rel_info.json.gz (96 entries, measured 2026-09-22 on wenceslaus; the file is a
    # top-level dict so the json builder cannot project it as rows, hence the frozen copy).
    # Values resolve through ScriptUtils.resolve_predicate: Wikidata names are mostly non-biolink
    # and keep native snake_case (sentence_rex precedent: native names train zero-shot breadth).
    RELATION_MAP: ClassVar[dict[str, str]] = {
        "P1001": "applies to jurisdiction",
        "P102": "member of political party",
        "P1056": "product or material produced",
        "P108": "employer",
        "P112": "founded by",
        "P118": "league",
        "P1198": "unemployment rate",
        "P123": "publisher",
        "P127": "owned by",
        "P131": "located in the administrative territorial entity",
        "P1336": "territory claimed by",
        "P1344": "participant of",
        "P136": "genre",
        "P1365": "replaces",
        "P1366": "replaced by",
        "P137": "operator",
        "P1376": "capital of",
        "P140": "religion",
        "P1412": "languages spoken, written or signed",
        "P1441": "present in work",
        "P150": "contains administrative territorial entity",
        "P155": "follows",
        "P156": "followed by",
        "P159": "headquarters location",
        "P161": "cast member",
        "P162": "producer",
        "P166": "award received",
        "P17": "country",
        "P170": "creator",
        "P171": "parent taxon",
        "P172": "ethnic group",
        "P175": "performer",
        "P176": "manufacturer",
        "P178": "developer",
        "P179": "series",
        "P19": "place of birth",
        "P190": "sister city",
        "P194": "legislative body",
        "P20": "place of death",
        "P205": "basin country",
        "P206": "located in or next to body of water",
        "P22": "father",
        "P241": "military branch",
        "P25": "mother",
        "P26": "spouse",
        "P264": "record label",
        "P27": "country of citizenship",
        "P272": "production company",
        "P276": "location",
        "P279": "subclass of",
        "P30": "continent",
        "P31": "instance of",
        "P3373": "sibling",
        "P35": "head of state",
        "P355": "subsidiary",
        "P36": "capital",
        "P361": "part of",
        "P364": "original language of work",
        "P37": "official language",
        "P39": "position held",
        "P40": "child",
        "P400": "platform",
        "P403": "mouth of the watercourse",
        "P449": "original network",
        "P463": "member of",
        "P488": "chairperson",
        "P495": "country of origin",
        "P50": "author",
        "P527": "has part",
        "P54": "member of sports team",
        "P551": "residence",
        "P569": "date of birth",
        "P57": "director",
        "P570": "date of death",
        "P571": "inception",
        "P576": "dissolved, abolished or demolished",
        "P577": "publication date",
        "P58": "screenwriter",
        "P580": "start time",
        "P582": "end time",
        "P585": "point in time",
        "P6": "head of government",
        "P607": "conflict",
        "P674": "characters",
        "P676": "lyrics by",
        "P69": "educated at",
        "P706": "located on terrain feature",
        "P710": "participant",
        "P737": "influenced by",
        "P740": "location of formation",
        "P749": "parent organization",
        "P800": "notable work",
        "P807": "separated from",
        "P840": "narrative location",
        "P86": "composer",
        "P937": "work location",
    }

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """columns_out for this ingest are [sents, vertexSet, labels], so values is that triple.

        The token stream is the text source: sents flatten into one list, sentence-relative
        end-exclusive pos pairs become global end-inclusive token spans, and every emitted surface
        is a join over its own span, so text containment holds by construction (gliner2's
        InputExample.validate rule). A row whose sentences are not lists of strings corrupts every
        offset at once, so it ships as the empty example the pipeline filters; everything else
        degrades per mention or per relation (skip-don't-coerce).
        """
        sents_value, vertex_value, labels_value = values
        sents: list[Any] = sents_value if isinstance(sents_value, list) else []
        vertex: list[Any] = vertex_value if isinstance(vertex_value, list) else []
        labels: list[Any] = labels_value if isinstance(labels_value, list) else []
        flattened: list[list[str]] | None = _flatten_sents(sents)
        if flattened is None:
            return TrainingExample(text="")
        offsets = _sentence_offsets(flattened)
        tokens: list[str] = [token for sentence in flattened for token in sentence]

        # per cluster: the resolved mentions that survived, plus the first survivor, which is the
        # canonical surface a relation field cites (DocRED relations hold between clusters, and
        # the first mention is the document's earliest well-formed reference)
        first_valid: list[str | None] = []
        resolved: list[ResolvedMention] = []
        for cluster in vertex:
            cluster_surfaces: list[str] = []
            if not isinstance(cluster, list):
                first_valid.append(None)
                continue
            for mention in cluster:
                span = _mention_span(mention, offsets, len(tokens))
                if span is None:
                    continue
                surface = ScriptUtils.join_tokens(tokens[span[0] : span[1] + 1])
                category = self._category_of(mention)
                resolved.append(ResolvedMention(mention=surface, category=category, origin="raw"))
                if surface not in cluster_surfaces:
                    cluster_surfaces.append(surface)
            first_valid.append(cluster_surfaces[0] if cluster_surfaces else None)

        relations: list[Relation] = []
        for label in labels:
            relation = self._relation_of(label, first_valid)
            if relation is not None:
                relations.append(relation)

        return TrainingExample(
            text=ScriptUtils.join_tokens(tokens),
            entities=ScriptUtils.group_entities(resolved),
            relations=relations,
        )

    def _category_of(self: Self, mention: Any) -> str:
        """biolink class for a mapped type, TitleCase raw category otherwise (TIME and NUM land
        here by design; an unknown future type degrades the same way instead of dropping gold)"""
        raw_type = mention.get("type") if isinstance(mention, dict) else None
        if isinstance(raw_type, str):
            mapped = self.TYPE_MAP.get(raw_type.lower())
            if mapped is not None:
                return mapped
            return raw_type.title()
        return "NamedThing"

    def _relation_of(self: Self, label: Any, first_valid: list[str | None]) -> Relation | None:
        """one gold (head cluster, tail cluster, predicate) triple, or None to drop: malformed
        entries, self-loops, out-of-range indices, clusters with no surviving mention, and
        P-ids outside the frozen map all drop individually (measured census: 0 self-loops, 0
        head-oob, 96/96 r coverage -- these guards exist for drift, not for the observed data)"""
        if not isinstance(label, dict):
            return None
        head, tail, relation_id = label.get("h"), label.get("t"), label.get("r")
        if isinstance(head, bool) or isinstance(tail, bool) or not isinstance(head, int) or not isinstance(tail, int):
            return None
        if head == tail:
            return None
        if not (0 <= head < len(first_valid)) or not (0 <= tail < len(first_valid)):
            return None
        if not isinstance(relation_id, str):
            return None
        relation_name = self.RELATION_MAP.get(relation_id)
        if relation_name is None:
            return None
        head_surface, tail_surface = first_valid[head], first_valid[tail]
        if head_surface is None or tail_surface is None:
            return None
        predicate, _ = ScriptUtils.resolve_predicate(relation_name)
        return Relation(
            name=predicate,
            fields=[RelationField(name="head", value=head_surface), RelationField(name="tail", value=tail_surface)],
            description=ScriptUtils.predicate_description(predicate),
            evidence="asserted",
            negated=False,
        )


def _flatten_sents(sents: list[Any]) -> list[list[str]] | None:
    """str-list sentences out, None when any sentence is not a list of strings: a malformed
    sentence shifts every later offset, so the whole row's spans would be fabricated"""
    flat: list[list[str]] = []
    for sentence in sents:
        if not isinstance(sentence, list) or any(not isinstance(token, str) for token in sentence):
            return None
        flat.append(sentence)
    return flat


def _sentence_offsets(flat: list[list[str]]) -> list[tuple[int, int]]:
    """global [start, end_exclusive) of every sentence inside the flattened token stream"""
    offsets: list[tuple[int, int]] = []
    start = 0
    for sentence in flat:
        offsets.append((start, start + len(sentence)))
        start += len(sentence)
    return offsets


def _mention_span(mention: Any, offsets: list[tuple[int, int]], total_tokens: int) -> tuple[int, int] | None:
    """one mention dict -> its global end-inclusive token span, or None to drop (skip-don't-coerce).

    Guards, in order: the mention must be a dict; pos a 2-element list/tuple of non-bool ints with
    0 <= start < end; sent_id a non-bool int inside the sentence table; the span inside its
    sentence (b <= len(sentence) is the measured out-of-bounds case: 0.19-0.56% of mentions).
    """
    if not isinstance(mention, dict):
        return None
    pos = mention.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    start, end = pos
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start:
        return None
    sent_id = mention.get("sent_id")
    if isinstance(sent_id, bool) or not isinstance(sent_id, int) or not (0 <= sent_id < len(offsets)):
        return None
    sent_start, sent_end = offsets[sent_id]
    global_start = sent_start + start
    global_end = sent_start + end
    if global_end > sent_end or global_start >= total_tokens or global_end > total_tokens:
        return None
    return (global_start, global_end - 1)


for category in DocredScript.TYPE_MAP.values():
    validate_category(category, DocredScript.NAME)

from __future__ import annotations

from typing import Any

import pytest

from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser
from relmedner.models import FullmapTask, HuggingFaceDataset
from relmedner.registry import build_stream
from relmedner.streams import DataStream, rebuild_task
from tests.test_ingests import EXPECTED_REDDIT_MATCH

# the exact qualifying corpus set: inclusion bar is MIT license AND total_rows >= 100,000.
# reddit_dataset_226 sits in the same tensorshield family but has only 22,572 rows, so after the
# health-community filter its projected yield is noise; its absence here is the lock that it
# stays out of the pipeline even though the dataset id looks like a sibling of the seven kept
EXPECTED_REDDIT_DATASETS: frozenset[str] = frozenset(
    {
        "tensorshield/reddit_dataset_157",
        "tensorshield/reddit_dataset_30",
        "tensorshield/reddit_dataset_84",
        "tensorshield/reddit_dataset_171",
        "tensorshield/reddit_dataset_85",
        "tensorshield/reddit_dataset_217",
        "tensorshield/reddit_dataset_237",
    }
)


def reddit_datasets() -> dict[str, HuggingFaceDataset]:
    return {
        entry.dataset: entry
        for entry in YamlIngestsParser().parse_ingests().datasets
        if isinstance(entry, HuggingFaceDataset) and entry.dataset.startswith("tensorshield/reddit_dataset")
    }


@pytest.mark.parametrize("dataset", sorted(EXPECTED_REDDIT_DATASETS))
def test_every_reddit_entry_rebuilds_to_a_fullmap_task_projecting_text(dataset: str) -> None:
    """each reddit entry must stay a fullmap task over raw text: rebuild_task round-trips the
    frozen field-order tuple the pipeline passes to build_stream (a tuple-shape drift would break
    every consumer of generate_tuples()), and columns_out ('text',) locks the projection so a
    column rename in the upstream corpus fails here instead of silently mining empty rows"""
    Source, Payload = reddit_datasets()[dataset].to_tuple()
    Stream: DataStream = build_stream(Source, Payload)
    Task = rebuild_task(Payload[0])

    assert isinstance(Stream, HuggingFaceDataStream)
    assert isinstance(Task, FullmapTask)
    assert Task.to_tuple() == Payload[0]
    assert Stream.columns_out == ("text",)


def test_the_declared_reddit_set_is_exactly_the_seven_qualifying_corpora() -> None:
    """the parsed yaml must name exactly the seven corpora meeting the inclusion bar (MIT license
    and total_rows >= 100,000): this is the behavioral lock on WHICH reddit corpora the pipeline
    mines, so adding a row-barren sibling (like reddit_dataset_226), dropping a kept corpus, or
    renaming one all fail here rather than silently changing the mined data diet"""
    assert set(reddit_datasets()) == EXPECTED_REDDIT_DATASETS


def test_every_reddit_entry_filters_communityName_with_one_shared_allowlist() -> None:
    """all seven entries must filter one shared allowlist on column 'communityName' whose values
    equal tests.test_ingests.EXPECTED_REDDIT_MATCH: importing that tuple instead of duplicating
    the ~42 names keeps the two test files unable to drift apart, and the shared-tuple check
    proves the yaml anchor (*biomed_communities) still fans out identically to every entry --
    a per-entry edit would fork the allowlists and this catches it"""
    Allowlists: set[tuple[str, tuple[str, ...]]] = set()
    for Name in sorted(reddit_datasets()):
        Source, Payload = reddit_datasets()[Name].to_tuple()
        Stream: HuggingFaceDataStream = build_stream(Source, Payload)

        assert len(Stream.match_on) == 1
        Column, Values = Stream.match_on[0]
        assert Column == "communityName"
        assert Values == frozenset(EXPECTED_REDDIT_MATCH[0][1])
        Entry: HuggingFaceDataset = reddit_datasets()[Name]
        Allowlists.add((Entry.match_on[0].column, tuple(Entry.match_on[0].values)))

    assert Allowlists == {(EXPECTED_REDDIT_MATCH[0][0], EXPECTED_REDDIT_MATCH[0][1])}


def test_apply_match_keeps_allowlisted_communities_and_drops_everything_else() -> None:
    """apply_match is exact membership on the raw row value, so casing must be the Reddit
    canonical form: 'r/AskDocs' is kept while 'r/askdocs' (wrong case) and 'r/madmen' (a
    non-health community present in the real corpus) are dropped. The inverse lock proves the
    filter is not a no-op -- without it, an allowlist accidentally matching every row would pass
    every other test here and leak the full general-Reddit firehose into the miner"""
    Source, Payload = reddit_datasets()["tensorshield/reddit_dataset_157"].to_tuple()
    Stream: DataStream = build_stream(Source, Payload)
    Kept: dict[str, Any] = {"communityName": "r/AskDocs"}
    WrongCase: dict[str, Any] = {"communityName": "r/askdocs"}
    Dropped: dict[str, Any] = {"communityName": "r/madmen"}

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.apply_match(Kept) is True
    assert Stream.apply_match(WrongCase) is False
    assert Stream.apply_match(Dropped) is False

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import apache_beam as beam
from apache_beam.io.avroio import WriteToAvro
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.runner import PipelineResult

from relmedner.constants import MAX_BATCH_ROWS, MIN_BATCH_ROWS, OUTPUTS_MOUNT
from relmedner.dedup import apply_dedup, format_dedup_summary
from relmedner.fullmap_mine import FullmapMiner
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig, TrainingExample, YamlIngests
from relmedner.registry import build_stream
from relmedner.streams import DataStream, StreamedRow, rebuild_task
from relmedner.types import DispatchedExample, Script

logger = logging.getLogger(__name__)

FULLMAP_TYPE: str = "fullmap"


def stream_rows(stream: DataStream, config: RunConfig) -> Iterator[StreamedRow]:
    return stream.stream(config)


def weighted(example: TrainingExample, weight: float) -> TrainingExample:
    """stamp the source-declared mixing weight onto a frozen example (avro provenance)"""
    return example.model_copy(update={"weight": weight})


def dispatch_row(row: StreamedRow, weights: dict[str, float]) -> DispatchedExample:
    """script tasks dispatch through the registry; the leading task value discriminates"""
    source, (task, values) = row
    task_model = rebuild_task(task)
    outputs = tuple(task_model.outputs)
    outputs, example = Script.dispatch(task_model.name, (outputs, values))
    return (outputs, weighted(example, weights[source]))


def resolve_rows(rows: list[StreamedRow], weights: dict[str, float]) -> Iterator[DispatchedExample]:
    """one shared redb round trip per batch of fullmap rows (batched upstream by BatchElements)"""
    tasks = [rebuild_task(task) for _source, (task, _values) in rows]
    sources = [source for source, _payload in rows]
    pairs = [(values[0], task) for (_source, (_task, values)), task in zip(rows, tasks, strict=True)]
    for source, task, example in zip(sources, tasks, FullmapMiner.resolve_batch(pairs), strict=True):
        yield (tuple(task.outputs), weighted(example, weights[source]))


def matches_declared_outputs(dispatched: DispatchedExample) -> bool:
    """permitted-shapes contract: a row ships when it produced at least one declared shape.
    Subset semantics in both directions -- rows may produce fewer shapes than declared (87% of
    mined rows carry no gazetteer relation; the post-training corpus keeps its entity-only NER
    rows too) and may produce extra shapes (the [entities]-only pile-ner declaration keeps rows
    whose gazetteer also fired; relations are free signal, not a contract violation)."""
    outputs, example = dispatched
    return bool(example.populated() & frozenset(outputs))


def to_record(example: TrainingExample) -> dict[str, Any]:
    return example.asdict()


class BeamPipeline:
    def __init__(self: Self, options: PipelineOptions | None = None) -> None:
        self.options: PipelineOptions | None = options
        self.result: PipelineResult | None = None
        """the finished run's result, set after the pipeline context closes; US-005 reads the
        dedup counters from it for the DirectRunner summary line"""

    def run(self: Self, config: RunConfig) -> None:
        Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
        # per-source mixing weight stamped onto every record each source emits (avro provenance;
        # stock gliner2 has no per-example weight, so duplication happens at JSONL export)
        Weights: dict[str, float] = Ingests.weights_by_source()
        # Flink user code runs in the sdkworker, so external runs write to its durable output mount.
        # DirectRunner keeps honoring the caller's ordinary local path for development and unit tests.
        Output: Path = Path(config.output) if self.options is None else Path(OUTPUTS_MOUNT) / config.artifact_name()

        with beam.Pipeline(options=self.options) as new_pipeline:
            rows = (
                new_pipeline
                | "load declarative ingests" >> beam.Create(Ingests.stream_args())
                | "initialize datastream classes" >> beam.MapTuple(build_stream)
                | "stream declared data" >> beam.FlatMap(stream_rows, config=config)
            )
            # one branch per task type: script dispatch is per-row, fullmap mining needs
            # BatchElements so one redb round trip serves many documents
            script_rows, fullmap_rows = rows | "split by task type" >> beam.Partition(lambda row, count: 1 if row[1][0][0] == FULLMAP_TYPE else 0, 2)
            dispatched = (
                script_rows
                | "dispatch rows to declared scripts" >> beam.Map(dispatch_row, weights=Weights)
                | "keep examples matching declared outputs" >> beam.Filter(matches_declared_outputs)
            )
            mined = (
                fullmap_rows
                | "buffer fullmap rows into batches" >> beam.BatchElements(min_batch_size=MIN_BATCH_ROWS, max_batch_size=MAX_BATCH_ROWS)
                | "resolve mined batches" >> beam.FlatMap(resolve_rows, weights=Weights)
                | "keep mined examples matching declared outputs" >> beam.Filter(matches_declared_outputs)
            )
            merged = (dispatched, mined) | "merge task branches" >> beam.Flatten() | "drop the declared output key" >> beam.Values()
            # dedup sits between the merge and the avro write; OFF returns `merged` unchanged,
            # so the graph reproduces the pre-dedup pipeline exactly
            deduped = apply_dedup(merged, config.dedup_mode)
            (
                deduped
                | "shape examples into avro records" >> beam.Map(to_record)
                | "write training data to avro"
                >> WriteToAvro(
                    file_path_prefix=str(Output.with_suffix("")),
                    file_name_suffix=Output.suffix,
                    num_shards=1,
                    shard_name_template="",
                    schema=TrainingExample.avro_schema_to_python(),
                )
            )
        # the with-block runs the pipeline on exit and stashes the result on the Pipeline
        # object; retaining it here is what lets US-005 query metrics after the block closes
        self.result = new_pipeline.result
        # REQ-INT-4: report what dedup rejected on a local run only (the Flink runner keeps
        # the same counters in the job UI/REST instead). The `options is None` check mirrors
        # the output-path branch above, so a cluster run never reaches metrics() here and an
        # absent result/counters just skips the line, never crashing the run
        if self.options is None:
            summary = format_dedup_summary(self.result)
            if summary is not None:
                logger.info(summary)

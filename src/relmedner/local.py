from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Self

from fastavro import reader

from relmedner.streams import DataStream, StreamedRow


class LocalAvroDataStream(DataStream):
    """streams records out of a local avro container declared by path in ingests.yaml

    The hf streams pull their rows from the hub; this one reads a file the operator built
    out-of-band (the CTKP interventions extract), so a dataset that cannot live on the hub --
    licensing, size, or because it is rebuilt per AACT snapshot -- still rides the same
    declarative ingest path as everything else.
    """

    SOURCE: ClassVar[str] = "local"

    def __init__(self: Self, task: tuple[Any, ...], weight: float, path: str) -> None:
        # parameter order must match DatasetBase.to_tuple's field order, because build_stream
        # unpacks the declared payload positionally: task, weight, then the local-specific path
        self.task: tuple[Any, ...] = tuple(task)
        self.weight: float = weight
        # the pipeline stamps every row with weights[source], so this key is LocalDataset.row_key
        # verbatim: the declared path, not its basename (two distinct files may share a name and
        # must still be able to declare different weights)
        self.name: str = path
        self.path: str = path

    def rows(self: Self) -> Iterator[StreamedRow]:
        # the whole record ships as a single value so the receiving script owns the shape;
        # avro's reader is already lazy, so a 1M-record container never lands in memory at once
        with Path(self.path).expanduser().open("rb") as handle:
            for record in reader(handle):
                yield (self.name, (self.task, (record,)))

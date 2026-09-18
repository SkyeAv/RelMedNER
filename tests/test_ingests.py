from __future__ import annotations

from relmedner.ingests import YamlIngestsParser


def test_generate_tuples_shape() -> None:
    IngestsParser: YamlIngestsParser = YamlIngestsParser()
    generated: tuple[tuple[str, tuple[object, ...]], ...] = IngestsParser.generate_tuples()

    assert generated == (
        (
            "hf",
            (
                ("script", "GlinerBiomedScript", ("entities", "relations")),
                "anthonyyazdaniml/gliner-biomed-pre-training",
                None,
                "train",
                None,
                ("tokenized_text", "ner"),
            ),
        ),
    )

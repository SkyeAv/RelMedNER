import apache_beam as beam


def build_pipeline() -> None:
    with beam.Pipeline as new_pipeline:
        (
            new_pipeline
            | "" >> beam.Create()
        )

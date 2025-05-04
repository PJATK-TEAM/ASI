"""Project pipelines."""

from kedro.pipeline import Pipeline
from pneumonia_discovery.pipelines import data_preparation


def register_pipelines() -> dict[str, Pipeline]:
    return {"__default__": data_preparation.create_pipeline()}

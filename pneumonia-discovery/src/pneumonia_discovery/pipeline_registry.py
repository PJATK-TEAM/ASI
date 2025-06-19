"""Project pipelines."""

from kedro.pipeline import Pipeline
from pneumonia_discovery.pipelines import data_preparation, training

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_preparation": data_preparation.create_pipeline(),
        "training": training.create_pipeline(),
        "__default__": data_preparation.create_pipeline() + training.create_pipeline(),
    }
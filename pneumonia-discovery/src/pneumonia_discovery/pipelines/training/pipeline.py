# pneumonia_discovery/pipelines/training/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_autogluon_model, train_autogluon_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_autogluon_model,
            inputs=["preprocessed_train_data", "preprocessed_val_data"],
            outputs="autogluon_model",
            name="train_autogluon_model_node"
        ),
        node(
            func=evaluate_autogluon_model,
            inputs=["autogluon_model", "preprocessed_test_data"],
            outputs="test_accuracy",
            name="evaluate_autogluon_model_node"
        )
    ])
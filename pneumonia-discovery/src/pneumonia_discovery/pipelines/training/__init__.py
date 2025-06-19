# pneumonia_discovery/pipelines/training/__init__.py

from .pipeline import create_pipeline  # <- to umożliwia import training_pipeline

__all__ = ["create_pipeline"]
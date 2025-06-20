from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (standardize_orientation,
                    load_images_from_directory,
                    preprocess_data,
                    normalize_images)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_images_from_directory,
                inputs="params:raw_train_normal_path",
                outputs="raw_train_normal",
                name="load_raw_train_normal_images",
            ),
            node(
                func=load_images_from_directory,
                inputs="params:raw_train_pneumonia_path",
                outputs="raw_train_pneumonia",
                name="load_raw_train_pneumonia_images",
            ),
            node(
                func=load_images_from_directory,
                inputs="params:raw_val_normal_path",
                outputs="raw_val_normal",
                name="load_raw_val_normal_images",
            ),
            node(
                func=load_images_from_directory,
                inputs="params:raw_val_pneumonia_path",
                outputs="raw_val_pneumonia",
                name="load_raw_val_pneumonia_images",
            ),
            node(
                func=load_images_from_directory,
                inputs="params:raw_test_normal_path",
                outputs="raw_test_normal",
                name="load_raw_test_normal_images",
            ),
            node(
                func=load_images_from_directory,
                inputs="params:raw_test_pneumonia_path",
                outputs="raw_test_pneumonia",
                name="load_raw_test_pneumonia_images",
            ),
            node(
                func=preprocess_data,
                inputs=["raw_train_normal", "raw_train_pneumonia"],
                outputs="basic_preprocessed_train_data",
                name="basic_preprocess_train_data_node",
            ),
            node(
                func=preprocess_data,
                inputs=["raw_val_normal", "raw_val_pneumonia"],
                outputs="basic_preprocessed_val_data",
                name="basic_preprocess_val_data_node",
            ),
            node(
                func=preprocess_data,
                inputs=["raw_test_normal", "raw_test_pneumonia"],
                outputs="basic_preprocessed_test_data",
                name="basic_preprocess_test_data_node",
            ),

            node(
                func=standardize_orientation,
                inputs="basic_preprocessed_train_data",
                outputs="train_std_orientation",
                name="train_standardize_orientation_node",
            ),
            node(
                func=normalize_images,
                inputs="train_std_orientation",
                outputs="preprocessed_train_data",
                name="train_normalize_images_node",
            ),

            #
            node(
                func=standardize_orientation,
                inputs="basic_preprocessed_val_data",
                outputs="val_std_orientation",
                name="val_standardize_orientation_node",
            ),
            node(
                func=normalize_images,
                inputs="val_std_orientation",
                outputs="preprocessed_val_data",
                name="val_normalize_images_node",
            ),

            node(
                func=standardize_orientation,
                inputs="basic_preprocessed_test_data",
                outputs="test_std_orientation",
                name="test_standardize_orientation_node",
            ),
            node(
                func=normalize_images,
                inputs="test_std_orientation",
                outputs="preprocessed_test_data",
                name="test_normalize_images_node",
            )
        ]
    )

import datetime
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
import os
import numpy as np
from typing import Dict, Any

def train_autogluon_model(train_data: Dict[str, Any], val_data: Dict[str, Any], output_path: str = "autogluon_model") -> MultiModalPredictor:
    os.makedirs("autogluon_train_images", exist_ok=True)

    def save_images(images, filenames, prefix):
        paths = []
        for img, fname in zip(images, filenames):
            path = os.path.join("autogluon_train_images", prefix + "_" + fname)
            img_uint8 = (img * 255).astype(np.uint8)
            if img.shape[-1] == 1:
                img_uint8 = img_uint8[:, :, 0]  # remove channel dimension
            import cv2
            cv2.imwrite(path, img_uint8)
            paths.append(path)
        return paths

    train_paths = save_images(train_data["images"], train_data["filenames"], "train")
    val_paths = save_images(val_data["images"], val_data["filenames"], "val")

    df_train = pd.DataFrame({
        "image": train_paths,
        "label": train_data["labels"]
    })

    df_val = pd.DataFrame({
        "image": val_paths,
        "label": val_data["labels"]
    })

    #output_path = f"autogluon_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
    
    predictor = MultiModalPredictor(label="label", problem_type="classification", eval_metric="accuracy")
    predictor.fit(
        train_data=df_train,
        tuning_data=df_val,
        time_limit=300,  # np. 5 minut
        presets="medium_quality",
        save_path=output_path
    )

    return predictor

def evaluate_autogluon_model(predictor: MultiModalPredictor, test_data: Dict[str, Any]) -> float:
    import pandas as pd
    import os

    os.makedirs("autogluon_test_images", exist_ok=True)

    paths = []
    for img, fname in zip(test_data["images"], test_data["filenames"]):
        path = os.path.join("autogluon_test_images", fname)
        img_uint8 = (img * 255).astype(np.uint8)
        if img.shape[-1] == 1:
            img_uint8 = img_uint8[:, :, 0]
        import cv2
        cv2.imwrite(path, img_uint8)
        paths.append(path)

    df_test = pd.DataFrame({
        "image": paths,
        "label": test_data["labels"]
    })

    score = predictor.evaluate(df_test)
    print(f"Test Accuracy: {score['accuracy']:.4f}")
    return score["accuracy"]

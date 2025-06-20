import os
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, filters


def load_images_from_directory(directory_path: str) -> Dict[str, Any]:
    images = []
    filenames = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                with Image.open(file_path) as img:
                    img = img.resize((224, 224)).convert("RGB")
                    images.append(np.array(img))
                    filenames.append(file_name)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return {"images": np.array(images), "filenames": filenames}

def preprocess_data(raw_normal: Dict[str, Any], raw_pneumonia: Dict[str, Any]) -> Dict[str, Any]:
    images = np.concatenate([raw_normal["images"], raw_pneumonia["images"]], axis=0)
    labels = [0] * len(raw_normal["images"]) + [1] * len(raw_pneumonia["images"])
    filenames = raw_normal["filenames"] + raw_pneumonia["filenames"]
    return {"images": images, "labels": labels, "filenames": filenames}

def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    elif img.shape[-1] == 1:
        return img[:, :, 0]
    else:
        return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

def standardize_orientation(data: Dict[str, Any]) -> Dict[str, Any]:
    images = data["images"]
    standardized_images = np.zeros_like(images)

    for i in range(len(images)):
        img = images[i]
        gray = convert_to_grayscale(img)

        left_side = gray[:, :gray.shape[1]//4]
        right_side = gray[:, 3*gray.shape[1]//4:]

        top_left = left_side[:left_side.shape[0]//4]
        top_right = right_side[:right_side.shape[0]//4]

        if np.mean(top_left) > np.mean(top_right):
            standardized_images[i] = np.flip(img, axis=1)
        else:
            standardized_images[i] = img

    return {
        "images": standardized_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }

def normalize_images(data: Dict[str, Any]) -> Dict[str, Any]:
    result = data.copy()
    images = data['images']

    normalized_images = images.astype(np.float32) / 255.0

    normalized_images = np.clip(normalized_images, 0, 1)

    result['images'] = normalized_images
    return result

def create_dataframe_from_preprocessed(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Tworzy DataFrame z przetworzonych danych.
    
    Args:
        data: Słownik z kluczami 'filenames' i 'labels'.
    
    Returns:
        pd.DataFrame z kolumnami 'filename' i 'label'.
    """
    filenames = data.get("filenames", [])
    labels = data.get("labels", [])

    df = pd.DataFrame({
        "filename": filenames,
        "label": labels
    })

    return df

def save_images_to_disk(data: dict, output_dir: str) -> dict:
    import os
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    images = data.get("images")
    filenames = data.get("filenames")

    for img, fname in zip(images, filenames):
        if img.dtype == np.float32 or img.dtype == np.float64:
            img_to_save = (img * 255).astype(np.uint8)
        else:
            img_to_save = img

        if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, img_to_save)

    print(f"Saved {len(filenames)} images to {output_dir}")

    return data
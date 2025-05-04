from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from skimage import exposure, filters

def preprocess_train_data(raw_train_normal: List[np.ndarray], raw_train_pneumonia: List[np.ndarray]) -> Dict[str, Any]:
    images = np.concatenate([np.array(raw_train_normal), np.array(raw_train_pneumonia)], axis=0)
    labels = [0] * len(raw_train_normal) + [1] * len(raw_train_pneumonia)
    return {"images": images, "labels": labels}

def preprocess_val_data(raw_val_normal: List[np.ndarray], raw_val_pneumonia: List[np.ndarray]) -> Dict[str, Any]:
    images = np.concatenate([np.array(raw_val_normal), np.array(raw_val_pneumonia)], axis=0)
    labels = [0] * len(raw_val_normal) + [1] * len(raw_val_pneumonia)
    return {"images": images, "labels": labels}

def preprocess_test_data(raw_test_normal: List[np.ndarray], raw_test_pneumonia: List[np.ndarray]) -> Dict[str, Any]:
    images = np.concatenate([np.array(raw_test_normal), np.array(raw_test_pneumonia)], axis=0)
    labels = [0] * len(raw_test_normal) + [1] * len(raw_test_pneumonia)
    return {"images": images, "labels": labels}

def convert_to_grayscale(img: Any) -> np.ndarray:
    if img.shape[-1] == 1:
        return (img[:, :, 0] * 255).astype(np.uint8)
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

def standardize_orientation(data: Dict[str, Any]) -> Dict[str, Any]:
    images = data["images"]
    standardized_images = np.zeros_like(images)

    for i in range(images.shape[0]):
        img = images[i]
        gray = convert_to_grayscale(img)

        left_side = gray[:, :gray.shape[1]//4]
        right_side = gray[:, 3*gray.shape[1]//4]

        top_left = left_side[:left_side.shape[0] // 4, :]
        top_right = right_side[:right_side.shape[0] // 4, :]

        if np.mean(top_left) > np.mean(top_right):
            standardized_images[i] = np.flip(img, axis=1)
        else:
            standardized_images[i] = img

    return {
        "images": standardized_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }

def crop_to_lungs(data: Dict[str, Any], margin: int = 10) -> Dict[str, Any]:
    images = data["images"]
    cropped_images = []

    for i in range(images.shape[0]):
        img = images[i]
        gray = convert_to_grayscale(img)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary

        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(gray.shape[1] - x, w + 2 * margin)
            h = min(gray.shape[0] - y, h + 2 * margin)

            if img.shape[-1] == 1:
                cropped = img[y:y + h, x:x + w, :]
            else:
                cropped = img[y:y + h, x:x + w, :]

            cropped = cv2.resize(cropped, (img.shape[1], img.shape[0]))
            cropped_images.append(cropped)
        else:
            cropped_images.append(img)

    cropped_images = np.array(cropped_images)

    return {
        "images": cropped_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def apply_lung_segmentation(data: Dict[str, Any]) -> Dict[str, Any]:
    images = data["images"]
    segmented_images = np.zeros_like(images)

    for i in range(images.shape[0]):
        img = images[i]
        gray = convert_to_grayscale(img)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(binary) > 127:
            binary = 255 - binary

        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                cv2.drawContours(mask, [contour], 0, 255, -1)

        segmented = cv2.bitwise_and(gray, gray, mask=mask)

        if img.shape[-1] == 1:
            segmented_images[i, :, :, 0] = segmented / 255.0
        else:
            for c in range(img.shape[-1]):
                segmented_images[i, :, :, c] = segmented / 255.0

    return {
        "images": segmented_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def enhance_contrast(data: Dict[str, Any], clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Dict[
    str, Any]:
    images = data["images"]
    enhanced_images = np.zeros_like(images)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    for i in range(images.shape[0]):
        img = images[i]

        if img.shape[-1] == 1:
            enhanced = clahe.apply((img[:, :, 0] * 255).astype(np.uint8))
            enhanced_images[i, :, :, 0] = enhanced / 255.0
        else:
            for c in range(img.shape[-1]):
                enhanced = clahe.apply((img[:, :, c] * 255).astype(np.uint8))
                enhanced_images[i, :, :, c] = enhanced / 255.0

    return {
        "images": enhanced_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def remove_text_markers(data: Dict[str, Any]) -> Dict[str, Any]:
    images = data["images"]
    cleaned_images = np.copy(images)

    for i in range(images.shape[0]):
        img = images[i]
        gray = convert_to_grayscale(img)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)

            if area < 200 and (aspect_ratio > 0.5 and aspect_ratio < 3.0):
                cv2.drawContours(mask, [contour], 0, 255, -1)

        inpainted = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

        if img.shape[-1] == 1:
            cleaned_images[i, :, :, 0] = inpainted / 255.0
        else:
            for c in range(img.shape[-1]):
                channel = (img[:, :, c] * 255).astype(np.uint8)
                inpainted_channel = cv2.inpaint(channel, mask, 3, cv2.INPAINT_TELEA)
                cleaned_images[i, :, :, c] = inpainted_channel / 255.0

    return {
        "images": cleaned_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def correct_illumination(data: Dict[str, Any], filter_size: int = 31) -> Dict[str, Any]:
    images = data["images"]
    corrected_images = np.zeros_like(images)

    for i in range(images.shape[0]):
        img = images[i]

        if img.shape[-1] == 1:
            img_8bit = (img[:, :, 0] * 255).astype(np.uint8)

            background = cv2.GaussianBlur(img_8bit, (filter_size, filter_size), 0)

            corrected = img_8bit.astype(np.float32) - background.astype(np.float32)
            corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)

            corrected = np.clip(corrected, 0, 1)
            corrected_images[i, :, :, 0] = corrected
        else:
            for c in range(img.shape[-1]):
                img_channel = (img[:, :, c] * 255).astype(np.uint8)
                background = cv2.GaussianBlur(img_channel, (filter_size, filter_size), 0)
                corrected = img_channel.astype(np.float32) - background.astype(np.float32)
                corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)
                corrected = np.clip(corrected, 0, 1)
                corrected_images[i, :, :, c] = corrected

    return {
        "images": corrected_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def normalize_histogram(data: Dict[str, Any]) -> Dict[str, Any]:
    images = data["images"]
    normalized_images = np.zeros_like(images)

    for i in range(images.shape[0]):
        img = images[i]

        if img.shape[-1] == 1:
            normalized_images[i, :, :, 0] = exposure.equalize_hist(img[:, :, 0])
        else:
            for c in range(img.shape[-1]):
                normalized_images[i, :, :, c] = exposure.equalize_hist(img[:, :, c])

    return {
        "images": normalized_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def apply_edge_enhancement(data: Dict[str, Any], sigma: float = 1.0) -> Dict[str, Any]:
    images = data["images"]
    enhanced_images = np.zeros_like(images)

    for i in range(images.shape[0]):
        img = images[i]

        if img.shape[-1] == 1:
            blurred = filters.gaussian(img[:, :, 0], sigma=sigma)
            enhanced = img[:, :, 0] + (img[:, :, 0] - blurred)
            enhanced = np.clip(enhanced, 0, 1)
            enhanced_images[i, :, :, 0] = enhanced
        else:
            for c in range(img.shape[-1]):
                blurred = filters.gaussian(img[:, :, c], sigma=sigma)
                enhanced = img[:, :, c] + (img[:, :, c] - blurred)
                enhanced = np.clip(enhanced, 0, 1)
                enhanced_images[i, :, :, c] = enhanced

    return {
        "images": enhanced_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }


def normalize_by_reference(data: Dict[str, Any]) -> Dict[str, Any]:
    images = data["images"]
    normalized_images = np.zeros_like(images)

    for i in range(images.shape[0]):
        img = images[i]

        if img.shape[-1] == 1:
            img_8bit = (img[:, :, 0] * 255).astype(np.uint8)

            height, width = img_8bit.shape
            spine_region = img_8bit[height // 4:3 * height // 4, width // 2 - 20:width // 2 + 20]

            reference_intensity = np.mean(spine_region)

            if reference_intensity > 0:
                normalized = img[:, :, 0] * (128 / reference_intensity)
                normalized = np.clip(normalized, 0, 1)
                normalized_images[i, :, :, 0] = normalized
            else:
                normalized_images[i, :, :, 0] = img[:, :, 0]
        else:
            for c in range(img.shape[-1]):
                img_channel = (img[:, :, c] * 255).astype(np.uint8)
                height, width = img_channel.shape
                spine_region = img_channel[height // 4:3 * height // 4, width // 2 - 20:width // 2 + 20]
                reference_intensity = np.mean(spine_region)

                if reference_intensity > 0:
                    normalized = img[:, :, c] * (128 / reference_intensity)
                    normalized = np.clip(normalized, 0, 1)
                    normalized_images[i, :, :, c] = normalized
                else:
                    normalized_images[i, :, :, c] = img[:, :, c]

    return {
        "images": normalized_images,
        "labels": data["labels"],
        "filenames": data["filenames"]
    }
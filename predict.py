"""
Predict whether a single image is a Cat or a Dog using the trained SVM model.

Usage:
    python predict.py --image path/to/image.jpg [--model svm_model.pkl]

Author : Pruthviraj
Date   : February 2026
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
from skimage.feature import hog
import joblib

# Must match training settings
IMG_SIZE = 64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

LABEL_MAP = {0: "Cat 🐱", 1: "Dog 🐶"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Cat or Dog from a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm_model.pkl",
        help="Path to the saved model file (default: svm_model.pkl).",
    )
    return parser.parse_args()


def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize, convert to grayscale, and extract HOG features."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    features = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
    )
    return features.reshape(1, -1)


def main():
    args = parse_args()

    # Load model + scaler
    try:
        bundle = joblib.load(args.model)
        model = bundle["model"]
        scaler = bundle["scaler"]
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {args.model}")
        print("  Train the model first:  python svm_cats_dogs.py")
        sys.exit(1)

    # Preprocess & predict
    features = preprocess_image(args.image)
    features = scaler.transform(features)
    prediction = model.predict(features)[0]

    print("=" * 40)
    print(f"  Image     : {args.image}")
    print(f"  Prediction: {LABEL_MAP[prediction]}")
    print("=" * 40)


if __name__ == "__main__":
    main()

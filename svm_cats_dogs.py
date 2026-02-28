"""
SVM Cats vs Dogs Image Classifier
===================================
Trains a Support Vector Machine (SVM) on HOG features extracted from the
Kaggle Dogs vs Cats dataset.

Usage:
    python svm_cats_dogs.py [--data_dir DATASET_PATH] [--max_samples N]

Author : Pruthviraj
Date   : February 2026
"""

from __future__ import annotations

import os
import argparse
import time
import warnings

import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
IMG_SIZE = 64          # Resize images to 64×64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
RANDOM_STATE = 42


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an SVM classifier on the Cats vs Dogs dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/training_set/training_set",
        help="Path to the training data directory containing 'cats' and 'dogs' sub-folders.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of images to load per class (useful for quick testing).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction of data reserved for testing (default: 0.20).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="svm_model.pkl",
        help="File path to save the trained model.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────
def load_images(data_dir: str, max_samples: int | None = None):
    """
    Load images from the dataset directory.

    Expected structure:
        data_dir/
            cats/
                cat.0.jpg
                ...
            dogs/
                dog.0.jpg
                ...

    Returns
    -------
    images : list of np.ndarray   — grayscale images resized to IMG_SIZE×IMG_SIZE
    labels : list of int          — 0 = Cat, 1 = Dog
    """
    categories = {"cats": 0, "dogs": 1}
    images, labels = [], []

    for category, label in categories.items():
        folder = os.path.join(data_dir, category)
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        file_list = os.listdir(folder)
        if max_samples is not None:
            file_list = file_list[:max_samples]

        print(f"Loading {category} images from: {folder}")
        for fname in tqdm(file_list, desc=f"  {category}"):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # skip corrupt / unreadable files
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    return images, labels


# ──────────────────────────────────────────────
#  Feature Extraction (HOG)
# ──────────────────────────────────────────────
def extract_hog_features(images: list[np.ndarray]) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from a list of
    grayscale images.

    Returns
    -------
    features : np.ndarray of shape (n_samples, n_features)
    """
    features = []
    print("\nExtracting HOG features...")
    for img in tqdm(images, desc="  HOG"):
        feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
        )
        features.append(feat)
    return np.array(features)


# ──────────────────────────────────────────────
#  Plotting Helpers
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_sample_predictions(X_test_imgs, y_true, y_pred, save_path="sample_predictions.png", n=10):
    """Save a grid of sample predictions."""
    indices = np.random.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    label_map = {0: "Cat", 1: "Dog"}

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, idx in zip(axes.flatten(), indices):
        ax.imshow(X_test_imgs[idx], cmap="gray")
        true_label = label_map[y_true[idx]]
        pred_label = label_map[y_pred[idx]]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample predictions saved to {save_path}")


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    start_time = time.time()

    # ── 1. Load images ──────────────────────
    print("=" * 60)
    print("   SVM Cats vs Dogs Classifier")
    print("=" * 60)
    images, labels = load_images(args.data_dir, args.max_samples)
    if len(images) == 0:
        print("\n[ERROR] No images loaded. Check --data_dir path.")
        print(f"  Expected structure: {args.data_dir}/cats/ and {args.data_dir}/dogs/")
        return

    print(f"\nTotal images loaded: {len(images)}")
    print(f"  Cats: {labels.count(0)}  |  Dogs: {labels.count(1)}")

    # ── 2. Extract HOG features ─────────────
    features = extract_hog_features(images)
    print(f"Feature vector length: {features.shape[1]}")

    # ── 3. Train/Test split ─────────────────
    labels = np.array(labels)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        features, labels, np.arange(len(labels)),
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"\nTrain set: {X_train.shape[0]}  |  Test set: {X_test.shape[0]}")

    # ── 4. Feature scaling ──────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── 5. SVM Training with GridSearchCV ───
    print("\nTraining SVM with GridSearchCV (this may take a few minutes)...")
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf"],
    }
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

    # ── 6. Evaluation ───────────────────────
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    # ── 7. Plots ────────────────────────────
    plot_confusion_matrix(y_test, y_pred)

    test_images = [images[i] for i in idx_test]
    plot_sample_predictions(test_images, y_test, y_pred)

    # ── 8. Save model + scaler ──────────────
    joblib.dump({"model": best_model, "scaler": scaler}, args.model_path)
    print(f"\nModel + scaler saved to: {args.model_path}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

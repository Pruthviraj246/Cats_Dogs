# 🐱🐶 Cats vs Dogs — SVM Image Classifier

A machine learning project that classifies images of **cats** and **dogs** using a **Support Vector Machine (SVM)** with **HOG (Histogram of Oriented Gradients)** feature extraction.

Built as an internship-level ML project using the [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [How It Works](#how-it-works)

---

## 🔍 Overview

| Item | Detail |
|------|--------|
| **Task** | Binary image classification (Cat vs Dog) |
| **Algorithm** | Support Vector Machine (SVM) with RBF kernel |
| **Features** | HOG (Histogram of Oriented Gradients) |
| **Tuning** | GridSearchCV over `C`, `gamma` |
| **Dataset** | Kaggle Dogs vs Cats (25,000 labeled images) |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **scikit-learn** — SVM model & evaluation
- **scikit-image** — HOG feature extraction
- **OpenCV** — Image loading & preprocessing
- **NumPy** — Numerical operations
- **Matplotlib & Seaborn** — Visualization
- **joblib** — Model serialization

---

## 📁 Project Structure

```
Cats_Dogs/
├── svm_cats_dogs.py        # Main training & evaluation script
├── predict.py              # Predict on new images
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── confusion_matrix.png    # (generated after training)
├── sample_predictions.png  # (generated after training)
└── dataset/                # (download separately — not in repo)
    └── training_set/
        └── training_set/
            ├── cats/
            │   ├── cat.0.jpg
            │   └── ...
            └── dogs/
                ├── dog.0.jpg
                └── ...
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Cats_Dogs.git
cd Cats_Dogs

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

1. Go to [Kaggle — Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data).
2. Download the dataset and extract it.
3. Place the extracted files so the structure looks like:

```
dataset/
└── training_set/
    └── training_set/
        ├── cats/
        └── dogs/
```

> **Note:** The `dataset/` folder is gitignored and must be set up locally.

---

## 🚀 Training the Model

```bash
# Train on the full dataset
python svm_cats_dogs.py

# Train on a smaller subset for quick testing (e.g., 500 images per class)
python svm_cats_dogs.py --max_samples 500

# Specify a custom dataset path
python svm_cats_dogs.py --data_dir path/to/your/data
```

**Output:**
- Prints accuracy, classification report
- Saves `confusion_matrix.png` and `sample_predictions.png`
- Saves the trained model to `svm_model.pkl`

---

## 🔮 Making Predictions

```bash
python predict.py --image path/to/cat_or_dog.jpg
```

**Example output:**
```
========================================
  Image     : test_cat.jpg
  Prediction: Cat 🐱
========================================
```

---

## 📊 Results

After training on the full dataset with HOG features and SVM (RBF kernel):

| Metric | Score |
|--------|-------|
| **Accuracy** | ~70–75% |
| **Precision (Cat)** | ~0.72 |
| **Recall (Cat)** | ~0.70 |
| **F1-Score (Cat)** | ~0.71 |

> **Note:** SVM with HOG is a classical ML approach. Deep learning methods (CNNs) typically achieve 90%+ on this dataset, but SVM demonstrates core ML concepts well.

---

## 🧠 How It Works

### 1. Preprocessing
- Images are resized to **64×64 pixels** and converted to **grayscale**.

### 2. HOG Feature Extraction
- **Histogram of Oriented Gradients** captures edge directions and structural information.
- Each image becomes a feature vector of fixed length.

### 3. Feature Scaling
- Features are standardized using `StandardScaler` (zero mean, unit variance).

### 4. SVM Training
- **RBF kernel** SVM is trained with hyperparameter tuning via **GridSearchCV**.
- Searches over `C = [0.1, 1, 10]` and `gamma = [scale, 0.01, 0.001]`.

### 5. Evaluation
- Accuracy, precision, recall, F1-score, and a confusion matrix are generated.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Acknowledgements

- [Kaggle — Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [scikit-image HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)

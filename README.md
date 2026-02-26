# Cat vs. Dog Image Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
  <a href="https://colab.research.google.com/drive/1VhiSW0U2YtVFtPiLbtTttm5Pb1ib5Nrf#scrollTo=nsl2AgTJIXk-">
    <img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  </a>
</p>

<p align="center">
  A production-ready binary image classification system using a custom Convolutional Neural Network (CNN) trained to distinguish between cats and dogs. Features an iterative model improvement pipeline with data augmentation, batch normalization, dropout regularization, and early stopping — reducing validation loss from <strong>0.7 → 0.3</strong> and improving accuracy by <strong>+2%</strong>.
</p>

---

## Demo

![App Screenshot](assets/demo.png)

> Built with **Streamlit** — upload any image and the model instantly classifies it as a cat or dog with a confidence score. Sample images are provided on the left panel for quick testing.

---

## Quick Access

| Resource | Link |
|----------|------|
| 📦 **Dataset (Google Drive)** | [Download Dataset](https://drive.google.com/drive/folders/1kBpVnKEFqxCNAWGfo-yXcRR7Nwd55i3p?usp=drive_link) |
| 📓 **Google Colab Notebook** | [Open in Colab](https://colab.research.google.com/drive/1VhiSW0U2YtVFtPiLbtTttm5Pb1ib5Nrf#scrollTo=nsl2AgTJIXk-) |

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Inference](#inference)
- [Running the App](#running-the-app)
- [Known Issues & Notes](#known-issues--notes)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project implements a full end-to-end deep learning pipeline for binary image classification. Two CNN models were developed and compared:

| Model | Description |
|-------|-------------|
| **Baseline CNN** | 3-block Conv2D architecture, trained for 10 epochs |
| **Augmented CNN** | Same backbone + data augmentation pipeline + EarlyStopping, trained up to 20 epochs |

The augmented model demonstrates significantly better generalization on unseen data, with validation loss reduced by more than half.

---

## Project Structure

```
PROJECT_ML/
├── .venv/                    # Isolated virtual environment
├── dataset/                  # Raw image dataset
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
├── model/                    # Persisted model artifacts
│   └── cat_dog_model.h5      # Saved augmented model (HDF5)
├── noteboooks/               # Experimental Jupyter notebooks
│   └── sample_data/          # Sample images used during prototyping
├── src/                      # Production source modules
│   ├── model_loader.py       # Model loading and initialization
│   ├── predict.py            # Inference pipeline
│   └── preprocess.py         # Image preprocessing utilities
├── app.py                    # Application entry point
└── requirement.txt           # Python dependencies
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PROJECT_ML.git
cd PROJECT_ML
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

**Core dependencies:**

```
tensorflow>=2.10
opencv-python
matplotlib
numpy
```

---

## Dataset

Images are organized by class in separate directories and loaded using `keras.utils.image_dataset_from_directory`.

```
dataset/
├── train/
│   ├── cats/     ← training cat images
│   └── dogs/     ← training dog images
└── test/
    ├── cats/     ← validation cat images
    └── dogs/     ← validation dog images
```

**Preprocessing applied:**
- Resized to **256 × 256 pixels**
- Pixel values normalized to `[0.0, 1.0]` by dividing by `255`
- Batched in groups of **32**
- Labels inferred automatically from directory names

---

## Model Architecture

Both models share the following CNN backbone:

```
Input (256, 256, 3)
│
├── Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPooling(2×2)
├── Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPooling(2×2)
├── Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPooling(2×2)
│
├── Flatten
├── Dense(128, ReLU) → Dropout(0.1)
├── Dense(64, ReLU)  → Dropout(0.1)
└── Dense(1, Sigmoid)   ← Binary output: 0 = Cat, 1 = Dog
```

**Loss function:** Binary Cross-Entropy  
**Optimizer:** Adam  
**Output activation:** Sigmoid (threshold = 0.5)

### Augmented Model Additions

The augmented model prepends an online augmentation block to the same architecture:

```python
keras.layers.RandomFlip("horizontal")
keras.layers.RandomRotation(0.2)
keras.layers.RandomZoom(0.2)
```

Dropout rates are raised to `0.2` in the augmented model to complement the regularization effect of augmentation.

---

## Training Pipeline

### Baseline Model

```python
model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

### Augmented Model with Early Stopping

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_aug.fit(
    train_ds,
    epochs=20,
    validation_data=validation_ds,
    callbacks=[early_stopping]
)
```

Early stopping prevents overfitting by halting training when validation loss stops improving, and automatically restores the weights from the best epoch.

---

## Results

| Metric | Baseline Model | Augmented Model | Δ Change |
|--------|:--------------:|:---------------:|:--------:|
| Validation Loss | 0.7 | **0.3** | ↓ 57% |
| Accuracy | baseline | **+2%** | ↑ improved |

The combination of data augmentation, increased dropout, and early stopping significantly improved the model's ability to generalize to unseen images.

---

## Inference

```python
import cv2
import numpy as np
from src.model_loader import load_model

model = load_model("model/cat_dog_model.h5")
class_names = ["cat", "dog"]

# Load and preprocess
img = cv2.imread("path/to/image.jpg")
img = cv2.resize(img, (256, 256))
img = img / 255.0
input_tensor = img.reshape((1, 256, 256, 3))

# Predict
prediction = model.predict(input_tensor)
label = class_names[1] if prediction[0][0] > 0.5 else class_names[0]
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")
```

> ⚠️ **Important:** Always normalize input images (`/ 255.0`) before inference to match the preprocessing applied during training.

---

## Running the App

```bash
python app.py
```

Refer to `app.py` for configurable parameters such as model path and input image directory.

---

## Known Issues & Notes

- The `.h5` format is supported but TensorFlow recommends migrating to the newer **SavedModel** format (`.keras`) for TF 2.x compatibility going forward.
- Ensure input images are normalized consistently at inference — skipping this step will lead to degraded or incorrect predictions.
- The `noteboooks/` directory name contains a typo (double `o`) — consider renaming to `notebooks/` for consistency.

---

## Future Improvements

- [ ] Migrate model serialization to `.keras` format
- [ ] Add transfer learning using **MobileNetV2** or **EfficientNetB0** for higher accuracy
- [ ] Integrate a REST API using **FastAPI** or **Flask** for serving predictions
- [ ] Add unit tests for `preprocess.py`, `predict.py`, and `model_loader.py`
- [ ] Containerize the application with **Docker**
- [ ] Set up a CI/CD pipeline with **GitHub Actions**
- [ ] Add model versioning with **MLflow** or **DVC**

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with TensorFlow & Keras</p>

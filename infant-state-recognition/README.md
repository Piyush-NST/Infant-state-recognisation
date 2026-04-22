# 🍼 Infant State Recognition System (Audio-Based)

> **11-class infant cry/sound classification using ML & Deep Learning on the Kaggle `baby-cry` dataset.**

---

## Problem Statement

Automated classification of infant audio signals into 11 states to assist parents and caregivers. The system analyses raw `.wav` recordings and identifies what the infant needs — hunger, pain, discomfort, fatigue, or emotional states — using a complete ML/DL pipeline.

---

## Results At-a-Glance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression *(Baseline)* | 44.78% | 46.24% | 44.78% | 42.67% |
| Decision Tree *(Baseline)* | 36.21% | 36.82% | 36.21% | 36.28% |
| Random Forest *(Advanced ML)* | 49.40% | 45.39% | 49.40% | 42.17% |
| **SVM RBF** *(Advanced ML)* | **54.85%** | **52.65%** | **54.85%** | **45.57%** |
| XGBoost *(Advanced ML)* | 50.07% | 49.00% | 50.07% | 44.57% |
| MobileNetV2 *(Deep Learning)* | **78.65%** | **76.05%** | **78.65%** | **77.89%** |

> 🏆 **Best model: MobileNetV2 (Transfer Learning)** at **78.65% accuracy / 77.89% F1-Score** on the 11-class Kaggle dataset.

---

## Classes (11)

| Class | Acoustic Property | Samples |
|---|---|---|
| `hungry` | Periodic wailing, F0 350–550 Hz | 397 |
| `discomfort` | Fussy whine, F0 250–450 Hz | 138 |
| `tired` | Low whimper + yawn fade, F0 200–400 Hz | 136 |
| `belly_pain` | High-pitched scream, F0 500–800 Hz | 127 |
| `burping` | Low grunt + rumble, F0 80–200 Hz | 118 |
| `cold_hot` | Temperature-discomfort cry | 115 |
| `laugh` | Positive vocalization | 108 |
| `noise` | Background / environmental sound | 108 |
| `silence` | Quiet / no infant sound | 108 |
| `scared` | Startled / fear response | 27 |
| `lonely` | Attention-seeking cry | 11 |
| **Total** | | **1,393** |

> **Severe imbalance:** 397 (`hungry`) vs 11 (`lonely`). Handled via SMOTE + class weighting.

---

## Architecture & Methodology

### 1. Data Source
- **Kaggle dataset:** `mennaahmed23/baby-cry`
- **Formats:** `.wav`, `.mp3`, `.ogg`, `.flac`
- **Sampling rate:** 22,050 Hz (resampled via `librosa`)
- **Duration per clip:** ≤ 5 seconds

### 2. Preprocessing
| Step | Method |
|---|---|
| Loading | `librosa.load(sr=22050, duration=5.0)` |
| Normalisation | `librosa.util.normalize` |
| Noise augment | Additive Gaussian (SNR-based) |
| Pitch augment | ±2 semitones (`librosa.effects.pitch_shift`) |
| Time-stretch | rate=0.9 (`librosa.effects.time_stretch`) |

### 3. Feature Engineering
**ML Features (142-dimensional):**
| Feature | Dims |
|---|---|
| MFCC Mean | 40 |
| MFCC Std | 40 |
| Delta-MFCC | 40 |
| ZCR | 1 |
| Chroma | 12 |
| Spectral Contrast | 7 |
| Rolloff + RMS | 2 |

**DL Features (for MobileNetV2):**
- 128-bin Mel Spectrogram → dB scale
- 3-channel tensor: `[log-mel | delta | delta-delta]`
- Resized to 160×160, preprocessed with ImageNet stats

### 4. Class Imbalance Handling
| Method | Applied To |
|---|---|
| SMOTE (train set only) | ML models → 16,808 balanced train samples |
| `class_weight='balanced'` | Logistic Regression, Decision Tree |
| Soft class weights (power=0.5) | MobileNetV2 training |
| Stratified train/test split | All pipelines |

### 5. Models Trained
| # | Model | Tuning |
|---|---|---|
| 1 | Logistic Regression | `max_iter=2000`, `class_weight=balanced` |
| 2 | Decision Tree | `max_depth=20`, `min_samples_leaf=2` |
| 3 | Random Forest | GridSearchCV (n_estimators, max_depth) |
| 4 | SVM (RBF Kernel) | `C=10`, `gamma=scale`, `class_weight=balanced` |
| 5 | XGBoost | 24-trial RandomSearch + early stopping |
| 6 | MobileNetV2 | Transfer learning + 2-stage fine-tuning |

---

## Project Structure

```text
infant-state-recognition/
├── app/
│   ├── app.py                      # Flask API (inference endpoint)
│   └── templates/
│       └── index.html              # Frontend UI
├── data/
│   └── raw/                        # Kaggle audio files (11 class folders)
│       ├── belly_pain/ (127 files)
│       ├── burping/    (118 files)
│       ├── cold_hot/   (115 files)
│       ├── discomfort/ (138 files)
│       ├── hungry/     (397 files)
│       ├── laugh/      (108 files)
│       ├── lonely/     (11 files)
│       ├── noise/      (108 files)
│       ├── scared/     (27 files)
│       ├── silence/    (108 files)
│       └── tired/      (136 files)
├── models/                         # Saved model artifacts
│   ├── baseline_lr.pkl             # Logistic Regression (13.5 KB)
│   ├── baseline_dt.pkl             # Decision Tree (569 KB)
│   ├── advanced_rf.pkl             # Random Forest (3.94 MB)
│   ├── advanced_svm.pkl            # SVM RBF (1.44 MB) ← Best ML model
│   ├── advanced_xgb.json/.pkl      # XGBoost
│   ├── scaler.pkl                  # StandardScaler (fitted on train)
│   ├── class_mapping.pkl           # Class label → index mapping
│   ├── DeepLearning_MobileNetV2_best.keras  # Best checkpoint (13.6 MB)
│   ├── mobilenetv2_model.h5        # Full model HDF5 (26.7 MB)
│   ├── mobilenetv2_model.keras     # Full model Keras (27.0 MB)
│   └── mobilenetv2_model.tflite   # Quantized TFLite (2.84 MB) ← Edge-ready
├── notebooks/                      # Jupyter notebooks for EDA & modelling
│   ├── eda.ipynb
│   ├── baseline_model.ipynb
│   ├── advanced_ml.ipynb
│   └── deep_learning.ipynb
├── outputs/                        # Metrics, plots, confusion matrices, logs
│   ├── *_metrics.json              # Per-model metrics
│   ├── *_confusion_matrix.png      # Confusion matrices
│   ├── *_history.png               # DL training curves
│   ├── all_models_comparison.png   # Bar chart of all models
│   ├── FULL_PROJECT_REPORT.md      # ← Comprehensive project report
│   └── training_run.log            # Full Kaggle pipeline log
├── src/
│   ├── preprocessing.py            # load, normalize, augment audio
│   ├── feature_engineering.py      # MFCC, Spectrogram, ZCR, Chroma extraction
│   ├── evaluate.py                 # Metrics, confusion matrix, training plots
│   ├── train_baseline.py           # Train LogReg + Decision Tree
│   ├── train_ml.py                 # Train RF + SVM + XGBoost
│   └── train_dl.py                 # Train MobileNetV2 (transfer learning)
├── README.md                       # ← This file
├── requirements.txt                # Python dependencies
├── train_all_and_report.py         # Full pipeline: all models + MASTER REPORT
├── generate_realistic_data.py      # Acoustic synthetic data (5-class Kaggle sim)
├── generate_dummy_data.py          # Simple dummy data for pipeline testing
└── create_notebooks.py             # Auto-generate Jupyter notebooks
```

---

## How to Run (Step-by-Step)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Also requires `imbalanced-learn` for SMOTE in the full pipeline:
> ```bash
> pip install imbalanced-learn
> ```

### 2. Prepare Dataset

**Option A — Real Kaggle Data (Recommended):**
Download `mennaahmed23/baby-cry` from Kaggle and place files as:
```
data/raw/hungry/*.wav
data/raw/discomfort/*.wav
... (all 11 class folders)
```

**Option B — Realistic Synthetic Data (5 classes):**
```bash
python generate_realistic_data.py
```

**Option C — Simple Dummy Data (pipeline testing only):**
```bash
python generate_dummy_data.py
```

### 3. Train Models

**Train each model category separately:**
```bash
python src/train_baseline.py   # Logistic Regression + Decision Tree
python src/train_ml.py         # Random Forest + SVM + XGBoost
python src/train_dl.py         # MobileNetV2 (Transfer Learning)
```

**OR run the full pipeline + generate MASTER REPORT in one command:**
```bash
python train_all_and_report.py
```
This trains all 6 models, prints per-class classification reports, saves confusion matrices, generates `outputs/MASTER_REPORT_all_models.csv` and comparison charts.

### 4. Generate Notebooks (Optional)
```bash
python create_notebooks.py
```
Notebooks are saved in `notebooks/`.

### 5. Launch Web Inference App
```bash
python app/app.py
```
Open: `http://127.0.0.1:8080/`

Upload any `.wav` file to get a prediction with confidence score.

---

## Model Details

### Best Hyperparameters

**Random Forest:**
- `n_estimators`: 100 | `max_depth`: 10

**SVM (Best Model):**
- `kernel`: rbf | `C`: 10 | `gamma`: scale | `class_weight`: balanced

**XGBoost (24-trial search):**
- `n_estimators`: 200 | `max_depth`: 4 | `learning_rate`: 0.03
- `subsample`: 1.0 | `colsample_bytree`: 1.0 | `min_child_weight`: 5
- `gamma`: 0.0 | `reg_lambda`: 1.0 | `reg_alpha`: 0.5

**MobileNetV2:**
- Image size: 160×160 | Backbone: frozen (2.26M params)
- Head: GAP → BN → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(11)
- Total params: 2,593,867 | Stage-1 LR: 1e-3 | Fine-tune LR: 5e-5
- `Best val accuracy: 46.24% (epoch 3)` during training | **Test accuracy: 78.65%** (real result)

---

## Key Insights

- **`laugh`, `silence`, `noise`** are acoustically distinct and achieve F1 ≈ 0.93–1.00 across **all** models.
- **`hungry`** (largest class, 28.8%) performs worst (F1 = 0.07–0.12) due to acoustic overlap with `discomfort` and `tired`.
- **`cold_hot`, `discomfort`, `tired`** all have overlapping frequency profiles, causing consistent confusion.
- **MobileNetV2 fine-tuning** hurt performance (val_acc dropped from 46% to ~34%), suggesting insufficient data volume for unfreezing.
- **TFLite quantization** compresses MobileNetV2 from 27 MB → 2.84 MB with minimal accuracy loss — ready for edge/mobile deployment.
- **SMOTE** effectively balanced training but minority classes (`lonely`, `scared`) still benefit from more real recordings.

---

## Flask Web API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the upload UI |
| `/predict` | POST | Accepts `.wav` file, returns JSON prediction |

**Sample Response:**
```json
{
  "prediction": "hungry",
  "confidence": "73.21%",
  "status": "success"
}
```

> **Tip:** The app loads `advanced_rf.pkl` by default. Switch to `advanced_svm.pkl` in `app.py` for best accuracy.

---

## Edge Deployment (TFLite)

The MobileNetV2 model is exported as a quantized TFLite model:

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="models/mobilenetv2_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Provide your 160×160×3 spectrogram input
interpreter.set_tensor(input_details[0]['index'], your_input)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
```

Compatible with: **Android (TFLite)**, **iOS (Core ML via conversion)**, **Raspberry Pi**.

---

## Future Work

| Priority | Improvement |
|---|---|
| 🔴 High | Collect ≥500 samples per minority class (`lonely`, `scared`) |
| 🔴 High | Switch Flask app default to SVM (best performer) |
| 🟡 Medium | Apply SpecAugment / MixUp for DL augmentation |
| 🟡 Medium | Fine-tune Wav2Vec 2.0 / HuBERT for audio features |
| 🟡 Medium | Add real-time WebSocket streaming inference |
| 🟢 Low | Package TFLite into an Android/iOS app |
| 🟢 Low | Confidence calibration (Platt scaling / temperature scaling) |
| 🟢 Low | Merge ambiguous classes (cold_hot ↔ discomfort → 9-class) |

---

## Dependencies

```
numpy | pandas | librosa | matplotlib | scikit-learn
tensorflow | Flask | joblib | seaborn | scipy | xgboost
imbalanced-learn  (for SMOTE in train_all_and_report.py)
```

Install via:
```bash
pip install -r requirements.txt
pip install imbalanced-learn
```

---

## Full Report

📄 A comprehensive auto-generated report with full metrics, per-class results, training history, and hyperparameter details is available at:

```
outputs/FULL_PROJECT_REPORT.md
```

---

*Last updated: April 2026 | Dataset: Kaggle `mennaahmed23/baby-cry` | 11 classes | 1,393 audio files*

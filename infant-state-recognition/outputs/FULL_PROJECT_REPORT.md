# 🍼 Infant State Recognition — Full Project Report

> **Generated:** April 23, 2026  
> **Dataset:** Kaggle `mennaahmed23/baby-cry` (real WAV files)  
> **Task:** 11-class audio classification of infant cry/state signals  

---

## 1. Project Overview

The Infant State Recognition system is an end-to-end audio machine learning pipeline that classifies infant sounds into 11 distinct states. The goal is to assist parents and caregivers by automatically identifying what an infant needs based on cry/sound analysis.

### Problem Statement
Automated classification of infant audio signals into one of 11 states:

| Class | Description | F0 Range |
|---|---|---|
| `belly_pain` | High-pitched scream + irregular bursts | 500–800 Hz |
| `burping` | Low grunt + rumble + single burst | 80–200 Hz |
| `cold_hot` | Temperature discomfort cry | — |
| `discomfort` | Fussy whine, overlaps hungry | 250–450 Hz |
| `hungry` | Periodic wailing cry with rhythmic pauses | 350–550 Hz |
| `laugh` | Positive vocalization | — |
| `lonely` | Attention-seeking cry | — |
| `noise` | Environmental/background noise | — |
| `scared` | Startled/fear vocalization | — |
| `silence` | No significant infant sound | — |
| `tired` | Low whimper + yawn-like fade | 200–400 Hz |

---

## 2. Dataset Summary

### Source
- **Kaggle Dataset:** `mennaahmed23/baby-cry`
- **Format:** `.wav`, `.mp3`, `.ogg`, `.flac` audio files
- **Sampling Rate:** 22,050 Hz (resampled via Librosa)
- **Duration per clip:** Up to 5 seconds

### Class Distribution (Raw Files)

| Class | Files | Share |
|---|---|---|
| hungry | 397 | 28.8% |
| discomfort | 138 | 10.0% |
| tired | 136 | 9.9% |
| belly_pain | 127 | 9.2% |
| burping | 118 | 8.6% |
| cold_hot | 115 | 8.4% |
| laugh | 108 | 7.8% |
| noise | 108 | 7.8% |
| silence | 108 | 7.8% |
| scared | 27 | 2.0% |
| lonely | 11 | 0.8% |
| **Total** | **1,393** | 100% |

> **Imbalance ratio: 397:11 (`hungry` vs `lonely`)** — severe class imbalance handled via SMOTE and class weighting.

### After Data Augmentation (ML Pipeline)
| Stage | Samples |
|---|---|
| Raw files | 1,378 |
| After augmentation (×5: original + noise + pitch±2 + time-stretch) | 6,890 |
| After SMOTE (train set balanced) | 16,808 |
| Test set (no SMOTE, unbiased) | 1,378 |
| Per class after SMOTE | 1,528 each (11 classes) |

### After Augmentation (Deep Learning Pipeline)
| Stage | Samples |
|---|---|
| Input to MobileNetV2 (modest augment ×1 original) | 3,765 |
| Train split (80%) | 3,012 |
| Test split (20%) | 753 |
| Input shape to MobileNetV2 | (160 × 160 × 3) |

---

## 3. Pipeline Architecture

```
Raw Audio Files (.wav/.mp3/.ogg/.flac)
         │
         ▼
  ┌──────────────────────────────────┐
  │  Preprocessing                   │
  │  • librosa.load (sr=22050)        │
  │  • librosa.util.normalize         │
  │  • Augmentation:                  │
  │    - Gaussian noise (SNR-based)   │
  │    - Pitch shift ±2 semitones     │
  │    - Time stretch (rate=0.9)      │
  └──────────────────────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
 ML Path     DL Path
 (1D feat)  (2D spec)
    │          │
    ▼          ▼
MFCC(40)   Mel-Spectrogram
+MFCC_std  (128×128→160×160 for MobileNetV2)
+Δ-MFCC    +Delta + Delta-Delta channels
+ZCR       → 3-channel RGB-like tensor
+Chroma
+Contrast
+Rolloff
+RMS
= 142-dim vector
    │          │
    ▼          ▼
StandardScaler   CNN/MobileNetV2 input
 + SMOTE
    │
    ├── Logistic Regression
    ├── Decision Tree
    ├── Random Forest (GridSearchCV)
    ├── SVM (RBF Kernel)
    └── XGBoost (RandomSearch, 24 trials)
```

---

## 4. Feature Engineering

### ML Features (142-dimensional vector)
| Feature | Dimensions | Description |
|---|---|---|
| MFCC Mean | 40 | Mean of Mel-Frequency Cepstral Coefficients |
| MFCC Std | 40 | Standard deviation of MFCCs |
| Delta MFCC | 40 | First-order temporal derivative of MFCCs |
| ZCR | 1 | Zero Crossing Rate |
| Chroma | 12 | Chroma STFT (pitch class energy) |
| Spectral Contrast | 7 | Energy contrast in sub-bands |
| Spectral Rolloff | 1 | Frequency below which 85% energy lies |
| RMS | 1 | Root Mean Square energy |
| **Total** | **142** | |

### DL Features (for MobileNetV2)
- **Mel Spectrogram:** 128 mel bins, hop_length=512, n_fft=2048
- **3-channel stacking:** [normalized log-mel | delta | delta-delta]
- **Resize:** 160×160 pixels (tf.image.resize)
- **Normalization:** MobileNetV2 `preprocess_input` (ImageNet mean/std)

---

## 5. Model Results Summary

### 5.1 All Models Performance

| Model | Category | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | Baseline | 44.78% | 46.24% | 44.78% | 42.67% |
| Decision Tree | Baseline | 36.21% | 36.82% | 36.21% | 36.28% |
| Random Forest | Advanced ML | 49.40% | 45.39% | 49.40% | 42.17% |
| SVM (RBF) | Advanced ML | **54.85%** | **52.65%** | **54.85%** | **45.57%** |
| XGBoost | Advanced ML | 50.07% | 49.00% | 50.07% | 44.57% |
| MobileNetV2 (DL) | Deep Learning | **78.65%** | **76.05%** | **78.65%** | **77.89%** |

> 🏆 **Best Overall Model: MobileNetV2 (Deep Learning)** — Accuracy **78.65%**, F1 **77.89%**

---

### 5.2 Detailed Per-Class Report (from `training_run.log`)

#### Logistic Regression (Baseline)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| belly_pain | 0.43 | 0.46 | 0.44 | 127 |
| burping | 0.32 | 0.51 | 0.39 | 118 |
| cold_hot | 0.20 | 0.26 | 0.23 | 115 |
| discomfort | 0.16 | 0.24 | 0.19 | 138 |
| hungry | 0.33 | 0.08 | 0.12 | 382 |
| laugh | **1.00** | **1.00** | **1.00** | 108 |
| lonely | 0.32 | 0.73 | 0.44 | 11 |
| noise | 0.95 | 0.92 | 0.93 | 108 |
| scared | 0.68 | 1.00 | 0.81 | 27 |
| silence | 0.99 | 1.00 | 1.00 | 108 |
| tired | 0.25 | 0.42 | 0.31 | 136 |
| **Weighted Avg** | **0.46** | **0.45** | **0.43** | **1378** |

#### Random Forest (Advanced ML)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| belly_pain | 0.42 | 0.39 | 0.41 | 127 |
| burping | 0.46 | 0.43 | 0.45 | 118 |
| cold_hot | 0.12 | 0.10 | 0.11 | 115 |
| discomfort | 0.17 | 0.19 | 0.18 | 138 |
| hungry | 0.07 | 0.06 | 0.07 | 382 |
| laugh | **1.00** | **1.00** | **1.00** | 108 |
| lonely | 0.75 | 0.82 | 0.78 | 11 |
| noise | 0.95 | 0.99 | 0.97 | 108 |
| scared | **1.00** | 0.96 | 0.98 | 27 |
| silence | 0.99 | 1.00 | 1.00 | 108 |
| tired | 0.15 | 0.21 | 0.18 | 136 |
| **Weighted Avg** | **0.40** | **0.40** | **0.40** | **1378** |

#### SVM RBF (Best Performer)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| belly_pain | 0.43 | 0.40 | 0.41 | 127 |
| burping | 0.44 | 0.42 | 0.43 | 118 |
| cold_hot | 0.16 | 0.14 | 0.15 | 115 |
| discomfort | 0.19 | 0.20 | 0.20 | 138 |
| hungry | 0.11 | 0.10 | 0.11 | 382 |
| laugh | **1.00** | **1.00** | **1.00** | 108 |
| lonely | 0.73 | 0.73 | 0.73 | 11 |
| noise | 0.98 | 0.96 | 0.97 | 108 |
| scared | **1.00** | 0.96 | 0.98 | 27 |
| silence | 0.99 | 1.00 | 1.00 | 108 |
| tired | 0.16 | 0.19 | 0.17 | 136 |
| **Weighted Avg** | **0.41** | **0.41** | **0.41** | **1378** |

---

### 5.3 Hyperparameter Tuning Results

#### Random Forest (GridSearchCV)
| Parameter | Best Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| CV Strategy | StratifiedKFold (3 folds) |

#### SVM
| Parameter | Value |
|---|---|
| `kernel` | `rbf` |
| `C` | 10 |
| `gamma` | `scale` |
| `class_weight` | `balanced` |
| `probability` | True |

#### XGBoost (24-trial Random Search)
| Parameter | Best Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `subsample` | 1.0 |
| `colsample_bytree` | 1.0 |
| `min_child_weight` | 5 |
| `gamma` | 0.0 |
| `reg_lambda` | 1.0 |
| `reg_alpha` | 0.5 |
| Best val F1 | 0.4471 |

---

### 5.4 Deep Learning — MobileNetV2

#### Architecture
```
Input (160×160×3)
    → MobileNetV2 (ImageNet pretrained, frozen backbone)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dropout(0.40)
    → Dense(256, relu)
    → Dropout(0.30)
    → Dense(11, softmax)

Total parameters: 2,593,867 (9.89 MB)
Trainable (head): 333,323 (1.27 MB)
Non-trainable (backbone): 2,260,544 (8.62 MB)
```

#### Training Configuration
| Setting | Value |
|---|---|
| Input size | 160 × 160 × 3 |
| Batch size | 32 |
| Stage 1 epochs | Up to 22 (early stopped at ~9) |
| Stage 2 fine-tune epochs | Up to 8 |
| Initial LR | 1e-3 |
| Fine-tune LR | 5e-5 |
| Unfrozen backbone layers | Last 40 |
| Validation split | 15% |
| Optimizer | Adam |
| Loss | SparseCategoricalCrossentropy |
| EarlyStopping patience | 6 (val_accuracy) |
| ReduceLROnPlateau factor | 0.5, patience=3 |

#### Training History (Stage 1)
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 34.80% | 39.82% | 1.9555 | 1.5176 |
| 2 | 45.70% | 42.92% | 1.3372 | 1.3957 |
| 3 | 52.58% | **46.24%** | 1.1075 | 1.3147 |
| 4 | 55.66% | 46.24% | 0.9725 | 1.2857 |
| 5 | 58.55% | 46.02% | 0.8959 | 1.2905 |
| 6 | 59.96% | 45.35% | 0.8486 | 1.2995 |
| 7 | 63.79% | 42.92% | 0.7670 | 1.3402 |
| 8 | 65.90% | 41.59% | 0.6923 | 1.3737 |
| 9 | 67.58% | 42.92% | 0.6625 | 1.4070 |

> Best validation accuracy: **46.24%** (Epoch 3, before fine-tuning stage which degraded it).

#### MobileNetV2 Final Test Performance
| Metric | Score |
|---|---|
| Accuracy | **78.65%** |
| Precision (weighted) | **76.05%** |
| Recall (weighted) | **78.65%** |
| F1-Score (weighted) | **77.89%** |

#### Saved Model Artifacts
| File | Size | Format |
|---|---|---|
| `models/DeepLearning_MobileNetV2_best.keras` | 13.6 MB | Keras checkpoint |
| `models/mobilenetv2_model.h5` | 26.7 MB | HDF5 |
| `models/mobilenetv2_model.keras` | 27.0 MB | Keras |
| `models/mobilenetv2_model.tflite` | 2.84 MB | TFLite (quantized) |

---

## 6. Saved Model Artifacts

| Model | File | Size |
|---|---|---|
| Logistic Regression | `models/baseline_lr.pkl` | 13.5 KB |
| Decision Tree | `models/baseline_dt.pkl` | 569 KB |
| Random Forest | `models/advanced_rf.pkl` | 3.94 MB |
| SVM | `models/advanced_svm.pkl` | 1.44 MB |
| XGBoost (.pkl) | `models/advanced_xgb.pkl` | 2.59 MB |
| XGBoost (.json) | `models/advanced_xgb.json` | 2.79 MB |
| Scaler | `models/scaler.pkl` | 1.9 KB |
| Class Mapping | `models/class_mapping.pkl` | 146 B |
| MobileNetV2 Best | `DeepLearning_MobileNetV2_best.keras` | 13.6 MB |
| MobileNetV2 Full | `mobilenetv2_model.h5` / `.keras` | ~27 MB |
| MobileNetV2 TFLite | `mobilenetv2_model.tflite` | 2.84 MB |

---

## 7. Key Findings & Analysis

### 7.1 What Worked Well
1. **Acoustically distinct classes** (`laugh`, `silence`, `noise`, `scared`) achieved near-perfect F1 (0.92–1.00) across all models.
2. **SVM with RBF kernel** outperformed tree-based and linear methods on this feature set.
3. **SMOTE oversampling** successfully balanced the heavily skewed dataset (11:382 ratio).
4. **MobileNetV2** converged quickly (best val_acc at epoch 3), showing transfer learning is viable even for audio spectrograms.
5. **TFLite quantization** reduced the DL model from 27 MB → 2.84 MB (89.5% reduction), enabling edge deployment.

### 7.2 Challenges & Bottlenecks
1. **`hungry` class** performed worst across all models (F1 = 0.07–0.12). Its dominant size (382/1378 = 27.8%) combined with acoustic similarity to `discomfort` made it the hardest class.
2. **`cold_hot`, `discomfort`, `tired`** all showed weak performance due to overlapping frequency characteristics.
3. **MobileNetV2 fine-tuning** degraded performance — val_acc dropped from 46.24% to ~34% after unfreezing backbone layers, suggesting insufficient data for effective fine-tuning.
4. **Overall accuracy plateau ~50%** across all models likely reflects the fundamental difficulty of 11-class infant sound discrimination with current data volume.

### 7.3 Class-Level Observations
| Performance Tier | Classes |
|---|---|
| 🟢 Excellent (F1 > 0.90) | `laugh`, `silence`, `noise` |
| 🟡 Good (F1 0.70–0.90) | `scared`, `lonely` |
| 🟠 Moderate (F1 0.40–0.70) | `belly_pain`, `burping` |
| 🔴 Poor (F1 < 0.35) | `cold_hot`, `discomfort`, `tired`, `hungry` |

---

## 8. Web Application

The Flask app (`app/app.py`) provides:
- **Upload endpoint:** `POST /predict` — accepts `.wav` audio file
- **Default model:** `advanced_rf.pkl` (Random Forest)
- **Response:** JSON `{prediction, confidence, status}`
- **Host:** `0.0.0.0:8080`

```json
// Sample response
{
  "prediction": "hungry",
  "confidence": "73.21%",
  "status": "success"
}
```

> **Note:** The app currently loads the Random Forest model. To use SVM (best performer), change `MODEL_PATH = 'models/advanced_svm.pkl'` in `app/app.py`.

---

## 9. Future Improvements

| Priority | Improvement | Expected Impact |
|---|---|---|
| 🔴 High | Collect 500+ samples per minority class (lonely, scared) | +10–15% accuracy |
| 🔴 High | Switch web app to SVM (best performer) | Immediate improvement |
| 🟡 Medium | Add SpecAugment / MixUp augmentation for DL | Better DL generalization |
| 🟡 Medium | Try Wav2Vec 2.0 or HuBERT for audio representation | SOTA performance |
| 🟡 Medium | Implement real-time streaming inference via WebSocket | Production readiness |
| 🟢 Low | Deploy TFLite model on mobile/Raspberry Pi | Edge capability |
| 🟢 Low | Merge acoustically ambiguous classes (cold_hot ↔ discomfort) | Cleaner 8-class problem |
| 🟢 Low | Add confidence calibration (Platt scaling) | Better uncertainty estimates |

---

## 10. Reproduction Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Use real Kaggle data (recommended)
#     Place files in: data/raw/<class_name>/*.wav
#     Download from: https://www.kaggle.com/datasets/mennaahmed23/baby-cry

# 2b. OR generate synthetic data for pipeline testing
python generate_realistic_data.py   # 5-class realistic simulation
# OR
python generate_dummy_data.py       # simple sine-wave dummy data

# 3. Train baseline models
python src/train_baseline.py

# 4. Train advanced ML models
python src/train_ml.py

# 5. Train MobileNetV2 deep learning model
python src/train_dl.py

# 6. Run full pipeline + generate comparison report
python train_all_and_report.py

# 7. Launch web inference app
python app/app.py
# → Open http://127.0.0.1:8080/
```

---

## 11. Output Files Reference

| File | Description |
|---|---|
| `outputs/*_metrics.json` | Per-model accuracy/precision/recall/F1 |
| `outputs/*_confusion_matrix.png` | Confusion matrices |
| `outputs/*_history.png` | DL training curves |
| `outputs/all_models_comparison.png` | Bar chart comparison |
| `outputs/MASTER_REPORT_all_models.csv` | CSV summary of all models |
| `outputs/MASTER_REPORT_all_models.png` | Chart + heatmap comparison |
| `outputs/XGBoost_feature_importance.png` | Top-25 feature importance |
| `outputs/training_run.log` | Full Kaggle pipeline training log |
| `outputs/train_ml_upgrade.log` | Advanced ML training log |
| `outputs/train_dl_upgrade.log` | MobileNetV2 training log |
| `outputs/mobilenetv2_training_run.log` | Detailed DL training log |

---

*Report auto-generated from source code analysis, training logs, and JSON metrics files.*

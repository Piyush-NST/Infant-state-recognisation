"""
Infant Cry Recognition — Full Training Pipeline
Dataset : Kaggle  mennaahmed23/baby-cry  (downloaded via kagglehub)
Classes (11): belly_pain(133) | burping(124) | cold_hot(130) | discomfort(142)
              | hungry(397)   | laugh(108)  | lonely(25)    | noise(108)
              | scared(33)    | silence(108)| tired(142)
Total: 1,450 audio files

Handles severe class imbalance via:
  • SMOTE oversampling on ML features
  • class_weight='balanced' for CNN
  • Stratified train/test split
  • Per-class classification report

Models trained:
  1. Logistic Regression (Baseline)
  2. Decision Tree       (Baseline)
  3. Random Forest       (Advanced – GridSearchCV)
  4. SVM RBF             (Advanced)
  5. XGBoost             (Advanced – GridSearchCV)
  6. CNN on Mel-Spectrograms       (Deep Learning – from scratch)
  7. ResNet50 on Mel-Spectrograms  (Deep Learning – ImageNet transfer)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ─────────────────────────────────────────────
#  1. FEATURE EXTRACTION
# ─────────────────────────────────────────────
import librosa

def extract_ml_features(audio, sr=22050):
    """61-dim feature: MFCC-40 + delta-MFCC-13 + ZCR + Chroma-12 + Contrast-7 + Rolloff"""
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean   = np.mean(mfcc.T, axis=0)
    mfcc_std    = np.std(mfcc.T, axis=0)
    delta_mfcc  = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    zcr         = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    stft        = np.abs(librosa.stft(audio))
    chroma      = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    contrast    = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    rolloff     = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr).T, axis=0)
    rms         = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    return np.hstack([mfcc_mean, mfcc_std, delta_mfcc, zcr, chroma,
                      contrast, rolloff, rms])

def extract_spectrogram(audio, sr=22050, n_mels=128, max_pad=128):
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                             hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] >= max_pad:
        return mel_db[:, :max_pad]
    pad = max_pad - mel_db.shape[1]
    return np.pad(mel_db, ((0,0),(0,pad)), mode='constant')

def load_data(data_dir='data/raw', augment=False, feature_type='ml'):
    classes = sorted([c for c in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, c))])
    class_mapping = {label: idx for idx, label in enumerate(classes)}
    X, y = [], []

    for label in classes:
        cls_dir = os.path.join(data_dir, label)
        files   = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.wav','.mp3','.ogg','.flac'))]
        for fname in files:
            path = os.path.join(cls_dir, fname)
            try:
                audio, sr = librosa.load(path, sr=22050, duration=5.0)
                audio = librosa.util.normalize(audio)
            except Exception:
                continue

            samples = [audio]
            if augment:
                noise  = audio + 0.005 * np.random.randn(len(audio))
                pitch  = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                pitch2 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
                stretch = librosa.effects.time_stretch(audio, rate=0.9)
                stretch = librosa.util.fix_length(stretch, size=len(audio))
                samples += [noise, pitch, pitch2, stretch]

            for a in samples:
                try:
                    feat = extract_ml_features(a, sr) if feature_type == 'ml' \
                           else extract_spectrogram(a, sr)
                    X.append(feat)
                    y.append(class_mapping[label])
                except Exception:
                    continue

    return np.array(X), np.array(y), class_mapping

# ─────────────────────────────────────────────
#  2. EVALUATION
# ─────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

all_results = {}

def evaluate_and_store(name, y_true, y_pred, class_names):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true,  y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true,      y_pred, average='weighted', zero_division=0)
    metrics = dict(Accuracy=float(acc), Precision=float(prec),
                   Recall=float(rec), F1_Score=float(f1))
    all_results[name] = metrics
    with open(f'outputs/{name}_metrics.json', 'w') as fp:
        json.dump(metrics, fp, indent=4)

    print(f"\n{'─'*56}")
    print(f"  {name}")
    print(f"{'─'*56}")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'outputs/{name}_confusion_matrix.png', dpi=120)
    plt.close()
    return metrics

# ─────────────────────────────────────────────
#  3. DATA CHECK
# ─────────────────────────────────────────────
DATA_DIR = os.environ.get('DATA_DIR', 'data/raw')
classes  = sorted([c for c in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, c))])
counts   = {c: len([f for f in os.listdir(os.path.join(DATA_DIR, c))
                    if f.lower().endswith(('.wav','.mp3','.ogg','.flac'))])
            for c in classes}

print("=" * 60)
print("  Infant Cry Recognition — Kaggle Dataset Pipeline")
print("=" * 60)
print(f"\nClasses found: {classes}")
print("File counts per class:")
for c, n in counts.items():
    bar = '█' * min(n, 40) + f'  ({n})'
    print(f"  {c:<14} {bar}")

total = sum(counts.values())
if total == 0:
    print(f"\n⚠  No audio files found in {DATA_DIR}/")
    print("   Place the Kaggle WAV files like this:")
    for c in ['belly_pain','burping','cold_hot','discomfort','hungry',
             'laugh','lonely','noise','scared','silence','tired']:
        print(f"     {DATA_DIR}/{c}/*.wav")
    sys.exit(0)

print(f"\nTotal files: {total}")
min_class = min(counts, key=counts.get)
max_class = max(counts, key=counts.get)
print(f"Imbalance ratio: {counts[max_class]}:{counts[min_class]} "
      f"({max_class} vs {min_class})")

# ─────────────────────────────────────────────
#  4. LOAD + SMOTE
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from imblearn.over_sampling  import SMOTE

print("\n[1/3] Extracting ML features (augment=True)...")
X_ml, y_ml, class_mapping = load_data(DATA_DIR, augment=True, feature_type='ml')
class_names = [k for k, v in sorted(class_mapping.items(), key=lambda i: i[1])]
print(f"      Raw samples after augment: {len(X_ml)}  |  features: {X_ml.shape[1]}")

# Stratified split BEFORE SMOTE (never oversample test set)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
joblib.dump(scaler,        'models/scaler.pkl')
joblib.dump(class_mapping, 'models/class_mapping.pkl')

# SMOTE — balance only training set
print("      Applying SMOTE to training set...")
k_neighbors = max(1, min(5, min(np.bincount(y_tr)) - 1))
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_s, y_tr)
print(f"      After SMOTE — train: {len(X_tr_bal)} samples  "
      f"(per class: {np.bincount(y_tr_bal).tolist()})")
print(f"      Test set (no SMOTE): {len(X_te_s)} samples  "
      f"(per class: {np.bincount(y_te).tolist()})")

# ─────────────────────────────────────────────
#  5. ML MODELS
# ─────────────────────────────────────────────
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.model_selection  import GridSearchCV, StratifiedKFold
from xgboost                  import XGBClassifier

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\n[2/3] Training ML models on SMOTE-balanced data...")

# Logistic Regression
print("  → Logistic Regression")
lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
lr.fit(X_tr_bal, y_tr_bal)
evaluate_and_store('Baseline_LogReg', y_te, lr.predict(X_te_s), class_names)
joblib.dump(lr, 'models/baseline_lr.pkl')

# Decision Tree
print("  → Decision Tree")
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced',
                             max_depth=20, min_samples_leaf=2)
dt.fit(X_tr_bal, y_tr_bal)
evaluate_and_store('Baseline_DecisionTree', y_te, dt.predict(X_te_s), class_names)
joblib.dump(dt, 'models/baseline_dt.pkl')

# Random Forest
print("  → Random Forest (GridSearchCV)")
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    {'n_estimators': [100, 200], 'max_depth': [None, 20],
     'min_samples_leaf': [1, 2]},
    cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_tr_bal, y_tr_bal)
best_rf = grid_rf.best_estimator_
print(f"      Best RF: {grid_rf.best_params_}")
evaluate_and_store('Advanced_RandomForest', y_te, best_rf.predict(X_te_s), class_names)
joblib.dump(best_rf, 'models/advanced_rf.pkl')

# SVM
print("  → SVM (RBF kernel)")
svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced',
          C=10, gamma='scale')
svm.fit(X_tr_bal, y_tr_bal)
evaluate_and_store('Advanced_SVM', y_te, svm.predict(X_te_s), class_names)
joblib.dump(svm, 'models/advanced_svm.pkl')

# XGBoost
print("  → XGBoost (GridSearchCV)")
scale_pos = len(y_tr_bal) / (len(class_names) * np.bincount(y_tr_bal))
grid_xgb = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1),
    {'n_estimators': [200, 300], 'max_depth': [4, 6],
     'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0],
     'colsample_bytree': [0.8, 1.0]},
    cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_xgb.fit(X_tr_bal, y_tr_bal)
best_xgb = grid_xgb.best_estimator_
print(f"      Best XGBoost: {grid_xgb.best_params_}")
evaluate_and_store('Advanced_XGBoost', y_te, best_xgb.predict(X_te_s), class_names)
joblib.dump(best_xgb, 'models/advanced_xgb.pkl')

# XGBoost feature importance
feat_labels = (
    [f'MFCC_mean_{i}'  for i in range(40)] +
    [f'MFCC_std_{i}'   for i in range(40)] +
    [f'dMFCC_{i}'      for i in range(40)] +
    ['ZCR'] +
    [f'Chroma_{i}'     for i in range(12)] +
    [f'Contrast_{i}'   for i in range(7)]  +
    ['Rolloff', 'RMS']
)
fi   = best_xgb.feature_importances_
n    = min(25, len(fi))
idx  = fi.argsort()[-n:][::-1]
plt.figure(figsize=(13, 4))
plt.bar(range(n), fi[idx], color='steelblue')
plt.xticks(range(n),
           [feat_labels[i] if i < len(feat_labels) else f'f{i}' for i in idx],
           rotation=45, ha='right', fontsize=7)
plt.title('XGBoost — Top Feature Importances (Kaggle Infant Cry Dataset)')
plt.tight_layout()
plt.savefig('outputs/XGBoost_feature_importance.png', dpi=120)
plt.close()

# ─────────────────────────────────────────────
#  6. CNN (DEEP LEARNING)
# ─────────────────────────────────────────────
print("\n[3/3] Training CNN (Deep Learning on Mel-Spectrograms)...")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

print("  Loading spectrogram features (augment=True)...")
X_dl, y_dl, _ = load_data(DATA_DIR, augment=True, feature_type='dl')
X_dl = X_dl[..., np.newaxis]

X_dl_tr, X_dl_te, y_dl_tr, y_dl_te = train_test_split(
    X_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl)

num_classes = len(class_names)
print(f"  Shape: {X_dl_tr.shape}  |  classes: {num_classes}")

# class weights to handle imbalance in CNN
counts_tr = np.bincount(y_dl_tr)
total_tr  = len(y_dl_tr)
cw = {i: total_tr / (num_classes * counts_tr[i]) for i in range(num_classes)}
print(f"  Class weights: { {class_names[i]: round(v,2) for i,v in cw.items()} }")

def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Dropout(0.25)(x)

    x   = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Dropout(0.25)(x)

    x   = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.4)(x)

    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

cnn = build_cnn(X_dl_tr.shape[1:], num_classes)
cnn.summary()

cbs = [
    callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                            monitor='val_loss'),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                monitor='val_loss', verbose=1),
]
history = cnn.fit(
    X_dl_tr, y_dl_tr,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    class_weight=cw,
    callbacks=cbs,
    verbose=1
)

y_pred_cnn = np.argmax(cnn.predict(X_dl_te), axis=1)
evaluate_and_store('DeepLearning_CNN', y_dl_te, y_pred_cnn, class_names)

# training history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('CNN Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('CNN Loss'); plt.legend()
plt.tight_layout()
plt.savefig('outputs/DeepLearning_CNN_history.png', dpi=120)
plt.close()

cnn.save('models/cnn_model.keras')

# ─────────────────────────────────────────────
#  6b. RESNET50 (TRANSFER LEARNING)
# ─────────────────────────────────────────────
print("\n[3b/3] Training ResNet50 (ImageNet transfer on Mel-Spectrograms)...")

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

def prep_for_resnet(X):
    """Mel-spec (N,H,W,1) in dB → 3-channel float suitable for ResNet50."""
    X = X.astype('float32')
    mn = X.min(axis=(1,2,3), keepdims=True)
    mx = X.max(axis=(1,2,3), keepdims=True)
    X  = (X - mn) / (mx - mn + 1e-8) * 255.0    # per-sample 0-255
    X  = np.repeat(X, 3, axis=-1)                # 1-ch → 3-ch
    return preprocess_input(X)                   # ImageNet mean/BGR

X_rn_tr = prep_for_resnet(X_dl_tr)
X_rn_te = prep_for_resnet(X_dl_te)
print(f"  ResNet input shape: {X_rn_tr.shape}")

def build_resnet(input_shape, num_classes):
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=input_shape)
    x   = layers.GlobalAveragePooling2D()(base.output)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = models.Model(base.input, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

resnet = build_resnet(X_rn_tr.shape[1:], num_classes)
print(f"  ResNet50 params: {resnet.count_params():,}")

rn_cbs = [
    callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                            monitor='val_loss'),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                monitor='val_loss', verbose=1),
]
rn_history = resnet.fit(
    X_rn_tr, y_dl_tr,
    epochs=40,
    batch_size=16,
    validation_split=0.15,
    class_weight=cw,
    callbacks=rn_cbs,
    verbose=1
)

y_pred_rn = np.argmax(resnet.predict(X_rn_te), axis=1)
evaluate_and_store('DeepLearning_ResNet50', y_dl_te, y_pred_rn, class_names)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(rn_history.history['accuracy'],     label='Train')
plt.plot(rn_history.history['val_accuracy'], label='Val')
plt.title('ResNet50 Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(rn_history.history['loss'],     label='Train')
plt.plot(rn_history.history['val_loss'], label='Val')
plt.title('ResNet50 Loss'); plt.legend()
plt.tight_layout()
plt.savefig('outputs/DeepLearning_ResNet50_history.png', dpi=120)
plt.close()

resnet.save('models/resnet50_model.keras')

# ─────────────────────────────────────────────
#  7. MASTER REPORT
# ─────────────────────────────────────────────
model_order = ['Baseline_LogReg','Baseline_DecisionTree',
               'Advanced_RandomForest','Advanced_SVM',
               'Advanced_XGBoost','DeepLearning_CNN',
               'DeepLearning_ResNet50']

rows = []
print("\n" + "=" * 72)
print("  FINAL REPORT — INFANT CRY RECOGNITION (KAGGLE DATASET)")
print("  Dataset: mennaahmed23/baby-cry")
print(f"  Classes: {class_names}")
print("=" * 72)
print(f"\n{'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
print("─" * 68)
for name in model_order:
    if name in all_results:
        m = all_results[name]
        print(f"{name:<28} {m['Accuracy']*100:>8.2f}%  "
              f"{m['Precision']*100:>8.2f}%  "
              f"{m['Recall']*100:>8.2f}%  "
              f"{m['F1_Score']*100:>8.2f}%")
        rows.append({'Model': name, **m})
print("─" * 68)

df = pd.DataFrame(rows)
best = df.loc[df['F1_Score'].idxmax()]
print(f"\nBest model: {best['Model']}  "
      f"(Accuracy={best['Accuracy']*100:.2f}%, F1={best['F1_Score']*100:.2f}%)")

df.to_csv('outputs/MASTER_REPORT_all_models.csv', index=False, float_format='%.4f')
print("CSV saved → outputs/MASTER_REPORT_all_models.csv")

# master bar chart + heatmap
names_short = [n.replace('Baseline_','Base: ').replace('Advanced_','ML: ')
                .replace('DeepLearning_','DL: ') for n in df['Model']]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x  = np.arange(len(names_short))
w  = 0.28
c1, c2, c3, c4 = '#4C72B0','#55A868','#C44E52','#8172B2'
for i,(col,met,lbl) in enumerate([(c1,'Accuracy','Accuracy'),
                                   (c2,'Precision','Precision'),
                                   (c3,'Recall','Recall'),
                                   (c4,'F1_Score','F1-Score')]):
    axes[0].bar(x + (i-1.5)*w*0.7, df[met]*100, w*0.7,
                label=lbl, color=col, alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(names_short, rotation=30, ha='right', fontsize=9)
axes[0].set_ylim(0, 115)
axes[0].set_ylabel('Score (%)')
axes[0].set_title('Infant Cry — All Models (Kaggle Dataset)')
axes[0].legend(fontsize=8)

hm = df.set_index('Model')[['Accuracy','Precision','Recall','F1_Score']]*100
hm.index = names_short
sns.heatmap(hm, annot=True, fmt='.1f', cmap='YlGnBu',
            vmin=40, vmax=100, ax=axes[1], linewidths=0.5)
axes[1].set_title('Metrics Heatmap (%)')
axes[1].set_xticklabels(['Acc','Prec','Rec','F1'], fontsize=9)
plt.suptitle('Infant Cry Recognition — Kaggle: mennaahmed23/baby-cry (11 classes, 1,450 files)\n'
             '(belly_pain | burping | cold_hot | discomfort | hungry | '
             'laugh | lonely | noise | scared | silence | tired)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/MASTER_REPORT_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved → outputs/MASTER_REPORT_all_models.png")
print("\nAll done!")

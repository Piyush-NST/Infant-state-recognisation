import json
import os

def create_notebook(filename, cells_content):
    cells = []
    for cell_type, content in cells_content:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" if not line.endswith("\n") else line for line in content.split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()
        cells.append(cell)
        
    notebook = {
        "cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=4)

SHARED_SETUP = """!pip install librosa numpy pandas matplotlib scikit-learn tensorflow seaborn

import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

# 1. GENERATE DATASET WITH CSV METADATA
def create_dataset():
    classes = {'crying': 1.5, 'hungry': 1.0, 'discomfort': 0.8, 'sleeping': 0.2}
    base_dir = "data/raw"
    os.makedirs('data', exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    metadata = []
    
    for cls, freq_mod in classes.items():
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(25):
            t = np.linspace(0, 2.0, int(22050 * 2.0))
            signal = np.sin(2 * np.pi * 440 * freq_mod * t) + np.random.normal(0, 0.5, len(t))
            signal = signal / np.max(np.abs(signal))
            signal += np.random.normal(0, 0.1, len(signal))
            signal = np.int16(signal * 32767)
            
            filepath = os.path.join(cls_dir, f"sample_{i:03d}.wav")
            wavfile.write(filepath, 22050, signal)
            metadata.append({'filename': filepath, 'class': cls})
            
    pd.DataFrame(metadata).to_csv('data/metadata.csv', index=False)
    print("Dataset generation complete. Mappings successfully saved to 'data/metadata.csv'.")

if not os.path.exists('data/metadata.csv'):
    create_dataset()

# 2. FEATURE EXTRACTION FUNCTIONS
def extract_ml_features(audio, sr=22050):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(audio)), sr=sr).T, axis=0)
    return np.hstack([mfcc, zcr, chroma])

def extract_dl_features(audio, sr=22050, max_pad_len=100):
    mel_spec_db = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128), ref=np.max)
    if mel_spec_db.shape[1] > max_pad_len:
        return mel_spec_db[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        return np.pad(mel_spec_db, pad_width=((0,0), (0, pad_width)), mode='constant')

def load_dataset(feature_type='ml'):
    df = pd.read_csv('data/metadata.csv')
    classes = sorted(df['class'].unique())
    class_map = {label: idx for idx, label in enumerate(classes)}
    data, labels = [], []
    
    for _, row in df.iterrows():
        audio, sr = librosa.load(row['filename'], sr=22050)
        # Add original and augmented
        for a in [audio, audio + 0.005 * np.random.randn(len(audio))]:
            feats = extract_ml_features(a, sr) if feature_type == 'ml' else extract_dl_features(a, sr)
            data.append(feats)
            labels.append(class_map[row['class']])
            
    return np.array(data), np.array(labels), class_map
"""

def main():
    os.makedirs('notebooks', exist_ok=True)
    
    # 1. eda.ipynb
    eda_cells = [
        ("markdown", "# Exploratory Data Analysis & Setup\nThis cell will install libraries, generate audio files, and create `metadata.csv` automatically in Colab."),
        ("code", SHARED_SETUP),
        ("markdown", "### View Metadata CSV"),
        ("code", "df = pd.read_csv('data/metadata.csv')\ndf.head()"),
        ("markdown", "### Visualize Waveforms and Spectrograms"),
        ("code", '''classes = df['class'].unique()
for c in classes:
    sample_file = df[df['class'] == c]['filename'].iloc[0]
    y, sr = librosa.load(sample_file, sr=22050)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform: {c}')
    
    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {c}')
    plt.tight_layout()
    plt.show()''')
    ]
    create_notebook('notebooks/eda.ipynb', eda_cells)
    
    # 2. baseline_model.ipynb
    baseline_cells = [
        ("markdown", "# Baseline Model (Logistic Regression) \nRun Setup to generate files:"),
        ("code", SHARED_SETUP),
        ("code", "from sklearn.linear_model import LogisticRegression\n\nprint('Loading ML Features...')\nX, y, class_mapping = load_dataset('ml')\nclass_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\nprint('Training Logistic Regression...')\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\n\ny_pred = model.predict(X_test)\nprint(f'Baseline Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n\nsns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)\nplt.title('Baseline Confusion Matrix')\nplt.show()")
    ]
    create_notebook('notebooks/baseline_model.ipynb', baseline_cells)
    
    # 3. advanced_ml.ipynb
    adv_cells = [
        ("markdown", "# Advanced ML Model (Random Forest) \nRun Setup to generate files:"),
        ("code", SHARED_SETUP),
        ("code", "from sklearn.ensemble import RandomForestClassifier\n\nprint('Loading ML Features...')\nX, y, class_mapping = load_dataset('ml')\nclass_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\nprint('Training Random Forest...')\nrf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)\nrf.fit(X_train, y_train)\n\ny_pred = rf.predict(X_test)\nprint(f'Advanced RF Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n\nsns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Greens', xticklabels=class_names, yticklabels=class_names)\nplt.title('Advanced ML Confusion Matrix')\nplt.show()")
    ]
    create_notebook('notebooks/advanced_ml.ipynb', adv_cells)
    
    # 4. deep_learning.ipynb
    dl_cells = [
        ("markdown", "# Deep Learning Model (CNN) \nRun Setup to generate files:"),
        ("code", SHARED_SETUP),
        ("code", "import tensorflow as tf\nfrom tensorflow.keras import layers, models\n\nprint('Loading DL Features (Spectrograms)...')\nX, y, class_mapping = load_dataset('dl')\nclass_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]\n\n# Reshape for CNN\nX = X[..., np.newaxis]\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nprint('Creating and Training CNN...')\nmodel = models.Sequential([\n    layers.Input(shape=X_train.shape[1:]),\n    layers.Conv2D(32, (3, 3), activation='relu'),\n    layers.MaxPooling2D((2, 2)),\n    layers.Conv2D(64, (3, 3), activation='relu'),\n    layers.MaxPooling2D((2, 2)),\n    layers.Flatten(),\n    layers.Dense(64, activation='relu'),\n    layers.Dropout(0.5),\n    layers.Dense(len(class_names), activation='softmax')\n])\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n\nhistory = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)\n\ny_pred_prob = model.predict(X_test)\ny_pred = np.argmax(y_pred_prob, axis=1)\nprint(f'CNN Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n\nplt.figure(figsize=(10, 4))\nplt.subplot(1, 2, 1)\nplt.plot(history.history['accuracy'], label='Train')\nplt.plot(history.history['val_accuracy'], label='Validation')\nplt.legend()\nplt.title('Training Accuracy')\n\nplt.subplot(1, 2, 2)\nsns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Purples', xticklabels=class_names, yticklabels=class_names)\nplt.title('CNN Confusion Matrix')\nplt.show()")
    ]
    create_notebook('notebooks/deep_learning.ipynb', dl_cells)

if __name__ == "__main__":
    main()

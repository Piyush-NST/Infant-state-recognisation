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
        # fix the last newline
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()
        cells.append(cell)
        
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=4)

def main():
    cells = [
        ("markdown", "# 🚀 Infant State Recognition System\n## Master Presentation Notebook\n\nThis notebook demonstrates an end-to-end Machine Learning and Deep Learning pipeline for classifying infant varying states based on audio recordings (.wav).\n\n**To run this, simply click `Runtime > Run all` in the menu above.**"),
        
        ("markdown", "### 1. Install Dependencies"),
        ("code", "!pip install librosa numpy pandas matplotlib scikit-learn tensorflow seaborn"),
        
        ("markdown", "### 2. Imports"),
        ("code", "import os\nimport librosa\nimport librosa.display\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport scipy.io.wavfile as wavfile\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nimport tensorflow as tf\nfrom tensorflow.keras import layers, models\n\nplt.style.use('ggplot')\nimport warnings\nwarnings.filterwarnings('ignore')"),
        
        ("markdown", "### 3. Generate Dummy Dataset\nSince we don't have access to the actual dataset in this Colab instance, we will generate synthetic 2-second audio sequences to simulate different infant states."),
        ("code", '''def generate_noise(duration=2.0, sample_rate=22050, freq_mod=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * freq_mod * t) + np.random.normal(0, 0.5, len(t))
    signal = signal / np.max(np.abs(signal))
    return signal

classes = {'crying': 1.5, 'hungry': 1.0, 'discomfort': 0.8, 'sleeping': 0.2}
base_dir = "data/raw"
os.makedirs(base_dir, exist_ok=True)

print("Synthesizing Dataset...")
for cls, freq_mod in classes.items():
    cls_dir = os.path.join(base_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for i in range(25): # 25 samples per class
        signal = generate_noise(duration=2.0, freq_mod=freq_mod)
        signal += np.random.normal(0, 0.1, len(signal))
        signal = np.int16(signal * 32767)
        wavfile.write(os.path.join(cls_dir, f"sample_{i:03d}.wav"), 22050, signal)

print("Dataset created successfully at `data/raw/` with classes:", list(classes.keys()))'''),

        ("markdown", "### 4. Exploratory Data Analysis (EDA)\nLet's visualize the waveforms and Spectrograms to understand the audio signals."),
        ("code", '''def plot_waveform_and_spectrogram(class_name):
    class_dir = os.path.join(base_dir, class_name)
    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    if not files: return
    file_path = os.path.join(class_dir, files[0])
    y, sr = librosa.load(file_path, sr=22050)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - {class_name}')
    
    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {class_name}')
    plt.tight_layout()
    plt.show()

for c in classes.keys():
    plot_waveform_and_spectrogram(c)'''),

        ("markdown", "### 5. Feature Engineering\nWe extract 1D features (MFCC, Zero Crossing Rate, Chroma) for Machine Learning, and 2D features (Mel Spectrograms) for Deep Learning."),
        ("code", '''def extract_ml_features(audio, sr=22050):
    mfcc_mean = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    stft = np.abs(librosa.stft(audio))
    chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    return np.hstack([mfcc_mean, zcr_mean, chroma_mean])

def extract_dl_features(audio, sr=22050, max_pad_len=100):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] > max_pad_len:
        mel_spec_db = mel_spec_db[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
    return mel_spec_db

def load_dataset(feature_type='ml'):
    data = []
    labels = []
    class_mapping = {label: idx for idx, label in enumerate(classes.keys())}
    
    for label in classes.keys():
        class_dir = os.path.join(base_dir, label)
        for val in os.listdir(class_dir):
            if not val.endswith('.wav'): continue
            file_path = os.path.join(class_dir, val)
            audio, sr = librosa.load(file_path, sr=22050)
            
            # Simple Augmentation inline (add noise)
            audio_noise = audio + 0.005 * np.random.randn(len(audio))
            
            for a in [audio, audio_noise]:
                if feature_type == 'ml':
                    data.append(extract_ml_features(a, sr))
                else:
                    data.append(extract_dl_features(a, sr))
                labels.append(class_mapping[label])
                
    return np.array(data), np.array(labels), class_mapping'''),

        ("markdown", "### 6. Machine Learning Model (Random Forest)\nLet's train a Random Forest on the 1D acoustic features."),
        ("code", '''print("Extracting ML Features...")
X_ml, y_ml, class_mapping = load_dataset('ml')
class_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]

X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_ml = scaler.fit_transform(X_train_ml)
X_test_ml = scaler.transform(X_test_ml)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
print("Training Random Forest...")
rf.fit(X_train_ml, y_train_ml)

y_pred_ml = rf.predict(X_test_ml)
print(f"Random Forest Accuracy: {accuracy_score(y_test_ml, y_pred_ml):.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_ml, y_pred_ml), annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Random Forest Confusion Matrix")
plt.show()'''),

        ("markdown", "### 7. Deep Learning Model (CNN)\nTraining a 2D Convolutional Neural Network directly on the extracted Mel Spectrogram images."),
        ("code", '''print("Extracting DL Features...")
X_dl, y_dl, _ = load_dataset('dl')

# Add channel dimension for CNN
X_dl = X_dl[..., np.newaxis]
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, random_state=42)

input_shape = X_train_dl.shape[1:]

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"CNN Model Summary:")
model.summary()'''),

        ("code", '''print("Training CNN...")
history = model.fit(X_train_dl, y_train_dl, epochs=15, batch_size=32, validation_split=0.2, verbose=1)'''),

        ("markdown", "### 8. Evaluating Deep Learning Model"),
        ("code", '''y_pred_dl = np.argmax(model.predict(X_test_dl), axis=1)
print(f"CNN Accuracy: {accuracy_score(y_test_dl, y_pred_dl):.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('CNN Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test_dl, y_pred_dl), annot=True, cmap='Purples', xticklabels=class_names, yticklabels=class_names)
plt.title('CNN Confusion Matrix')
plt.tight_layout()
plt.show()

print("Presentation complete. Models have successfully trained and inferred infant states from audio.")''')
    ]
    
    create_notebook('Infant_State_Recognition_Master_Colab.ipynb', cells)
    print("Master Colab Notebook created.")

if __name__ == "__main__":
    main()

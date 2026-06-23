import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/Users/piyush/Infant-state-recognisation/infant-state-recognition/.mplconfig",
)

import joblib
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.feature_engineering import extract_spectrogram
from src.preprocessing import preprocess_pipeline

MODEL_PATH = BASE_DIR / "models" / "DeepLearning_MobileNetV2_best.keras"
CLASS_MAP_PATH = BASE_DIR / "models" / "class_mapping.pkl"


def build_mobilenet_input(audio, sr=22050, image_size=160):
    spectrogram = extract_spectrogram(audio, sr=sr).astype(np.float32)
    mean = spectrogram.mean()
    std = spectrogram.std() + 1e-6
    mel = (spectrogram - mean) / std

    delta = librosa.feature.delta(mel).astype(np.float32)
    delta2 = librosa.feature.delta(mel, order=2).astype(np.float32)
    stacked = np.stack([mel, delta, delta2], axis=-1)

    min_v = stacked.min()
    max_v = stacked.max()
    stacked = (stacked - min_v) / (max_v - min_v + 1e-8) * 255.0
    resized = tf.image.resize(stacked[np.newaxis, ...], (image_size, image_size)).numpy()
    return preprocess_input(resized)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: mobilenet_runner.py <audio_path>")

    audio_path = sys.argv[1]
    audios, sr = preprocess_pipeline(audio_path, augment=False)
    model_input = build_mobilenet_input(audios[0], sr=sr)
    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(model_input, verbose=0)[0].astype(float)

    class_mapping = joblib.load(CLASS_MAP_PATH)
    inv_class_mapping = {int(idx): label for label, idx in class_mapping.items()}
    pred_idx = int(np.argmax(probabilities))

    print(
        json.dumps(
            {
                "pred_idx": pred_idx,
                "prediction": inv_class_mapping.get(pred_idx, str(pred_idx)),
                "probabilities": probabilities.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()

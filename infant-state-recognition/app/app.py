import json
import os
import importlib.util
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import extract_features_for_dim, extract_spectrogram
from src.preprocessing import preprocess_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "inference_inputs"
LOG_PATH = UPLOAD_DIR / "prediction_log.jsonl"
CLASS_MAP_PATH = MODELS_DIR / "class_mapping.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

MODEL_SPECS = {
    "baseline_lr": {
        "label": "Logistic Regression",
        "category": "Baseline ML",
        "type": "sklearn",
        "path": MODELS_DIR / "baseline_lr.pkl",
    },
    "baseline_dt": {
        "label": "Decision Tree",
        "category": "Baseline ML",
        "type": "sklearn",
        "path": MODELS_DIR / "baseline_dt.pkl",
    },
    "advanced_rf": {
        "label": "Random Forest",
        "category": "Advanced ML",
        "type": "sklearn",
        "path": MODELS_DIR / "advanced_rf.pkl",
    },
    "advanced_svm": {
        "label": "SVM (RBF)",
        "category": "Advanced ML",
        "type": "sklearn",
        "path": MODELS_DIR / "advanced_svm.pkl",
    },
    "advanced_xgb": {
        "label": "XGBoost",
        "category": "Advanced ML",
        "type": "xgboost",
        "path": MODELS_DIR / "advanced_xgb.json",
    },
    "mobilenet_v2": {
        "label": "MobileNetV2",
        "category": "Deep Learning",
        "type": "keras",
        "path": MODELS_DIR / "DeepLearning_MobileNetV2_best.keras",
    },
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_class_mapping():
    if not CLASS_MAP_PATH.exists():
        return {}
    class_mapping = joblib.load(CLASS_MAP_PATH)
    return {int(idx): label for label, idx in class_mapping.items()}


def load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


def load_available_models():
    loaded_models = {}
    for model_key, spec in MODEL_SPECS.items():
        if not spec["path"].exists():
            continue

        loaded_spec = dict(spec)
        loaded_spec["artifact"] = None

        if spec["type"] == "sklearn":
            loaded_spec["artifact"] = joblib.load(spec["path"])
        elif spec["type"] == "xgboost":
            try:
                import xgboost as xgb
            except ImportError:
                continue
            booster = xgb.Booster()
            booster.load_model(str(spec["path"]))
            loaded_spec["artifact"] = booster
        elif spec["type"] == "keras":
            # Lazy-load Keras model only when selected for inference.
            loaded_spec["artifact"] = None

        loaded_models[model_key] = loaded_spec

    return loaded_models


INV_CLASS_MAPPING = load_class_mapping()
SCALER = load_scaler()
AVAILABLE_MODELS = load_available_models()
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def get_keras_model(model_key):
    spec = AVAILABLE_MODELS[model_key]
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not installed in this Python environment, so MobileNetV2 inference is unavailable."
        )
    if spec["artifact"] is None:
        import tensorflow as tf

        spec["artifact"] = tf.keras.models.load_model(spec["path"])
    return spec["artifact"]


def build_mobilenet_input(audio, sr=22050, image_size=160):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    spectrogram = extract_spectrogram(audio, sr=sr).astype(np.float32)
    mean = spectrogram.mean()
    std = spectrogram.std() + 1e-6
    mel = (spectrogram - mean) / std

    delta = extract_delta(mel)
    delta2 = extract_delta(mel, order=2)
    stacked = np.stack([mel, delta, delta2], axis=-1)

    min_v = stacked.min()
    max_v = stacked.max()
    stacked = (stacked - min_v) / (max_v - min_v + 1e-8) * 255.0
    resized = tf.image.resize(stacked[np.newaxis, ...], (image_size, image_size)).numpy()
    return preprocess_input(resized)


def extract_delta(sample, order=1):
    import librosa

    return librosa.feature.delta(sample, order=order).astype(np.float32)


def save_upload(file_storage):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    safe_name = secure_filename(file_storage.filename or "audio.wav")
    dated_dir = UPLOAD_DIR / datetime.utcnow().strftime("%Y-%m-%d")
    dated_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{timestamp}_{safe_name}"
    destination = dated_dir / stored_name
    file_storage.save(destination)
    return destination


def append_log(record):
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


def read_recent_logs(limit=8):
    if not LOG_PATH.exists():
        return []

    with LOG_PATH.open("r", encoding="utf-8") as log_file:
        rows = [json.loads(line) for line in log_file if line.strip()]

    return list(reversed(rows[-limit:]))


def get_audio_and_features(filepath):
    audios, sr = preprocess_pipeline(str(filepath), augment=False)
    return audios[0], sr


def predict_with_model(model_key, filepath):
    if model_key not in AVAILABLE_MODELS:
        raise ValueError("Selected model is not available in the repo.")

    audio, sr = get_audio_and_features(filepath)
    model_spec = AVAILABLE_MODELS[model_key]

    if model_spec["type"] in {"sklearn", "xgboost"}:
        if SCALER is None:
            raise RuntimeError("Scaler not found. Re-train or restore ML artifacts first.")

        expected_dim = getattr(SCALER, "n_features_in_", None)
        features = extract_features_for_dim(audio, sr, n_features=expected_dim)
        scaled_features = SCALER.transform([features])

        if model_spec["type"] == "sklearn":
            model = model_spec["artifact"]
            pred_idx = int(model.predict(scaled_features)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_features)[0]
            else:
                scores = model.decision_function(scaled_features)
                exp_scores = np.exp(scores - np.max(scores))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                probabilities = probabilities[0]
        else:
            import xgboost as xgb

            dmatrix = xgb.DMatrix(scaled_features)
            probabilities = model_spec["artifact"].predict(dmatrix)[0]
            pred_idx = int(np.argmax(probabilities))

        return pred_idx, np.asarray(probabilities, dtype=float)

    model = get_keras_model(model_key)
    model_input = build_mobilenet_input(audio, sr=sr)
    probabilities = model.predict(model_input, verbose=0)[0]
    pred_idx = int(np.argmax(probabilities))
    return pred_idx, np.asarray(probabilities, dtype=float)


@app.route("/", methods=["GET"])
def index():
    recommended_key = "mobilenet_v2" if TENSORFLOW_AVAILABLE and "mobilenet_v2" in AVAILABLE_MODELS else "advanced_svm"
    if recommended_key not in AVAILABLE_MODELS and AVAILABLE_MODELS:
        recommended_key = next(iter(AVAILABLE_MODELS))

    model_options = [
        {
            "key": key,
            "label": spec["label"],
            "category": spec["category"],
            "recommended": key == recommended_key,
            "runtime_note": (
                "Requires TensorFlow runtime"
                if spec["type"] == "keras" and not TENSORFLOW_AVAILABLE
                else ""
            ),
        }
        for key, spec in AVAILABLE_MODELS.items()
    ]
    return render_template(
        "index.html",
        models=model_options,
        default_model_label=AVAILABLE_MODELS[recommended_key]["label"] if recommended_key else "Unavailable",
        tensorflow_available=TENSORFLOW_AVAILABLE,
        recent_predictions=read_recent_logs(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not AVAILABLE_MODELS:
        return jsonify({"error": "No trained models were found in the repo."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No audio file was provided."}), 400

    file = request.files["file"]
    model_key = request.form.get("model_key", "mobilenet_v2")

    if file.filename == "":
        return jsonify({"error": "Please choose an audio file."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {
                "error": "Unsupported file type. Use wav, mp3, ogg, flac, or m4a."
            }
        ), 400

    saved_path = save_upload(file)

    try:
        pred_idx, probabilities = predict_with_model(model_key, saved_path)
        prediction = INV_CLASS_MAPPING.get(pred_idx, str(pred_idx))
        confidence = float(np.max(probabilities))

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "label": INV_CLASS_MAPPING.get(int(idx), str(idx)),
                "confidence": round(float(probabilities[idx]) * 100, 2),
            }
            for idx in top_indices
        ]

        model_spec = AVAILABLE_MODELS[model_key]
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        record = {
            "timestamp": timestamp,
            "model_key": model_key,
            "model_label": model_spec["label"],
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "saved_path": str(saved_path.relative_to(BASE_DIR)),
            "original_filename": file.filename,
        }
        append_log(record)

        return jsonify(
            {
                "status": "success",
                "prediction": prediction,
                "confidence": f"{confidence * 100:.2f}%",
                "model": model_spec["label"],
                "saved_path": record["saved_path"],
                "timestamp": timestamp,
                "top_predictions": top_predictions,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/recent-predictions", methods=["GET"])
def recent_predictions():
    return jsonify({"items": read_recent_logs()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

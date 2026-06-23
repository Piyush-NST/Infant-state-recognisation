import json
import os
import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"),
)

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import extract_features_for_dim, extract_spectrogram
from src.preprocessing import preprocess_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "inference_inputs"
LOG_PATH = UPLOAD_DIR / "prediction_log.jsonl"
CLASS_MAP_PATH = MODELS_DIR / "class_mapping.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
TF_PYTHON_PATH = PROJECT_ROOT / ".venv-tf" / "bin" / "python"
MOBILENET_RUNNER_PATH = BASE_DIR / "app" / "mobilenet_runner.py"
DERIVED_SCALER_TEMPLATE = "scaler_{n_features}_auto.pkl"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
DATASET_AUDIO_EXTENSIONS = {".wav"}
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

MODEL_SPECS = {
    "baseline_lr": {
        "label": "Logistic Regression",
        "category": "Baseline ML",
        "type": "sklearn",
        "path": MODELS_DIR / "baseline_lr.pkl",
    },
    # "baseline_dt": {
    #     "label": "Decision Tree",
    #     "category": "Baseline ML",
    #     "type": "sklearn",
    #     "path": MODELS_DIR / "baseline_dt.pkl",
    # },
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


def can_delegate_keras():
    return TF_PYTHON_PATH.exists() and MOBILENET_RUNNER_PATH.exists()


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
        if spec["type"] == "keras" and not TENSORFLOW_AVAILABLE and not can_delegate_keras():
            continue

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
            # Lazy-load locally, or delegate to the TensorFlow venv at prediction time.
            loaded_spec["artifact"] = None

        loaded_models[model_key] = loaded_spec

    return loaded_models


def build_model_options():
    available_model_key = get_default_model_key()
    options = []

    for model_key, spec in MODEL_SPECS.items():
        exists_on_disk = spec["path"].exists()
        is_available = model_key in AVAILABLE_MODELS

        status_note = ""
        if spec["type"] == "keras" and not TENSORFLOW_AVAILABLE and can_delegate_keras():
            status_note = "Runs through TensorFlow environment"
        elif spec["type"] == "keras" and not TENSORFLOW_AVAILABLE:
            status_note = "TensorFlow environment missing"
        elif not exists_on_disk:
            status_note = "Model file missing"

        options.append(
            {
                "key": model_key,
                "label": spec["label"],
                "category": spec["category"],
                "recommended": model_key == available_model_key,
                "disabled": not is_available,
                "status_note": status_note,
            }
        )

    return options


INV_CLASS_MAPPING = load_class_mapping()
SCALER = load_scaler()
AVAILABLE_MODELS = load_available_models()
SCALER_CACHE = {}
if SCALER is not None and getattr(SCALER, "n_features_in_", None) is not None:
    SCALER_CACHE[int(SCALER.n_features_in_)] = SCALER


def get_default_model_key():
    preferred_order = ["advanced_svm", "advanced_rf", "advanced_xgb", "baseline_lr", "baseline_dt"]
    for model_key in preferred_order:
        if model_key in AVAILABLE_MODELS:
            return model_key
    return next(iter(AVAILABLE_MODELS), None)


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


def run_keras_via_tensorflow_env(filepath):
    if not can_delegate_keras():
        raise RuntimeError("TensorFlow environment is not ready for MobileNetV2 inference.")

    completed = subprocess.run(
        [str(TF_PYTHON_PATH), str(MOBILENET_RUNNER_PATH), str(filepath)],
        cwd=str(BASE_DIR),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return int(payload["pred_idx"]), np.asarray(payload["probabilities"], dtype=float)


def iter_dataset_audio_files():
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        return

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for filepath in sorted(class_dir.iterdir()):
            if filepath.suffix.lower() in DATASET_AUDIO_EXTENSIONS:
                yield filepath


def get_scaler_for_dim(n_features):
    if n_features is None:
        return None

    n_features = int(n_features)
    if n_features in SCALER_CACHE:
        return SCALER_CACHE[n_features]

    scaler_path = MODELS_DIR / DERIVED_SCALER_TEMPLATE.format(n_features=n_features)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        SCALER_CACHE[n_features] = scaler
        return scaler

    from sklearn.preprocessing import StandardScaler

    feature_rows = []
    for filepath in iter_dataset_audio_files() or []:
        try:
            audios, sr = preprocess_pipeline(str(filepath), augment=False)
            feature_rows.append(extract_features_for_dim(audios[0], sr, n_features=n_features))
        except Exception:
            continue

    if not feature_rows:
        raise RuntimeError(
            f"Could not build a scaler for {n_features} features because no dataset audio could be processed."
        )

    scaler = StandardScaler().fit(np.asarray(feature_rows))
    joblib.dump(scaler, scaler_path)
    SCALER_CACHE[n_features] = scaler
    return scaler


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
        if model_spec["type"] == "sklearn":
            expected_dim = getattr(model_spec["artifact"], "n_features_in_", None)
        elif model_spec["type"] == "xgboost":
            expected_dim = model_spec["artifact"].num_features()
        else:
            expected_dim = getattr(SCALER, "n_features_in_", 53) if SCALER is not None else 53

        features = extract_features_for_dim(audio, sr, n_features=expected_dim)
        scaler = get_scaler_for_dim(expected_dim)
        scaled_features = scaler.transform([features]) if scaler is not None else np.asarray([features])

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

    if not TENSORFLOW_AVAILABLE:
        return run_keras_via_tensorflow_env(filepath)

    model = get_keras_model(model_key)
    model_input = build_mobilenet_input(audio, sr=sr)
    probabilities = model.predict(model_input, verbose=0)[0]
    pred_idx = int(np.argmax(probabilities))
    return pred_idx, np.asarray(probabilities, dtype=float)


@app.route("/", methods=["GET"])
def index():
    recommended_key = get_default_model_key()
    return render_template(
        "index.html",
        models=build_model_options(),
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
    model_key = request.form.get("model_key") or get_default_model_key()

    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": "Selected model is not available in this environment."}), 400

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
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

import json
import os
import random
import sys
from dataclasses import dataclass

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate_model, plot_training_history
from src.feature_engineering import load_data


@dataclass
class TrainingConfig:
    data_dir: str = "data/raw"
    model_name: str = "DeepLearning_MobileNetV2"
    image_size: int = 160
    batch_size: int = 32
    initial_epochs: int = 22
    fine_tune_epochs: int = 8
    validation_split: float = 0.15
    test_size: float = 0.20
    random_state: int = 42
    augment: bool = True
    unfreeze_last_layers: int = 40
    use_class_weights: bool = True
    class_weight_power: float = 0.5
    initial_lr: float = 1e-3
    fine_tune_lr: float = 5e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_model_comparison():
    """Print and plot metric comparison for all available trained models."""
    metrics_dir = "outputs"
    model_order = [
        "Baseline_LogReg",
        "Baseline_DecisionTree",
        "Advanced_RandomForest",
        "Advanced_SVM",
        "Advanced_XGBoost",
        "DeepLearning_CNN",
        "DeepLearning_MobileNetV2",
        "DeepLearning_ResNet50",
    ]

    results = {}
    for name in model_order:
        path = os.path.join(metrics_dir, f"{name}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)

    if not results:
        print("No metrics found. Run training scripts first.")
        return

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("=" * 70)
    for name, metrics in results.items():
        print(
            f"{name:<30} "
            f"{metrics['Accuracy']:>10.4f} "
            f"{metrics['Precision']:>10.4f} "
            f"{metrics['Recall']:>10.4f} "
            f"{metrics['F1_Score']:>10.4f}"
        )
    print("=" * 70)

    names = list(results.keys())
    accuracies = [results[name]["Accuracy"] for name in names]
    f1_scores = [results[name]["F1_Score"] for name in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - 0.2, accuracies, width=0.35, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + 0.2, f1_scores, width=0.35, label="F1-Score", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Infant Cry Recognition — All Models Comparison")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=7)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=7)

    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/all_models_comparison.png", dpi=150)
    plt.close()
    print("\nComparison chart saved to outputs/all_models_comparison.png")


def make_mobilenet_inputs(spectrograms, image_size):
    """
    Build 3-channel spectrogram images:
    channel-0: normalized log-mel
    channel-1: delta
    channel-2: delta-delta
    """
    specs = spectrograms.astype(np.float32)
    mean = specs.mean(axis=(1, 2), keepdims=True)
    std = specs.std(axis=(1, 2), keepdims=True) + 1e-6
    mel = (specs - mean) / std

    delta = np.stack([librosa.feature.delta(sample) for sample in mel], axis=0).astype(np.float32)
    delta2 = np.stack([librosa.feature.delta(sample, order=2) for sample in mel], axis=0).astype(np.float32)
    stacked = np.stack([mel, delta, delta2], axis=-1)

    min_v = stacked.min(axis=(1, 2, 3), keepdims=True)
    max_v = stacked.max(axis=(1, 2, 3), keepdims=True)
    stacked = (stacked - min_v) / (max_v - min_v + 1e-8) * 255.0

    resized = tf.image.resize(stacked, (image_size, image_size)).numpy()
    return preprocess_input(resized)


def build_model(input_shape, num_classes, learning_rate):
    """Build MobileNetV2 transfer-learning model."""
    try:
        backbone = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        print("Loaded ImageNet weights for MobileNetV2.")
    except Exception as exc:
        print(f"Warning: could not load ImageNet weights ({exc}). Using random init.")
        backbone = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=input_shape,
        )

    backbone.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, backbone


def get_class_weights(y_train, power=0.5):
    """Soften class weights with exponent to avoid over-correcting imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = np.power(weights, power)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_callbacks(model_name):
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", f"{model_name}_best.keras")
    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return checkpoint_path, callback_list


def merge_histories(history_a, history_b):
    merged = {}
    for key, values in history_a.history.items():
        merged[key] = list(values)
    for key, values in history_b.history.items():
        merged.setdefault(key, [])
        merged[key].extend(values)

    class History:
        pass

    history = History()
    history.history = merged
    return history


def split_with_full_class_coverage(X, y, test_size, random_state):
    """
    Build train/test split while ensuring training side contains all labels.
    """
    all_labels = np.unique(y)
    for offset in range(30):
        rs = random_state + offset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=rs,
            stratify=y,
        )
        if np.array_equal(np.unique(y_train), all_labels):
            return X_train, X_test, y_train, y_test

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def main():
    config = TrainingConfig()
    set_seed(config.random_state)

    print("Loading Data for Deep Learning (all supported audio formats)...")
    X, y, class_mapping = load_data(
        data_dir=config.data_dir,
        augment=config.augment,
        feature_type="dl",
    )

    if len(X) == 0:
        print("No data found. Place dataset folders under data/raw/<class_name>/")
        return

    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Samples: {len(X)}")

    X = make_mobilenet_inputs(X, image_size=config.image_size)

    X_train, X_test, y_train, y_test = split_with_full_class_coverage(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Creating MobileNetV2 model for {num_classes} classes...")
    model, backbone = build_model(X_train.shape[1:], num_classes, config.initial_lr)
    model.summary()

    class_weight = None
    if config.use_class_weights:
        class_weight = get_class_weights(y_train, power=config.class_weight_power)
        print(f"Class weights (soft): {class_weight}")

    best_checkpoint_path, callback_list = get_callbacks(config.model_name)

    print("\nStage 1: training classifier head...")
    head_history = model.fit(
        X_train,
        y_train,
        epochs=config.initial_epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        class_weight=class_weight,
        callbacks=callback_list,
        verbose=2,
    )

    print("\nStage 2: fine-tuning top backbone layers...")
    backbone.trainable = True
    for layer in backbone.layers[:-config.unfreeze_last_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        X_train,
        y_train,
        epochs=config.fine_tune_epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        class_weight=class_weight,
        callbacks=callback_list,
        verbose=2,
    )

    history = merge_histories(head_history, fine_tune_history)

    if os.path.exists(best_checkpoint_path):
        print("Loading best validation checkpoint before final evaluation...")
        model = tf.keras.models.load_model(best_checkpoint_path)

    print("\nEvaluating MobileNetV2...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    evaluate_model(y_test, y_pred, config.model_name, class_names)
    plot_training_history(history, config.model_name)

    print("Saving MobileNetV2 model (.h5, .keras) and TFLite...")
    os.makedirs("models", exist_ok=True)
    model.save("models/mobilenetv2_model.h5")
    model.save("models/mobilenetv2_model.keras")
    joblib.dump(class_mapping, "models/class_mapping.pkl")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("models/mobilenetv2_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("\n--- All Models Comparison ---")
    print_model_comparison()
    print("Done!")


if __name__ == "__main__":
    main()

import json
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, ParameterSampler, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate_model
from src.feature_engineering import load_data


@dataclass
class MLConfig:
    data_dir: str = "data/raw"
    test_size: float = 0.20
    validation_size: float = 0.15
    random_state: int = 42
    cv_folds: int = 3
    xgb_trials: int = 24
    xgb_early_stopping_rounds: int = 40


def get_safe_cv(y_train, requested_folds=3, random_state=42):
    _, counts = np.unique(y_train, return_counts=True)
    min_count = int(np.min(counts))
    folds = max(2, min(requested_folds, min_count))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state), folds


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def split_with_full_class_coverage(X, y, test_size, random_state):
    """
    Create a split where training part keeps all classes.
    If class counts are very low, reduce test size progressively.
    """
    all_labels = np.unique(y)
    for shrink in [1.0, 0.8, 0.6, 0.4, 0.25]:
        current_size = max(0.05, test_size * shrink)
        for offset in range(20):
            rs = random_state + offset
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y,
                test_size=current_size,
                random_state=rs,
                stratify=y,
            )
            if np.array_equal(np.unique(y_tr), all_labels):
                return X_tr, X_val, y_tr, y_val
    return X, X, y, y


def main():
    config = MLConfig()
    print("Loading Data for Advanced ML...")
    X, y, class_mapping = load_data(data_dir=config.data_dir, augment=True, feature_type="ml")

    if len(X) == 0:
        print("No data found. Place dataset folders under data/raw/<class_name>/")
        return

    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"Classes found: {class_names} ({num_classes} classes, {len(X)} samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv_splitter, used_folds = get_safe_cv(
        y_train, requested_folds=config.cv_folds, random_state=config.random_state
    )
    print(f"Using stratified CV with {used_folds} folds")

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(random_state=config.random_state, n_jobs=-1)
    param_grid_rf = {
        "n_estimators": [50, 100, 300],
        "max_depth": [None, 10, 20],
    }
    grid_rf = GridSearchCV(
        rf,
        param_grid_rf,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1,
    )
    grid_rf.fit(X_train_scaled, y_train)
    best_rf = grid_rf.best_estimator_
    print(f"Best RF Params: {grid_rf.best_params_}")
    y_pred_rf = best_rf.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_rf, "Advanced_RandomForest", class_names)
    save_json("outputs/Advanced_RandomForest_best_params.json", grid_rf.best_params_)

    print("\nTraining SVM...")
    best_svm = SVC(kernel="rbf", probability=True, random_state=config.random_state)
    best_svm.fit(X_train_scaled, y_train)
    y_pred_svm = best_svm.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_svm, "Advanced_SVM", class_names)
    save_json(
        "outputs/Advanced_SVM_best_params.json",
        {"kernel": "rbf", "probability": True, "random_state": config.random_state},
    )

    print("\nTraining XGBoost (manual robust search + early stopping)...")
    X_tr, X_val, y_tr, y_val = split_with_full_class_coverage(
        X_train_scaled,
        y_train,
        test_size=config.validation_size,
        random_state=config.random_state,
    )

    param_space = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
        "reg_lambda": [1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5],
    }

    best_score = -1.0
    best_params = None
    best_model = None

    for i, params in enumerate(
        ParameterSampler(param_space, n_iter=config.xgb_trials, random_state=config.random_state),
        start=1,
    ):
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val if X_tr is not X_val else X_tr, label=y_val if X_tr is not X_val else y_tr)

        xgb_params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": config.random_state,
            "eta": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "gamma": params["gamma"],
            "lambda": params["reg_lambda"],
            "alpha": params["reg_alpha"],
            "nthread": -1,
        }

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dvalid, "valid")],
            early_stopping_rounds=None if X_tr is X_val else config.xgb_early_stopping_rounds,
            verbose_eval=False,
        )
        y_val_prob = model.predict(dvalid)
        y_val_pred = np.argmax(y_val_prob, axis=1)
        target_val = y_val if X_tr is not X_val else y_tr
        score = f1_score(target_val, y_val_pred, average="weighted", zero_division=0)
        print(f"XGBoost trial {i}/{config.xgb_trials} | val_f1={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    if best_model is None:
        raise RuntimeError("XGBoost search failed to produce a model.")

    print(f"Best XGBoost Params: {best_params}")
    print(f"Best XGBoost validation F1: {best_score:.4f}")
    dtest = xgb.DMatrix(X_test_scaled)
    y_pred_xgb = np.argmax(best_model.predict(dtest), axis=1)
    evaluate_model(y_test, y_pred_xgb, "Advanced_XGBoost", class_names)
    save_json("outputs/Advanced_XGBoost_best_params.json", best_params)

    print("\nSaving models...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/advanced_rf.pkl")
    joblib.dump(best_svm, "models/advanced_svm.pkl")
    try:
        joblib.dump(best_model, "models/advanced_xgb.pkl")
    except Exception:
        pass
    best_model.save_model("models/advanced_xgb.json")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(class_mapping, "models/class_mapping.pkl")
    print("Done! Models saved to models/")


if __name__ == "__main__":
    main()

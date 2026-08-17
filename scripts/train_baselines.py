#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ["Low", "Moderate", "High"]
NUM_CLASSES = 3


# ============================================================
# DATA LOADING
# ============================================================

def load_split(
    root: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all NPZ sequences from one split.

    Each original sequence:
        X.shape = (T, F), e.g. (6, 26)

    It is converted to a fixed-size feature vector:
        mean over time -> F
        std over time  -> F

    Final:
        2 * F = 52 features
    """

    paths = sorted(
        (root / split).glob("*.npz")
    )

    if not paths:
        raise RuntimeError(
            f"No NPZ files found in {root / split}"
        )

    X_all = []
    y_all = []
    session_ids = []

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        X = data["X"].astype(
            np.float32
        )

        y = int(
            data["y"]
        )

        # -----------------------------
        # Handle possible NaN / Inf
        # -----------------------------

        X = np.where(
            np.isfinite(X),
            X,
            np.nan,
        )

        mean_features = np.nanmean(
            X,
            axis=0,
        )

        std_features = np.nanstd(
            X,
            axis=0,
        )

        vector = np.concatenate(
            [
                mean_features,
                std_features,
            ],
            axis=0,
        )

        # Remaining missing values
        # will later be replaced using
        # training statistics.
        X_all.append(vector)
        y_all.append(y)

        if "session_id" in data.files:
            session_ids.append(
                str(data["session_id"])
            )
        else:
            session_ids.append(
                path.stem
            )

    return (
        np.asarray(
            X_all,
            dtype=np.float32,
        ),
        np.asarray(
            y_all,
            dtype=np.int64,
        ),
        session_ids,
    )


# ============================================================
# IMPUTATION
# ============================================================

def compute_imputation_values(
    X_train: np.ndarray,
) -> np.ndarray:
    """
    Median imputation values calculated
    using training data only.
    """

    values = np.nanmedian(
        X_train,
        axis=0,
    )

    values = np.where(
        np.isfinite(values),
        values,
        0.0,
    )

    return values.astype(
        np.float32
    )


def apply_imputation(
    X: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:

    X = X.copy()

    invalid = ~np.isfinite(X)

    if invalid.any():
        rows, cols = np.where(
            invalid
        )

        X[rows, cols] = values[
            cols
        ]

    return X.astype(
        np.float32
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "balanced_accuracy": float(
            bal_acc
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report,
    }


def print_metrics(
    model_name: str,
    split_name: str,
    metrics: Dict,
):

    print(
        f"\n===== {model_name} - "
        f"{split_name} ====="
    )

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision:   "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:      "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    print(
        "\nPer-class:"
    )

    report = metrics[
        "classification_report"
    ]

    for class_name in CLASS_NAMES:

        cls = report[
            class_name
        ]

        print(
            f"{class_name:10s} "
            f"P={cls['precision']:.4f} "
            f"R={cls['recall']:.4f} "
            f"F1={cls['f1-score']:.4f}"
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
) -> Dict:

    print(
        "\n======================================"
    )
    print(
        f"TRAINING: {name}"
    )
    print(
        "======================================"
    )

    model.fit(
        X_train,
        y_train,
    )

    # Validation
    val_pred = model.predict(
        X_val
    )

    val_metrics = compute_metrics(
        y_val,
        val_pred,
    )

    print_metrics(
        name,
        "VALIDATION",
        val_metrics,
    )

    # Test
    test_pred = model.predict(
        X_test
    )

    test_metrics = compute_metrics(
        y_test,
        test_pred,
    )

    print_metrics(
        name,
        "TEST",
        test_metrics,
    )

    # Probabilities
    if hasattr(
        model,
        "predict_proba",
    ):
        test_prob = model.predict_proba(
            X_test
        )
    else:
        test_prob = np.full(
            (
                len(y_test),
                NUM_CLASSES,
            ),
            np.nan,
            dtype=np.float32,
        )

    model_file = (
        save_dir
        / f"{name.lower().replace(' ', '_')}.joblib"
    )

    joblib.dump(
        model,
        model_file,
    )

    predictions_file = (
        save_dir
        / (
            f"{name.lower().replace(' ', '_')}"
            "_test_predictions.npz"
        )
    )

    np.savez_compressed(
        predictions_file,
        y_true=y_test,
        y_pred=test_pred,
        probabilities=test_prob,
        class_names=np.asarray(
            CLASS_NAMES
        ),
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "model_file": str(
            model_file
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/processed/drozy",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="runs/fatigue_drozy_baselines",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    root = Path(
        args.features_dir
    )

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    X_train, y_train, _ = load_split(
        root,
        "train",
    )

    X_val, y_val, _ = load_split(
        root,
        "val",
    )

    X_test, y_test, _ = load_split(
        root,
        "test",
    )

    print(
        "======================================"
    )
    print(
        "DROZY MULTIMODAL BASELINES"
    )
    print(
        "======================================"
    )

    print(
        "Train:",
        X_train.shape,
        np.bincount(
            y_train,
            minlength=3,
        ),
    )

    print(
        "Val:  ",
        X_val.shape,
        np.bincount(
            y_val,
            minlength=3,
        ),
    )

    print(
        "Test: ",
        X_test.shape,
        np.bincount(
            y_test,
            minlength=3,
        ),
    )

    print(
        "\nOriginal sequence: "
        "(6, 26)"
    )

    print(
        f"Baseline input size: "
        f"{X_train.shape[1]}"
    )

    # ========================================================
    # IMPUTATION
    # ========================================================

    imputation_values = (
        compute_imputation_values(
            X_train
        )
    )

    X_train = apply_imputation(
        X_train,
        imputation_values,
    )

    X_val = apply_imputation(
        X_val,
        imputation_values,
    )

    X_test = apply_imputation(
        X_test,
        imputation_values,
    )

    # ========================================================
    # MODELS
    # ========================================================

    logistic_regression = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    C=1.0,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    random_forest = (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
    )

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    results = {}

    results[
        "Logistic Regression"
    ] = evaluate_model(
        name="Logistic Regression",
        model=logistic_regression,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        save_dir=save_dir,
    )

    results[
        "Random Forest"
    ] = evaluate_model(
        name="Random Forest",
        model=random_forest,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        save_dir=save_dir,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n======================================"
    )
    print(
        "FINAL TEST SUMMARY"
    )
    print(
        "======================================"
    )

    print(
        f"{'Model':25s} "
        f"{'Acc':>8s} "
        f"{'BalAcc':>8s} "
        f"{'MacroF1':>8s}"
    )

    print(
        "-" * 55
    )

    for (
        model_name,
        result,
    ) in results.items():

        test = result[
            "test"
        ]

        print(
            f"{model_name:25s} "
            f"{test['accuracy']:8.4f} "
            f"{test['balanced_accuracy']:8.4f} "
            f"{test['macro_f1']:8.4f}"
        )

    # Save experiment summary
    summary_path = (
        save_dir
        / "baseline_metrics.json"
    )

    summary_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save imputation parameters
    np.save(
        save_dir
        / "imputation_values.npy",
        imputation_values,
    )

    print(
        f"\nSaved results to: "
        f"{save_dir}"
    )


if __name__ == "__main__":
    main()
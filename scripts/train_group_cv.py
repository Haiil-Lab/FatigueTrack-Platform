#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fatiguetrack.dataset import (
    NPZSequenceDataset,
    collate_batch,
)

from fatiguetrack.model_ns import AttentionLSTM


CLASS_NAMES = ["Low", "Moderate", "High"]
NUM_CLASSES = 3

# First seven columns in your current multimodal representation
# are the behavioral features.
N_BEHAVIORAL_FEATURES = 7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD DATA
# ============================================================

def load_all_samples(
    features_dir: Path,
    drop_missing_behavioral: bool = True,
) -> Tuple[List[Path], np.ndarray, np.ndarray]:

    paths = sorted(
        features_dir.rglob("*.npz")
    )

    valid_paths = []
    labels = []
    groups = []

    dropped = 0

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        X = data["X"].astype(
            np.float32
        )

        y = int(data["y"])

        subject_id = int(
            data["subject_id"]
        )

        # Optionally remove samples where the entire
        # behavioral branch is missing.
        if drop_missing_behavioral:

            behavioral = X[
                :,
                :N_BEHAVIORAL_FEATURES,
            ]

            if np.isnan(
                behavioral
            ).all():

                dropped += 1
                continue

        valid_paths.append(path)
        labels.append(y)
        groups.append(subject_id)

    print(
        f"Loaded samples: {len(valid_paths)}"
    )

    print(
        f"Dropped fully-missing behavioral samples: "
        f"{dropped}"
    )

    print(
        f"Subjects: {sorted(set(groups))}"
    )

    print(
        f"Class distribution: "
        f"{Counter(labels)}"
    )

    return (
        valid_paths,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            groups,
            dtype=np.int64,
        ),
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    accuracy = accuracy_score(
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
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    per_p, per_r, per_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
    )

    # For a fixed 3-class problem this is the
    # class-balanced recall over Low/Moderate/High.
    balanced_accuracy = float(
        macro_recall
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    metrics = {
        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "confusion_matrix":
            cm.tolist(),
    }

    for i, name in enumerate(
        CLASS_NAMES
    ):

        metrics[
            f"{name.lower()}_precision"
        ] = float(per_p[i])

        metrics[
            f"{name.lower()}_recall"
        ] = float(per_r[i])

        metrics[
            f"{name.lower()}_f1"
        ] = float(per_f1[i])

        metrics[
            f"{name.lower()}_support"
        ] = int(support[i])

    return metrics


def print_metrics(
    title: str,
    metrics: Dict,
):

    print(f"\n--- {title} ---")

    print(
        f"Accuracy:     "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Acc: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:     "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        np.asarray(
            metrics["confusion_matrix"]
        )
    )


# ============================================================
# BASELINE FEATURE AGGREGATION
# ============================================================

def sequence_to_baseline_vector(
    X: np.ndarray,
) -> np.ndarray:
    """
    Convert:
        (T, 26)

    into:
        mean(26) + std(26) = 52 features.
    """

    X = X.astype(
        np.float32
    )

    with np.errstate(
        all="ignore"
    ):

        means = np.nanmean(
            X,
            axis=0,
        )

        stds = np.nanstd(
            X,
            axis=0,
        )

    return np.concatenate(
        [
            means,
            stds,
        ]
    ).astype(
        np.float32
    )


def build_baseline_matrix(
    paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:

    X_all = []
    y_all = []

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

        X_all.append(
            sequence_to_baseline_vector(
                X
            )
        )

        y_all.append(y)

    return (
        np.asarray(
            X_all,
            dtype=np.float32,
        ),
        np.asarray(
            y_all,
            dtype=np.int64,
        ),
    )


def median_impute(
    X_train: np.ndarray,
    X_test: np.ndarray,
):

    medians = np.nanmedian(
        X_train,
        axis=0,
    )

    medians = np.where(
        np.isfinite(medians),
        medians,
        0.0,
    )

    def apply(X):

        X = X.copy()

        invalid = ~np.isfinite(X)

        rows, cols = np.where(
            invalid
        )

        if len(rows) > 0:

            X[
                rows,
                cols,
            ] = medians[cols]

        return X

    return (
        apply(X_train),
        apply(X_test),
    )


# ============================================================
# BASELINES
# ============================================================

def run_logistic_regression(
    train_paths: List[Path],
    test_paths: List[Path],
    seed: int,
) -> Dict:

    X_train, y_train = (
        build_baseline_matrix(
            train_paths
        )
    )

    X_test, y_test = (
        build_baseline_matrix(
            test_paths
        )
    )

    X_train, X_test = (
        median_impute(
            X_train,
            X_test,
        )
    )

    model = Pipeline(
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
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    return compute_metrics(
        y_test,
        y_pred,
    )


def run_random_forest(
    train_paths: List[Path],
    test_paths: List[Path],
    seed: int,
) -> Dict:

    X_train, y_train = (
        build_baseline_matrix(
            train_paths
        )
    )

    X_test, y_test = (
        build_baseline_matrix(
            test_paths
        )
    )

    X_train, X_test = (
        median_impute(
            X_train,
            X_test,
        )
    )

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    return compute_metrics(
        y_test,
        y_pred,
    )


# ============================================================
# LSTM NORMALIZATION
# ============================================================

def compute_norm_stats(
    paths: List[Path],
):

    sum_ = None
    sumsq = None
    count = None

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        X = data["X"].astype(
            np.float32
        )

        if sum_ is None:

            F = X.shape[1]

            sum_ = np.zeros(
                F,
                dtype=np.float64,
            )

            sumsq = np.zeros(
                F,
                dtype=np.float64,
            )

            count = np.zeros(
                F,
                dtype=np.float64,
            )

        mask = np.isfinite(X)

        X0 = np.where(
            mask,
            X,
            0.0,
        ).astype(
            np.float64
        )

        sum_ += X0.sum(
            axis=0
        )

        sumsq += (
            X0 * X0
        ).sum(
            axis=0
        )

        count += mask.sum(
            axis=0
        )

    mean = (
        sum_
        / np.maximum(
            count,
            1.0,
        )
    )

    var = (
        sumsq
        / np.maximum(
            count,
            1.0,
        )
        - mean * mean
    )

    var = np.maximum(
        var,
        1e-6,
    )

    std = np.sqrt(var)

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def class_weights_from_paths(
    paths: List[Path],
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for path in paths:

        y = int(
            np.load(
                path,
                allow_pickle=True,
            )["y"]
        )

        counts[y] += 1

    total = counts.sum()

    weights = (
        total
        / (
            NUM_CLASSES
            * np.maximum(
                counts,
                1.0,
            )
        )
    )

    return (
        counts,
        weights.astype(
            np.float32
        ),
    )


# ============================================================
# INNER SUBJECT VALIDATION
# ============================================================

def make_inner_split(
    paths: List[Path],
    seed: int,
):

    labels = []
    groups = []

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        labels.append(
            int(data["y"])
        )

        groups.append(
            int(
                data["subject_id"]
            )
        )

    labels = np.asarray(
        labels
    )

    groups = np.asarray(
        groups
    )

    indices = np.arange(
        len(paths)
    )

    fallback = None

    # Try several seeds to obtain an inner split
    # containing all three classes.
    for attempt in range(50):

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=(
                seed + attempt
            ),
        )

        train_idx, val_idx = next(
            splitter.split(
                indices,
                labels,
                groups,
            )
        )

        if fallback is None:
            fallback = (
                train_idx,
                val_idx,
            )

        train_classes = set(
            labels[train_idx]
        )

        val_classes = set(
            labels[val_idx]
        )

        if (
            train_classes
            == {0, 1, 2}
            and val_classes
            == {0, 1, 2}
        ):
            return (
                [paths[i] for i in train_idx],
                [paths[i] for i in val_idx],
            )

    train_idx, val_idx = fallback

    print(
        "WARNING: inner validation split "
        "does not contain all 3 classes."
    )

    return (
        [paths[i] for i in train_idx],
        [paths[i] for i in val_idx],
    )


# ============================================================
# LSTM HELPERS
# ============================================================

@torch.no_grad()
def neural_predictions(
    model,
    loader,
    device,
):

    model.eval()

    true = []
    pred = []

    for X, lengths, y in loader:

        X = X.to(device)

        lengths = lengths.to(
            device
        )

        logits = model(
            X,
            lengths,
        )

        y_pred = torch.argmax(
            logits,
            dim=1,
        )

        true.append(
            y.numpy()
        )

        pred.append(
            y_pred
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(true),
        np.concatenate(pred),
    )


def train_neural_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
):

    model.train()

    total_loss = 0.0
    n_items = 0

    for X, lengths, y in loader:

        X = X.to(device)

        lengths = lengths.to(
            device
        )

        y = y.to(
            device
        ).long()

        optimizer.zero_grad()

        logits = model(
            X,
            lengths,
        )

        loss = loss_fn(
            logits,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * X.size(0)
        )

        n_items += X.size(0)

    return (
        total_loss
        / max(
            n_items,
            1,
        )
    )


def make_loader(
    paths,
    mean,
    std,
    batch_size,
    shuffle,
):

    dataset = NPZSequenceDataset(
        paths,
        mean=mean,
        std=std,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_batch,
    )


# ============================================================
# BILSTM + ATTENTION
# ============================================================

def run_bilstm(
    outer_train_paths: List[Path],
    outer_test_paths: List[Path],
    device,
    fold_seed: int,
    hidden_size: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
) -> Dict:

    # --------------------------------------------------------
    # 1. Inner subject-level validation
    # --------------------------------------------------------

    inner_train, inner_val = (
        make_inner_split(
            outer_train_paths,
            fold_seed,
        )
    )

    print(
        f"    Inner train: {len(inner_train)} | "
        f"Inner val: {len(inner_val)}"
    )

    mean, std = compute_norm_stats(
        inner_train
    )

    _, weights = (
        class_weights_from_paths(
            inner_train
        )
    )

    dl_train = make_loader(
        inner_train,
        mean,
        std,
        batch_size,
        True,
    )

    dl_val = make_loader(
        inner_val,
        mean,
        std,
        batch_size,
        False,
    )

    first_dataset = (
        NPZSequenceDataset(
            inner_train,
            mean=mean,
            std=std,
        )
    )

    X0, _, _ = first_dataset[0]

    input_size = int(
        X0.shape[1]
    )

    set_seed(
        fold_seed
    )

    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
        num_classes=NUM_CLASSES,
    ).to(device)

    loss_fn = (
        torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                weights,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_epoch = 1
    patience_left = patience

    # --------------------------------------------------------
    # 2. Select best epoch using inner validation only
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        loss = train_neural_epoch(
            model,
            dl_train,
            optimizer,
            loss_fn,
            device,
        )

        y_true, y_pred = (
            neural_predictions(
                model,
                dl_val,
                device,
            )
        )

        val_metrics = (
            compute_metrics(
                y_true,
                y_pred,
            )
        )

        score = (
            val_metrics[
                "macro_f1"
            ]
        )

        if score > (
            best_f1 + 1e-6
        ):

            best_f1 = score
            best_epoch = epoch
            patience_left = patience

        else:

            patience_left -= 1

            if patience_left <= 0:
                break

    print(
        f"    Selected epoch: {best_epoch} | "
        f"inner val Macro F1: {best_f1:.4f}"
    )

    # --------------------------------------------------------
    # 3. Retrain fresh model using COMPLETE outer train fold
    # --------------------------------------------------------

    full_mean, full_std = (
        compute_norm_stats(
            outer_train_paths
        )
    )

    _, full_weights = (
        class_weights_from_paths(
            outer_train_paths
        )
    )

    dl_full_train = make_loader(
        outer_train_paths,
        full_mean,
        full_std,
        batch_size,
        True,
    )

    dl_test = make_loader(
        outer_test_paths,
        full_mean,
        full_std,
        batch_size,
        False,
    )

    set_seed(
        fold_seed + 1000
    )

    final_model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
        num_classes=NUM_CLASSES,
    ).to(device)

    final_loss_fn = (
        torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                full_weights,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    final_optimizer = (
        torch.optim.Adam(
            final_model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
    )

    for _ in range(
        best_epoch
    ):

        train_neural_epoch(
            final_model,
            dl_full_train,
            final_optimizer,
            final_loss_fn,
            device,
        )

    # --------------------------------------------------------
    # 4. Outer test
    # --------------------------------------------------------

    y_true, y_pred = (
        neural_predictions(
            final_model,
            dl_test,
            device,
        )
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    metrics[
        "selected_epoch"
    ] = int(best_epoch)

    metrics[
        "inner_val_macro_f1"
    ] = float(best_f1)

    return metrics


# ============================================================
# SUMMARY
# ============================================================

def summarize_model(
    fold_metrics: List[Dict],
) -> Dict:

    summary = {}

    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "low_f1",
        "moderate_f1",
        "high_f1",
    ]

    for key in keys:

        values = np.asarray(
            [
                fold[key]
                for fold
                in fold_metrics
            ],
            dtype=np.float64,
        )

        summary[key] = {
            "mean":
                float(
                    np.mean(values)
                ),

            "std":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if len(values) > 1
                else 0.0,
        }

    return summary


def format_mean_std(
    summary: Dict,
    key: str,
):

    mean = summary[
        key
    ]["mean"]

    std = summary[
        key
    ]["std"]

    return (
        f"{mean:.3f} +/- {std:.3f}"
    )


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
        default="runs/fatigue_drozy_group_cv",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--keep_missing_behavioral",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load ALL processed samples.
    # Existing train/val/test folder names are ignored for CV.
    # --------------------------------------------------------

    paths, labels, groups = (
        load_all_samples(
            Path(
                args.features_dir
            ),
            drop_missing_behavioral=(
                not args.keep_missing_behavioral
            ),
        )
    )

    unique_subjects = np.unique(
        groups
    )

    print(
        f"Unique subjects: "
        f"{len(unique_subjects)}"
    )

    if args.folds > len(
        unique_subjects
    ):
        raise ValueError(
            "Number of folds exceeds "
            "number of subjects."
        )

    splitter = GroupKFold(
        n_splits=args.folds
    )

    model_results = {
        "Logistic Regression": [],
        "Random Forest": [],
        "BiLSTM + Temporal Attention": [],
    }

    indices = np.arange(
        len(paths)
    )

    # ========================================================
    # OUTER CV
    # ========================================================

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            indices,
            labels,
            groups,
        ),
        start=1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"FOLD {fold}/{args.folds}"
        )

        print(
            "========================================"
        )

        train_paths = [
            paths[i]
            for i in train_idx
        ]

        test_paths = [
            paths[i]
            for i in test_idx
        ]

        train_subjects = sorted(
            set(
                groups[
                    train_idx
                ]
            )
        )

        test_subjects = sorted(
            set(
                groups[
                    test_idx
                ]
            )
        )

        print(
            f"Train subjects: "
            f"{train_subjects}"
        )

        print(
            f"Test subjects:  "
            f"{test_subjects}"
        )

        print(
            "Train classes:",
            Counter(
                labels[
                    train_idx
                ]
            ),
        )

        print(
            "Test classes:",
            Counter(
                labels[
                    test_idx
                ]
            ),
        )

        fold_seed = (
            args.seed + fold
        )

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        lr_metrics = (
            run_logistic_regression(
                train_paths,
                test_paths,
                fold_seed,
            )
        )

        print_metrics(
            "Logistic Regression",
            lr_metrics,
        )

        model_results[
            "Logistic Regression"
        ].append(
            lr_metrics
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        rf_metrics = (
            run_random_forest(
                train_paths,
                test_paths,
                fold_seed,
            )
        )

        print_metrics(
            "Random Forest",
            rf_metrics,
        )

        model_results[
            "Random Forest"
        ].append(
            rf_metrics
        )

        # ----------------------------------------------------
        # BiLSTM + Temporal Attention
        # ----------------------------------------------------

        bilstm_metrics = (
            run_bilstm(
                outer_train_paths=
                    train_paths,

                outer_test_paths=
                    test_paths,

                device=device,

                fold_seed=
                    fold_seed,

                hidden_size=
                    args.hidden_size,

                epochs=
                    args.epochs,

                patience=
                    args.patience,

                batch_size=
                    args.batch_size,

                lr=
                    args.lr,
            )
        )

        print_metrics(
            "BiLSTM + Temporal Attention",
            bilstm_metrics,
        )

        model_results[
            "BiLSTM + Temporal Attention"
        ].append(
            bilstm_metrics
        )

    # ========================================================
    # FINAL MEAN +/- STD
    # ========================================================

    summaries = {}

    for (
        model_name,
        folds,
    ) in model_results.items():

        summaries[
            model_name
        ] = summarize_model(
            folds
        )

    print(
        "\n\n"
        "============================================================"
    )

    print(
        "5-FOLD SUBJECT-INDEPENDENT GROUPED CV"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Model':32s}"
        f"{'Accuracy':20s}"
        f"{'Balanced Acc':20s}"
        f"{'Macro F1':20s}"
    )

    print(
        "-" * 92
    )

    for (
        model_name,
        summary,
    ) in summaries.items():

        print(
            f"{model_name:32s}"
            f"{format_mean_std(summary, 'accuracy'):20s}"
            f"{format_mean_std(summary, 'balanced_accuracy'):20s}"
            f"{format_mean_std(summary, 'macro_f1'):20s}"
        )

    # --------------------------------------------------------
    # Save complete fold-level results
    # --------------------------------------------------------

    payload = {
        "protocol":
            (
                f"{args.folds}-fold "
                "subject-independent GroupKFold"
            ),

        "num_classes":
            NUM_CLASSES,

        "class_names":
            CLASS_NAMES,

        "num_samples":
            len(paths),

        "num_subjects":
            int(
                len(unique_subjects)
            ),

        "fold_results":
            model_results,

        "summary":
            summaries,
    }

    output_path = (
        save_dir
        / "group_cv_results.json"
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
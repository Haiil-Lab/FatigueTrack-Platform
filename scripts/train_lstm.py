#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fatiguetrack.dataset import (
    load_npz_paths,
    NPZSequenceDataset,
    collate_batch,
)

from fatiguetrack.model_ns import AttentionLSTM


CLASS_NAMES = [
    "Low",
    "Moderate",
    "High",
]

NUM_CLASSES = 3


# ============================================================
# NORMALIZATION
# ============================================================

def compute_norm_stats(
    train_paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NaN-safe feature mean/std using TRAINING data only.
    """

    sum_ = None
    sumsq = None
    count = None

    for path in train_paths:

        npz = np.load(
            path,
            allow_pickle=True,
        )

        X = npz["X"].astype(
            np.float32
        )

        if sum_ is None:

            feature_dim = X.shape[1]

            sum_ = np.zeros(
                feature_dim,
                dtype=np.float64,
            )

            sumsq = np.zeros(
                feature_dim,
                dtype=np.float64,
            )

            count = np.zeros(
                feature_dim,
                dtype=np.float64,
            )

        mask = np.isfinite(X)

        X0 = np.where(
            mask,
            X,
            0.0,
        ).astype(np.float64)

        sum_ += X0.sum(axis=0)

        sumsq += (
            X0 * X0
        ).sum(axis=0)

        count += mask.sum(axis=0)

    mean = (
        sum_
        / np.maximum(count, 1.0)
    )

    variance = (
        sumsq
        / np.maximum(count, 1.0)
        - mean * mean
    )

    variance = np.maximum(
        variance,
        1e-6,
    )

    std = np.sqrt(
        variance
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# ============================================================
# CLASS DISTRIBUTION / WEIGHTS
# ============================================================

def get_class_counts(
    paths: List[Path],
) -> np.ndarray:

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    for path in paths:

        npz = np.load(
            path,
            allow_pickle=True,
        )

        y = int(
            npz["y"]
        )

        counts[y] += 1

    return counts


def compute_class_weights(
    class_counts: np.ndarray,
) -> np.ndarray:
    """
    Balanced class weighting:

        weight_c =
            N / (K * n_c)

    where:
        N = total training samples
        K = number of classes
        n_c = samples in class c
    """

    total = class_counts.sum()

    weights = (
        total
        / (
            NUM_CLASSES
            * np.maximum(
                class_counts,
                1,
            )
        )
    )

    return weights.astype(
        np.float32
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
):

    model.eval()

    all_true = []
    all_pred = []
    all_probs = []

    for X, lengths, y in loader:

        X = X.to(device)

        lengths = lengths.to(
            device
        )

        logits = model(
            X,
            lengths,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_true.append(
            y.cpu().numpy()
        )

        all_pred.append(
            predictions
            .cpu()
            .numpy()
        )

        all_probs.append(
            probabilities
            .cpu()
            .numpy()
        )

    y_true = np.concatenate(
        all_true
    ).astype(np.int64)

    y_pred = np.concatenate(
        all_pred
    ).astype(np.int64)

    y_prob = np.concatenate(
        all_probs
    ).astype(np.float32)

    return (
        y_true,
        y_pred,
        y_prob,
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

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_precision":
            float(
                precision_macro
            ),

        "macro_recall":
            float(
                recall_macro
            ),

        "macro_f1":
            float(
                f1_macro
            ),

        "weighted_precision":
            float(
                precision_weighted
            ),

        "weighted_recall":
            float(
                recall_weighted
            ),

        "weighted_f1":
            float(
                f1_weighted
            ),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report,
    }


# ============================================================
# PRINTING
# ============================================================

def print_split_stats(
    name: str,
    paths: List[Path],
):

    counts = get_class_counts(
        paths
    )

    print(
        f"{name}: "
        f"total={len(paths)} | "
        f"Low={counts[0]} | "
        f"Moderate={counts[1]} | "
        f"High={counts[2]}"
    )


def print_metrics(
    title: str,
    metrics: Dict,
):

    print(
        f"\n===== {title} ====="
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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_dir",
        type=str,
        default=(
            "data/processed/drozy"
        ),
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=(
            "runs/fatigue_drozy_bilstm"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--hidden_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--bidirectional",
        action="store_true",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
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

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    features_dir = Path(
        args.features_dir
    )

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_paths = load_npz_paths(
        features_dir,
        "train",
    )

    val_paths = load_npz_paths(
        features_dir,
        "val",
    )

    test_paths = load_npz_paths(
        features_dir,
        "test",
    )

    if not train_paths:
        raise RuntimeError(
            "No training samples found."
        )

    print_split_stats(
        "TRAIN",
        train_paths,
    )

    print_split_stats(
        "VAL",
        val_paths,
    )

    print_split_stats(
        "TEST",
        test_paths,
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    mean, std = compute_norm_stats(
        train_paths
    )

    norm_payload = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    norm_path = (
        save_dir / "norm.json"
    )

    norm_path.write_text(
        json.dumps(
            norm_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    train_counts = (
        get_class_counts(
            train_paths
        )
    )

    class_weights = (
        compute_class_weights(
            train_counts
        )
    )

    print(
        "\nTraining class counts:",
        train_counts.tolist(),
    )

    print(
        "Class weights:",
        class_weights.tolist(),
    )

    class_weights_tensor = (
        torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ds_train = NPZSequenceDataset(
        train_paths,
        mean=mean,
        std=std,
    )

    ds_val = NPZSequenceDataset(
        val_paths,
        mean=mean,
        std=std,
    )

    ds_test = NPZSequenceDataset(
        test_paths,
        mean=mean,
        std=std,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    X0, _, _ = ds_train[0]

    input_size = int(
        X0.shape[1]
    )

    print(
        f"\nInput size: {input_size}"
    )

    print(
        f"Sequence length: {X0.shape[0]}"
    )

    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_classes=NUM_CLASSES,
    ).to(device)

    print(
        "\nModel:"
    )

    print(model)

    # --------------------------------------------------------
    # Loss / Optimizer
    # --------------------------------------------------------

    loss_fn = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_score = -1.0

    best_epoch = 0

    best_path = (
        save_dir / "best.pt"
    )

    patience_left = (
        args.patience
    )

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        n_items = 0

        for (
            X,
            lengths,
            y,
        ) in dl_train:

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
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += (
                float(loss.item())
                * X.size(0)
            )

            n_items += X.size(0)

        train_loss = (
            total_loss
            / max(n_items, 1)
        )

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        (
            y_val_true,
            y_val_pred,
            _,
        ) = collect_predictions(
            model,
            dl_val,
            device,
        )

        val_metrics = (
            compute_metrics(
                y_val_true,
                y_val_pred,
            )
        )

        # Macro F1 is main model-selection metric.
        score = float(
            val_metrics[
                "macro_f1"
            ]
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc="
            f"{val_metrics['accuracy']:.4f} | "
            f"val_bal_acc="
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1="
            f"{val_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    float(train_loss),
                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],
                "val_balanced_accuracy":
                    val_metrics[
                        "balanced_accuracy"
                    ],
                "val_macro_f1":
                    val_metrics[
                        "macro_f1"
                    ],
            }
        )

        # ---------------------------------------------
        # Save best model
        # ---------------------------------------------

        if score > (
            best_score + 1e-6
        ):

            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "input_size":
                        input_size,

                    "num_classes":
                        NUM_CLASSES,

                    "class_names":
                        CLASS_NAMES,

                    "best_epoch":
                        best_epoch,

                    "best_val_macro_f1":
                        best_score,

                    "args":
                        vars(args),
                },
                best_path,
            )

            patience_left = (
                args.patience
            )

        else:

            patience_left -= 1

            if patience_left <= 0:

                print(
                    "Early stopping."
                )

                break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    print(
        f"\nBest epoch: "
        f"{checkpoint['best_epoch']}"
    )

    print(
        f"Best validation Macro F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    (
        y_val_true,
        y_val_pred,
        y_val_prob,
    ) = collect_predictions(
        model,
        dl_val,
        device,
    )

    final_val_metrics = (
        compute_metrics(
            y_val_true,
            y_val_pred,
        )
    )

    print_metrics(
        "FINAL VALIDATION",
        final_val_metrics,
    )

    # ========================================================
    # TEST
    # ========================================================

    (
        y_test_true,
        y_test_pred,
        y_test_prob,
    ) = collect_predictions(
        model,
        dl_test,
        device,
    )

    test_metrics = compute_metrics(
        y_test_true,
        y_test_pred,
    )

    print_metrics(
        "TEST RESULTS",
        test_metrics,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary = {
        "model":
            "BiLSTM + Temporal Attention"
            if args.bidirectional
            else "LSTM + Temporal Attention",

        "features":
            "Behavioral + EEG + ECG",

        "input_size":
            input_size,

        "num_classes":
            NUM_CLASSES,

        "class_names":
            CLASS_NAMES,

        "class_counts": {
            "train":
                get_class_counts(
                    train_paths
                ).tolist(),

            "val":
                get_class_counts(
                    val_paths
                ).tolist(),

            "test":
                get_class_counts(
                    test_paths
                ).tolist(),
        },

        "class_weights":
            class_weights.tolist(),

        "best_epoch":
            int(best_epoch),

        "best_val_macro_f1":
            float(best_score),

        "validation":
            final_val_metrics,

        "test":
            test_metrics,

        "history":
            history,
    }

    metrics_path = (
        save_dir
        / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save raw predictions for later plots/statistics.
    np.savez_compressed(
        save_dir
        / "test_predictions.npz",

        y_true=y_test_true,

        y_pred=y_test_pred,

        probabilities=y_test_prob,

        class_names=np.asarray(
            CLASS_NAMES
        ),
    )

    print(
        "\nSaved:"
    )

    print(
        f"  Model:   {best_path}"
    )

    print(
        f"  Norm:    {norm_path}"
    )

    print(
        f"  Metrics: {metrics_path}"
    )

    print(
        f"  Preds:   "
        f"{save_dir / 'test_predictions.npz'}"
    )


if __name__ == "__main__":
    main()
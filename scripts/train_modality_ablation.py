#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fatiguetrack.dataset import collate_batch
from fatiguetrack.model_ns import AttentionLSTM


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Low",
    "Moderate",
    "High",
]

NUM_CLASSES = 3


# Original multimodal feature layout:
#
# 0:7    Behavioral
# 7:19   EEG
# 19:26  ECG

BEHAVIORAL_IDX = list(
    range(0, 7)
)

EEG_IDX = list(
    range(7, 19)
)

ECG_IDX = list(
    range(19, 26)
)


MODALITY_CONFIGS = {
    "Behavioral":
        BEHAVIORAL_IDX,

    "EEG":
        EEG_IDX,

    "ECG":
        ECG_IDX,

    "Behavioral + EEG":
        BEHAVIORAL_IDX
        + EEG_IDX,

    "Behavioral + ECG":
        BEHAVIORAL_IDX
        + ECG_IDX,

    "EEG + ECG":
        EEG_IDX
        + ECG_IDX,

    "Behavioral + EEG + ECG":
        BEHAVIORAL_IDX
        + EEG_IDX
        + ECG_IDX,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD SAMPLE METADATA
# ============================================================

def load_all_samples(
    features_dir: Path,
) -> Tuple[
    List[Path],
    np.ndarray,
    np.ndarray,
]:
    """
    Load all processed NPZ paths.

    We deliberately remove samples where the entire
    behavioral block is missing so that all seven
    modality configurations are evaluated on exactly
    the same samples and folds.
    """

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

        X = data[
            "X"
        ].astype(
            np.float32
        )

        y = int(
            data["y"]
        )

        subject_id = int(
            data["subject_id"]
        )

        behavioral = X[
            :,
            BEHAVIORAL_IDX,
        ]

        # Remove only sequences where ALL behavioral
        # values across the full 60-s sequence are missing.
        if np.isnan(
            behavioral
        ).all():

            dropped += 1
            continue

        valid_paths.append(
            path
        )

        labels.append(
            y
        )

        groups.append(
            subject_id
        )

    print(
        f"Loaded samples: "
        f"{len(valid_paths)}"
    )

    print(
        "Dropped fully-missing "
        f"behavioral samples: {dropped}"
    )

    print(
        "Subjects:",
        sorted(
            set(groups)
        ),
    )

    print(
        "Class distribution:",
        Counter(
            labels
        ),
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
# NORMALIZATION
# ============================================================

def compute_norm_stats(
    paths: Sequence[Path],
    feature_indices: Sequence[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Compute mean/std from TRAINING DATA ONLY
    for the selected modality configuration.
    """

    feature_indices = list(
        feature_indices
    )

    n_features = len(
        feature_indices
    )

    sum_ = np.zeros(
        n_features,
        dtype=np.float64,
    )

    sumsq = np.zeros(
        n_features,
        dtype=np.float64,
    )

    count = np.zeros(
        n_features,
        dtype=np.float64,
    )

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        X = data[
            "X"
        ].astype(
            np.float32
        )

        X = X[
            :,
            feature_indices,
        ]

        mask = np.isfinite(
            X
        )

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

    variance = (
        sumsq
        / np.maximum(
            count,
            1.0,
        )
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
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


# ============================================================
# DATASET
# ============================================================

class ModalitySequenceDataset(
    Dataset
):
    """
    Loads one processed NPZ sequence and retains only
    the requested modality feature columns.

    Original:
        X = (6, 26)

    Examples:
        Behavioral              -> (6, 7)
        EEG                     -> (6, 12)
        ECG                     -> (6, 7)
        Behavioral + EEG        -> (6, 19)
        Behavioral + ECG        -> (6, 14)
        EEG + ECG               -> (6, 19)
        Behavioral + EEG + ECG  -> (6, 26)
    """

    def __init__(
        self,
        paths: Sequence[Path],
        feature_indices: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
    ):

        self.paths = list(
            paths
        )

        self.feature_indices = list(
            feature_indices
        )

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        )

        self.std = np.asarray(
            std,
            dtype=np.float32,
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.paths
        )

    def __getitem__(
        self,
        index: int,
    ):

        data = np.load(
            self.paths[index],
            allow_pickle=True,
        )

        X = data[
            "X"
        ].astype(
            np.float32
        )

        X = X[
            :,
            self.feature_indices,
        ]

        length = int(
            data["length"]
        )

        y = int(
            data["y"]
        )

        # Normalize with training-fold statistics.
        X = (
            X - self.mean
        ) / self.std

        # Missing values become 0 after normalization,
        # corresponding to the training mean.
        X = np.where(
            np.isfinite(X),
            X,
            0.0,
        ).astype(
            np.float32
        )

        return (
            torch.tensor(
                X,
                dtype=torch.float32,
            ),

            torch.tensor(
                length,
                dtype=torch.long,
            ),

            torch.tensor(
                y,
                dtype=torch.long,
            ),
        )


def make_loader(
    paths: Sequence[Path],
    feature_indices: Sequence[int],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:

    dataset = (
        ModalitySequenceDataset(
            paths=paths,
            feature_indices=
                feature_indices,
            mean=mean,
            std=std,
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_batch,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def class_weights_from_paths(
    paths: Sequence[Path],
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        y = int(
            data["y"]
        )

        counts[
            y
        ] += 1

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
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:

    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
            ],
            average="macro",
            zero_division=0,
        )
    )

    per_precision, per_recall, per_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
            ],
            average=None,
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

    result = {
        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                macro_recall
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "confusion_matrix":
            cm.tolist(),
    }

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        key = (
            class_name
            .lower()
        )

        result[
            f"{key}_precision"
        ] = float(
            per_precision[
                i
            ]
        )

        result[
            f"{key}_recall"
        ] = float(
            per_recall[
                i
            ]
        )

        result[
            f"{key}_f1"
        ] = float(
            per_f1[
                i
            ]
        )

        result[
            f"{key}_support"
        ] = int(
            support[
                i
            ]
        )

    return result


# ============================================================
# MODEL PREDICTIONS
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

    for (
        X,
        lengths,
        y,
    ) in loader:

        X = X.to(
            device
        )

        lengths = lengths.to(
            device
        )

        logits = model(
            X,
            lengths,
        )

        pred = torch.argmax(
            logits,
            dim=1,
        )

        all_true.append(
            y.numpy()
        )

        all_pred.append(
            pred
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(
            all_true
        ),

        np.concatenate(
            all_pred
        ),
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
) -> float:

    model.train()

    total_loss = 0.0
    n_items = 0

    for (
        X,
        lengths,
        y,
    ) in loader:

        X = X.to(
            device
        )

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
            float(
                loss.item()
            )
            * X.size(0)
        )

        n_items += X.size(
            0
        )

    return (
        total_loss
        / max(
            n_items,
            1,
        )
    )


# ============================================================
# INNER SUBJECT-INDEPENDENT VALIDATION
# ============================================================

def make_inner_split(
    paths: Sequence[Path],
    seed: int,
) -> Tuple[
    List[Path],
    List[Path],
]:

    labels = []
    groups = []

    for path in paths:

        data = np.load(
            path,
            allow_pickle=True,
        )

        labels.append(
            int(
                data["y"]
            )
        )

        groups.append(
            int(
                data[
                    "subject_id"
                ]
            )
        )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    groups = np.asarray(
        groups,
        dtype=np.int64,
    )

    indices = np.arange(
        len(paths)
    )

    fallback = None

    # Search for an inner subject split
    # containing all 3 classes in both partitions.
    for attempt in range(
        50
    ):

        splitter = (
            GroupShuffleSplit(
                n_splits=1,
                test_size=0.20,
                random_state=(
                    seed
                    + attempt
                ),
            )
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
            labels[
                train_idx
            ].tolist()
        )

        val_classes = set(
            labels[
                val_idx
            ].tolist()
        )

        if (
            train_classes
            == {
                0,
                1,
                2,
            }
            and
            val_classes
            == {
                0,
                1,
                2,
            }
        ):

            return (
                [
                    paths[i]
                    for i
                    in train_idx
                ],

                [
                    paths[i]
                    for i
                    in val_idx
                ],
            )

    print(
        "WARNING: Could not create "
        "inner validation split "
        "containing all three classes."
    )

    train_idx, val_idx = (
        fallback
    )

    return (
        [
            paths[i]
            for i
            in train_idx
        ],

        [
            paths[i]
            for i
            in val_idx
        ],
    )


# ============================================================
# RUN ONE MODALITY / ONE OUTER FOLD
# ============================================================

def run_modality_fold(
    modality_name: str,
    feature_indices: Sequence[int],
    outer_train_paths: Sequence[Path],
    outer_test_paths: Sequence[Path],
    device,
    seed: int,
    hidden_size: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
) -> Dict:

    # ========================================================
    # INNER MODEL SELECTION
    # ========================================================

    (
        inner_train,
        inner_val,
    ) = make_inner_split(
        outer_train_paths,
        seed,
    )

    inner_mean, inner_std = (
        compute_norm_stats(
            inner_train,
            feature_indices,
        )
    )

    _, inner_weights = (
        class_weights_from_paths(
            inner_train
        )
    )

    train_loader = make_loader(
        paths=inner_train,
        feature_indices=
            feature_indices,
        mean=inner_mean,
        std=inner_std,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        paths=inner_val,
        feature_indices=
            feature_indices,
        mean=inner_mean,
        std=inner_std,
        batch_size=batch_size,
        shuffle=False,
    )

    input_size = len(
        feature_indices
    )

    set_seed(
        seed
    )

    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
        num_classes=NUM_CLASSES,
    ).to(
        device
    )

    loss_fn = (
        torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                inner_weights,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
    )

    best_macro_f1 = -1.0
    best_epoch = 1

    patience_left = (
        patience
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        y_true, y_pred = (
            collect_predictions(
                model,
                val_loader,
                device,
            )
        )

        metrics = compute_metrics(
            y_true,
            y_pred,
        )

        score = metrics[
            "macro_f1"
        ]

        if score > (
            best_macro_f1
            + 1e-6
        ):

            best_macro_f1 = (
                score
            )

            best_epoch = (
                epoch
            )

            patience_left = (
                patience
            )

        else:

            patience_left -= 1

            if (
                patience_left
                <= 0
            ):
                break

    print(
        f"      Selected epoch: "
        f"{best_epoch} | "
        f"inner val Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    # ========================================================
    # RETRAIN ON COMPLETE OUTER TRAIN
    # ========================================================

    full_mean, full_std = (
        compute_norm_stats(
            outer_train_paths,
            feature_indices,
        )
    )

    _, full_weights = (
        class_weights_from_paths(
            outer_train_paths
        )
    )

    full_train_loader = (
        make_loader(
            paths=
                outer_train_paths,

            feature_indices=
                feature_indices,

            mean=
                full_mean,

            std=
                full_std,

            batch_size=
                batch_size,

            shuffle=True,
        )
    )

    test_loader = (
        make_loader(
            paths=
                outer_test_paths,

            feature_indices=
                feature_indices,

            mean=
                full_mean,

            std=
                full_std,

            batch_size=
                batch_size,

            shuffle=False,
        )
    )

    # Different initialization from inner model,
    # but deterministic for this fold/configuration.
    set_seed(
        seed + 1000
    )

    final_model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
        num_classes=NUM_CLASSES,
    ).to(
        device
    )

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

        train_epoch(
            model=final_model,
            loader=
                full_train_loader,
            optimizer=
                final_optimizer,
            loss_fn=
                final_loss_fn,
            device=device,
        )

    # ========================================================
    # OUTER TEST
    # ========================================================

    y_true, y_pred = (
        collect_predictions(
            final_model,
            test_loader,
            device,
        )
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    metrics[
        "selected_epoch"
    ] = int(
        best_epoch
    )

    metrics[
        "inner_val_macro_f1"
    ] = float(
        best_macro_f1
    )

    metrics[
        "input_features"
    ] = int(
        input_size
    )

    return metrics


# ============================================================
# SUMMARY
# ============================================================

def summarize_folds(
    fold_results: List[Dict],
) -> Dict:

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "low_precision",
        "low_recall",
        "low_f1",
        "moderate_precision",
        "moderate_recall",
        "moderate_f1",
        "high_precision",
        "high_recall",
        "high_f1",
    ]

    summary = {}

    for metric_name in (
        metric_names
    ):

        values = np.asarray(
            [
                fold[
                    metric_name
                ]
                for fold
                in fold_results
            ],
            dtype=np.float64,
        )

        summary[
            metric_name
        ] = {
            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "std":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if len(
                    values
                ) > 1
                else 0.0,
        }

    return summary


def mean_std_text(
    summary: Dict,
    metric: str,
) -> str:

    mean = (
        summary[
            metric
        ]["mean"]
    )

    std = (
        summary[
            metric
        ]["std"]
    )

    return (
        f"{mean:.3f} "
        f"+/- "
        f"{std:.3f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

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
            "runs/"
            "fatigue_drozy_"
            "modality_ablation"
        ),
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
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    (
        all_paths,
        labels,
        groups,
    ) = load_all_samples(
        Path(
            args.features_dir
        )
    )

    unique_subjects = (
        np.unique(
            groups
        )
    )

    if args.folds > len(
        unique_subjects
    ):

        raise ValueError(
            "Number of folds cannot "
            "exceed number of subjects."
        )

    print(
        f"Unique subjects: "
        f"{len(unique_subjects)}"
    )

    print(
        "\nModality configurations:"
    )

    for (
        name,
        indices,
    ) in MODALITY_CONFIGS.items():

        print(
            f"  {name:28s} "
            f"{len(indices)} features"
        )

    # --------------------------------------------------------
    # Construct outer folds ONCE.
    #
    # Every modality uses exactly the same subjects
    # in exactly the same folds.
    # --------------------------------------------------------

    indices = np.arange(
        len(all_paths)
    )

    splitter = GroupKFold(
        n_splits=args.folds
    )

    outer_folds = list(
        splitter.split(
            indices,
            labels,
            groups,
        )
    )

    results = {
        modality: []
        for modality
        in MODALITY_CONFIGS
    }

    # ========================================================
    # RUN ALL 7 CONFIGURATIONS
    # ========================================================

    for modality_index, (
        modality_name,
        feature_indices,
    ) in enumerate(
        MODALITY_CONFIGS.items(),
        start=1,
    ):

        print(
            "\n\n"
            "============================================================"
        )

        print(
            f"MODALITY "
            f"{modality_index}/"
            f"{len(MODALITY_CONFIGS)}: "
            f"{modality_name}"
        )

        print(
            f"Input dimension: "
            f"{len(feature_indices)}"
        )

        print(
            "============================================================"
        )

        for fold_number, (
            train_idx,
            test_idx,
        ) in enumerate(
            outer_folds,
            start=1,
        ):

            print(
                "\n"
                "----------------------------------------"
            )

            print(
                f"{modality_name} | "
                f"Fold "
                f"{fold_number}/"
                f"{args.folds}"
            )

            print(
                "----------------------------------------"
            )

            train_paths = [
                all_paths[i]
                for i
                in train_idx
            ]

            test_paths = [
                all_paths[i]
                for i
                in test_idx
            ]

            train_subjects = sorted(
                set(
                    groups[
                        train_idx
                    ].tolist()
                )
            )

            test_subjects = sorted(
                set(
                    groups[
                        test_idx
                    ].tolist()
                )
            )

            print(
                "    Train subjects:",
                train_subjects,
            )

            print(
                "    Test subjects: ",
                test_subjects,
            )

            print(
                "    Train classes:",
                Counter(
                    labels[
                        train_idx
                    ].tolist()
                ),
            )

            print(
                "    Test classes:",
                Counter(
                    labels[
                        test_idx
                    ].tolist()
                ),
            )

            # Same basic fold seed across modality
            # configurations to keep experiments
            # reproducible.
            fold_seed = (
                args.seed
                + fold_number
            )

            metrics = (
                run_modality_fold(
                    modality_name=
                        modality_name,

                    feature_indices=
                        feature_indices,

                    outer_train_paths=
                        train_paths,

                    outer_test_paths=
                        test_paths,

                    device=
                        device,

                    seed=
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

            results[
                modality_name
            ].append(
                metrics
            )

            print(
                f"      Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"      Balanced Accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"      Macro F1: "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                "      Confusion matrix:"
            )

            print(
                np.asarray(
                    metrics[
                        "confusion_matrix"
                    ]
                )
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summaries = {}

    for (
        modality_name,
        fold_results,
    ) in results.items():

        summaries[
            modality_name
        ] = summarize_folds(
            fold_results
        )

    print(
        "\n\n"
        "=========================================================================="
    )

    print(
        "MODALITY ABLATION - "
        "5-FOLD SUBJECT-INDEPENDENT GROUPED CV"
    )

    print(
        "=========================================================================="
    )

    print(
        f"{'Configuration':30s}"
        f"{'Accuracy':20s}"
        f"{'Balanced Acc':20s}"
        f"{'Macro F1':20s}"
    )

    print(
        "-" * 90
    )

    for modality_name in (
        MODALITY_CONFIGS
    ):

        summary = summaries[
            modality_name
        ]

        print(
            f"{modality_name:30s}"
            f"{mean_std_text(summary, 'accuracy'):20s}"
            f"{mean_std_text(summary, 'balanced_accuracy'):20s}"
            f"{mean_std_text(summary, 'macro_f1'):20s}"
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {
        "protocol":
            (
                f"{args.folds}-fold "
                "subject-independent "
                "GroupKFold"
            ),

        "model":
            (
                "BiLSTM + "
                "Temporal Attention"
            ),

        "num_samples":
            int(
                len(
                    all_paths
                )
            ),

        "num_subjects":
            int(
                len(
                    unique_subjects
                )
            ),

        "class_names":
            CLASS_NAMES,

        "modality_feature_indices": {
            name:
                list(indices_)
            for (
                name,
                indices_,
            )
            in MODALITY_CONFIGS.items()
        },

        "fold_results":
            results,

        "summary":
            summaries,
    }

    output_path = (
        save_dir
        / "modality_ablation_results.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
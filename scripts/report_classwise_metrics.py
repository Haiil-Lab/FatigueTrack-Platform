from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["Low", "Moderate", "High"]


def metrics_from_confusion_matrix(cm: np.ndarray):
    rows = []

    for i, class_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "class": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    return rows


def normalized_confusion_matrix(cm: np.ndarray):
    row_sums = cm.sum(axis=1, keepdims=True)

    return np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float,
        ),
        where=row_sums != 0,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
):
    cm_norm = normalized_confusion_matrix(cm)

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm_norm,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(
            len(CLASS_NAMES)
        )
    )

    ax.set_yticks(
        np.arange(
            len(CLASS_NAMES)
        )
    )

    ax.set_xticklabels(
        CLASS_NAMES
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        "BiLSTM + Temporal Attention\n"
        "Pooled Subject-Independent Confusion Matrix"
    )

    for i in range(
        len(CLASS_NAMES)
    ):
        for j in range(
            len(CLASS_NAMES)
        ):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100.0

            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Row-normalized proportion",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "runs/fatigue_drozy_group_cv/"
            "group_cv_results.json"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "runs/fatigue_drozy_group_cv/"
            "classwise"
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    all_results = {}

    for model_name, fold_results in (
        data["fold_results"].items()
    ):
        pooled_cm = np.zeros(
            (3, 3),
            dtype=int,
        )

        for fold in fold_results:
            fold_cm = np.asarray(
                fold[
                    "confusion_matrix"
                ],
                dtype=int,
            )

            pooled_cm += fold_cm

        class_metrics = (
            metrics_from_confusion_matrix(
                pooled_cm
            )
        )

        all_results[
            model_name
        ] = {
            "confusion_matrix":
                pooled_cm.tolist(),
            "class_metrics":
                class_metrics,
        }

        print(
            "\n"
            "===================================="
        )

        print(
            model_name
        )

        print(
            "===================================="
        )

        print(
            "\nPooled confusion matrix:"
        )

        print(
            pooled_cm
        )

        print(
            "\nPer-class metrics:"
        )

        print(
            f"{'Class':10s} "
            f"{'Precision':>10s} "
            f"{'Recall':>10s} "
            f"{'F1':>10s} "
            f"{'Support':>10s}"
        )

        for row in class_metrics:
            print(
                f"{row['class']:10s} "
                f"{row['precision']:10.4f} "
                f"{row['recall']:10.4f} "
                f"{row['f1']:10.4f} "
                f"{row['support']:10d}"
            )

        csv_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("+", "plus")
        )

        csv_path = (
            output_dir
            / f"{csv_name}_class_metrics.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class",
                    "precision",
                    "recall",
                    "f1",
                    "support",
                ],
            )

            writer.writeheader()
            writer.writerows(
                class_metrics
            )

        if (
            model_name
            == "BiLSTM + Temporal Attention"
        ):
            plot_confusion_matrix(
                pooled_cm,
                output_dir
                / "bilstm_confusion_matrix.png",
            )

    json_path = (
        output_dir
        / "pooled_classwise_results.json"
    )

    json_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved results to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()
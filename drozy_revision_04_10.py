#!/usr/bin/env python
"""Audit DROZY features, export existing CV results, or run modality ablations.

Place this file in the FatigueTrack project root. It reuses the supplied
scripts/train_group_cv.py and fatiguetrack package. Original data and code are
not edited. Export/audit require NumPy; training also needs the project's
PyTorch and scikit-learn environment.

Examples (run from the project root):
  python drozy_revision_04_10.py --existing runs/fatigue_drozy_group_cv/group_cv_results.json
  python drozy_revision_04_10.py --run --modes all_modalities
  python drozy_revision_04_10.py --run

The bootstrap describes uncertainty conditional on these held-out predictions.
It resamples whole participants, does not retrain models, and does not provide
a hypothesis test or account for every source of training uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


CLASSES = ["Low", "Moderate", "High"]
MODES = {
    "behavioral": list(range(7)),
    "eeg": list(range(7, 19)),
    "ecg": list(range(19, 26)),
    "behavioral_eeg": list(range(19)),
    "behavioral_ecg": list(range(7)) + list(range(19, 26)),
    "eeg_ecg": list(range(7, 26)),
    "all_modalities": list(range(26)),
}
EXPECTED_FEATURES = [
    "ear_mean", "ear_min", "blink_rate", "eye_closed_ratio", "mar_mean",
    "mar_max", "yawn_ratio", "eeg_delta_power_mean", "eeg_theta_power_mean",
    "eeg_alpha_power_mean", "eeg_beta_power_mean", "eeg_delta_relative_mean",
    "eeg_theta_relative_mean", "eeg_alpha_relative_mean", "eeg_beta_relative_mean",
    "eeg_theta_alpha_ratio_mean", "eeg_theta_beta_ratio_mean",
    "eeg_alpha_beta_ratio_mean", "eeg_spectral_entropy_mean", "ecg_hr_mean",
    "ecg_hr_min", "ecg_hr_max", "ecg_rr_mean", "ecg_sdnn", "ecg_rmssd",
    "ecg_pnn50",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(root, output):
    paths = sorted(Path(root).rglob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz files found in {root}")
    kept, rows, errors, powers = [], [], [], []
    finite_counts = np.zeros(26, dtype=int)
    total_steps = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            required = {"X", "y", "length", "subject_id", "session_id", "kss", "feature_names"}
            if not required.issubset(data.files):
                errors.append(f"{path.name}: missing keys {sorted(required - set(data.files))}")
                continue
            X = np.asarray(data["X"], dtype=np.float32)
            names = [str(name) for name in data["feature_names"]]
            if X.shape != (6, 26) or int(data["length"]) != 6 or names != EXPECTED_FEATURES:
                errors.append(f"{path.name}: expected 6 x 26 and original feature order")
                continue
            y, kss = int(data["y"]), int(data["kss"])
            expected = 0 if kss <= 3 else (1 if kss <= 6 else 2)
            if not 1 <= kss <= 9 or y != expected:
                errors.append(f"{path.name}: invalid/mismatched KSS={kss}, y={y}; 0 means missing")
            drop = bool(np.isnan(X[:, :7]).all())
            row = {
                "file": str(path.resolve()), "subject_id": int(data["subject_id"]),
                "session_id": str(data["session_id"].item()), "kss": kss, "class_id": y,
                "class_name": CLASSES[y] if 0 <= y <= 2 else "INVALID",
                "excluded_missing_behavioral": drop,
            }
            if "start_sec" in data and "end_sec" in data:
                row.update(start_sec=float(data["start_sec"]), end_sec=float(data["end_sec"]))
            rows.append(row)
            if not drop:
                kept.append(path.resolve())
                finite_counts += np.isfinite(X).sum(axis=0)
                total_steps += X.shape[0]
                powers.extend(X[:, 7:11].reshape(-1).tolist())
    valid_rows = [row for row in rows if not row["excluded_missing_behavioral"]]
    session_labels = {}
    for row in valid_rows:
        previous = session_labels.setdefault(row["session_id"], row["class_name"])
        if previous != row["class_name"]:
            errors.append(f"Conflicting session labels: {row['session_id']}")
    class_counts = Counter(row["class_name"] for row in valid_rows)
    session_counts = Counter(session_labels.values())
    p = np.asarray(powers, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    median_power = float(np.median(p)) if p.size else None
    summary = {
        "initial_sequences": len(paths), "evaluated_sequences": len(kept),
        "excluded_fully_missing_behavioral": len(rows) - len(valid_rows),
        "participants": sorted({row["subject_id"] for row in valid_rows}),
        "sessions": len(session_labels),
        "sequence_class_distribution": {name: class_counts[name] for name in CLASSES},
        "session_class_distribution": {name: session_counts[name] for name in CLASSES},
        "missing_fraction_by_feature": {
            name: float(1 - count / max(total_steps, 1))
            for name, count in zip(EXPECTED_FEATURES, finite_counts)
        },
        "median_positive_eeg_absolute_band_power": median_power,
        "eeg_ratio_regularizer_in_supplied_code": 1e-8,
        "eeg_ratio_regularizer_may_dominate": bool(median_power is not None and median_power < 1e-8),
        "errors": errors,
    }
    write_json(output / "data_audit.json", summary)
    write_csv(output / "sequence_inventory.csv", rows)
    print(json.dumps(summary, indent=2))
    if errors:
        raise ValueError("Data audit failed; read data_audit.json before using/retraining results.")
    return kept, valid_rows


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    true, pred, correct = cm.sum(-1), cm.sum(-2), np.diagonal(cm, axis1=-2, axis2=-1)
    recall = np.divide(correct, true, out=np.zeros_like(correct), where=true > 0)
    f1 = np.divide(2 * correct, true + pred, out=np.zeros_like(correct), where=(true + pred) > 0)
    return {
        "accuracy": np.divide(correct.sum(-1), cm.sum((-2, -1))),
        "balanced_accuracy": recall.mean(-1), "macro_f1": f1.mean(-1),
    }


def export_results(payload, output):
    if payload.get("num_classes") != 3 or payload.get("class_names") != CLASSES:
        raise ValueError("Expected original three-class CV result schema.")
    folds, per_class, matrices, summary = [], [], [], []
    expected_samples = int(payload["num_samples"])
    for model, results in payload["fold_results"].items():
        pooled = np.zeros((3, 3), dtype=int)
        for fold, row in enumerate(results, 1):
            cm = np.asarray(row["confusion_matrix"], dtype=int)
            if cm.shape != (3, 3):
                raise ValueError(f"Invalid confusion matrix: {model}, fold {fold}")
            pooled += cm
            fold_row = {"model": model, "fold": fold, "test_sequences": int(cm.sum())}
            fold_row.update({key: value for key, value in row.items() if key != "confusion_matrix"})
            folds.append(fold_row)
            for i, name in enumerate(CLASSES):
                per_class.append({"model": model, "aggregation": "fold", "fold": fold,
                                  "class": name, "precision": row[f"{name.lower()}_precision"],
                                  "recall": row[f"{name.lower()}_recall"],
                                  "f1": row[f"{name.lower()}_f1"],
                                  "support": int(cm[i].sum())})
        if int(pooled.sum()) != expected_samples:
            raise ValueError(f"{model}: total confusion-matrix support differs from num_samples")
        for i, name in enumerate(CLASSES):
            true, pred, tp = int(pooled[i].sum()), int(pooled[:, i].sum()), int(pooled[i, i])
            per_class.append({"model": model, "aggregation": "pooled_held_out", "fold": "",
                              "class": name, "precision": tp / pred if pred else 0,
                              "recall": tp / true if true else 0,
                              "f1": 2 * tp / (true + pred) if true + pred else 0, "support": true})
            for j, predicted in enumerate(CLASSES):
                matrices.append({"model": model, "true_class": name, "predicted_class": predicted,
                                 "count": int(pooled[i, j]),
                                 "row_normalized": float(pooled[i, j] / true) if true else 0})
        pooled_metrics = metrics_from_cm(pooled)
        for metric, value in pooled_metrics.items():
            reported = payload["summary"][model][metric]
            summary.append({"model": model, "metric": metric, "fold_mean": reported["mean"],
                            "fold_sample_sd": reported["std"], "pooled_held_out": float(value)})
    write_csv(output / "fold_metrics.csv", folds)
    write_csv(output / "per_class_metrics.csv", per_class)
    write_csv(output / "pooled_confusion_matrices.csv", matrices)
    write_csv(output / "summary_metrics.csv", summary)


def environment(output, cv_script):
    versions = {}
    for package in ["numpy", "scipy", "scikit-learn", "torch", "mediapipe", "opencv-python", "mne", "neurokit2"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed under this interpreter"
    write_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "cv_script": str(cv_script.resolve()),
        "cv_script_sha256": hashlib.sha256(cv_script.read_bytes()).hexdigest() if cv_script.exists() else None,
    })


def bootstrap(rows, output, seed, repetitions):
    """Paired participant bootstrap; comparisons are descriptive, not p-values."""
    subjects = sorted({row["subject_id"] for row in rows})
    sid = {subject: i for i, subject in enumerate(subjects)}
    grouped = {}
    for row in rows:
        key = (row["mode"], row["model"])
        if key not in grouped:
            grouped[key] = np.zeros((len(subjects), 3, 3), dtype=int)
        grouped[key][sid[row["subject_id"]], row["y_true"], row["y_pred"]] += 1
    keys = list(grouped)
    counts = [cm.sum((1, 2)) for cm in grouped.values()]
    if not all(np.array_equal(counts[0], count) for count in counts):
        raise ValueError("Bootstrap requires identical participant cohorts for all configurations")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(repetitions, len(subjects)))
    weights = np.stack([np.bincount(draw, minlength=len(subjects)) for draw in draws])
    estimates, intervals = {}, []
    valid = None
    for key, cm in grouped.items():
        drawn_cm = (weights @ cm.reshape(len(subjects), 9)).reshape(-1, 3, 3)
        current_valid = (drawn_cm.sum(-1) > 0).all(-1)
        valid = current_valid if valid is None else valid & current_valid
        estimates[key] = metrics_from_cm(drawn_cm)
    if int(valid.sum()) < repetitions // 2:
        raise ValueError("Too many participant-bootstrap draws omit classes; do not report these intervals")
    for key, samples in estimates.items():
        point = metrics_from_cm(grouped[key].sum(0))
        for metric, sample in samples.items():
            low, high = np.quantile(sample[valid], [.025, .975])
            intervals.append({"mode": key[0], "model": key[1], "metric": metric,
                              "point_estimate": float(point[metric]), "ci_low": float(low),
                              "ci_high": float(high), "valid_bootstrap_draws": int(valid.sum())})
    comparisons = []
    pairs = [(a, b) for a, b in itertools.combinations(keys, 2) if a[0] == b[0]]
    reference = ("all_modalities", "BiLSTM + Temporal Attention")
    if reference in grouped:
        pairs += [(reference, key) for key in keys if key[0] != reference[0] and key[1] == reference[1]]
    for a, b in pairs:
        for metric in ["balanced_accuracy", "macro_f1"]:
            delta = estimates[a][metric] - estimates[b][metric]
            low, high = np.quantile(delta[valid], [.025, .975])
            point = float(metrics_from_cm(grouped[a].sum(0))[metric] - metrics_from_cm(grouped[b].sum(0))[metric])
            comparisons.append({"first_mode": a[0], "first_model": a[1], "second_mode": b[0],
                                "second_model": b[1], "metric": metric,
                                "point_difference_first_minus_second": point,
                                "ci_low": float(low), "ci_high": float(high)})
    write_csv(output / "participant_bootstrap_intervals.csv", intervals)
    write_csv(output / "paired_participant_bootstrap_differences.csv", comparisons)
    write_json(output / "bootstrap_method.json", {
        "resampling_unit": "participant (all held-out sequences sampled together)",
        "repetitions": repetitions, "seed": seed, "valid_draws": int(valid.sum()),
        "missing_class_draws_excluded": int((~valid).sum()),
        "interval": "descriptive 95% percentile interval, conditional on fixed held-out predictions",
        "limitations": "No model retraining or p-value; overlapping CV training sets, small cohort and training uncertainty limit inferential conclusions.",
    })


def run_experiments(args, paths, inventory, output):
    import torch
    from sklearn.model_selection import GroupKFold

    if not args.cv_script.is_file():
        raise FileNotFoundError(f"Missing {args.cv_script}; set --cv_script to the original train_group_cv.py")
    spec = importlib.util.spec_from_file_location("drozy_original_cv", args.cv_script.resolve())
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    y = np.array([row["class_id"] for row in inventory])
    groups = np.array([row["subject_id"] for row in inventory])
    if len(np.unique(groups)) < args.folds:
        raise ValueError("More folds than participants")
    split_indices = list(GroupKFold(n_splits=args.folds).split(np.arange(len(paths)), y, groups))
    membership = []
    for fold, (train, test) in enumerate(split_indices, 1):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Subject leakage")
        if set(y[train]) != {0, 1, 2} or set(y[test]) != {0, 1, 2}:
            raise ValueError(f"Fold {fold} lacks a class; inspect subject/label distributions before proceeding")
        for role, indices in [("train", train), ("test", test)]:
            for index in indices:
                membership.append({"fold": fold, "role": role, **inventory[index]})
    write_csv(output / "fold_membership.csv", membership)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_compute = cv.compute_metrics
    captured = {}

    def capture(y_true, y_pred):
        captured["true"] = np.array(y_true, copy=True)
        captured["pred"] = np.array(y_pred, copy=True)
        return original_compute(y_true, y_pred)

    cv.compute_metrics = capture
    oof = []
    ablation_rows = []
    for mode in args.modes:
        print(f"\nMODE {mode}: {len(MODES[mode])} features", flush=True)
        results = {"Logistic Regression": [], "Random Forest": [],
                   "BiLSTM + Temporal Attention": [], "Majority baseline": []}
        mode_output = output / mode
        mode_output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="feature_subset_", dir=output) as tmp:
            subset_paths = []
            for index, path in enumerate(paths):
                with np.load(path, allow_pickle=True) as data:
                    values = {name: data[name] for name in data.files}
                columns = MODES[mode]
                values["X"] = values["X"][:, columns]
                values["feature_names"] = values["feature_names"][columns]
                target = Path(tmp) / f"{index:05d}.npz"
                np.savez_compressed(target, **values)
                subset_paths.append(target)
            for fold, (train, test) in enumerate(split_indices, 1):
                print(f"  Fold {fold}/{args.folds}; test subjects {sorted(set(groups[test]))}", flush=True)
                train_paths = [subset_paths[index] for index in train]
                test_paths = [subset_paths[index] for index in test]
                fold_seed = args.seed + fold
                runners = {
                    "Logistic Regression": lambda: cv.run_logistic_regression(train_paths, test_paths, fold_seed),
                    "Random Forest": lambda: cv.run_random_forest(train_paths, test_paths, fold_seed),
                    "BiLSTM + Temporal Attention": lambda: cv.run_bilstm(
                        outer_train_paths=train_paths, outer_test_paths=test_paths, device=device,
                        fold_seed=fold_seed, hidden_size=args.hidden_size, epochs=args.epochs,
                        patience=args.patience, batch_size=args.batch_size, lr=args.lr),
                    "Majority baseline": lambda: capture(y[test], np.full(len(test), np.bincount(y[train], minlength=3).argmax())),
                }
                for model, runner in runners.items():
                    captured.clear()
                    row = runner()
                    if not np.array_equal(captured["true"], y[test]):
                        raise AssertionError("Held-out predictions do not match test sequence ordering")
                    results[model].append(row)
                    for index, truth, prediction in zip(test, captured["true"], captured["pred"]):
                        oof.append({"mode": mode, "model": model, "fold": fold,
                                    "subject_id": int(groups[index]),
                                    "session_id": inventory[index]["session_id"],
                                    "file": str(paths[index]), "y_true": int(truth), "y_pred": int(prediction)})
                    print(f"    {model}: balanced accuracy {row['balanced_accuracy']:.3f}, macro F1 {row['macro_f1']:.3f}", flush=True)
                write_csv(output / "oof_predictions.csv", oof)
        payload = {"protocol": f"{args.folds}-fold subject-independent GroupKFold", "mode": mode,
                   "num_classes": 3, "class_names": CLASSES, "num_samples": len(paths),
                   "num_subjects": len(np.unique(groups)), "fold_results": results,
                   "summary": {model: cv.summarize_model(rows) for model, rows in results.items()}}
        write_json(mode_output / "group_cv_results.json", payload)
        export_results(payload, mode_output)
        for model, summary in payload["summary"].items():
            ablation_rows.append({
                "mode": mode, "features_per_timestep": len(MODES[mode]), "model": model,
                **{f"{metric}_{stat}": summary[metric][stat]
                   for metric in ["accuracy", "balanced_accuracy", "macro_f1"]
                   for stat in ["mean", "std"]},
            })
        write_csv(output / "ablation_summary.csv", ablation_rows)
    bootstrap(oof, output, args.seed, args.bootstrap)
    write_json(output / "run_configuration.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device), "cohort_policy": "same full-modality eligible cohort and outer folds for every ablation",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_dir", type=Path, default=Path("data/processed/drozy"))
    parser.add_argument("--cv_script", type=Path, default=Path("scripts/train_group_cv.py"))
    parser.add_argument("--out", type=Path, default=Path("runs/drozy_revision"))
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.existing and args.run:
        parser.error("Use --existing or --run, not both")
    args.out.mkdir(parents=True, exist_ok=True)
    environment(args.out, args.cv_script)
    paths, inventory = audit(args.features_dir, args.out)
    if args.existing:
        payload = json.loads(args.existing.read_text(encoding="utf-8"))
        if int(payload["num_samples"]) != len(paths):
            raise ValueError("Existing report and audited feature set have different sample counts")
        if int(payload["num_subjects"]) != len({row["subject_id"] for row in inventory}):
            raise ValueError("Existing report and audited feature set have different participant counts")
        for model, folds in payload["fold_results"].items():
            support = np.sum([np.asarray(fold["confusion_matrix"]).sum(1) for fold in folds], axis=0)
            expected = np.bincount([row["class_id"] for row in inventory], minlength=3)
            if not np.array_equal(support, expected):
                raise ValueError(f"{model}: existing report class support differs from audited cohort")
        export_results(payload, args.out)
        print("Existing results exported without retraining. No participant bootstrap is inferred from aggregate matrices.")
    elif args.run:
        run_experiments(args, paths, inventory, args.out)
    print(f"Saved reports in {args.out.resolve()}")


if __name__ == "__main__":
    main()

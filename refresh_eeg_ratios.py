#!/usr/bin/env python
"""Refresh three EEG ratios from EDF recordings in a new copy of processed data.

Run from the FatigueTrack project root after replacing
fatiguetrack/eeg_features.py with the supplied correction:
  python refresh_eeg_ratios.py

Existing video, ECG, labels, windows and other EEG features are preserved.
This script verifies that the other nine EEG descriptors match the original
extraction before replacing ratios. It needs the project's MNE environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD = "positive_denominator_v1"
RATIO_NAMES = ["eeg_theta_alpha_ratio_mean", "eeg_theta_beta_ratio_mean", "eeg_alpha_beta_ratio_mean"]


def read_sample(path):
    with np.load(path, allow_pickle=True) as data:
        values = {key: data[key].copy() for key in data.files}
    required = {"X", "feature_names", "length", "start_sec", "end_sec", "session_id", "subject_id", "y", "kss"}
    if not required.issubset(values):
        raise ValueError(f"{path.name}: missing {sorted(required - values.keys())}")
    if values["X"].shape != (6, 26) or int(values["length"]) != 6:
        raise ValueError(f"{path.name}: expected a 6 x 26 sequence")
    return values


def refresh_session(paths, source_root, target_root, raw, eeg_module, window_sec=10.0):
    """Shared implementation; raw is an MNE recording, testable in memory."""
    channels = [channel for channel in eeg_module.EEG_CHANNELS if channel in raw.ch_names]
    if not channels:
        raise ValueError("No expected EEG channels in recording")
    selected = raw.copy().pick(channels)
    fs = float(selected.info["sfreq"])
    signal = selected.get_data()
    cache = {}
    updated = 0
    newly_missing_ratios = 0
    eeg_names = eeg_module.compact_eeg_feature_names()
    ratio_positions = [eeg_names.index(name) for name in RATIO_NAMES]
    other_positions = [i for i in range(12) if i not in ratio_positions]
    for path in paths:
        values = read_sample(path)
        names = [str(name) for name in values["feature_names"]]
        if names[7:19] != eeg_names:
            raise ValueError(f"{path.name}: EEG feature ordering differs from supplied extractor")
        target_columns = [names.index(name) for name in RATIO_NAMES]
        start, end = float(values["start_sec"]), float(values["end_sec"])
        if not np.isclose(end - start, 6 * window_sec, atol=1e-4):
            raise ValueError(f"{path.name}: temporal window settings differ")
        original = values["X"]
        corrected = original.copy()
        for step in range(6):
            t = start + step * window_sec
            first, last = int(t * fs), int((t + window_sec) * fs)
            if first < 0 or last > signal.shape[1]:
                raise ValueError(f"{path.name}: requested EEG interval is outside EDF duration")
            interval = (first, last)
            if interval not in cache:
                features = eeg_module.extract_eeg_features(signal[:, first:last], fs, channels)
                cache[interval] = eeg_module.compact_eeg_vector(features)
            vector = cache[interval]
            before = original[step, 7:19][other_positions]
            after = vector[other_positions]
            # Relative tolerance applies even to tiny powers: no large absolute floor.
            if not np.allclose(before, after, rtol=2e-4, atol=0.0, equal_nan=True):
                raise ValueError(
                    f"{path.name}, step {step}: non-ratio EEG features do not match. "
                    "Check original EDF files and preprocessing environment before using this correction."
                )
            ratios = vector[ratio_positions]
            if np.isinf(ratios).any():
                raise ValueError(f"{path.name}: corrected ratio overflows float32")
            old_ratios = original[step, target_columns]
            newly_missing_ratios += int((np.isfinite(old_ratios) & ~np.isfinite(ratios)).sum())
            corrected[step, target_columns] = ratios
        untouched = [i for i in range(26) if i not in target_columns]
        if not np.array_equal(original[:, untouched], corrected[:, untouched], equal_nan=True):
            raise AssertionError("Unexpected change outside the three ratio columns")
        values["X"] = corrected
        values["eeg_ratio_method"] = np.asarray(METHOD)
        values["eeg_ratio_additive_epsilon"] = np.float64(0.0)
        target = target_root / path.relative_to(source_root)
        if target.exists():
            raise FileExistsError(f"Will not overwrite {target}; choose an empty output folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(partial, **values)
        partial.replace(target)
        updated += 1
    return {"sequences": updated, "unique_windows": len(cache), "channels": channels,
            "newly_missing_ratio_values": newly_missing_ratios}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_dir", type=Path, default=Path("data/processed/drozy"))
    parser.add_argument("--drozy_dir", type=Path, default=Path("data/raw/drozy"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/drozy_eeg_corrected"))
    parser.add_argument("--eeg_script", type=Path, default=Path("fatiguetrack/eeg_features.py"))
    args = parser.parse_args()
    source, target = args.features_dir.resolve(), args.output_dir.resolve()
    if source == target or source in target.parents or target in source.parents:
        parser.error("Source and output must be separate folders, with neither nested in the other")
    if target.exists() and any(target.iterdir()):
        parser.error("Output folder is not empty; choose a new output directory")
    if not args.eeg_script.is_file():
        parser.error(f"Missing corrected EEG module: {args.eeg_script}")
    spec = importlib.util.spec_from_file_location("drozy_corrected_eeg", args.eeg_script.resolve())
    eeg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eeg)
    if getattr(eeg, "EEG_RATIO_METHOD", None) != METHOD:
        parser.error("Replace fatiguetrack/eeg_features.py with the supplied corrected file first")
    paths = sorted(source.rglob("*.npz"))
    if not paths:
        parser.error(f"No processed samples found in {source}")
    grouped = defaultdict(list)
    for path in paths:
        values = read_sample(path)
        grouped[str(values["session_id"].item())].append(path)
    import mne
    report = {"method": METHOD, "source": str(source), "output": str(target),
              "eeg_source_sha256": hashlib.sha256(args.eeg_script.read_bytes()).hexdigest(),
              "initial_sequences": len(paths), "sessions": {}}
    for session, session_paths in sorted(grouped.items()):
        if Path(session).name != session or any(c not in "0123456789-" for c in session):
            raise ValueError(f"Unexpected session identifier: {session}")
        edf = args.drozy_dir / "psg" / f"{session}.edf"
        if not edf.is_file():
            raise FileNotFoundError(edf)
        print(f"Refreshing EEG ratios: session {session}", flush=True)
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
        try:
            report["sessions"][session] = refresh_session(session_paths, source, target, raw, eeg)
        finally:
            raw.close()
    report["updated_sequences"] = sum(item["sequences"] for item in report["sessions"].values())
    report["newly_missing_ratio_values"] = sum(item["newly_missing_ratio_values"] for item in report["sessions"].values())
    target.mkdir(parents=True, exist_ok=True)
    (target / "eeg_ratio_refresh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Updated {report['updated_sequences']} sequences. Saved: {target}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mne
import numpy as np

from fatiguetrack.behavioral_features import (
    FatigueBehavioralExtractor,
    BehavioralWindowAggregator,
    behavioral_feature_names,
    behavioral_feature_vector,
)

from fatiguetrack.eeg_features_04_10_2026 import (
    EEG_CHANNELS,
    extract_eeg_features,
    compact_eeg_feature_names,
    compact_eeg_vector,
)

from fatiguetrack.ecg_features import (
    extract_ecg_features,
    ecg_feature_names,
    ecg_feature_vector,
)

from fatiguetrack.labels import (
    load_kss_labels,
)


# ============================================================
# SUBJECT-INDEPENDENT SPLIT
# ============================================================

TRAIN_SUBJECTS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9
}

VAL_SUBJECTS = {
    10, 11
}

TEST_SUBJECTS = {
    12, 13, 14
}


# ============================================================
# FEATURE NAMES
# ============================================================

MULTIMODAL_FEATURE_NAMES = (
    behavioral_feature_names()
    + compact_eeg_feature_names()
    + ecg_feature_names()
)


def get_split(subject_id: int) -> str:
    """
    Subject-independent train / validation / test split.
    """

    if subject_id in TRAIN_SUBJECTS:
        return "train"

    if subject_id in VAL_SUBJECTS:
        return "val"

    if subject_id in TEST_SUBJECTS:
        return "test"

    raise ValueError(
        f"Subject {subject_id} is not assigned "
        f"to train/val/test."
    )


def parse_session_id(
    session_id: str,
) -> Tuple[int, int]:
    """
    Example:
        1-3 -> subject=1, test=3
    """

    subject_str, test_str = session_id.split("-")

    return (
        int(subject_str),
        int(test_str),
    )


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

def extract_behavioral_windows(
    video_path: Path,
    window_sec: float,
    target_fps: float,
    max_duration_sec: float,
) -> np.ndarray:
    """
    Extract behavioral features in fixed temporal windows.

    Returns
    -------
    Array:
        [num_windows, 7]
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    video_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if video_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid FPS for {video_path}"
        )

    # Reduce frame processing cost.
    # Example:
    # original 30 FPS, target 10 FPS -> process every 3rd frame.
    frame_stride = max(
        1,
        int(round(
            video_fps / target_fps
        )),
    )

    n_windows = int(
        max_duration_sec // window_sec
    )

    aggregators = [
        BehavioralWindowAggregator(
            window_seconds=window_sec
        )
        for _ in range(n_windows)
    ]

    extractor = (
        FatigueBehavioralExtractor()
    )

    frame_index = 0
    processed_frames = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            t_sec = (
                frame_index / video_fps
            )

            frame_index += 1

            if t_sec >= max_duration_sec:
                break

            # Downsample video processing.
            if (
                (frame_index - 1)
                % frame_stride
                != 0
            ):
                continue

            window_idx = int(
                t_sec // window_sec
            )

            if (
                window_idx < 0
                or window_idx >= n_windows
            ):
                continue

            frame_features = (
                extractor.extract(frame)
            )

            aggregators[
                window_idx
            ].add(frame_features)

            processed_frames += 1

    finally:

        cap.release()
        extractor.close()

    windows = []

    for aggregator in aggregators:

        features = (
            aggregator.compute()
        )

        vector = (
            behavioral_feature_vector(
                features
            )
        )

        windows.append(vector)

    result = np.stack(
        windows,
        axis=0,
    ).astype(np.float32)

    print(
        f"    Behavioral: "
        f"{result.shape}, "
        f"processed_frames="
        f"{processed_frames}"
    )

    return result


# ============================================================
# EEG FEATURES
# ============================================================

def extract_eeg_windows(
    raw,
    window_sec: float,
    max_duration_sec: float,
) -> np.ndarray:
    """
    Extract compact EEG features from:

        Fz, Cz, C3, C4, Pz, Oz

    Returns
    -------
    [num_windows, 12]
    """

    # Use only EEG channels that are actually present
    available_channels = [
        ch for ch in EEG_CHANNELS
        if ch in raw.ch_names
    ]

    missing_channels = [
        ch for ch in EEG_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing_channels:
        print(
            f"    WARNING: missing EEG channels: "
            f"{missing_channels}"
        )

    if len(available_channels) == 0:
        raise RuntimeError(
            "No expected EEG channels found "
            f"in recording. Available: {raw.ch_names}"
        )

    eeg_raw = raw.copy().pick(
        available_channels
    )

    fs = float(
        eeg_raw.info["sfreq"]
    )

    eeg_data = eeg_raw.get_data()

    print(
        f"    EEG channels used: "
        f"{available_channels}"
    )

    n_windows = int(
        max_duration_sec // window_sec
    )

    vectors = []

    for window_idx in range(
        n_windows
    ):

        start_sec = (
            window_idx * window_sec
        )

        end_sec = (
            start_sec + window_sec
        )

        start_sample = int(
            start_sec * fs
        )

        end_sample = int(
            end_sec * fs
        )

        eeg_window = eeg_data[
            :,
            start_sample:end_sample,
        ]

        features = (
            extract_eeg_features(
                eeg_window,
                fs,
                available_channels,
            )
        )

        vector = compact_eeg_vector(
            features
        )

        vectors.append(vector)

    result = np.stack(
        vectors,
        axis=0,
    ).astype(np.float32)

    print(
        f"    EEG:        "
        f"{result.shape}"
    )

    return result


# ============================================================
# ECG FEATURES
# ============================================================

def extract_ecg_windows(
    raw,
    window_sec: float,
    ecg_context_sec: float,
    max_duration_sec: float,
) -> np.ndarray:
    """
    Compute one ECG vector for every temporal step.

    Behavioral and EEG windows are 10 s.

    ECG uses a wider context (default 30 s)
    to obtain more stable HR/HRV estimates.

    Example for temporal window 30-40 s:

        ECG context approximately 20-50 s

    Returns
    -------
    [num_windows, 7]
    """

    ecg_raw = (
        raw.copy().pick(["ECG"])
    )

    fs = float(
        ecg_raw.info["sfreq"]
    )

    ecg_data = (
        ecg_raw
        .get_data()[0]
    )

    n_windows = int(
        max_duration_sec // window_sec
    )

    half_context = (
        ecg_context_sec / 2.0
    )

    vectors = []

    for window_idx in range(
        n_windows
    ):

        # Center of corresponding
        # behavioral / EEG window.
        center_sec = (
            window_idx * window_sec
            + window_sec / 2.0
        )

        start_sec = max(
            0.0,
            center_sec - half_context,
        )

        end_sec = min(
            max_duration_sec,
            center_sec + half_context,
        )

        start_sample = int(
            start_sec * fs
        )

        end_sample = int(
            end_sec * fs
        )

        ecg_window = ecg_data[
            start_sample:end_sample
        ]

        features = (
            extract_ecg_features(
                ecg_window,
                fs,
            )
        )

        vector = (
            ecg_feature_vector(
                features
            )
        )

        vectors.append(vector)

    result = np.stack(
        vectors,
        axis=0,
    ).astype(np.float32)

    print(
        f"    ECG:        "
        f"{result.shape}"
    )

    return result


# ============================================================
# SESSION PROCESSING
# ============================================================

def process_session(
    session_id: str,
    video_path: Path,
    edf_path: Path,
    output_root: Path,
    label_info: Dict,
    window_sec: float,
    sequence_windows: int,
    sequence_stride: int,
    target_behavior_fps: float,
    ecg_context_sec: float,
    overwrite: bool,
) -> int:
    """
    Process one DROZY session.

    Final temporal representation:

        10-second window
            ->
        26 features

    Then groups:

        6 windows
            ->
        X.shape = (6, 26)

    which corresponds to a 60-second sequence
    when window_sec=10 and sequence_windows=6.
    """

    subject_id, test_id = (
        parse_session_id(
            session_id
        )
    )

    split = get_split(
        subject_id
    )

    kss = int(
        label_info["kss"]
    )

    class_id = int(
        label_info["class_id"]
    )

    class_name = str(
        label_info["class_name"]
    )

    print(
        f"\n[{session_id}] "
        f"subject={subject_id} "
        f"KSS={kss} "
        f"class={class_name} "
        f"split={split}"
    )

    # -------------------------------------------------------
    # Load EDF once
    # -------------------------------------------------------

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    print(
        f"    EDF channels: {raw.ch_names}"
    )

    edf_duration = (
        raw.n_times
        / float(raw.info["sfreq"])
    )

    # -------------------------------------------------------
    # Video duration
    # -------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    video_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = float(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if video_fps <= 0:
        raise RuntimeError(
            f"Invalid video FPS: "
            f"{video_path}"
        )

    video_duration = (
        frame_count / video_fps
    )

    # Use only the portion common
    # to video and physiological recording.
    max_duration = min(
        video_duration,
        edf_duration,
    )

    # Only complete windows.
    max_duration = (
        int(max_duration // window_sec)
        * window_sec
    )

    print(
        f"    video_duration="
        f"{video_duration:.2f}s"
    )

    print(
        f"    edf_duration="
        f"{edf_duration:.2f}s"
    )

    print(
        f"    usable_duration="
        f"{max_duration:.2f}s"
    )

    if max_duration < (
        window_sec
        * sequence_windows
    ):
        print(
            "    SKIP: session too short."
        )
        return 0

    # -------------------------------------------------------
    # Extract three modalities
    # -------------------------------------------------------

    behavioral = (
        extract_behavioral_windows(
            video_path=video_path,
            window_sec=window_sec,
            target_fps=target_behavior_fps,
            max_duration_sec=max_duration,
        )
    )

    eeg = extract_eeg_windows(
        raw=raw,
        window_sec=window_sec,
        max_duration_sec=max_duration,
    )

    ecg = extract_ecg_windows(
        raw=raw,
        window_sec=window_sec,
        ecg_context_sec=ecg_context_sec,
        max_duration_sec=max_duration,
    )

    # -------------------------------------------------------
    # Temporal alignment
    # -------------------------------------------------------

    n_windows = min(
        len(behavioral),
        len(eeg),
        len(ecg),
    )

    behavioral = behavioral[
        :n_windows
    ]

    eeg = eeg[
        :n_windows
    ]

    ecg = ecg[
        :n_windows
    ]

    # Early multimodal feature fusion.
    multimodal = np.concatenate(
        [
            behavioral,
            eeg,
            ecg,
        ],
        axis=1,
    ).astype(np.float32)

    print(
        f"    Multimodal: "
        f"{multimodal.shape}"
    )

    if (
        multimodal.shape[1]
        != len(
            MULTIMODAL_FEATURE_NAMES
        )
    ):
        raise RuntimeError(
            "Feature dimension mismatch: "
            f"{multimodal.shape[1]} "
            f"vs expected "
            f"{len(MULTIMODAL_FEATURE_NAMES)}"
        )

    # -------------------------------------------------------
    # Generate sequence samples
    # -------------------------------------------------------

    split_dir = (
        output_root / split
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0

    max_start = (
        n_windows
        - sequence_windows
    )

    for start_idx in range(
        0,
        max_start + 1,
        sequence_stride,
    ):

        end_idx = (
            start_idx
            + sequence_windows
        )

        X = multimodal[
            start_idx:end_idx
        ]

        if (
            X.shape[0]
            != sequence_windows
        ):
            continue

        start_sec = (
            start_idx
            * window_sec
        )

        end_sec = (
            end_idx
            * window_sec
        )

        filename = (
            f"{session_id}"
            f"_seq_{start_idx:03d}"
            f".npz"
        )

        output_path = (
            split_dir
            / filename
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            continue

        np.savez_compressed(
            output_path,

            X=X.astype(
                np.float32
            ),

            length=np.int64(
                X.shape[0]
            ),

            y=np.int64(
                class_id
            ),

            kss=np.int64(
                kss
            ),

            subject_id=np.int64(
                subject_id
            ),

            test_id=np.int64(
                test_id
            ),

            session_id=np.asarray(
                session_id
            ),

            class_name=np.asarray(
                class_name
            ),

            start_sec=np.float32(
                start_sec
            ),

            end_sec=np.float32(
                end_sec
            ),

            feature_names=np.asarray(
                MULTIMODAL_FEATURE_NAMES
            ),
        )

        saved += 1

    print(
        f"    Saved sequences: "
        f"{saved}"
    )

    return saved


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--drozy_dir",
        type=str,
        default="data/raw/drozy",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/drozy",
    )

    parser.add_argument(
        "--window_sec",
        type=float,
        default=10.0,
        help=(
            "Behavioral/EEG temporal "
            "window length."
        ),
    )

    parser.add_argument(
        "--sequence_windows",
        type=int,
        default=6,
        help=(
            "Number of temporal windows "
            "per BiLSTM sample."
        ),
    )

    parser.add_argument(
        "--sequence_stride",
        type=int,
        default=6,
        help=(
            "Stride in temporal windows. "
            "6 gives non-overlapping "
            "60-second samples when "
            "window_sec=10."
        ),
    )

    parser.add_argument(
        "--behavior_fps",
        type=float,
        default=10.0,
        help=(
            "Video processing FPS. "
            "Original videos are 30 FPS."
        ),
    )

    parser.add_argument(
        "--ecg_context_sec",
        type=float,
        default=30.0,
        help=(
            "ECG context used to estimate "
            "HR/HRV for each temporal step."
        ),
    )

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help=(
            "Useful for testing. "
            "Example: --max_sessions 1"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    drozy_dir = Path(
        args.drozy_dir
    )

    output_dir = Path(
        args.output_dir
    )

    video_dir = (
        drozy_dir / "videos_i8"
    )

    psg_dir = (
        drozy_dir / "psg"
    )

    kss_path = (
        drozy_dir / "KSS.txt"
    )

    if not video_dir.exists():
        raise FileNotFoundError(
            video_dir
        )

    if not psg_dir.exists():
        raise FileNotFoundError(
            psg_dir
        )

    if not kss_path.exists():
        raise FileNotFoundError(
            kss_path
        )

    labels = load_kss_labels(
        kss_path
    )

    # Use EDF files as canonical session list.
    edf_paths = sorted(
        psg_dir.glob("*.edf"),
        key=lambda p: tuple(
            int(x)
            for x
            in p.stem.split("-")
        ),
    )

    sessions = []

    for edf_path in edf_paths:

        session_id = (
            edf_path.stem
        )

        video_path = (
            video_dir
            / f"{session_id}.mp4"
        )

        if not video_path.exists():
            print(
                f"WARNING: video missing "
                f"for {session_id}"
            )
            continue

        if session_id not in labels:
            print(
                f"WARNING: KSS label missing "
                f"for {session_id}"
            )
            continue

        sessions.append(
            (
                session_id,
                video_path,
                edf_path,
            )
        )

    if args.max_sessions is not None:
        sessions = sessions[
            :args.max_sessions
        ]

    print(
        "===================================="
    )
    print(
        "DROZY MULTIMODAL PREPROCESSING"
    )
    print(
        "===================================="
    )

    print(
        f"Sessions: {len(sessions)}"
    )

    print(
        f"Behavioral features: "
        f"{len(behavioral_feature_names())}"
    )

    print(
        f"EEG features: "
        f"{len(compact_eeg_feature_names())}"
    )

    print(
        f"ECG features: "
        f"{len(ecg_feature_names())}"
    )

    print(
        f"Total features: "
        f"{len(MULTIMODAL_FEATURE_NAMES)}"
    )

    print(
        f"Window: "
        f"{args.window_sec}s"
    )

    print(
        f"Sequence: "
        f"{args.sequence_windows} windows"
    )

    print(
        f"Sequence duration: "
        f"{args.window_sec * args.sequence_windows}s"
    )

    total_saved = 0

    class_counter = Counter()
    split_counter = Counter()

    for (
        session_id,
        video_path,
        edf_path,
    ) in sessions:

        label_info = labels[
            session_id
        ]

        subject_id, _ = (
            parse_session_id(
                session_id
            )
        )

        split = get_split(
            subject_id
        )

        n_saved = process_session(
            session_id=session_id,
            video_path=video_path,
            edf_path=edf_path,
            output_root=output_dir,
            label_info=label_info,
            window_sec=args.window_sec,
            sequence_windows=args.sequence_windows,
            sequence_stride=args.sequence_stride,
            target_behavior_fps=args.behavior_fps,
            ecg_context_sec=args.ecg_context_sec,
            overwrite=args.overwrite,
        )

        total_saved += n_saved

        class_counter[
            label_info["class_name"]
        ] += n_saved

        split_counter[
            split
        ] += n_saved

    print(
        "\n===================================="
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "===================================="
    )

    print(
        f"Total sequences: "
        f"{total_saved}"
    )

    print(
        "\nBy split:"
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        print(
            f"  {split:5s}: "
            f"{split_counter[split]}"
        )

    print(
        "\nBy class:"
    )

    for class_name in [
        "Low",
        "Moderate",
        "High",
    ]:
        print(
            f"  {class_name:8s}: "
            f"{class_counter[class_name]}"
        )

    print(
        "\nFeature dimension:"
    )

    print(
        f"  F = "
        f"{len(MULTIMODAL_FEATURE_NAMES)}"
    )

    print(
        "\nOutput:"
    )

    print(
        f"  {output_dir}"
    )


if __name__ == "__main__":
    main()
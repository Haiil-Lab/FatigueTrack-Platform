from pathlib import Path

import mne

from fatiguetrack.ecg_features import (
    extract_ecg_features,
    ecg_feature_vector,
    ecg_feature_names,
)


ROOT = Path(__file__).resolve().parents[1]

EDF_PATH = (
    ROOT
    / "data"
    / "raw"
    / "drozy"
    / "psg"
    / "1-1.edf"
)


def main():

    raw = mne.io.read_raw_edf(
        EDF_PATH,
        preload=True,
        verbose=False,
    )

    fs = float(raw.info["sfreq"])

    print("Sampling rate:", fs)

    print("Available channels:")
    print(raw.ch_names)

    # Select DROZY ECG channel
    ecg_raw = raw.copy().pick(["ECG"])

    ecg_data = ecg_raw.get_data()[0]

    print(
        "ECG samples:",
        len(ecg_data),
    )

    print(
        "Duration:",
        len(ecg_data) / fs,
        "seconds",
    )

    # -----------------------------------------
    # Test first 10-second ECG window
    # -----------------------------------------
    window_seconds = 10

    n_samples = int(
        window_seconds * fs
    )

    ecg_window = ecg_data[
        :n_samples
    ]

    features = extract_ecg_features(
        ecg_window,
        fs,
    )

    vector = ecg_feature_vector(
        features
    )

    print(
        "\n=== ECG FEATURES ==="
    )

    for name, value in zip(
        ecg_feature_names(),
        vector,
    ):

        print(
            f"{name:25s}: "
            f"{value:.4f}"
        )

    print(
        "\nVector shape:",
        vector.shape,
    )


if __name__ == "__main__":
    main()
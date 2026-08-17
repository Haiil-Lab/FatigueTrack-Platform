from pathlib import Path

import mne

from fatiguetrack.eeg_features import (
    EEG_CHANNELS,
    extract_eeg_features,
    compact_eeg_vector,
    compact_eeg_feature_names,
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

    print("Sampling rate:", raw.info["sfreq"])

    eeg_raw = raw.copy().pick(
        EEG_CHANNELS
    )

    eeg_data = eeg_raw.get_data()

    fs = float(
        eeg_raw.info["sfreq"]
    )

    print(
        "EEG shape:",
        eeg_data.shape,
    )

    # For this test use only first 10 seconds.
    samples_10_sec = int(
        fs * 10
    )

    eeg_window = eeg_data[
        :,
        :samples_10_sec
    ]

    features = extract_eeg_features(
        eeg_window,
        fs,
        EEG_CHANNELS,
    )

    vector = compact_eeg_vector(
        features
    )

    print(
        "\n=== COMPACT EEG FEATURES ==="
    )

    for name, value in zip(
        compact_eeg_feature_names(),
        vector,
    ):
        print(
            f"{name:35s}: "
            f"{value:.6f}"
        )

    print(
        "\nVector shape:",
        vector.shape,
    )


if __name__ == "__main__":
    main()
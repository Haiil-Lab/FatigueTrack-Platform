from pathlib import Path

import mne


ROOT = Path(__file__).resolve().parents[1]
EDF_PATH = ROOT / "data" / "raw" / "drozy" / "psg" / "1-1.edf"


def main():
    print(f"Reading: {EDF_PATH}")

    raw = mne.io.read_raw_edf(
        EDF_PATH,
        preload=False,
        verbose=False,
    )

    print("\n=== GENERAL INFO ===")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")

    print("\n=== CHANNELS ===")
    for i, channel in enumerate(raw.ch_names):
        print(f"{i:02d}: {channel}")


if __name__ == "__main__":
    main()
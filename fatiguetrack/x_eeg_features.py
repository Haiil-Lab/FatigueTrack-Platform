from typing import Dict

import numpy as np
from scipy.signal import welch


EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def compute_band_power(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
) -> float:
    """
    Compute EEG band power using Welch PSD.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size < 2:
        return np.nan

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=min(len(signal), int(fs * 2)),
    )

    mask = (freqs >= low) & (freqs < high)

    if not np.any(mask):
        return np.nan

    return float(np.trapz(psd[mask], freqs[mask]))


def extract_eeg_features(
    signal: np.ndarray,
    fs: float,
) -> Dict[str, float]:

    powers = {}

    for name, (low, high) in EEG_BANDS.items():
        powers[name] = compute_band_power(
            signal,
            fs,
            low,
            high,
        )

    total_power = sum(
        value for value in powers.values()
        if np.isfinite(value)
    )

    eps = 1e-8

    features = {
        "eeg_delta_power": powers["delta"],
        "eeg_theta_power": powers["theta"],
        "eeg_alpha_power": powers["alpha"],
        "eeg_beta_power": powers["beta"],

        "eeg_theta_alpha_ratio":
            powers["theta"] / (powers["alpha"] + eps),

        "eeg_theta_beta_ratio":
            powers["theta"] / (powers["beta"] + eps),

        "eeg_alpha_beta_ratio":
            powers["alpha"] / (powers["beta"] + eps),
    }

    if total_power > 0:
        for name, value in powers.items():
            features[f"eeg_{name}_relative"] = (
                value / total_power
            )

    return features
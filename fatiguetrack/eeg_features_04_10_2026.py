from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import welch


EEG_CHANNELS: List[str] = [
    "Fz",
    "Cz",
    "C3",
    "C4",
    "Pz",
    "Oz",
]

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

EPS = 1e-8


def _safe_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> float:
    """
    Integrate PSD within a frequency band.
    """
    mask = (freqs >= low) & (freqs < high)

    if not np.any(mask):
        return float("nan")

    return float(
        np.trapz(
            psd[mask],
            freqs[mask],
        )
    )


def _spectral_entropy(psd: np.ndarray) -> float:
    """
    Normalized spectral entropy in range approximately [0, 1].
    """
    psd = np.asarray(psd, dtype=np.float64)

    psd = psd[np.isfinite(psd)]
    psd = psd[psd >= 0]

    if psd.size == 0:
        return float("nan")

    total = np.sum(psd)

    if total <= 0:
        return float("nan")

    p = psd / total

    entropy = -np.sum(
        p * np.log(p + EPS)
    )

    if len(p) > 1:
        entropy /= np.log(len(p))

    return float(entropy)


def extract_eeg_channel_features(
    signal: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    """
    Extract fatigue-related EEG features from one EEG channel.

    Features:
        delta power
        theta power
        alpha power
        beta power

        relative delta
        relative theta
        relative alpha
        relative beta

        theta / alpha
        theta / beta
        alpha / beta

        spectral entropy
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).reshape(-1)

    signal = signal[np.isfinite(signal)]

    if len(signal) < 2:
        return {}

    # Remove DC component.
    signal = signal - np.mean(signal)

    nperseg = min(
        len(signal),
        int(fs * 2),
    )

    if nperseg < 2:
        return {}

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    powers: Dict[str, float] = {}

    for band_name, (low, high) in EEG_BANDS.items():
        powers[band_name] = _safe_band_power(
            freqs,
            psd,
            low,
            high,
        )

    total_band_power = sum(
        value
        for value in powers.values()
        if np.isfinite(value)
    )

    features: Dict[str, float] = {
        "delta_power": powers["delta"],
        "theta_power": powers["theta"],
        "alpha_power": powers["alpha"],
        "beta_power": powers["beta"],
    }

    # Relative band powers.
    if total_band_power > 0:
        features["delta_relative"] = (
            powers["delta"] / total_band_power
        )
        features["theta_relative"] = (
            powers["theta"] / total_band_power
        )
        features["alpha_relative"] = (
            powers["alpha"] / total_band_power
        )
        features["beta_relative"] = (
            powers["beta"] / total_band_power
        )
    else:
        features["delta_relative"] = float("nan")
        features["theta_relative"] = float("nan")
        features["alpha_relative"] = float("nan")
        features["beta_relative"] = float("nan")

    # Fatigue-related spectral ratios.
    features["theta_alpha_ratio"] = (
        powers["theta"] /
        (powers["alpha"] + EPS)
    )

    features["theta_beta_ratio"] = (
        powers["theta"] /
        (powers["beta"] + EPS)
    )

    features["alpha_beta_ratio"] = (
        powers["alpha"] /
        (powers["beta"] + EPS)
    )

    # Entropy calculated over 0.5-30 Hz only.
    entropy_mask = (
        (freqs >= 0.5)
        & (freqs <= 30.0)
    )

    features["spectral_entropy"] = _spectral_entropy(
        psd[entropy_mask]
    )

    return features


def extract_eeg_features(
    eeg_data: np.ndarray,
    fs: float,
    channel_names: List[str] | None = None,
) -> Dict[str, float]:
    """
    Extract EEG features from all DROZY EEG channels.

    Parameters
    ----------
    eeg_data:
        Array with shape:
            [n_channels, n_samples]

    fs:
        EEG sampling frequency.
        DROZY uses 512 Hz.

    channel_names:
        Names corresponding to rows in eeg_data.

    Returns
    -------
    Dictionary containing:

        1. Per-channel features, e.g.
           Fz_theta_power
           Cz_alpha_power

        2. Mean EEG features across all available channels, e.g.
           eeg_theta_power_mean
           eeg_theta_alpha_ratio_mean
    """

    eeg_data = np.asarray(
        eeg_data,
        dtype=np.float64,
    )

    if eeg_data.ndim != 2:
        raise ValueError(
            "eeg_data must have shape "
            "[n_channels, n_samples]"
        )

    if channel_names is None:
        channel_names = EEG_CHANNELS

    if eeg_data.shape[0] != len(channel_names):
        raise ValueError(
            f"Number of EEG channels in data "
            f"({eeg_data.shape[0]}) does not match "
            f"channel_names ({len(channel_names)})."
        )

    all_features: Dict[str, float] = {}

    channel_feature_dicts = []

    for channel_idx, channel_name in enumerate(channel_names):

        channel_signal = eeg_data[channel_idx]

        channel_features = extract_eeg_channel_features(
            channel_signal,
            fs,
        )

        channel_feature_dicts.append(
            channel_features
        )

        for feature_name, value in channel_features.items():

            key = (
                f"{channel_name}_"
                f"{feature_name}"
            )

            all_features[key] = float(value)

    # -------------------------------------------------------
    # Mean representation across EEG channels.
    # This is the compact representation we will initially
    # feed into the multimodal BiLSTM.
    # -------------------------------------------------------

    if channel_feature_dicts:

        feature_names = list(
            channel_feature_dicts[0].keys()
        )

        for feature_name in feature_names:

            values = np.asarray(
                [
                    channel_features.get(
                        feature_name,
                        np.nan,
                    )
                    for channel_features
                    in channel_feature_dicts
                ],
                dtype=np.float64,
            )

            if np.isfinite(values).any():

                mean_value = float(
                    np.nanmean(values)
                )

            else:
                mean_value = float("nan")

            all_features[
                f"eeg_{feature_name}_mean"
            ] = mean_value

    return all_features


def compact_eeg_feature_names() -> List[str]:
    """
    Feature names used for the first compact fatigue model.
    """

    return [
        "eeg_delta_power_mean",
        "eeg_theta_power_mean",
        "eeg_alpha_power_mean",
        "eeg_beta_power_mean",

        "eeg_delta_relative_mean",
        "eeg_theta_relative_mean",
        "eeg_alpha_relative_mean",
        "eeg_beta_relative_mean",

        "eeg_theta_alpha_ratio_mean",
        "eeg_theta_beta_ratio_mean",
        "eeg_alpha_beta_ratio_mean",

        "eeg_spectral_entropy_mean",
    ]


def compact_eeg_vector(
    features: Dict[str, float],
) -> np.ndarray:
    """
    Convert extracted EEG dictionary into a fixed-order vector.

    Output shape:
        (12,)
    """

    names = compact_eeg_feature_names()

    return np.asarray(
        [
            features.get(
                name,
                np.nan,
            )
            for name in names
        ],
        dtype=np.float32,
    )
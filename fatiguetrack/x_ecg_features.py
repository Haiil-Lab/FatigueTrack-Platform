from typing import Dict

import numpy as np
import neurokit2 as nk


def extract_ecg_features(
    signal: np.ndarray,
    fs: float,
) -> Dict[str, float]:

    signal = np.asarray(signal, dtype=np.float64)

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=fs,
    )

    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=fs,
    )

    peaks = info["ECG_R_Peaks"]

    if len(peaks) < 3:
        return {}

    rr_seconds = np.diff(peaks) / fs
    rr_ms = rr_seconds * 1000.0

    hr = 60.0 / rr_seconds

    diff_rr = np.diff(rr_ms)

    features = {
        "ecg_hr_mean": float(np.mean(hr)),
        "ecg_hr_min": float(np.min(hr)),
        "ecg_hr_max": float(np.max(hr)),

        "ecg_rr_mean": float(np.mean(rr_ms)),

        "ecg_sdnn": float(
            np.std(rr_ms, ddof=1)
        ),

        "ecg_rmssd": float(
            np.sqrt(np.mean(diff_rr ** 2))
        ),

        "ecg_pnn50": float(
            np.mean(np.abs(diff_rr) > 50.0) * 100.0
        ),
    }

    return features
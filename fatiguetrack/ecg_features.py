from __future__ import annotations

from typing import Dict, List

import numpy as np
import neurokit2 as nk


EPS = 1e-8


def extract_ecg_features(
    signal: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    """
    Extract ECG / HRV features from a single ECG window.

    Features:
        mean HR
        min HR
        max HR
        mean RR
        SDNN
        RMSSD
        pNN50

    Notes:
        - RR/NN intervals are expressed in milliseconds.
        - LF/HF features are intentionally excluded from short
          windows and can be added later using longer segments.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).reshape(-1)

    # Remove non-finite values
    signal = signal[np.isfinite(signal)]

    if signal.size < int(fs * 2):
        return _empty_features()

    try:
        # --------------------------------------------------
        # 1. Clean raw ECG
        # --------------------------------------------------
        cleaned = nk.ecg_clean(
            signal,
            sampling_rate=int(fs),
            method="neurokit",
        )

        # --------------------------------------------------
        # 2. Detect R peaks
        # --------------------------------------------------
        _, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=int(fs),
            method="neurokit",
            correct_artifacts=True,
        )

        peaks = np.asarray(
            info.get("ECG_R_Peaks", []),
            dtype=np.int64,
        )

        if peaks.size < 3:
            return _empty_features()

        # --------------------------------------------------
        # 3. RR intervals
        # --------------------------------------------------
        rr_seconds = np.diff(peaks) / float(fs)
        rr_ms = rr_seconds * 1000.0

        # Keep physiologically plausible RR intervals.
        # 300 ms  ~ 200 bpm
        # 2000 ms ~ 30 bpm
        valid = (
            np.isfinite(rr_ms)
            & (rr_ms >= 300.0)
            & (rr_ms <= 2000.0)
        )

        rr_ms = rr_ms[valid]

        if rr_ms.size < 2:
            return _empty_features()

        # --------------------------------------------------
        # 4. Heart rate
        # --------------------------------------------------
        hr = 60000.0 / rr_ms

        # --------------------------------------------------
        # 5. HRV time-domain features
        # --------------------------------------------------
        rr_diff = np.diff(rr_ms)

        mean_hr = float(np.mean(hr))
        min_hr = float(np.min(hr))
        max_hr = float(np.max(hr))

        mean_rr = float(np.mean(rr_ms))

        sdnn = (
            float(np.std(rr_ms, ddof=1))
            if rr_ms.size > 1
            else float("nan")
        )

        rmssd = (
            float(
                np.sqrt(
                    np.mean(rr_diff ** 2)
                )
            )
            if rr_diff.size > 0
            else float("nan")
        )

        pnn50 = (
            float(
                np.mean(
                    np.abs(rr_diff) > 50.0
                ) * 100.0
            )
            if rr_diff.size > 0
            else float("nan")
        )

        return {
            "ecg_hr_mean": mean_hr,
            "ecg_hr_min": min_hr,
            "ecg_hr_max": max_hr,
            "ecg_rr_mean": mean_rr,
            "ecg_sdnn": sdnn,
            "ecg_rmssd": rmssd,
            "ecg_pnn50": pnn50,
        }

    except Exception:
        # During preprocessing we prefer NaNs for a bad window
        # instead of crashing the entire subject/session.
        return _empty_features()


def _empty_features() -> Dict[str, float]:
    """
    Return the expected ECG feature structure filled with NaN.
    """

    return {
        "ecg_hr_mean": float("nan"),
        "ecg_hr_min": float("nan"),
        "ecg_hr_max": float("nan"),
        "ecg_rr_mean": float("nan"),
        "ecg_sdnn": float("nan"),
        "ecg_rmssd": float("nan"),
        "ecg_pnn50": float("nan"),
    }


def ecg_feature_names() -> List[str]:
    """
    Fixed ordering used by the multimodal model.
    """

    return [
        "ecg_hr_mean",
        "ecg_hr_min",
        "ecg_hr_max",
        "ecg_rr_mean",
        "ecg_sdnn",
        "ecg_rmssd",
        "ecg_pnn50",
    ]


def ecg_feature_vector(
    features: Dict[str, float],
) -> np.ndarray:
    """
    Convert ECG feature dictionary into fixed-order vector.

    Output:
        shape = (7,)
    """

    return np.asarray(
        [
            features.get(name, np.nan)
            for name in ecg_feature_names()
        ],
        dtype=np.float32,
    )
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------
# MediaPipe FaceMesh landmark indices
# ---------------------------------------------------------

# Left eye
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Right eye
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ---------------------------------------------------------
# Fatigue thresholds
# ---------------------------------------------------------

# These are initial operational thresholds.
# We can later calibrate them on DROZY if needed.
EAR_CLOSED_TH = 0.20
EAR_OPEN_TH = 0.23

MAR_YAWN_TH = 0.60


def _pt(
    landmarks,
    idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert normalized MediaPipe landmark coordinates
    into image pixel coordinates.
    """

    lm = landmarks[idx]

    return np.asarray(
        [
            float(lm.x * width),
            float(lm.y * height),
        ],
        dtype=np.float64,
    )


def _distance(
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Euclidean distance between two 2D points.
    """

    return float(
        np.linalg.norm(p1 - p2)
    )


def eye_aspect_ratio(
    landmarks,
    eye_indices: Sequence[int],
    width: int,
    height: int,
) -> float:
    """
    Compute Eye Aspect Ratio (EAR).

    Expected order of six landmarks:

        p1, p2, p3, p4, p5, p6

    EAR =
        (||p2-p6|| + ||p3-p5||)
        --------------------------------
               2 * ||p1-p4||
    """

    if len(eye_indices) != 6:
        raise ValueError(
            "EAR requires exactly 6 eye landmarks."
        )

    p1 = _pt(
        landmarks,
        eye_indices[0],
        width,
        height,
    )

    p2 = _pt(
        landmarks,
        eye_indices[1],
        width,
        height,
    )

    p3 = _pt(
        landmarks,
        eye_indices[2],
        width,
        height,
    )

    p4 = _pt(
        landmarks,
        eye_indices[3],
        width,
        height,
    )

    p5 = _pt(
        landmarks,
        eye_indices[4],
        width,
        height,
    )

    p6 = _pt(
        landmarks,
        eye_indices[5],
        width,
        height,
    )

    vertical_1 = _distance(
        p2,
        p6,
    )

    vertical_2 = _distance(
        p3,
        p5,
    )

    horizontal = _distance(
        p1,
        p4,
    )

    if horizontal <= 1e-8:
        return float("nan")

    return float(
        (
            vertical_1
            + vertical_2
        )
        /
        (
            2.0
            * horizontal
        )
    )


def mouth_aspect_ratio(
    landmarks,
    width: int,
    height: int,
) -> float:
    """
    Compute Mouth Aspect Ratio (MAR).

    Uses MediaPipe mouth landmarks:

        horizontal:
            61 ---- 291

        vertical pairs:
            13 ---- 14
            82 ---- 87

    MAR =
        (vertical_1 + vertical_2)
        -------------------------
             2 * horizontal
    """

    left = _pt(
        landmarks,
        61,
        width,
        height,
    )

    right = _pt(
        landmarks,
        291,
        width,
        height,
    )

    upper_1 = _pt(
        landmarks,
        13,
        width,
        height,
    )

    lower_1 = _pt(
        landmarks,
        14,
        width,
        height,
    )

    upper_2 = _pt(
        landmarks,
        82,
        width,
        height,
    )

    lower_2 = _pt(
        landmarks,
        87,
        width,
        height,
    )

    vertical_1 = _distance(
        upper_1,
        lower_1,
    )

    vertical_2 = _distance(
        upper_2,
        lower_2,
    )

    horizontal = _distance(
        left,
        right,
    )

    if horizontal <= 1e-8:
        return float("nan")

    return float(
        (
            vertical_1
            + vertical_2
        )
        /
        (
            2.0
            * horizontal
        )
    )


def classify_mar(
    mar: float,
) -> int:
    """
    Binary yawning indicator based on MAR.
    """

    if not np.isfinite(mar):
        return 0

    return int(
        mar >= MAR_YAWN_TH
    )


def classify_ear(
    ear: float,
) -> int:
    """
    Simple fatigue-oriented eye state:

        0 = closed
        1 = partially open / transitional
        2 = open
    """

    if not np.isfinite(ear):
        return -1

    if ear < EAR_CLOSED_TH:
        return 0

    if ear < EAR_OPEN_TH:
        return 1

    return 2
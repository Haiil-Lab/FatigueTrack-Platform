from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import rules


@dataclass
class BehavioralFrame:
    """
    Behavioral fatigue indicators extracted from one video frame.
    """
    ear: float
    mar: float
    blink_event: float
    eye_closed: float
    yawning: float
    face_ok: float


class FatigueBehavioralExtractor:
    """
    Extracts fatigue-related behavioral indicators from a video frame.

    Used features:
        - Eye Aspect Ratio (EAR)
        - Mouth Aspect Ratio (MAR)
        - Blink event
        - Eye closure
        - Yawning

    Attention-specific features such as gaze, head pose,
    body motion and attention rules are intentionally excluded.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Used for blink hysteresis.
        self._ear_below = False

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()

    def extract(self, frame_bgr: np.ndarray) -> BehavioralFrame:
        """
        Extract behavioral fatigue indicators from one BGR video frame.
        """

        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        result = self.face_mesh.process(rgb)

        # Defaults when face is unavailable.
        ear = np.nan
        mar = np.nan
        blink_event = 0.0
        eye_closed = 0.0
        yawning = 0.0
        face_ok = 0.0

        if result.multi_face_landmarks:

            face_ok = 1.0

            lm = (
                result
                .multi_face_landmarks[0]
                .landmark
            )

            # ---------------------------------------
            # EAR
            # ---------------------------------------

            ear_left = rules.eye_aspect_ratio(
                lm,
                rules.LEFT_EYE_IDX,
                w,
                h,
            )

            ear_right = rules.eye_aspect_ratio(
                lm,
                rules.RIGHT_EYE_IDX,
                w,
                h,
            )

            ear = float(
                (ear_left + ear_right) / 2.0
            )

            # Eye-closure indicator.
            eye_closed = float(
                ear < rules.EAR_CLOSED_TH
            )

            # Blink event using hysteresis.
            blink_event = float(
                self._update_blink(ear)
            )

            # ---------------------------------------
            # MAR
            # ---------------------------------------

            mar = float(
                rules.mouth_aspect_ratio(
                    lm,
                    w,
                    h,
                )
            )

            yawning = float(
                rules.classify_mar(mar)
            )

        return BehavioralFrame(
            ear=ear,
            mar=mar,
            blink_event=blink_event,
            eye_closed=eye_closed,
            yawning=yawning,
            face_ok=face_ok,
        )

    def _update_blink(
        self,
        ear: float,
    ) -> int:
        """
        Detect a completed blink using EAR hysteresis.

        Blink begins when EAR falls below EAR_CLOSED_TH
        and is completed when EAR rises above EAR_OPEN_TH.
        """

        if (
            ear < rules.EAR_CLOSED_TH
            and not self._ear_below
        ):
            self._ear_below = True

        if (
            ear >= rules.EAR_OPEN_TH
            and self._ear_below
        ):
            self._ear_below = False
            return 1

        return 0


class BehavioralWindowAggregator:
    """
    Aggregates frame-level behavioral indicators over a
    temporal window, e.g. 10 seconds.

    Output features:
        1. EAR mean
        2. EAR minimum
        3. Blink rate (blinks/minute)
        4. Eye-closed ratio
        5. MAR mean
        6. MAR maximum
        7. Yawn ratio
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
    ):
        self.window_seconds = float(
            window_seconds
        )

        self.reset()

    def reset(self):
        self.ear_values: List[float] = []
        self.mar_values: List[float] = []

        self.blink_events: List[float] = []
        self.eye_closed_values: List[float] = []
        self.yawning_values: List[float] = []

        self.valid_frames = 0
        self.total_frames = 0

    def add(
        self,
        frame_features: BehavioralFrame,
    ):
        """
        Add one frame to the current temporal window.
        """

        self.total_frames += 1

        if frame_features.face_ok <= 0:
            return

        self.valid_frames += 1

        if np.isfinite(frame_features.ear):
            self.ear_values.append(
                frame_features.ear
            )

        if np.isfinite(frame_features.mar):
            self.mar_values.append(
                frame_features.mar
            )

        self.blink_events.append(
            frame_features.blink_event
        )

        self.eye_closed_values.append(
            frame_features.eye_closed
        )

        self.yawning_values.append(
            frame_features.yawning
        )

    def compute(
        self,
    ) -> Dict[str, float]:
        """
        Compute behavioral features for the current window.
        """

        if self.valid_frames == 0:
            return _empty_behavioral_features()

        # ---------------------------------------
        # EAR
        # ---------------------------------------

        ear_mean = (
            float(np.mean(self.ear_values))
            if self.ear_values
            else float("nan")
        )

        ear_min = (
            float(np.min(self.ear_values))
            if self.ear_values
            else float("nan")
        )

        # ---------------------------------------
        # Blink dynamics
        # ---------------------------------------

        blink_count = float(
            np.sum(self.blink_events)
        )

        # Convert to blinks per minute.
        blink_rate = (
            blink_count
            / self.window_seconds
            * 60.0
        )

        # ---------------------------------------
        # Eye closure
        # ---------------------------------------

        eye_closed_ratio = (
            float(
                np.mean(
                    self.eye_closed_values
                )
            )
            if self.eye_closed_values
            else float("nan")
        )

        # ---------------------------------------
        # MAR / yawning
        # ---------------------------------------

        mar_mean = (
            float(np.mean(self.mar_values))
            if self.mar_values
            else float("nan")
        )

        mar_max = (
            float(np.max(self.mar_values))
            if self.mar_values
            else float("nan")
        )

        yawn_ratio = (
            float(
                np.mean(
                    self.yawning_values
                )
            )
            if self.yawning_values
            else float("nan")
        )

        return {
            "ear_mean": ear_mean,
            "ear_min": ear_min,
            "blink_rate": blink_rate,
            "eye_closed_ratio": eye_closed_ratio,
            "mar_mean": mar_mean,
            "mar_max": mar_max,
            "yawn_ratio": yawn_ratio,
        }


def _empty_behavioral_features() -> Dict[str, float]:
    """
    Expected behavioral feature structure when
    no valid face is detected.
    """

    return {
        "ear_mean": float("nan"),
        "ear_min": float("nan"),
        "blink_rate": float("nan"),
        "eye_closed_ratio": float("nan"),
        "mar_mean": float("nan"),
        "mar_max": float("nan"),
        "yawn_ratio": float("nan"),
    }


def behavioral_feature_names() -> List[str]:
    """
    Fixed feature order used by the multimodal model.
    """

    return [
        "ear_mean",
        "ear_min",
        "blink_rate",
        "eye_closed_ratio",
        "mar_mean",
        "mar_max",
        "yawn_ratio",
    ]


def behavioral_feature_vector(
    features: Dict[str, float],
) -> np.ndarray:
    """
    Convert behavioral feature dictionary to a fixed-order vector.

    Output:
        shape = (7,)
    """

    return np.asarray(
        [
            features.get(name, np.nan)
            for name
            in behavioral_feature_names()
        ],
        dtype=np.float32,
    )
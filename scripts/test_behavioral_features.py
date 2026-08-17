from pathlib import Path

import cv2

from fatiguetrack.behavioral_features import (
    FatigueBehavioralExtractor,
    BehavioralWindowAggregator,
    behavioral_feature_names,
    behavioral_feature_vector,
)


ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    ROOT
    / "data"
    / "raw"
    / "drozy"
    / "videos_i8"
    / "1-1.mp4"
)


def main():

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    print("Video:", VIDEO_PATH)
    print("FPS:", fps)

    extractor = (
        FatigueBehavioralExtractor()
    )

    aggregator = (
        BehavioralWindowAggregator(
            window_seconds=10.0
        )
    )

    # Analyze only first 10 seconds.
    max_frames = int(
        fps * 10
    )

    frame_count = 0

    while (
        frame_count < max_frames
    ):

        ok, frame = cap.read()

        if not ok:
            break

        features = extractor.extract(
            frame
        )

        aggregator.add(
            features
        )

        frame_count += 1

    result = aggregator.compute()

    vector = behavioral_feature_vector(
        result
    )

    print(
        "\n=== BEHAVIORAL FEATURES ==="
    )

    for name, value in zip(
        behavioral_feature_names(),
        vector,
    ):
        print(
            f"{name:25s}: "
            f"{value:.6f}"
        )

    print(
        "\nVector shape:",
        vector.shape,
    )

    print(
        "Frames processed:",
        frame_count,
    )

    print(
        "Valid face frames:",
        aggregator.valid_frames,
    )

    extractor.close()
    cap.release()


if __name__ == "__main__":
    main()
from pathlib import Path
from typing import Dict

import numpy as np


def kss_to_class(kss: int) -> int:
    """
    Fatigue classes:
        0 = Low
        1 = Moderate
        2 = High
    """
    if kss <= 3:
        return 0
    elif kss <= 6:
        return 1
    else:
        return 2


CLASS_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "High",
}


def load_kss_labels(kss_path: str | Path) -> Dict[str, Dict[str, int | str]]:
    """
    KSS.txt:
        rows    -> subject IDs 1..14
        columns -> test IDs 1..3
    """

    values = np.loadtxt(kss_path, dtype=int)

    labels = {}

    for subject_idx in range(values.shape[0]):
        subject_id = subject_idx + 1

        for test_idx in range(values.shape[1]):
            test_id = test_idx + 1

            kss = int(values[subject_idx, test_idx])
            class_id = kss_to_class(kss)

            session_id = f"{subject_id}-{test_id}"

            labels[session_id] = {
                "kss": kss,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            }

    return labels
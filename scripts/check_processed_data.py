from pathlib import Path
from collections import Counter

import numpy as np


ROOT = Path("data/processed/drozy")


for split in ["train", "val", "test"]:

    paths = list(
        (ROOT / split).glob("*.npz")
    )

    total_values = 0
    nan_values = 0
    all_nan_features = Counter()

    print(f"\n===== {split.upper()} =====")

    for path in paths:

        d = np.load(
            path,
            allow_pickle=True,
        )

        X = d["X"]

        total_values += X.size
        nan_values += np.isnan(X).sum()

        names = d["feature_names"]

        for j, name in enumerate(names):

            if np.isnan(X[:, j]).all():

                all_nan_features[
                    str(name)
                ] += 1

    print(
        "Files:",
        len(paths),
    )

    print(
        "NaN values:",
        nan_values,
        "/",
        total_values,
    )

    print(
        "NaN percentage:",
        100 * nan_values / max(total_values, 1),
    )

    print(
        "Fully missing features:"
    )

    for name, count in (
        all_nan_features
        .most_common()
    ):
        print(
            f"  {name}: {count}"
        )
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afib.data_preparation import load_prepared_split
from afib.evaluation import binary_metrics
from afib.preprocessing import SCENARIOS
from models.histogram_gradient_boosting.model import create_model


GRID = [7, 8, 9]
REPEATS = 3
SCENARIO_NAMES = list(SCENARIOS)
PROCESSED = ROOT / "processed_data"

DEFAULT_OUTPUT = (
    ROOT / "results"
    / "histogram_gradient_boosting"
    / "retuned_7_8_9_runs"
)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def inner_cv(arm, outer):
    """Select depth using inner folds only."""

    rows = []
    splits = {}

    for inner in range(3):
        splits[inner] = load_prepared_split(
            PROCESSED,
            outer_fold=outer,
            inner_fold=inner,
            arm=arm,
            scenario="observed"
        )

    for depth in GRID:
        for inner in range(3):

            split = splits[inner]

            model = create_model(max_depth=depth)

            model.fit(
                split["X_train"],
                split["y_train"],
                sample_weight=split["sample_weight"]
            )

            proba = model.predict_proba(
                split["X_valid"]
            )[:, 1]

            metrics = binary_metrics(
                split["y_valid"],
                proba
            )

            rows.append({
                "arm": arm,
                "outer_fold": outer,
                "max_depth": depth,
                "inner_fold": inner,
                "macro_f1": metrics["macro_f1"]
            })

            print(
                f"  inner depth={depth}, fold={inner}, "
                f"macro-F1={metrics['macro_f1']:.4f}",
                flush=True
            )

            del model
            gc.collect()

    tuning = pd.DataFrame(rows)

    assert len(tuning) == 9
    assert np.isfinite(tuning["macro_f1"]).all()

    means = (
        tuning.groupby("max_depth")["macro_f1"]
        .mean()
    )

    # Higher mean F1 wins; ties prefer smaller depth.
    winner = min(
        GRID,
        key=lambda depth: (-float(means[depth]), depth)
    )

    del splits
    gc.collect()

    return winner, tuning


def run_one(arm, outer, output):

    folder = output / f"{arm}_fold{outer}"
    folder.mkdir(parents=True, exist_ok=True)

    marker = folder / "complete.json"
    metrics_file = folder / "metrics.csv"
    predictions_file = folder / "predictions.csv"
    tuning_file = folder / "inner_tuning.csv"

    expected_rows = 31

    # Completed-fold checkpoint
    if marker.exists():

        info = json.loads(
            marker.read_text(encoding="utf-8")
        )

        valid = (
            info.get("arm") == arm
            and info.get("outer_fold") == outer
            and info.get("grid") == GRID
            and info.get("repeats") == REPEATS
            and info.get("scenario_names") == SCENARIO_NAMES
            and metrics_file.is_file()
            and predictions_file.is_file()
            and tuning_file.is_file()
        )

        if valid:
            m = pd.read_csv(metrics_file)
            p = pd.read_csv(predictions_file)
            t = pd.read_csv(tuning_file)

            valid = (
                len(m) == expected_rows
                and len(p) > 0
                and len(t) == 9
                and (m["arm"] == arm).all()
                and (m["outer_fold"] == outer).all()
                and (m["max_depth"] == info["best_depth"]).all()
            )

        if valid:
            print(
                f"SKIP: {arm} outer={outer} complete",
                flush=True
            )
            return

        raise RuntimeError(
            f"Inconsistent saved output: {folder}"
        )

    print(
        f"\n=== ARM {arm}, OUTER FOLD {outer} ===",
        flush=True
    )

    # Step 1: tune only on inner folds
    winner, tuning = inner_cv(arm, outer)

    print(
        f"Selected max_depth={winner}",
        flush=True
    )

    # Step 2: prepare outer train/test data
    base = load_prepared_split(
        PROCESSED,
        outer_fold=outer,
        inner_fold=None,
        arm=arm,
        scenario="observed"
    )

    # Step 3: fit selected model
    model = create_model(max_depth=winner)

    model.fit(
        base["X_train"],
        base["y_train"],
        sample_weight=base["sample_weight"]
    )

    y = np.asarray(base["y_valid"], dtype=int)
    ids = base["y_valid"].index.to_numpy()

    metrics_rows = []
    prediction_rows = []

    # Step 4: evaluate observed + corruption scenarios
    for scenario in SCENARIO_NAMES:

        repeats = (
            [0] if scenario == "observed"
            else range(REPEATS)
        )

        for repeat in repeats:

            if scenario == "observed":
                split = base
            else:
                split = load_prepared_split(
                    PROCESSED,
                    outer_fold=outer,
                    inner_fold=None,
                    arm=arm,
                    scenario=scenario,
                    repeat=repeat
                )

            assert np.array_equal(
                split["y_valid"].index.to_numpy(),
                ids
            ), "Held-out ECG order mismatch"

            proba = model.predict_proba(
                split["X_valid"]
            )[:, 1]

            metrics = binary_metrics(y, proba)

            metrics_rows.append({
                "arm": arm,
                "outer_fold": outer,
                "scenario": scenario,
                "repeat": repeat,
                "max_depth": winner,
                **metrics
            })

            prediction_rows.extend(
                {
                    "arm": arm,
                    "outer_fold": outer,
                    "scenario": scenario,
                    "repeat": repeat,
                    "ecg_id": int(ids[i]),
                    "y_true": int(y[i]),
                    "prob_afib": float(proba[i])
                }
                for i in range(len(ids))
            )

            print(
                f"  {scenario} repeat={repeat}: "
                f"macro-F1={metrics['macro_f1']:.4f}",
                flush=True
            )

            if scenario != "observed":
                del split

    # Step 5: save results
    mdf = pd.DataFrame(metrics_rows)
    pdf = pd.DataFrame(prediction_rows)

    assert len(mdf) == expected_rows
    assert len(pdf) == expected_rows * len(ids)

    save_csv(mdf, metrics_file)
    save_csv(pdf, predictions_file)
    save_csv(tuning, tuning_file)

    metadata = {
        "arm": arm,
        "outer_fold": outer,
        "best_depth": winner,
        "grid": GRID,
        "repeats": REPEATS,
        "scenario_names": SCENARIO_NAMES,
        "selection_metric": "macro_f1"
    }

    temp = folder / "complete.tmp"

    temp.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    os.replace(temp, marker)

    print(
        f"COMPLETED: {arm} outer={outer}",
        flush=True
    )

    del model, base, tuning, mdf, pdf
    del metrics_rows, prediction_rows
    gc.collect()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list("ABCD"),
        default=["A"]
    )

    parser.add_argument(
        "--outer-folds",
        nargs="+",
        type=int,
        default=[0]
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    assert all(
        fold in range(10)
        for fold in args.outer_folds
    ), "Outer folds must be 0..9"

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    assert PROCESSED.is_dir(), (
        f"Missing processed_data: {PROCESSED}"
    )

    print("=== HGB RETUNED EXPERIMENT ===")
    print("Depth grid:", GRID)
    print("Arms:", args.arms)
    print("Outer folds:", args.outer_folds)
    print("Output:", args.output)

    for arm in args.arms:
        for outer in args.outer_folds:
            run_one(arm, outer, args.output)

    print("\nREQUESTED FOLDS FINISHED")
    print("Results saved to:", args.output)


if __name__ == "__main__":
    main()

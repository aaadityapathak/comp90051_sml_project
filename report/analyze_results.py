"""Verify the saved runs and rebuild the report's comparison tables and figure.

Run from any directory with ``python report/analyze_results.py``. This uses
saved model outputs only: it neither downloads data nor fits a classifier.
"""

from itertools import product
from pathlib import Path
import json
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(REPOSITORY / "src"))
from afib.evaluation import binary_metrics

ARMS = tuple("ABCD")
SCENARIOS = (
    "observed", "missing_1", "missing_3", "missing_6", "noise_0.05",
    "noise_0.10", "noise_0.20", "gain_0.05", "gain_0.10", "gain_0.20",
    "combined_3lead_noise0.10",
)
METRICS = (
    "macro_f1", "sensitivity", "specificity", "precision", "npv", "accuracy",
    "balanced_accuracy", "auroc", "average_precision", "brier",
)
COUNTS = ("n", "n_positive", "n_negative", "tp", "tn", "fp", "fn")
CONDITION_KEYS = ["variant", "outer_fold", "scenario", "repeat"]
FOLD_KEYS = ["model", "variant", "outer_fold", "scenario"]
RETUNED = REPOSITORY / "results/histogram_gradient_boosting/retuned7_8_9"


def check_close(actual, expected):
    """CSV round trips may alter the final floating-point digit."""
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                               equal_nan=True)


def metric_frame(model):
    """Keep the fixed 0.5 operating point separate from threshold tuning."""
    frame = pd.read_csv(REPOSITORY / f"results/{model}/output_data/metrics.csv")
    frame = frame.loc[frame.threshold_policy.eq("fixed_0.50")].copy()
    frame["model"] = model
    return frame


def verify_predictions(predictions, saved, expected_records):
    """Check identity, labels, condition coverage, and every reported metric."""
    recalculated = []
    for key, condition in predictions.groupby(CONDITION_KEYS, sort=True):
        variant, outer_fold, scenario, repeat = key
        records = expected_records.loc[expected_records.outer_fold.eq(outer_fold)]
        if condition.ecg_id.duplicated().any():
            raise ValueError(f"Duplicated ECG in {key}")
        if set(condition.ecg_id) != set(records.ecg_id):
            raise ValueError(f"Incomplete held-out cohort in {key}")
        truth = records.set_index("ecg_id").loc[condition.ecg_id, "y_true"]
        check_close(condition.y_true.to_numpy(), truth.to_numpy())
        recalculated.append(dict(zip(CONDITION_KEYS, key),
                                 **binary_metrics(condition.y_true, condition.score)))
    actual = pd.DataFrame(recalculated)
    paired = actual.merge(saved, on=CONDITION_KEYS, validate="one_to_one",
                          suffixes=("_computed", "_saved"))
    if len(actual) != len(saved) or len(paired) != len(saved):
        raise ValueError("Saved metric conditions differ from prediction conditions")
    max_error = 0.0
    for metric in (*METRICS, *COUNTS):
        check_close(paired[f"{metric}_computed"], paired[f"{metric}_saved"])
        max_error = max(max_error, float(np.nanmax(np.abs(
            paired[f"{metric}_computed"] - paired[f"{metric}_saved"]))))
    return {"prediction_rows": len(predictions), "conditions": len(actual),
            "maximum_absolute_metric_difference": max_error}


def main():
    OUTPUT.mkdir(exist_ok=True)
    frames = [metric_frame(model) for model in ("logistic_regression", "tabm")]
    # The held-out prediction identifiers provide a data-free comparison map.
    # This checks agreement across result bundles, not source-data authenticity.
    expected = []
    for outer_fold in range(10):
        path = REPOSITORY / ("results/logistic_regression/output_data/folds/"
                             f"{outer_fold:02d}_A/outer_predictions.csv.gz")
        frame = pd.read_csv(path)
        expected.append(frame.loc[frame.scenario.eq("observed"),
                                   ["ecg_id", "outer_fold", "y_true"]])
    expected = pd.concat(expected, ignore_index=True)
    if len(expected) != 21799 or expected.ecg_id.duplicated().any():
        raise ValueError("Expected one observed prediction for each of 21,799 ECGs")

    validation = {"training_performed": False, "raw_data_required": False,
                  "aggregation": "Average repeats within fold, then ten-fold mean and sample SD.",
                  "prediction_checks": {}}
    # Verify all LR/TabM variants, including the controls used in the discussion.
    for model, saved in zip(("logistic_regression", "tabm"), frames):
        checks = []
        for (variant, outer_fold), conditions in saved.groupby(["variant", "outer_fold"]):
            path = REPOSITORY / (f"results/{model}/output_data/folds/"
                                 f"{outer_fold:02d}_{variant}/outer_predictions.csv.gz")
            predictions = pd.read_csv(path)
            checks.append(verify_predictions(predictions, conditions, expected))
        validation["prediction_checks"][model] = {
            "prediction_rows": sum(row["prediction_rows"] for row in checks),
            "conditions": sum(row["conditions"] for row in checks),
            "maximum_absolute_metric_difference": max(
                row["maximum_absolute_metric_difference"] for row in checks),
        }
        print(f"Verified {model} predictions", flush=True)

    hgb = pd.read_csv(RETUNED / "HGB_ABCD_all_metrics.csv").rename(columns={"arm": "variant"})
    expected_conditions = {(arm, fold, scenario, repeat)
        for arm, fold, scenario in product(ARMS, range(10), SCENARIOS)
        for repeat in ([0] if scenario == "observed" else range(3))}
    if set(hgb[CONDITION_KEYS].itertuples(index=False, name=None)) != expected_conditions:
        raise ValueError("Retuned HGB condition coverage is incomplete")
    hgb_checks = []
    for arm in ARMS:
        with zipfile.ZipFile(RETUNED / f"HGB_retuned_{arm}_predictions.zip") as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError("Expected one prediction CSV per arm archive")
            with archive.open(members[0]) as stream:
                predictions = pd.read_csv(stream).rename(columns={"arm": "variant", "prob_afib": "score"})
        hgb_checks.append(verify_predictions(predictions, hgb.loc[hgb.variant.eq(arm)], expected))
    validation["prediction_checks"]["histogram_gradient_boosting_retuned"] = {
        "prediction_rows": sum(row["prediction_rows"] for row in hgb_checks),
        "conditions": sum(row["conditions"] for row in hgb_checks),
        "maximum_absolute_metric_difference": max(row["maximum_absolute_metric_difference"] for row in hgb_checks),
    }
    hgb["model"] = "histogram_gradient_boosting_retuned"
    frames.append(hgb)

    tuning = pd.read_csv(RETUNED / "HGB_ABCD_all_inner_tuning.csv")
    keys = ["arm", "outer_fold", "max_depth", "inner_fold"]
    if tuning.duplicated(keys).any() or set(tuning[keys].itertuples(index=False, name=None)) != set(product(ARMS, range(10), (7, 8, 9), range(3))):
        raise ValueError("Expected exactly 360 unique HGB inner tuning scores")
    candidate_means = tuning.groupby(["arm", "outer_fold", "max_depth"]).macro_f1.mean().reset_index()
    winners = candidate_means.sort_values(["macro_f1", "max_depth"], ascending=[False, True]).drop_duplicates(["arm", "outer_fold"])
    selections = pd.concat([pd.read_csv(RETUNED / f"{arm}_selected_depths.csv") for arm in ARMS])
    joined = winners.merge(selections, on=["arm", "outer_fold"], suffixes=("_computed", "_saved"), validate="one_to_one")
    check_close(joined.max_depth_computed, joined.max_depth_saved)
    check_close(joined.macro_f1_computed, joined.macro_f1_saved)
    counts = winners.max_depth.value_counts().sort_index()
    validation["hgb_tuning"] = {"inner_scores": len(tuning), "verified_selections": len(joined),
                                 "selected_depth_counts": {str(key): int(value) for key, value in counts.items()}}
    validation["hgb_tuning"]["mean_inner_f1_by_arm_depth"] = candidate_means.groupby(["arm", "max_depth"]).macro_f1.mean().reset_index().to_dict("records")

    historical = pd.read_csv(REPOSITORY / "results/histogram_gradient_boosting/final_all_metrics.csv").rename(columns={"arm": "variant"})
    historical["model"] = "histogram_gradient_boosting_earlier"
    frames.append(historical)
    all_metrics = pd.concat(frames, ignore_index=True)
    fold_means = all_metrics.groupby(FOLD_KEYS)[list(METRICS)].mean().reset_index()
    summary = fold_means.melt(id_vars=FOLD_KEYS, value_vars=METRICS, var_name="metric", value_name="value").groupby(["model", "variant", "scenario", "metric"]).value.agg(mean="mean", sd="std", n_folds="count").reset_index()
    summary.to_csv(OUTPUT / "model_summary.csv", index=False)

    paired_rows = []
    for model, model_folds in fold_means.groupby("model"):
        for scenario, scenario_folds in model_folds.groupby("scenario"):
            matrix = scenario_folds.pivot(index="outer_fold", columns="variant", values="macro_f1")
            for upper, lower in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C")):
                difference = matrix[upper] - matrix[lower]
                paired_rows.append({"model": model, "scenario": scenario, "contrast": f"{upper}-{lower}", "mean": difference.mean(), "sd": difference.std(ddof=1)})
    differences = pd.DataFrame(paired_rows)
    differences.to_csv(OUTPUT / "paired_differences.csv", index=False)
    side_by_side = summary.loc[summary.variant.isin(ARMS) & summary.metric.eq("macro_f1")].pivot(index=["scenario", "variant"], columns="model", values=["mean", "sd"])
    side_by_side.to_csv(OUTPUT / "side_by_side_macro_f1.csv")

    # Chart values come from paired folds, so their SD is not the subtraction
    # of two separate SDs and does not treat three stress repeats as new folds.
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(6.85, 1.45))
    model_labels = [("logistic_regression", "LR"), ("histogram_gradient_boosting_retuned", "HGB"), ("tabm", "TabM")]
    for axis, scenario, title in zip(axes, ("observed", "missing_6"), ("Observed", "Six hidden leads")):
        for position, (model, label) in enumerate(model_labels):
            result = differences.loc[differences.model.eq(model) & differences.scenario.eq(scenario) & differences.contrast.eq("D-B")].iloc[0]
            axis.errorbar(result["mean"], 2-position, xerr=result["sd"], fmt="o", color="#14516d", capsize=3, markersize=4)
        axis.axvline(0, color="#888888", linewidth=.7)
        axis.set_yticks([2, 1, 0], [label for _, label in model_labels])
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Macro-F1 change (D - B)", fontsize=10)
        axis.set_ylim(-.5, 2.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=.3, w_pad=2)
    figure.savefig(OUTPUT / "augmentation_effect.png", dpi=300)
    plt.close(figure)
    validation["limitations"] = [
        "Prediction labels and folds are checked against the LR result bundle, not raw source data.",
        "Historical HGB metrics are retained as published; this script does not recover its 7z predictions.",
        "Matching ECG/fold keys do not prove identical HGB stress realizations; historical perturbation manifests are absent.",
        "Scores verify the saved outputs, not an independent retraining of the published experiments.",
    ]
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation["prediction_checks"], indent=2))
    print("HGB depth selections:", validation["hgb_tuning"]["selected_depth_counts"])


if __name__ == "__main__":
    main()

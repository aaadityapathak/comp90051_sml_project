"""Manual evaluation for the AFIB robustness study.

No model fitting, cross-validation helpers or library metrics are used here.
NaN means a metric is undefined, not zero performance. The sensitivity target
used for threshold selection is an exploratory study choice, not a clinical
acceptance criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_NAMES = ("logistic_regression", "hist_gradient_boosting", "tabm")
FEATURE_SETS = ("original", "enhanced")
TRAINING_ARMS = ("observed", "augmented")


def _binary_arrays(y_true, scores):
    y = np.asarray(y_true)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not len(y):
        raise ValueError("Labels and scores must be non-empty aligned 1D arrays.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Scores must be finite AFIB probabilities between 0 and 1.")
    return y.astype(np.int64), score


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def auroc(y_true, scores):
    """Probability a positive ranks above a negative; ties receive half credit."""
    y, score = _binary_arrays(y_true, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ordered_y, ordered_score = y[order], score[order]
    negative_below = 0
    pair_credit = 0.0
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and ordered_score[stop] == ordered_score[start]:
            stop += 1
        group_positive = int(ordered_y[start:stop].sum())
        group_negative = stop - start - group_positive
        pair_credit += group_positive * (negative_below + .5 * group_negative)
        negative_below += group_negative
        start = stop
    return float(pair_credit / (positives * negatives))


def average_precision(y_true, scores):
    """Non-interpolated PR summary, including tied scores as one threshold group.

    This is average precision (AP), not trapezoidal area under a PR curve.
    With no positive labels it is undefined. With only positives it is 1,
    although subgroup reporting flags one-class discrimination as unavailable.
    """
    y, score = _binary_arrays(y_true, scores)
    positives = int(y.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ordered_y, ordered_score = y[order], score[order]
    true_positive, result, start = 0, 0.0, 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and ordered_score[stop] == ordered_score[start]:
            stop += 1
        added_positive = int(ordered_y[start:stop].sum())
        true_positive += added_positive
        result += (added_positive / positives) * (true_positive / stop)
        start = stop
    return float(result)


def binary_metrics(y_true, scores, threshold=.5):
    """Compute confusion metrics, tied-score AUROC/AP, and the Brier score.

    Thresholds may be just above 1 to represent predicting no positive records.
    Macro-F1 requires both per-class F1 denominators to be defined; balanced
    accuracy requires both classes. Precision/NPV are NaN if never predicted.
    """
    y, score = _binary_arrays(y_true, scores)
    if not np.isfinite(threshold):
        raise ValueError("The decision threshold must be finite.")
    prediction = score >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    tn = int(np.sum(~prediction & (y == 0)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    sensitivity = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    positive_f1 = _divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _divide(2 * tn, 2 * tn + fp + fn)
    return {
        "n": len(y), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": _divide(tp, tp + fp), "npv": _divide(tn, tn + fn),
        "positive_f1": positive_f1, "negative_f1": negative_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "auroc": auroc(y, score), "average_precision": average_precision(y, score),
        "brier": float(np.mean((score - y) ** 2)),
        "threshold": float(threshold),
    }


def choose_sensitivity_threshold(inner_y, inner_oof_scores, target_sensitivity=.90):
    """Choose a threshold using inner out-of-fold predictions only.

    Maximise specificity subject to observed sensitivity >= the target; break
    ties by higher sensitivity and then higher threshold. Call separately for
    each outer-training partition and freeze the result for every test stress.
    The caller must establish provenance: outer test scores are forbidden here.
    """
    y, score = _binary_arrays(inner_y, inner_oof_scores)
    if not 0 < target_sensitivity <= 1:
        raise ValueError("Target sensitivity must be in (0, 1].")
    if len(np.unique(y)) != 2:
        return {"threshold": float("nan"), "status": "unavailable_one_class",
                "target_sensitivity": float(target_sensitivity)}
    # Sweep tied scores together in O(n log n); the all-negative rule cannot
    # meet a strictly positive sensitivity target and need not be evaluated.
    order = np.argsort(-score, kind="stable")
    ordered_score, ordered_y = score[order], y[order]
    positive_count, negative_count = int(y.sum()), int((y == 0).sum())
    true_positive, false_positive, start, best = 0, 0, 0, None
    while start < len(y):
        stop = start + 1
        while stop < len(y) and ordered_score[stop] == ordered_score[start]:
            stop += 1
        added_positive = int(ordered_y[start:stop].sum())
        true_positive += added_positive
        false_positive += stop - start - added_positive
        threshold = ordered_score[start]
        sensitivity = true_positive / positive_count
        specificity = 1 - false_positive / negative_count
        if sensitivity + 1e-12 >= target_sensitivity:
            key = (specificity, sensitivity, float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold))
        start = stop
    assert best is not None  # The all-positive rule reaches sensitivity 1.
    return {"threshold": best[1], "status": "selected_on_inner_oof",
            "target_sensitivity": float(target_sensitivity),
            "inner_sensitivity": best[0][1], "inner_specificity": best[0][0]}


def majority_baseline(y_train, y_test):
    """Choose the constant label and prevalence from training labels only."""
    training, _ = _binary_arrays(y_train, np.zeros(len(y_train)))
    prevalence = float(training.mean())
    majority_label = int(prevalence > .5)  # A tied training count chooses 0.
    probability = np.full(len(y_test), prevalence)
    threshold = .5 if majority_label else np.nextafter(.5, np.inf)
    result = binary_metrics(y_test, probability, threshold)
    result.update(training_prevalence=prevalence, majority_label=majority_label)
    return result


def assert_patient_disjoint(training_patients, evaluation_patients):
    """Check patient separation independently of score and metric computation."""
    train = np.asarray(training_patients)
    test = np.asarray(evaluation_patients)
    if train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test):
        raise ValueError("Patient IDs must be non-empty 1D arrays.")
    if pd.isna(train).any() or pd.isna(test).any():
        raise ValueError("Patient IDs must not be missing.")
    if set(train).intersection(test):
        raise ValueError("Training and evaluation records share patient IDs.")


def quality_subgroup_metrics(y_true, scores, subgroup, threshold=.5):
    """Report existing quality groups without changing the frozen threshold.

    Missing group labels form an explicit 'unknown' group. Empty categories
    are absent. For a one-class group, AUROC, AP, macro-F1 and balanced accuracy
    are NaN; confusion counts and other defined metrics remain available.
    These are descriptive record-level groups, not independent validation sets.
    """
    y, score = _binary_arrays(y_true, scores)
    groups = pd.Series(subgroup, dtype="object").fillna("unknown").astype(str).to_numpy()
    if len(groups) != len(y):
        raise ValueError("Subgroups must align with labels.")
    rows = []
    for group in sorted(set(groups)):
        selected = groups == group
        result = binary_metrics(y[selected], score[selected], threshold)
        both_classes = bool(result["n_positive"] and result["n_negative"])
        if not both_classes:
            for name in ("auroc", "average_precision", "macro_f1", "balanced_accuracy"):
                result[name] = float("nan")
        result.update(subgroup=group, both_classes_present=both_classes)
        rows.append(result)
    return pd.DataFrame(rows)


def source_quality_groups(metadata):
    """Build a fixed, label-independent group from PTB-XL quality annotations.

    An empty annotation means no recorded flag, not proof of a clean signal.
    These fields are only for post-prediction subgroup reporting, not model
    inputs, split selection, threshold choice or corruption assignment.
    """
    columns = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
    frame = pd.DataFrame(metadata)
    if not set(["ecg_id"] + columns).issubset(frame.columns):
        raise ValueError("Source quality metadata lacks required annotation columns.")
    if frame.ecg_id.isna().any() or not frame.ecg_id.is_unique:
        raise ValueError("Source quality metadata requires unique non-missing ECG IDs.")
    flags = frame[columns].fillna("").astype(str).apply(lambda column: column.str.strip().ne(""))
    groups = np.where(flags.any(axis=1), "recorded_quality_flag", "no_recorded_quality_flag")
    return pd.Series(groups, index=frame.ecg_id.to_numpy(), name="quality_group")


def validate_scenario_repeats(records, scenario_names, repeats=(0, 1, 2), reference="observed"):
    """Require the full prespecified scenario/repeat set in one arm and fold.

    Validation against the declared protocol catches conditions or repeats
    missing uniformly from every arm, which matching observed groups cannot.
    The unperturbed reference appears only at repeat 0.
    """
    frame = pd.DataFrame(records)
    scenarios = set(scenario_names)
    if reference not in scenarios or not set(repeats):
        raise ValueError("The protocol must include its reference and stress repeats.")
    if not {"scenario", "repeat"}.issubset(frame.columns):
        raise ValueError("Records require scenario and repeat columns.")
    if frame[["scenario", "repeat"]].isna().any().any():
        raise ValueError("Scenario and repeat identifiers must be present.")
    expected = {(reference, 0)} | {(scenario, repeat) for scenario in scenarios
                                  if scenario != reference for repeat in repeats}
    actual = set(map(tuple, frame[["scenario", "repeat"]].to_numpy()))
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"Scenario/repeat protocol mismatch: missing={missing}; extra={extra}.")


def validate_experiment_arms(records):
    """Require all three models and four feature/training arms in ten folds.

    Names are fixed by the study protocol. Compare against the declared grid,
    so uniformly missing models or arms cannot pass by agreeing with each other.
    """
    columns = ["model", "feature_set", "training_arm", "outer_fold"]
    frame = pd.DataFrame(records)
    if not set(columns).issubset(frame.columns) or frame[columns].isna().any().any():
        raise ValueError("Experiment groups require model, feature_set, training_arm and outer_fold.")
    expected = {(model, features, training, fold) for model in MODEL_NAMES
                for features in FEATURE_SETS for training in TRAINING_ARMS for fold in range(10)}
    actual = set(map(tuple, frame[columns].to_numpy()))
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"Experiment model/arm protocol mismatch: missing={missing}; extra={extra}.")


def summarise_paired_drops(records, metric, reference="observed",
                          higher_is_better=True, minimum_folds=10):
    """Average repeats within each fold, then summarise paired deterioration.

    records contains outer_fold, scenario, repeat and the named metric, for
    ONE model/feature/augmentation arm at a time. Positive deterioration means
    worse under stress. Error bars are sample SD across folds, not confidence
    intervals. Shared training data make folds dependent. Missing metrics
    propagate rather than being silently excluded. Return (folds, summary).
    """
    frame = pd.DataFrame(records).copy()
    required = {"outer_fold", "scenario", "repeat", metric}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Non-empty records require columns {sorted(required)}.")
    if minimum_folds < 2:
        raise ValueError("At least two folds are needed for sample SD.")
    keys = ["outer_fold", "scenario", "repeat"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Each fold/scenario/repeat must be present and unique.")
    if np.isinf(frame[metric].to_numpy(dtype=float)).any():
        raise ValueError("Metrics must be finite or NaN, never infinite.")
    # Different folds must not silently average different subsets of repeats.
    # The full scenario list is validated separately against the study protocol.
    for scenario, condition in frame.groupby("scenario"):
        repeat_sets = condition.groupby("outer_fold")["repeat"].agg(lambda values: frozenset(values))
        if len(set(repeat_sets)) != 1:
            raise ValueError(f"Scenario {scenario} has different repeat sets across outer folds.")
    rows = []
    for (fold, scenario), group in frame.groupby(["outer_fold", "scenario"], sort=True):
        values = group[metric].to_numpy(dtype=float)
        rows.append({"outer_fold": fold, "scenario": scenario,
                     "value": float(np.mean(values)), "n_repeats": len(values)})
    per_fold = pd.DataFrame(rows)
    baseline = per_fold[per_fold.scenario.eq(reference)].set_index("outer_fold")
    if len(baseline) < minimum_folds:
        raise ValueError(f"At least {minimum_folds} reference outer folds are required.")
    effects, summaries = [], []
    for scenario, group in per_fold.groupby("scenario", sort=True):
        if scenario == reference:
            continue
        group = group.set_index("outer_fold").sort_index()
        if set(group.index) != set(baseline.index):
            raise ValueError("Every stress scenario must use the same reference folds.")
        clean = baseline.loc[group.index, "value"].to_numpy()
        stress = group.value.to_numpy()
        drop = clean - stress if higher_is_better else stress - clean
        for i, fold in enumerate(group.index):
            effects.append({"outer_fold": fold, "scenario": scenario, "metric": metric,
                            "reference_value": clean[i], "stress_value": stress[i],
                            "deterioration": drop[i], "n_repeats": int(group.loc[fold, "n_repeats"])})
        summaries.append({"scenario": scenario, "metric": metric, "n_folds": len(drop),
                          "n_defined_folds": int(np.isfinite(drop).sum()),
                          "mean_deterioration": float(np.mean(drop)),
                          "sd_deterioration": float(np.std(drop, ddof=1)),
                          "mean_reference": float(np.mean(clean)),
                          "mean_stress": float(np.mean(stress)),
                          "error_bar": "sample SD across outer folds; descriptive"})
    return pd.DataFrame(effects), pd.DataFrame(summaries)

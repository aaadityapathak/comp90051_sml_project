"""Validate previously saved HGB results and summarise them without retraining."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.evaluation import binary_metrics

ROOT = PROJECT / "results/histogram_gradient_boosting/retuned7_8_9"
METRICS = ["macro_f1", "sensitivity", "specificity", "precision", "npv",
           "balanced_accuracy", "auroc", "average_precision", "brier"]
STRESSES = ["missing_1", "missing_3", "missing_6", "noise_0.05",
            "noise_0.10", "noise_0.20", "gain_0.05", "gain_0.10",
            "gain_0.20", "combined_3lead_noise0.10"]
EXPECTED = {("observed", 0)} | {(s, r) for s in STRESSES for r in range(3)}


def check_retuned(root):
    """Recompute the consolidated retuned metrics from its saved predictions.

    This check needs no dataset or fitted estimator. It checks every held-out
    ECG, scenario and repeat, then checks the selected depth against all inner
    scores using the experiment's smaller-depth tie rule.
    """
    metrics = pd.read_csv(root / 'HGB_ABCD_all_metrics.csv')
    tuning = pd.read_csv(root / 'HGB_ABCD_all_inner_tuning.csv')
    metric_keys = ['arm', 'outer_fold', 'scenario', 'repeat']
    tuning_keys = ['arm', 'outer_fold', 'max_depth', 'inner_fold']
    if len(metrics) != 1240 or metrics.duplicated(metric_keys).any():
        raise ValueError('Expected 1,240 distinct outer evaluation records.')
    if len(tuning) != 360 or tuning.duplicated(tuning_keys).any():
        raise ValueError('Expected 360 distinct inner tuning records.')
    if set(metrics.arm) != set('ABCD') or set(tuning.arm) != set('ABCD'):
        raise ValueError('Both metric and tuning tables must cover arms A-D.')
    if not np.isfinite(tuning.macro_f1).all():
        raise ValueError('Inner tuning scores must be finite.')

    reference_folds = {}
    prediction_count = 0
    for arm in 'ABCD':
        predictions = pd.read_csv(root / f'HGB_retuned_{arm}_predictions.zip')
        selected = pd.read_csv(root / f'{arm}_selected_depths.csv')
        if (len(selected) != 10 or set(selected.outer_fold) != set(range(10))
                or not selected.arm.eq(arm).all()):
            raise ValueError(f'{arm}: selected-depth table must cover all ten folds.')
        if (not predictions.arm.eq(arm).all()
                or set(predictions.outer_fold) != set(range(10))
                or predictions.duplicated(metric_keys + ['ecg_id']).any()
                or not predictions.prob_afib.between(0, 1).all()
                or not predictions.y_true.isin([0, 1]).all()):
            raise ValueError(f'{arm}: invalid prediction coverage, labels or probabilities.')
        for fold in range(10):
            label = f'{arm}/fold{fold}'
            fold_metrics = metrics[metrics.arm.eq(arm) & metrics.outer_fold.eq(fold)]
            fold_tuning = tuning[tuning.arm.eq(arm) & tuning.outer_fold.eq(fold)]
            fold_predictions = predictions[predictions.outer_fold.eq(fold)]
            if set(zip(fold_metrics.scenario, fold_metrics.repeat)) != EXPECTED:
                raise ValueError(f'{label}: missing evaluation condition.')
            candidates = {(depth, inner) for depth in (7, 8, 9) for inner in range(3)}
            if set(zip(fold_tuning.max_depth, fold_tuning.inner_fold)) != candidates:
                raise ValueError(f'{label}: incomplete depth search.')
            means = fold_tuning.groupby('max_depth').macro_f1.mean()
            winner = min((7, 8, 9), key=lambda depth: (-means[depth], depth))
            chosen = selected[selected.outer_fold.eq(fold)].iloc[0]
            if not fold_metrics.max_depth.eq(winner).all() or int(chosen.max_depth) != winner:
                raise ValueError(f'{label}: depth differs from inner-CV selection.')
            np.testing.assert_allclose(chosen.macro_f1, means[winner], rtol=1e-10, atol=1e-12)
            observed = fold_predictions[fold_predictions.scenario.eq('observed')]
            reference = observed[['ecg_id', 'y_true']].sort_values('ecg_id').reset_index(drop=True)
            if reference.empty or reference.ecg_id.duplicated().any():
                raise ValueError(f'{label}: observed ECGs are missing or duplicated.')
            if arm == 'A':
                reference_folds[fold] = reference
            elif not reference.equals(reference_folds[fold]):
                raise ValueError(f'{label}: held-out patients differ between arms.')
            grouped = fold_predictions.groupby(['scenario', 'repeat'])
            if set(grouped.groups) != EXPECTED:
                raise ValueError(f'{label}: prediction conditions do not match the protocol.')
            saved = fold_metrics.set_index(['scenario', 'repeat'])
            for condition, rows in grouped:
                identities = rows[['ecg_id', 'y_true']].sort_values('ecg_id').reset_index(drop=True)
                if not identities.equals(reference):
                    raise ValueError(f'{label}/{condition}: ECG IDs or labels changed under corruption.')
                calculated = binary_metrics(rows.y_true.to_numpy(), rows.prob_afib.to_numpy())
                for metric, value in calculated.items():
                    np.testing.assert_allclose(value, saved.loc[condition, metric],
                                               rtol=1e-10, atol=1e-12, equal_nan=True,
                                               err_msg=f'{label}/{condition}/{metric}')
        prediction_count += len(predictions)
        print(f'Verified retuned arm {arm}: 10 folds, {len(predictions):,} predictions.')
    combined_ids = pd.concat(reference_folds.values(), ignore_index=True).ecg_id
    if combined_ids.duplicated().any():
        raise ValueError('An ECG appears in more than one held-out outer fold.')
    observed = metrics[metrics.scenario.eq('observed')].groupby('arm').macro_f1.agg(['mean', 'std'])
    print('\nObserved macro-F1 (mean and sample SD):\n', observed.to_string())
    return {'outer_metric_records': len(metrics), 'inner_tuning_records': len(tuning),
            'prediction_records': prediction_count, 'all_metrics_recomputed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT,
                        help='Consolidated retuned results, or runner output with A_fold0 through D_fold9.')
    args = parser.parse_args()
    root = args.output.resolve()
    if (root / 'HGB_ABCD_all_inner_tuning.csv').is_file():
        print(json.dumps(check_retuned(root), indent=2))
        return
    records = []
    for arm in "ABCD":
        for fold in range(10):
            location = root / f"{arm}_fold{fold}"
            for name in ["complete.json", "inner_tuning.csv", "metrics.csv", "predictions.csv"]:
                if not (location / name).is_file():
                    raise FileNotFoundError(f"{location}: missing {name}")
            meta = json.loads((location / "complete.json").read_text(encoding="utf-8"))
            if meta.get("arm") != arm or meta.get("outer_fold") != fold or meta.get("repeats") != 3:
                raise ValueError(f"{location}: completion metadata mismatch")
            m = pd.read_csv(location / "metrics.csv")
            p = pd.read_csv(location / "predictions.csv")
            t = pd.read_csv(location / "inner_tuning.csv")
            for df, name in [(m, "metrics"), (p, "predictions"), (t, "inner_tuning")]:
                if not ((df["arm"] == arm) & (df["outer_fold"] == fold)).all():
                    raise ValueError(f"{location}: {name} has mismatched arm/fold")
            pairs = list(zip(m["scenario"], m["repeat"]))
            if len(pairs) != 31 or set(pairs) != EXPECTED or len(set(pairs)) != 31:
                raise ValueError(f"{location}: incomplete or duplicate scenario/repeat rows")
            candidates = set(t['max_depth'])
            expected_tuning = {(depth, inner) for depth in candidates for inner in range(3)}
            if (len(t) not in (9, 12) or len(t) != len(expected_tuning)
                    or set(zip(t.max_depth, t.inner_fold)) != expected_tuning):
                raise ValueError(f"{location}: unexpected tuning coverage")
            n = int(m.loc[m["scenario"].eq("observed"), "n"].iloc[0])
            if n <= 0 or len(p) != 31 * n:
                raise ValueError(f"{location}: prediction count mismatch")
            count = p.groupby(["scenario", "repeat"]).size()
            if set(count.index) != EXPECTED or not (count == n).all():
                raise ValueError(f"{location}: incomplete prediction scenarios")
            if not p["prob_afib"].between(0, 1).all() or p["prob_afib"].isna().any():
                raise ValueError(f"{location}: invalid probabilities")
            if not p["y_true"].isin([0, 1]).all():
                raise ValueError(f"{location}: invalid labels")
            base = p.loc[(p["scenario"] == "observed") & (p["repeat"] == 0),
                         ["ecg_id", "y_true"]].sort_values("ecg_id").reset_index(drop=True)
            if base["ecg_id"].duplicated().any():
                raise ValueError(f"{location}: duplicate ECG IDs")
            for (_, _), chunk in p.groupby(["scenario", "repeat"]):
                target = chunk[["ecg_id", "y_true"]].sort_values("ecg_id").reset_index(drop=True)
                if not target.equals(base):
                    raise ValueError(f"{location}: scenario labels or ECG IDs differ")
            if not all(metric in m for metric in METRICS):
                raise ValueError(f"{location}: missing evaluation columns")
            records.append(m)
            print(f"Verified {arm}_fold{fold}")

    all_metrics = pd.concat(records, ignore_index=True)
    observed = all_metrics.loc[(all_metrics["scenario"] == "observed") &
                               (all_metrics["repeat"] == 0)]
    summary = observed.groupby("arm")[METRICS].agg(["mean", "std"])
    fold_means = all_metrics.groupby(["arm", "outer_fold", "scenario"])["macro_f1"].mean().unstack("scenario")
    stress_drops = pd.DataFrame({f"drop_{s}": fold_means["observed"] - fold_means[s] for s in STRESSES})
    drops = stress_drops.groupby(level="arm").agg(["mean", "std"])
    wide_observed = observed.pivot(index="outer_fold", columns="arm", values="macro_f1")
    comparisons = {"B_minus_A": ("B", "A"), "C_minus_A": ("C", "A"),
                   "D_minus_B": ("D", "B"), "D_minus_C": ("D", "C")}
    differences = pd.DataFrame({name: wide_observed[a] - wide_observed[b]
                                for name, (a, b) in comparisons.items()})
    paired = differences.agg(["mean", "std"]).T
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "final_observed_metrics.csv")
    drops.to_csv(output / "final_stress_drops.csv")
    paired.to_csv(output / "final_paired_method_differences.csv")
    all_metrics.to_csv(output / "final_all_metrics.csv", index=False)
    print("\n=== OBSERVED MACRO-F1 (mean +/- sample SD) ===")
    print(summary["macro_f1"].round(4).to_string())
    print("\nAll 40 folds and 31 scenarios per fold verified; final summaries saved.")


if __name__ == "__main__":
    main()

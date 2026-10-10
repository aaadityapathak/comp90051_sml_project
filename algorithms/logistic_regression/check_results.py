"""Check the completed experiment without fitting or changing any model."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.data_preparation import load_processed_data
from afib.evaluation import choose_sensitivity_threshold
from afib.preprocessing import SCENARIOS


def reference_metrics(y, scores, threshold):
    """Recalculate metrics directly from saved probabilities."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    divide = lambda a, b: a / b if b else np.nan
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)

    # Sort once and evaluate complete tied-score groups.
    order = np.argsort(-scores, kind='stable')
    ranked_y, ranked_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)) + 1, len(y)]
    starts = np.r_[0, ends[:-1]]
    cumulative_positive = np.r_[0, np.cumsum(ranked_y)]
    positives_per_group = cumulative_positive[ends] - cumulative_positive[starts]
    negatives_per_group = ends - starts - positives_per_group
    positives = int(y.sum())
    negatives = len(y) - positives
    ap = (float(np.sum(positives_per_group / positives * cumulative_positive[ends] / ends))
          if positives else np.nan)
    negative_below = negatives - (ends - cumulative_positive[ends])
    auc = (float(np.sum(positives_per_group * (negative_below + .5 * negatives_per_group))
                 / (positives * negatives)) if positives and negatives else np.nan)
    return {
        'n': len(y), 'n_positive': positives, 'n_negative': negatives,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'macro_f1': .5 * (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)),
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': divide(tp, tp + fp), 'npv': divide(tn, tn + fn),
        'balanced_accuracy': .5 * (sensitivity + specificity),
        'accuracy': (tp + tn) / len(y), 'auroc': auc, 'average_precision': ap,
        'brier': float(np.mean((scores - y) ** 2)),
    }


def check(output):
    package = load_processed_data(PROJECT / 'processed_data')
    folds = package['folds']
    completion = json.loads((output / 'completion.json').read_text())
    config = json.loads((output / 'run_config.json').read_text())
    assert completion['complete_logistic_plan'] and completion['data_unchanged']
    assert package['manifest'] == config['manifest']
    for path in [Path(__file__).with_name('model.py'), Path(__file__).with_name('run_experiment.py'),
                 PROJECT / 'src/afib/preprocessing.py', PROJECT / 'src/afib/imputation.py',
                 PROJECT / 'src/afib/evaluation.py']:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config['code'][path.name], path.name

    checked_conditions = 0
    fitted_models = 0
    for outer in range(10):
        test_ids = folds.index[folds.outer_fold.eq(outer)]
        train_ids = folds.index[folds.outer_fold.ne(outer)]
        assert set(folds.loc[test_ids, 'patient_id']).isdisjoint(folds.loc[train_ids, 'patient_id'])
        for variant in completion['all_variants']:
            location = output / 'folds' / f'{outer:02d}_{variant}'
            assert json.loads((location / 'complete.json').read_text())['completed']
            selection = json.loads((location / 'selection.json').read_text())
            audit = json.loads((location / 'fit_audit.json').read_text())
            assert all(f['converged'] and f['iterations'] <= 2000 and f['max_abs_gradient'] <= 1e-4 for f in audit)
            fitted_models += len(audit)

            oof = pd.read_csv(location / 'inner_oof_predictions.csv.gz')
            assert oof.ecg_id.is_unique and set(oof.ecg_id) == set(train_ids)
            reference = folds.loc[oof.ecg_id]
            assert np.array_equal(oof.inner_fold, reference[f'inner_fold_when_outer_{outer}'])
            assert np.array_equal(oof.patient_id, reference.patient_id)
            assert np.array_equal(oof.y_true, reference.target_afib)
            threshold = choose_sensitivity_threshold(oof.y_true, oof.score, .90)
            np.testing.assert_allclose(threshold['threshold'], selection['secondary_threshold']['threshold'], atol=1e-12)

            tuning = pd.read_csv(location / 'tuning.csv')
            means = tuning.groupby('C').macro_f1.mean()
            grid = selection['final_grid']
            selected = min(grid, key=lambda c: (-means.loc[c], c))
            assert selected == selection['C']
            assert selection['selected_middle'] == (selected == sorted(grid)[1])
            for _, candidate in tuning.groupby('C'):
                assert set(candidate.inner_fold) == {0, 1, 2} and len(candidate) == 3

            predictions = pd.read_csv(location / 'outer_predictions.csv.gz')
            metrics = pd.read_csv(location / 'metrics.csv').set_index(['scenario', 'repeat', 'threshold_policy'])
            conditions = list(SCENARIOS) if variant != 'ridge' else ['observed', 'missing_3']
            expected = {(scenario, repeat) for scenario in conditions
                        for repeat in ([0] if scenario == 'observed' else range(3))}
            assert set(predictions.groupby(['scenario', 'repeat']).groups) == expected
            for (scenario, repeat), rows in predictions.groupby(['scenario', 'repeat']):
                assert rows.ecg_id.is_unique and set(rows.ecg_id) == set(test_ids)
                reference = folds.loc[rows.ecg_id]
                assert np.array_equal(rows.patient_id, reference.patient_id)
                assert np.array_equal(rows.y_true, reference.target_afib)
                assert np.isfinite(rows.score).all() and rows.score.between(0, 1).all()
                for policy, decision in [('fixed_0.50', .5), ('inner_sensitivity_0.90', threshold['threshold'])]:
                    result = reference_metrics(rows.y_true, rows.score, decision)
                    saved = metrics.loc[(scenario, repeat, policy)]
                    for metric, value in result.items():
                        np.testing.assert_allclose(value, saved[metric], rtol=1e-11, atol=1e-12,
                                                   equal_nan=True, err_msg=f'{outer}/{variant}/{scenario}/{metric}')
                checked_conditions += 1
    assert fitted_models == completion['n_fits']

    metrics = pd.read_csv(output / 'metrics.csv')
    metric_names = ['macro_f1', 'sensitivity', 'specificity', 'precision', 'npv',
                    'balanced_accuracy', 'accuracy', 'auroc', 'average_precision', 'brier']
    per_fold = metrics.groupby(['variant', 'outer_fold', 'scenario', 'threshold_policy'])[metric_names].mean()
    summary = pd.read_csv(output / 'summary.csv')
    for row in summary.itertuples():
        values = per_fold.xs((row.variant, row.scenario, row.threshold_policy),
                                level=['variant', 'scenario', 'threshold_policy'])[row.metric].to_numpy()
        assert len(values) == row.n_folds == 10
        np.testing.assert_allclose([values.mean(), values.std(ddof=1)], [row.mean, row.sd], atol=1e-12, equal_nan=True)
    return {'status': 'passed', 'model_training_performed': False,
            'checkpoints': 90, 'outer_conditions': checked_conditions,
            'fitted_models_in_saved_run': fitted_models,
            'metric_values_independently_recalculated': True,
            'patient_separation_and_oof_coverage': True,
            'selection_and_thresholds_verified': True,
            'fold_mean_and_sd_verified': True, 'processed_data_hashes_verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'reports/logistic_regression')
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    result = check(args.output)
    text = json.dumps(result, indent=2)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text + '\n', encoding='utf-8')
    print(text)

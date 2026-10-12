"""Independently validate saved TabM output without fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))
from afib.data_preparation import load_processed_data
from afib.evaluation import choose_sensitivity_threshold, source_quality_groups
from algorithms.tabm.reporting import (FOLD_FILES, METRICS, completed_locations,
                                      expected_conditions, full_protocol_settings, read_json)
from algorithms.tabm.source_verification import RUNNER_SOURCE, verify_runner_migration


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_metrics(y, scores, threshold):
    """Compute confusion, ranking and probability metrics from first principles."""
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    require(len(y) > 0 and set(y).issubset({0, 1}), 'Invalid binary targets')
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), 'Invalid probabilities')
    predicted = scores >= threshold
    tp, tn = int(((y == 1) & predicted).sum()), int(((y == 0) & ~predicted).sum())
    fp, fn = int(((y == 0) & predicted).sum()), int(((y == 1) & ~predicted).sum())
    divide = lambda a, b: a / b if b else np.nan
    sensitivity, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    positive_f1, negative_f1 = divide(2 * tp, 2 * tp + fp + fn), divide(2 * tn, 2 * tn + fp + fn)
    order = np.argsort(-scores, kind='stable')
    ranked_y, ranked_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)) + 1, len(y)]
    starts = np.r_[0, ends[:-1]]
    cumulative_positive = np.r_[0, np.cumsum(ranked_y)]
    group_positive = cumulative_positive[ends] - cumulative_positive[starts]
    group_negative = ends - starts - group_positive
    positives, negatives = int(y.sum()), int((y == 0).sum())
    negative_below = negatives - (ends - cumulative_positive[ends])
    return {'n': len(y), 'n_positive': positives, 'n_negative': negatives,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'positive_f1': positive_f1, 'negative_f1': negative_f1,
            'macro_f1': .5 * (positive_f1 + negative_f1),
            'sensitivity': sensitivity, 'specificity': specificity,
            'precision': divide(tp, tp + fp), 'npv': divide(tn, tn + fn),
            'balanced_accuracy': .5 * (sensitivity + specificity), 'accuracy': (tp + tn) / len(y),
            'auroc': (float(np.sum(group_positive * (negative_below + .5 * group_negative))
                            / (positives * negatives)) if positives and negatives else np.nan),
            'average_precision': (float(np.sum(group_positive / positives * cumulative_positive[ends] / ends))
                                  if positives else np.nan),
            'brier': float(np.mean((scores - y) ** 2)), 'threshold': threshold}


def _weighted_metrics(rows, threshold):
    weights = rows.patient_id.map(1 / rows.patient_id.value_counts()).to_numpy(dtype=float)
    y, score = rows.y_true.to_numpy(), rows.score.to_numpy()
    prediction = score >= threshold
    tp, tn = weights[(y == 1) & prediction].sum(), weights[(y == 0) & ~prediction].sum()
    fp, fn = weights[(y == 0) & prediction].sum(), weights[(y == 1) & ~prediction].sum()
    divide = lambda a, b: float(a / b) if b else np.nan
    return {'n_patients': rows.patient_id.nunique(), 'weighted_tp': tp, 'weighted_tn': tn,
            'weighted_fp': fp, 'weighted_fn': fn,
            'macro_f1': .5 * (divide(2*tp, 2*tp+fp+fn) + divide(2*tn, 2*tn+fp+fn)),
            'sensitivity': divide(tp, tp+fn), 'specificity': divide(tn, tn+fp),
            'balanced_accuracy': .5 * (divide(tp, tp+fn) + divide(tn, tn+fp)),
            'accuracy': divide(tp+tn, weights.sum()),
            'brier': float(weights @ ((score-y)**2) / weights.sum())}


def _compare(expected, row, label):
    require(isinstance(row, pd.Series), f'Duplicate saved rows: {label}')
    for metric, value in expected.items():
        require(metric in row.index, f'Missing metric {metric}: {label}')
        np.testing.assert_allclose(value, row[metric], rtol=1e-10, atol=1e-12, equal_nan=True,
                                   err_msg=f'{label}/{metric}')


def _hash_files(location, marker):
    files = marker.get('files', {})
    require(bool(files), f'No checkpoint file hashes: {location}')
    for name, digest in files.items():
        path = (location / name).resolve()
        require(path.parent == location.resolve(), f'Unsafe checkpoint hash path: {name}')
        require(path.is_file(), f'Missing saved artifact: {path}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Artifact hash changed: {path}')


def _verify_history(history, audit, config, label):
    require(audit.get('epochs') == config['epochs'], f'Wrong epoch budget: {label}')
    require(audit.get('epochs_completed', audit['epochs']) == config['epochs'], f'Incomplete fit: {label}')
    require(history.epoch.tolist() == list(range(1, config['epochs'] + 1)), f'Incomplete epoch history: {label}')
    require(np.isfinite(history.weighted_bce).all(), f'Non-finite loss: {label}')
    require((history.weighted_bce >= 0).all(), f'Negative BCE loss: {label}')
    if 'batch_size' in audit:
        require(audit['batch_size'] == config['batch_size'], f'Wrong batch size: {label}')
    if 'early_stopping' in audit:
        require(audit['early_stopping'] is False, f'Early stopping not in protocol: {label}')
    if 'architecture' in audit:
        architecture = audit['architecture']
        require(all(architecture.get(key) == value for key, value in
                    {'arch_type': 'tabm', 'd_block': 128, 'k': 16, 'dropout': .1,
                     'num_embeddings': None}.items()), f'Architecture not in protocol: {label}')
    if 'optimizer' in audit:
        require(audit['optimizer'] == {'name': 'AdamW', 'lr': .0003, 'weight_decay': .0001},
                f'Optimizer not in protocol: {label}')


def _verify_predictions(rows, reference, ids, label):
    require(rows.ecg_id.is_unique and set(rows.ecg_id) == set(ids), f'Prediction coverage: {label}')
    expected = reference.loc[rows.ecg_id]
    require(np.array_equal(rows.patient_id, expected.patient_id), f'Patient mapping: {label}')
    require(np.array_equal(rows.y_true, expected.target_afib), f'Target mapping: {label}')
    require(np.isfinite(rows.score).all() and rows.score.between(0, 1).all(), f'Probabilities: {label}')


def _verify_summary(output, metric_frames):
    path = output / 'output_data' / 'summary.csv'
    if not path.exists():
        return False
    summary = pd.read_csv(path)
    if not metric_frames:
        require(summary.empty, 'Summary contains results with no completed outer folds')
        return True
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    keys = ['variant', 'scenario', 'threshold_policy']
    expected_keys = set()
    for group_key, group in all_metrics.groupby(keys):
        for metric in METRICS:
            per_fold = [np.mean(rows[metric].to_numpy(dtype=float)) for _, rows in group.groupby('outer_fold')]
            values = np.asarray(per_fold)
            mask = np.logical_and.reduce([summary[key].eq(value) for key, value in zip(keys, group_key)])
            saved = summary[mask & summary.metric.eq(metric)]
            require(len(saved) == 1, f'Summary coverage: {group_key}/{metric}')
            row = saved.iloc[0]
            expected = {'mean': float(np.mean(values)),
                        'sd': float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                        'n_folds': len(values), 'n_defined': int(np.isfinite(values).sum())}
            _compare(expected, row, f'summary/{group_key}/{metric}')
            expected_keys.add((*group_key, metric))
    require(len(summary) == len(expected_keys), 'Unexpected summary rows')
    return True


def check(output, allow_partial=False, package=None):
    output = Path(output)
    config = read_json(output / 'run_config.json')
    require(config.get('model') == 'tabm', 'This checker validates TabM only')
    package = load_processed_data(PROJECT / 'processed_data') if package is None else package
    require(package['manifest'] == config['manifest'], 'Processed-data manifest differs from saved run')
    source_count = 0
    preserved_sources = {}
    normalization = config.get('source_hash_normalization', 'raw')
    require(normalization in ('raw', 'LF'), 'Unsupported source hash normalization')
    for name, digest in config.get('code', {}).items():
        path = (PROJECT / name).resolve()
        require(path.is_relative_to(PROJECT), f'Unsafe source path: {name}')
        require(path.is_file(), f'Missing training source: {name}')
        source = path.read_bytes()
        if normalization == 'LF':
            source = source.replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != digest:
            require(name == RUNNER_SOURCE, f'Training source changed since saved run: {name}')
            preserved_sources[name] = verify_runner_migration(PROJECT, output, digest, normalization)
        source_count += 1
    locations, pending = completed_locations(output, config)
    if not allow_partial:
        require(not pending, f'Run incomplete: {len(pending)} requested outer-fold/variant checkpoints missing')
    report_completion = output / 'reports' / 'completion.json'
    if report_completion.exists():
        completion = read_json(report_completion)
        require(not completion.get('complete_tabm_plan') or (not pending and full_protocol_settings(config)),
                'Report falsely claims a complete 100-epoch TabM plan')
    folds = package['folds']
    quality = source_quality_groups(package['metadata'].reset_index()) if 'metadata' in package else None
    checked_conditions, checked_candidates, checked_fits = 0, 0, set()
    candidate_scores, candidate_predictions = {}, {}
    metric_frames = []
    for outer in config['outer_folds']:
        training, test = folds[folds.outer_fold.ne(outer)], folds[folds.outer_fold.eq(outer)]
        require(set(training.patient_id).isdisjoint(test.patient_id), f'Patient leakage: outer {outer}')
        for inner in range(config['n_inner_folds']):
            column = f'inner_fold_when_outer_{outer}'
            fit = training[training[column].ne(inner)]
            validation = training[training[column].eq(inner)]
            require(set(fit.patient_id).isdisjoint(validation.patient_id), f'Patient leakage: outer {outer} inner {inner}')
    # Verify completed tuning even before the first outer checkpoint exists.
    for marker_path in sorted((output / 'output_data' / 'tuning').glob('*/blocks_*/inner_*/complete.json')):
        location = marker_path.parent
        marker = read_json(marker_path)
        if marker.get('completed') is not True:
            continue
        outer_text, variant = location.parent.parent.name.split('_', 1)
        outer, inner = int(outer_text), int(location.name.removeprefix('inner_'))
        blocks = int(location.parent.name.removeprefix('blocks_'))
        if outer not in config['outer_folds'] or variant not in config['variants']:
            continue
        _hash_files(location, marker)
        history, audit = pd.read_csv(location / 'history.csv'), read_json(location / 'audit.json')
        _verify_history(history, audit, config, str(location))
        rows = pd.read_csv(location / 'predictions.csv.gz')
        ids = folds.index[folds.outer_fold.ne(outer) & folds[f'inner_fold_when_outer_{outer}'].eq(inner)]
        _verify_predictions(rows, folds, ids, str(location))
        require(audit['n_blocks'] == blocks, f'Candidate architecture: {location}')
        candidate_key = (outer, variant, blocks, inner)
        candidate_scores[candidate_key] = reference_metrics(rows.y_true, rows.score, .5)['macro_f1']
        candidate_predictions[candidate_key] = rows
        if 'macro_f1' in audit:
            np.testing.assert_allclose(audit['macro_f1'], candidate_scores[candidate_key], atol=1e-12)
        checked_candidates += 1
        checked_fits.add((outer, variant, 'inner', blocks, inner))
    for location in locations:
        outer_text, variant = location.name.split('_', 1)
        outer = int(outer_text)
        marker = read_json(location / 'complete.json')
        _hash_files(location, marker)
        require(set(FOLD_FILES).issubset(marker['files']), f'Missing committed file hashes: {location}')
        selection = read_json(location / 'selection.json')
        train_ids, test_ids = folds.index[folds.outer_fold.ne(outer)], folds.index[folds.outer_fold.eq(outer)]
        oof = pd.read_csv(location / 'inner_oof_predictions.csv.gz')
        _verify_predictions(oof, folds, train_ids, str(location / 'inner_oof_predictions.csv.gz'))
        require(np.array_equal(oof.inner_fold, folds.loc[oof.ecg_id, f'inner_fold_when_outer_{outer}']),
                f'Inner OOF fold assignment: {location}')
        threshold = choose_sensitivity_threshold(oof.y_true, oof.score, .90)
        require(threshold['status'] == 'selected_on_inner_oof', f'Cannot select secondary threshold: {location}')
        np.testing.assert_allclose(threshold['threshold'], selection['secondary_threshold']['threshold'], atol=1e-12)
        tuning = pd.read_csv(location / 'tuning.csv')
        require(not tuning.duplicated(['n_blocks', 'inner_fold']).any(), f'Duplicate tuning scores: {location}')
        for blocks, candidate in tuning.groupby('n_blocks'):
            require(set(candidate.inner_fold) == set(range(config['n_inner_folds'])), f'Incomplete candidate: {location}/{blocks}')
            for row in candidate.itertuples():
                candidate_key = (outer, variant, int(blocks), int(row.inner_fold))
                require(candidate_key in candidate_scores, f'Missing verified tuning predictions: {candidate_key}')
                np.testing.assert_allclose(row.macro_f1, candidate_scores[candidate_key], atol=1e-12)
        means = {int(blocks): float(np.mean(rows.macro_f1)) for blocks, rows in tuning.groupby('n_blocks')}
        initial = sorted(config['initial_grid'])
        require(set(initial).issubset(means), f'Initial grid not completed: {location}')
        first_best = min(initial, key=lambda blocks: (-means[blocks], blocks))
        final_grid = initial
        if config['refine'] and first_best in (initial[0], initial[-1]):
            final_grid = ([max(1, initial[0] - 1), initial[0], initial[1]] if first_best == initial[0]
                          else [initial[-2], initial[-1], initial[-1] + 1])
            final_grid = sorted(set(final_grid))
        require(selection['final_grid'] == final_grid, f'Wrong refinement grid: {location}')
        require(set(final_grid).issubset(means), f'Refined grid not completed: {location}')
        selected = min(final_grid, key=lambda blocks: (-means[blocks], blocks))
        require(selection['n_blocks'] == selected, f'Hyperparameter selection: {location}')
        require(selection['selected_middle'] == (selected == final_grid[len(final_grid)//2]), f'Middle selection: {location}')
        selected_predictions = pd.concat([candidate_predictions[(outer, variant, selected, inner)]
                                          for inner in range(config['n_inner_folds'])]).set_index('ecg_id')
        np.testing.assert_allclose(oof.score, selected_predictions.loc[oof.ecg_id, 'score'], atol=1e-12)
        _verify_history(pd.read_csv(location / 'training_history.csv'), read_json(location / 'outer_fit_audit.json'),
                        config, str(location))
        audits = read_json(location / 'fit_audit.json')
        for audit in audits:
            require(audit['epochs'] == config['epochs'], f'Epoch budget in fit audit: {location}')
            phase = 'outer_refit' if audit['phase'] == 'outer_refit' else 'inner'
            checked_fits.add((outer, variant, phase, int(audit['n_blocks']), int(audit.get('inner_fold', -1))))
        predictions = pd.read_csv(location / 'outer_predictions.csv.gz')
        expected = expected_conditions(config['repeats'])
        require(set(predictions.groupby(['scenario', 'repeat']).groups) == expected, f'Expected scenario/repeat coverage: {location}')
        metrics = pd.read_csv(location / 'metrics.csv')
        metric_frames.append(metrics)
        metrics = metrics.set_index(['scenario', 'repeat', 'threshold_policy'])
        patient = pd.read_csv(location / 'patient_weighted_metrics.csv').set_index(['scenario', 'repeat', 'threshold_policy'])
        subgroups = pd.read_csv(location / 'quality_subgroups.csv').set_index(['scenario', 'repeat', 'threshold_policy', 'subgroup'])
        reliability = pd.read_csv(location / 'reliability.csv').set_index(['scenario', 'repeat', 'bin'])
        for (scenario, repeat), rows in predictions.groupby(['scenario', 'repeat']):
            _verify_predictions(rows, folds, test_ids, f'{location}/{scenario}/{repeat}')
            if quality is not None:
                require(np.array_equal(rows.quality_group, quality.loc[rows.ecg_id]), f'Quality groups: {location}')
            for policy, decision in [('fixed_0.50', .5), ('inner_sensitivity_0.90', threshold['threshold'])]:
                key = (scenario, repeat, policy)
                _compare(reference_metrics(rows.y_true, rows.score, decision), metrics.loc[key], f'{location}/{key}')
                _compare(_weighted_metrics(rows, decision), patient.loc[key], f'{location}/patient/{key}')
                for group, subset in rows.groupby('quality_group'):
                    values = reference_metrics(subset.y_true, subset.score, decision)
                    if len(set(subset.y_true)) < 2:
                        for metric in ('auroc', 'average_precision', 'macro_f1', 'balanced_accuracy'):
                            values[metric] = np.nan
                    _compare(values, subgroups.loc[(*key, group)], f'{location}/subgroup/{group}/{key}')
            bins = np.minimum((rows.score.to_numpy() * 10).astype(int), 9)
            for index in range(10):
                subset = rows[bins == index]
                values = {'n': len(subset), 'mean_probability': subset.score.mean() if len(subset) else np.nan,
                          'fraction_positive': subset.y_true.mean() if len(subset) else np.nan}
                _compare(values, reliability.loc[(scenario, repeat, index)], f'{location}/reliability/{scenario}/{repeat}/{index}')
            checked_conditions += 1
        baseline = read_json(location / 'majority_baseline.json')
        prevalence = float(folds.loc[train_ids, 'target_afib'].mean())
        majority_threshold = .5 if prevalence > .5 else np.nextafter(.5, np.inf)
        expected_baseline = reference_metrics(folds.loc[test_ids, 'target_afib'],
                                             np.full(len(test_ids), prevalence), majority_threshold)
        # JSON preserves unavailable baseline metrics as null rather than inventing zeros.
        saved_baseline = pd.Series({key: np.nan if value is None else value for key, value in baseline.items()})
        _compare(expected_baseline, saved_baseline, f'{location}/majority_baseline')
    summary_verified = _verify_summary(output, metric_frames)
    return {'status': 'passed' if not pending else 'passed_available_outputs_run_incomplete',
            'complete_tabm_plan': bool(not pending and locations and full_protocol_settings(config)),
            'requested_scope_complete': not pending, 'model_training_performed': False,
            'completed_outer_fold_variants_verified': len(locations), 'pending_outer_fold_variants': len(pending),
            'outer_conditions_verified': checked_conditions, 'completed_inner_checkpoints_verified': checked_candidates,
            'n_fits': len(checked_fits), 'source_files_verified': source_count,
            'source_verification': {'mode': 'preserved_layout_v1_sources' if preserved_sources else 'exact_saved_hashes',
                                    'preserved_sources': preserved_sources,
                                    'training_calculations_unchanged': True},
            'processed_data_manifest_verified': True, 'patient_separation_verified': True,
            'prediction_coverage_and_metrics_recomputed': bool(locations),
            'inner_selection_and_thresholds_verified': bool(locations), 'fold_mean_and_sd_verified': summary_verified}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results' / 'tabm')
    parser.add_argument('--allow-partial', action='store_true', help='Verify available checkpoints without claiming completion.')
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    result = check(args.output, allow_partial=args.allow_partial)
    text = json.dumps(result, indent=2)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text + '\n', encoding='utf-8')
    print(text)

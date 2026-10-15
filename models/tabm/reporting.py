"""Rebuild honest, resumable TabM reports from saved experiment artifacts.

This module never trains a model. Only completed outer checkpoints enter model
performance summaries; candidate histories may be shown while tuning is running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.data_preparation import load_processed_data
from afib.evaluation import binary_metrics, majority_baseline
from afib.preprocessing import SCENARIOS

VARIANTS = ('A', 'B', 'C', 'D', 'availability_observed',
            'availability_augmented', 'duplicate_original', 'duplicate_enhanced')
METRICS = ('macro_f1', 'sensitivity', 'specificity', 'precision', 'npv',
           'balanced_accuracy', 'accuracy', 'auroc', 'average_precision', 'brier')
FOLD_FILES = ('metrics.csv', 'patient_weighted_metrics.csv', 'quality_subgroups.csv',
              'reliability.csv', 'inner_oof_predictions.csv.gz',
              'outer_predictions.csv.gz', 'tuning.csv', 'fit_audit.json',
              'perturbation_audit.json', 'selection.json', 'training_history.csv',
              'outer_fit_audit.json', 'majority_baseline.json')
COMPARISONS = (('B', 'A'), ('D', 'C'), ('C', 'A'), ('D', 'B'),
               ('B', 'availability_observed'), ('D', 'availability_augmented'),
               ('duplicate_original', 'A'), ('duplicate_enhanced', 'B'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def expected_conditions(repeats=3):
    return {(scenario, repeat) for scenario in SCENARIOS
            for repeat in ([0] if scenario == 'observed' else range(repeats))}


def full_protocol_settings(config):
    """A reduced run can finish its requested scope but cannot finish the plan."""
    architecture = config.get('architecture', {})
    optimizer = config.get('optimizer', {})
    return (config.get('model') == 'tabm'
            and set(config.get('variants', [])) == set(VARIANTS)
            and set(config.get('outer_folds', [])) == set(range(10))
            and config.get('n_inner_folds') == 3 and config.get('epochs') == 100
            and config.get('repeats') == 3 and config.get('batch_size') == 256
            and sorted(config.get('initial_grid', [])) == [2, 3, 4]
            and config.get('refine') is True
            and architecture == {'width': 128, 'members': 16, 'dropout': .1, 'numerical_embeddings': False}
            and optimizer == {'name': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001})


def completed_locations(output, config):
    """Exclude interrupted and malformed checkpoints, even if some CSVs exist."""
    locations, pending = [], []
    for fold in config['outer_folds']:
        for variant in config['variants']:
            name = f'{fold:02d}_{variant}'
            path = Path(output) / 'output_data' / 'folds' / name
            reason = 'not completed'
            if (path / 'complete.json').exists():
                try:
                    marker = read_json(path / 'complete.json')
                    missing = [file for file in FOLD_FILES if not (path / file).is_file()]
                    if marker.get('completed') is True and not missing:
                        metrics = pd.read_csv(path / 'metrics.csv')
                        conditions = set(map(tuple, metrics[['scenario', 'repeat']].to_numpy()))
                        policies = {'fixed_0.50', 'inner_sensitivity_0.90'}
                        valid = (conditions == expected_conditions(config['repeats'])
                                 and set(metrics.threshold_policy) == policies
                                 and len(metrics) == 2 * len(conditions)
                                 and not metrics.duplicated(['scenario', 'repeat', 'threshold_policy']).any()
                                 and metrics.variant.eq(variant).all()
                                 and metrics.outer_fold.eq(fold).all())
                        if valid:
                            locations.append(path)
                            continue
                        reason = 'complete marker has incomplete or inconsistent metric coverage'
                    elif missing:
                        reason = 'missing files: ' + ', '.join(missing)
                except (ValueError, KeyError, OSError) as error:
                    reason = f'unreadable checkpoint: {error}'
            pending.append({'outer_fold': fold, 'variant': variant, 'reason': reason})
    return locations, pending


def fold_summaries(frame, labels=None, metrics=METRICS):
    """Mean repeats within a fold, then mean and sample SD of those fold means.

    np.mean deliberately propagates NaN, unlike pandas' default mean/SD.
    """
    labels = labels or ['variant', 'scenario', 'threshold_policy']
    available = [metric for metric in metrics if metric in frame]
    fold_columns = labels + ['outer_fold'] + available
    summary_columns = labels + ['metric', 'mean', 'sd', 'n_folds', 'n_defined']
    if frame.empty:
        return pd.DataFrame(columns=fold_columns), pd.DataFrame(columns=summary_columns)
    rows = []
    for key, values in frame.groupby(labels + ['outer_fold'], sort=True, dropna=False):
        row = dict(zip(labels + ['outer_fold'], key))
        row.update({metric: float(np.mean(values[metric].to_numpy(dtype=float))) for metric in available})
        rows.append(row)
    folds = pd.DataFrame(rows, columns=fold_columns)
    summaries = []
    for key, group in folds.groupby(labels, sort=True, dropna=False):
        for metric in available:
            values = group[metric].to_numpy(dtype=float)
            summaries.append({**dict(zip(labels, key)), 'metric': metric,
                              'mean': float(np.mean(values)),
                              'sd': float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                              'n_folds': len(values), 'n_defined': int(np.isfinite(values).sum())})
    return folds, pd.DataFrame(summaries, columns=summary_columns)


def paired_effects(folds):
    """Use only the intersection of completed folds for each paired contrast."""
    drops, comparisons = [], []
    if not folds.empty:
        for (variant, policy), group in folds.groupby(['variant', 'threshold_policy']):
            clean = group[group.scenario.eq('observed')].set_index('outer_fold')
            for scenario, stress in group.groupby('scenario'):
                if scenario == 'observed':
                    continue
                stress = stress.set_index('outer_fold')
                shared = clean.index.intersection(stress.index).sort_values()
                for metric in METRICS:
                    if metric not in group:
                        continue
                    delta = clean.loc[shared, metric].to_numpy() - stress.loc[shared, metric].to_numpy()
                    if metric == 'brier':
                        delta = -delta
                    drops.append({'variant': variant, 'threshold_policy': policy, 'scenario': scenario,
                                  'metric': metric, **_effect_stats(delta, 'deterioration')})
        for left, right in COMPARISONS:
            a = folds[folds.variant.eq(left)]
            b = folds[folds.variant.eq(right)]
            for (scenario, policy), left_group in a.groupby(['scenario', 'threshold_policy']):
                right_group = b[b.scenario.eq(scenario) & b.threshold_policy.eq(policy)]
                a_indexed, b_indexed = left_group.set_index('outer_fold'), right_group.set_index('outer_fold')
                shared = a_indexed.index.intersection(b_indexed.index).sort_values()
                if not len(shared):
                    continue
                for metric in METRICS:
                    if metric not in folds:
                        continue
                    delta = a_indexed.loc[shared, metric].to_numpy() - b_indexed.loc[shared, metric].to_numpy()
                    comparisons.append({'comparison': f'{left} minus {right}', 'scenario': scenario,
                                        'threshold_policy': policy, 'metric': metric,
                                        'direction': 'lower better' if metric == 'brier' else 'higher better',
                                        **_effect_stats(delta, 'difference')})
    drop_columns = ['variant', 'threshold_policy', 'scenario', 'metric', 'n_folds',
                    'n_defined', 'mean_deterioration', 'sd_deterioration']
    comparison_columns = ['comparison', 'scenario', 'threshold_policy', 'metric',
                          'direction', 'n_folds', 'n_defined', 'mean_difference', 'sd_difference']
    return pd.DataFrame(drops, columns=drop_columns), pd.DataFrame(comparisons, columns=comparison_columns)


def _effect_stats(values, name):
    return {'n_folds': len(values), 'n_defined': int(np.isfinite(values).sum()),
            f'mean_{name}': float(np.mean(values)),
            f'sd_{name}': float(np.std(values, ddof=1)) if len(values) > 1 else np.nan}


def candidate_progress(output, config):
    rows, histories = [], []
    root = Path(output) / 'output_data' / 'tuning'
    for location in sorted(root.glob('*/blocks_*/inner_*')):
        try:
            fold, variant = location.parent.parent.name.split('_', 1)
            blocks = int(location.parent.name.removeprefix('blocks_'))
            inner = int(location.name.removeprefix('inner_'))
            if int(fold) not in config['outer_folds'] or variant not in config['variants']:
                continue
            complete = ((location / 'complete.json').exists()
                        and read_json(location / 'complete.json').get('completed') is True)
            key = {'outer_fold': int(fold), 'variant': variant, 'n_blocks': blocks, 'inner_fold': inner}
            history_path = location / 'history.csv'
            history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
            if not history.empty:
                histories.append(history.assign(**key, completed=complete))
            audit = read_json(location / 'audit.json') if (location / 'audit.json').exists() else {}
            row = {**key, 'completed': complete, 'epochs_recorded': len(history),
                   'seconds': audit.get('train_seconds', audit.get('seconds', np.nan)), 'macro_f1': np.nan}
            predictions = location / 'predictions.csv.gz'
            if complete and predictions.exists():
                prediction = pd.read_csv(predictions)
                row['macro_f1'] = binary_metrics(prediction.y_true, prediction.score)['macro_f1']
            rows.append(row)
        except (ValueError, KeyError, OSError, pd.errors.EmptyDataError):
            # A concurrently written, unfinished candidate is not completion evidence.
            continue
    columns = ['outer_fold', 'variant', 'n_blocks', 'inner_fold', 'completed',
               'epochs_recorded', 'seconds', 'macro_f1']
    return pd.DataFrame(rows, columns=columns), (pd.concat(histories, ignore_index=True)
                                               if histories else pd.DataFrame())


def _table(frame, limit=40):
    if frame.empty:
        return 'No completed results are available yet.'
    # Do not require the optional tabulate dependency.
    display = frame.head(limit).copy()
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return 'undefined' if not np.isfinite(value) else f'{value:.4f}'
        return str(value).replace('|', '/')
    lines = ['| ' + ' | '.join(display.columns) + ' |',
             '| ' + ' | '.join(['---'] * len(display.columns)) + ' |']
    lines.extend('| ' + ' | '.join(format_value(value) for value in row) + ' |'
                 for row in display.itertuples(index=False, name=None))
    if len(frame) > limit:
        lines.append(f'\nShowing {limit} of {len(frame)} rows; full values are in output_data/.')
    return '\n'.join(lines)


def _figures(reports, config, summary, reliability, progress, histories, locations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    requested = len(config['outer_folds']) * len(config['variants'])
    if full_protocol_settings(config):
        scope = f"{config['epochs']}-epoch protocol · {len(locations)}/{requested} outer checkpoints complete"
    else:
        scope = (f"{config['epochs']}-epoch pilot · {len(locations)}/{requested} requested outer checkpoints; "
                 'full protocol incomplete')

    def save(fig, filename):
        fig.suptitle(scope, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(reports / filename, dpi=150)
        plt.close(fig)

    def empty(axis, message):
        axis.text(.5, .5, message, ha='center', va='center', transform=axis.transAxes, wrap=True)
        axis.set_xticks([])
        axis.set_yticks([])

    fig, axis = plt.subplots(figsize=(10, 4))
    counts = {variant: sum(path.name.split('_', 1)[1] == variant for path in locations)
              for variant in config['variants']}
    axis.bar(range(len(counts)), list(counts.values()), color='#287c8e')
    axis.axhline(len(config['outer_folds']), color='#777777', ls='--', label='Requested outer folds')
    axis.set_xticks(range(len(counts)), list(counts), rotation=25, ha='right')
    axis.set_ylabel('Completed outer folds')
    axis.set_ylim(0, max(1, len(config['outer_folds'])) + .5)
    axis.set_title('TabM progress: completed checkpoints only')
    axis.legend()
    save(fig, 'progress.png')

    for metric, ylabel in [('macro_f1', 'Macro-F1 at threshold 0.50'),
                           ('sensitivity', 'Sensitivity at threshold 0.50'),
                           ('brier', 'Brier score (lower is better)')]:
        fig, axis = plt.subplots(figsize=(11, 5))
        selected = summary[summary.threshold_policy.eq('fixed_0.50') & summary.metric.eq(metric)]
        if selected.empty:
            empty(axis, 'No completed outer-fold model results yet.\nTuning results are not test performance.')
        else:
            for variant in config['variants']:
                rows = selected[selected.variant.eq(variant)].set_index('scenario').reindex(list(SCENARIOS))
                if rows['mean'].notna().any():
                    axis.plot(range(len(SCENARIOS)), rows['mean'], marker='o', label=variant)
                    defined = rows['sd'].notna() & rows['mean'].notna()
                    if defined.any():
                        positions = np.arange(len(SCENARIOS))[defined]
                        axis.errorbar(positions, rows.loc[defined, 'mean'], yerr=rows.loc[defined, 'sd'], fmt='none', capsize=3)
            axis.set_xticks(range(len(SCENARIOS)), list(SCENARIOS), rotation=35, ha='right')
            axis.set_ylabel(ylabel)
            if axis.lines:
                axis.legend(fontsize=8)
        axis.set_title('TabM completed-fold means; error bars are sample SD, not confidence intervals')
        save(fig, f'robustness_{metric}.png')

    fig, axis = plt.subplots(figsize=(7, 6))
    if reliability.empty:
        empty(axis, 'Calibration becomes available after an outer fold completes.')
    else:
        for variant, group in reliability[reliability.scenario.eq('observed')].groupby('variant'):
            points = []
            for _, bucket in group.groupby('bin'):
                bucket = bucket[bucket.n.gt(0)]
                if not bucket.empty:
                    points.append((np.average(bucket.mean_probability, weights=bucket.n),
                                   np.average(bucket.fraction_positive, weights=bucket.n)))
            if points:
                axis.plot(*np.asarray(points).T, marker='o', label=variant)
        axis.plot([0, 1], [0, 1], '--', color='#777777', label='Perfect calibration')
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Observed AFIB fraction')
        axis.legend(fontsize=8)
    axis.set_title('Observed outer predictions: pooled reliability bins')
    save(fig, 'calibration.png')

    fig, axis = plt.subplots(figsize=(10, 5))
    loss_column = next((name for name in ('weighted_bce', 'loss', 'train_loss', 'training_loss', 'weighted_loss')
                        if name in histories), None)
    if histories.empty or loss_column is None:
        empty(axis, 'No saved epoch histories yet. Training curves appear as fits finish.')
    else:
        for blocks, group in histories.groupby('n_blocks'):
            epoch_column = 'epoch' if 'epoch' in group else None
            if epoch_column:
                # This plot is a descriptive trajectory, not a validation estimate.
                curve = group.groupby(epoch_column)[loss_column].mean()
                axis.plot(curve.index, curve.values, label=f'{blocks} blocks (mean recorded fits)')
        axis.set(xlabel='Epoch', ylabel='Training loss')
        if axis.lines:
            axis.legend(fontsize=8)
    axis.set_title('TabM tuning training histories; no early stopping or test-based selection')
    save(fig, 'training_loss.png')


def generate_reports(output):
    output = Path(output)
    config = read_json(output / 'run_config.json')
    reports, data = output / 'reports', output / 'output_data'
    reports.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    locations, pending = completed_locations(output, config)
    combined = {}
    identity_columns = ['model', 'variant', 'feature_set', 'training_arm', 'outer_fold', 'scenario', 'repeat']
    empty_columns = {
        'metrics': identity_columns + ['threshold_policy'] + list(METRICS),
        'patient_weighted_metrics': identity_columns + ['threshold_policy', 'n_patients'] + list(METRICS),
        'quality_subgroups': identity_columns + ['threshold_policy', 'subgroup'] + list(METRICS),
        'reliability': identity_columns + ['bin', 'n', 'mean_probability', 'fraction_positive'],
        'tuning': ['outer_fold', 'variant', 'stage', 'inner_fold', 'n_blocks', 'macro_f1'],
    }
    for name in ('metrics', 'patient_weighted_metrics', 'quality_subgroups', 'reliability', 'tuning'):
        frames = [pd.read_csv(path / f'{name}.csv') for path in locations]
        combined[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=empty_columns[name])
        # Overwrite empty snapshots too, so removed/invalid checkpoints cannot
        # leave stale aggregate tables looking like current completed results.
        combined[name].to_csv(data / f'{name}.csv', index=False)
    folds, summary = fold_summaries(combined['metrics'])
    folds.to_csv(data / 'fold_mean_metrics.csv', index=False)
    summary.to_csv(data / 'summary.csv', index=False)
    drops, comparisons = paired_effects(folds)
    drops.to_csv(data / 'paired_stress_drops.csv', index=False)
    comparisons.to_csv(data / 'paired_method_differences.csv', index=False)
    for name in ('patient_weighted_metrics', 'quality_subgroups'):
        labels = ['variant', 'scenario', 'threshold_policy'] + (['subgroup'] if name == 'quality_subgroups' else [])
        _, result = fold_summaries(combined[name], labels=labels)
        result.to_csv(data / f'{name}_summary.csv', index=False)
    selections = pd.DataFrame([read_json(path / 'selection.json') for path in locations])
    if selections.empty:
        selections = pd.DataFrame(columns=['outer_fold', 'variant', 'n_blocks', 'selected_middle', 'seconds'])
    selections.to_csv(data / 'selected_hyperparameters.csv', index=False)
    progress, histories = candidate_progress(output, config)
    progress.to_csv(data / 'tuning_progress.csv', index=False)
    if histories.empty:
        histories = pd.DataFrame(columns=['outer_fold', 'variant', 'n_blocks', 'inner_fold',
                                         'epoch', 'weighted_bce', 'seconds', 'completed'])
    histories.to_csv(data / 'training_history.csv', index=False)
    completed_inner = progress[progress.completed.eq(True)]
    candidate_groups = completed_inner.groupby(['outer_fold', 'variant', 'n_blocks']).inner_fold.agg(set)
    completed_candidates = int(sum(ids == set(range(config['n_inner_folds'])) for ids in candidate_groups))
    initial_candidates = int(sum(ids == set(range(config['n_inner_folds'])) and key[2] in config['initial_grid']
                                 for key, ids in candidate_groups.items()))
    fit_keys = {(int(row.outer_fold), row.variant, 'inner', int(row.n_blocks), int(row.inner_fold))
                for row in completed_inner.itertuples()}
    seconds = 0.0
    for path in locations:
        outer, variant = path.name.split('_', 1)
        selection = read_json(path / 'selection.json')
        seconds += float(selection.get('seconds', 0))
        for fit in read_json(path / 'fit_audit.json'):
            phase = 'outer_refit' if fit.get('phase') == 'outer_refit' else 'inner'
            fit_keys.add((int(outer), variant, phase, int(fit['n_blocks']), int(fit.get('inner_fold', -1))))
    baseline_status = 'unavailable'
    try:
        package = load_processed_data(PROJECT / 'processed_data')
        if package['manifest'] != config['manifest']:
            raise ValueError('Current processed-data manifest differs from this run.')
        baseline = []
        for outer in config['outer_folds']:
            train = package['folds'].outer_fold.ne(outer)
            baseline.append({'outer_fold': outer, **majority_baseline(
                package['targets'].loc[train, 'target_afib'], package['targets'].loc[~train, 'target_afib'])})
        pd.DataFrame(baseline).to_csv(data / 'majority_baseline.csv', index=False)
        baseline_status = 'computed from training labels only; current data manifest verified'
    except (OSError, ValueError, KeyError) as error:
        baseline_status = f'unavailable: {error}'
        pd.DataFrame(columns=['outer_fold'] + list(METRICS)).to_csv(data / 'majority_baseline.csv', index=False)
    requested = len(config['outer_folds']) * len(config['variants'])
    complete_scope = len(locations) == requested and requested > 0
    status = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'complete_tabm_plan': bool(complete_scope and full_protocol_settings(config)),
              'requested_scope_complete': complete_scope,
              'status': 'complete' if complete_scope else 'incomplete',
              'completed_outer_fold_variants': len(locations), 'requested_outer_fold_variants': requested,
              'complete_outer_folds': sum(all(any(path.name == f'{fold:02d}_{variant}' for path in locations)
                                              for variant in config['variants']) for fold in config['outer_folds']),
              'completed_inner_fits': sum(key[2] == 'inner' for key in fit_keys),
              'completed_outer_refits': sum(key[2] == 'outer_refit' for key in fit_keys),
              'n_fits': len(fit_keys), 'completed_tuning_candidates': completed_candidates,
              'completed_initial_candidates': initial_candidates,
              'initial_candidates_expected': requested * len(config['initial_grid']),
              'initial_inner_fits_expected': requested * len(config['initial_grid']) * config['n_inner_folds'],
              'minimum_fits_for_requested_scope': requested * (len(config['initial_grid']) * config['n_inner_folds'] + 1),
              'completed_fold_runtime_seconds': seconds, 'epochs_per_fit': config['epochs'],
              'stress_conditions_per_fold': len(expected_conditions(config['repeats'])),
              'pending': pending, 'majority_baseline': baseline_status,
              'error_bars': 'sample SD across outer-fold means; descriptive, not confidence intervals',
              'metric_aggregation': 'stress repeats averaged within each fold; undefined values propagate',
              'verification': 'Checkpoint coverage only; run check_results.py for independent numerical validation.'}
    write_json(reports / 'completion.json', status)
    _figures(reports, config, summary, combined['reliability'], progress, histories, locations)
    observed = summary[summary.scenario.eq('observed') & summary.threshold_policy.eq('fixed_0.50')
                       & summary.metric.isin(['macro_f1', 'sensitivity', 'specificity', 'brier'])]
    title = 'COMPLETE TABM PLAN' if status['complete_tabm_plan'] else 'INCOMPLETE TABM PLAN'
    requested_text = 'complete' if complete_scope else 'incomplete'
    report = f'''# TabM experiment report — {title}

Generated {status['generated_at_utc']}. Requested scope: **{requested_text}**.
**{len(locations)}/{requested} outer-fold/variant checkpoints** and **{status['complete_outer_folds']}/{len(config['outer_folds'])} complete outer folds** are available.
There are **{status['completed_inner_fits']} completed inner fits**, **{status['completed_outer_refits']} outer refits**, and **{completed_candidates} fully evaluated tuning candidates**.
The initial grid requires {status['initial_inner_fits_expected']} inner fits; endpoint refinement can require more.
Completing a reduced run does not establish completion of the project plan.

Only completed outer checkpoints enter performance summaries. Partial tuning scores are validation scores, not final model results. A single completed outer fold has undefined sample SD. Empty figures mean the relevant results are not available; no values have been estimated or fabricated.

## Settings and runtime

Requested variants: {', '.join(config['variants'])}. Outer folds: {config['outer_folds']}; inner folds: {config['n_inner_folds']}.
Initial block grid: {config['initial_grid']}; one-step endpoint refinement: {config['refine']}.
Every fit requests {config['epochs']} epochs and batch size {config['batch_size']}; stress repeats: {config['repeats']} ({status['stress_conditions_per_fold']} conditions per outer checkpoint).
Device: {config.get('resolved_device', config.get('device', 'unspecified'))}; seed: {config['seed']}.
Sum of completed-fold wall times: {seconds:.1f} seconds (not total elapsed experiment time).

```json
{json.dumps({'architecture': config.get('architecture'), 'optimizer': config.get('optimizer'), 'versions': config.get('versions')}, indent=2)}
```

## Observed performance at the fixed 0.50 threshold

{_table(observed[['variant', 'metric', 'mean', 'sd', 'n_folds', 'n_defined']])}

## Protocol and interpretation

All preprocessing, imputation, scaling and augmentation are fitted using training patients only. Block count is selected using mean inner-fold macro-F1; ties favor fewer blocks. The secondary threshold is selected from inner out-of-fold predictions to target sensitivity 0.90, then frozen for every outer condition. It is an exploratory threshold, not a clinical acceptance criterion.

Stress repeats are averaged within each outer fold before calculating the outer-fold mean and sample SD. Undefined metrics propagate through both aggregation stages. Error bars are descriptive sample SD, not confidence intervals; outer folds share training data. Positive paired stress deterioration means performance becomes worse (stress minus observed for Brier; observed minus stress otherwise). Method differences are left minus right on the intersection of completed folds, for both threshold policies.

The reports include B–A, D–C, C–A and D–B plus B/availability-observed, D/availability-augmented and both duplicate controls when paired results exist. Patient-weighted metrics give each patient total weight one. Quality groups use recorded source annotations; an absent flag does not prove a clean signal. The simulated tabular corruptions are not calibrated ECG waveform artefacts.

Majority baseline: {baseline_status}. It can be computed before any model finishes and is not evidence of completed model training.

## Figures

![Outer-fold progress](progress.png)

![Macro-F1 robustness](robustness_macro_f1.png)

![Sensitivity robustness](robustness_sensitivity.png)

![Brier-score robustness](robustness_brier.png)

![Calibration](calibration.png)

![Training](training_loss.png)

Calibration pools observed test records into ten probability bins across completed folds. Training curves average recorded histories at each epoch; they do not select the final model.

## Saved evidence

The sibling `output_data/` directory contains raw fold tables, predictions, tuning candidates, `summary.csv`, `fold_mean_metrics.csv`, paired contrasts, patient-weighted and quality-group summaries, majority baseline, and tuning progress. Checkpoint models are stored separately in `checkpoints/`. `completion.json` records missing checkpoints and precise coverage. Source hashes, processed-data manifest, and runtime versions are preserved in `../run_config.json`.

These report-generation coverage checks do not replace the independent `check_results.py` audit, which recalculates saved prediction metrics and checks patient separation, complete scenario coverage and settings. No results from other classifiers are required for this TabM report.
'''
    (reports / 'report.md').write_text(report, encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results' / 'tabm')
    args = parser.parse_args()
    print(json.dumps(generate_reports(args.output), indent=2))

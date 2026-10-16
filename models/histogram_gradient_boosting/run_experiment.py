"""COMP90051 HGB experiment runner. Run from repository via absolute path or relative path.

Needs models/histogram_gradient_boosting/model.py with create_model(max_depth).
Uses the project's training-fold-only processing and manual evaluation functions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from afib.data_preparation import load_prepared_split
from afib.paths import processed_directory
from afib.evaluation import binary_metrics
from afib.preprocessing import SCENARIOS
from models.histogram_gradient_boosting.model import create_model


def grid_near_boundary(grid, winner):
    """One same-size shifted three-point grid, derived only from inner folds."""
    if winner == grid[-1]:
        step = grid[-1] - grid[-2]
        return sorted(set([winner - step, winner, winner + step]))
    if winner == grid[0]:
        step = grid[1] - grid[0]
        return sorted(set([max(1, winner - step), winner, winner + step]))
    return None


def inner_scores(arm, outer, depths, cache):
    rows = []
    for depth in depths:
        for inner in range(3):
            if inner not in cache:
                cache[inner] = load_prepared_split(processed_directory(ROOT), outer_fold=outer,
                                                  inner_fold=inner, arm=arm)
            split = cache[inner]
            model = create_model(max_depth=int(depth))
            model.fit(split['X_train'], split['y_train'], sample_weight=split['sample_weight'])
            probabilities = model.predict_proba(split['X_valid'])[:, 1]
            metrics = binary_metrics(split['y_valid'], probabilities)
            rows.append({'arm': arm, 'outer_fold': outer, 'max_depth': int(depth),
                         'inner_fold': inner, 'macro_f1': metrics['macro_f1']})
            print(f'  depth={depth} inner={inner}: macro-F1={metrics["macro_f1"]:.4f}', flush=True)
    return rows


def choose_depth(rows, candidates):
    scores = pd.DataFrame(rows).groupby('max_depth')['macro_f1'].mean()
    # Ties prefer lower depth (simpler model).
    return min(candidates, key=lambda d: (-float(scores[d]), d))


def run_one(arm, outer, output, refine=True, repeats=3, scenario_names=None):
    folder = output / f'{arm}_fold{outer}'
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / 'complete.json'
    if completed.exists():
        print(f'Skipping complete {arm} fold {outer}', flush=True)
        return
    print(f'\nRunning arm={arm}, outer_fold={outer}', flush=True)
    cache = {}
    initial = [2, 3, 4]
    tuning = inner_scores(arm, outer, initial, cache)
    winner = choose_depth(tuning, initial)
    extra_grid = grid_near_boundary(initial, winner) if refine else None
    if extra_grid is not None:
        added = [depth for depth in extra_grid if depth not in initial]
        if added:
            print(f'Refining inner grid with {added}; no outer-test results used.', flush=True)
            tuning.extend(inner_scores(arm, outer, added, cache))
        # Select from shifted 3-value grid (not arbitrary expansion).
        winner = choose_depth(tuning, extra_grid)
    pd.DataFrame(tuning).to_csv(folder / 'inner_tuning.csv', index=False)
    del cache

    base = load_prepared_split(processed_directory(ROOT), outer_fold=outer,
                               inner_fold=None, arm=arm, scenario='observed')
    model = create_model(max_depth=winner)
    model.fit(base['X_train'], base['y_train'], sample_weight=base['sample_weight'])
    y = np.asarray(base['y_valid'], dtype=int)
    ids = base['y_valid'].index.to_numpy()
    scenarios = scenario_names if scenario_names is not None else list(SCENARIOS)
    metric_rows = []
    prediction_rows = []
    for scenario in scenarios:
        repeat_ids = [0] if scenario == 'observed' else list(range(repeats))
        for repeat in repeat_ids:
            split = base if scenario == 'observed' else load_prepared_split(
                processed_directory(ROOT), outer_fold=outer, inner_fold=None,
                arm=arm, scenario=scenario, repeat=repeat)
            if not np.array_equal(split['y_valid'].index.to_numpy(), ids):
                raise RuntimeError('Held-out ECG order changed between scenarios.')
            proba = model.predict_proba(split['X_valid'])[:, 1]
            metrics = binary_metrics(y, proba)
            metric_rows.append({'arm': arm, 'outer_fold': outer, 'scenario': scenario,
                                'repeat': repeat, 'max_depth': winner, **metrics})
            prediction_rows.extend({'arm': arm, 'outer_fold': outer, 'scenario': scenario,
                                    'repeat': repeat, 'ecg_id': int(ids[i]),
                                    'y_true': int(y[i]), 'prob_afib': float(proba[i])}
                                   for i in range(len(ids)))
            print(f'  {scenario} repeat={repeat}: macro-F1={metrics["macro_f1"]:.4f}', flush=True)
    pd.DataFrame(metric_rows).to_csv(folder / 'metrics.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(folder / 'predictions.csv', index=False)
    completed.write_text(json.dumps({'arm': arm, 'outer_fold': outer, 'best_depth': winner,
                                     'scenario_names': scenarios, 'repeats': repeats,
                                     'refine': refine}, indent=2), encoding='utf-8')
    print(f'Saved {folder}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=list('ABCD'), default=['A'])
    parser.add_argument('--outer-folds', nargs='+', type=int, default=[0])
    parser.add_argument('--scenarios', nargs='+', choices=list(SCENARIOS), default=None)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'histogram_gradient_boosting' / 'formal')
    args = parser.parse_args()
    if any(k not in range(10) for k in args.outer_folds) or not 1 <= args.repeats <= 3:
        parser.error('Outer folds must be 0..9, repeats must be 1..3.')
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for outer in args.outer_folds:
            run_one(arm, outer, args.output, refine=not args.no_refine,
                    repeats=args.repeats, scenario_names=args.scenarios)
    files = sorted(args.output.glob('*_fold*/metrics.csv'))
    if files:
        all_metrics = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        all_metrics.to_csv(args.output / 'all_metrics.csv', index=False)
        observed = all_metrics[all_metrics.scenario.eq('observed')]
        stats = observed.groupby('arm')['macro_f1'].agg(['count', 'mean', 'std'])
        print('\nObserved scores (descriptive; report only complete 10 folds):\n', stats)


if __name__ == '__main__':
    main()

"""Run the patient-grouped logistic regression experiments.

The default run uses 10 outer and 3 inner folds, arms A-D, feature-availability
and duplicate controls, and the separate ridge-filling experiment.
Completed fold/variant checkpoints are reused when the run configuration matches.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

PROJECT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.preprocessing import (ARMS, SCENARIOS, RAW_COLUMNS, build_features,
    construct_training_views, transform_features, perturb_measurements, fit_preprocessor)
from afib.data_preparation import load_processed_data
from afib.evaluation import (binary_metrics, choose_sensitivity_threshold,
    assert_patient_disjoint, source_quality_groups, quality_subgroup_metrics,
    majority_baseline, summarise_paired_drops, validate_scenario_repeats)
from afib.imputation import fit_imputer, impute_measurements, mask_observed_measurements, score_recovery
from model import LogisticRegression

VARIANTS = ['A', 'B', 'C', 'D', 'availability_observed', 'availability_augmented',
            'duplicate_original', 'duplicate_enhanced', 'ridge']
METRICS = ['macro_f1', 'sensitivity', 'specificity', 'precision', 'npv',
           'balanced_accuracy', 'accuracy', 'auroc', 'average_precision', 'brier']


def json_write(path, value):
    def convert(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert), encoding='utf-8')


def split_ids(package, outer, inner=None):
    folds = package['folds']
    training = folds.outer_fold.ne(outer)
    validation = ~training
    if inner is not None:
        assignments = folds[f'inner_fold_when_outer_{outer}']
        validation = training & assignments.eq(inner)
        training &= assignments.ne(inner)
    train_ids, valid_ids = folds.index[training], folds.index[validation]
    assert_patient_disjoint(folds.loc[train_ids, 'patient_id'], folds.loc[valid_ids, 'patient_id'])
    if inner is not None:
        assert not folds.loc[train_ids.union(valid_ids), 'outer_fold'].eq(outer).any()
    return train_ids, valid_ids


def identity(variant, outer):
    enhanced = variant in ['B', 'D', 'duplicate_enhanced'] or variant.startswith('availability_')
    augmented = variant in ['C', 'D', 'availability_augmented']
    return {'model': 'logistic_regression', 'variant': variant,
            'feature_set': 'enhanced' if enhanced else 'original',
            'training_arm': 'augmented' if augmented else 'observed', 'outer_fold': outer}


def availability_features(raw):
    features = build_features(raw, True)
    # Keep derived missingness patterns while removing their numerical values.
    derived = [c for c in features if c not in RAW_COLUMNS]
    features[derived] = features[derived].where(features[derived].isna(), 0.0)
    return features


def training_pipeline(package, train_ids, variant):
    raw = package['raw'].loc[train_ids]
    y = package['targets'].loc[train_ids, 'target_afib']
    seed = package['manifest']['seed']
    if variant in ARMS or variant.startswith('duplicate_') or variant.startswith('availability_'):
        enhanced = variant in ('B', 'D', 'duplicate_enhanced') or variant.startswith('availability_')
        augmented = variant in ('C', 'D', 'availability_augmented')
        train = construct_training_views(raw, y, enhanced=enhanced, augment=augmented, seed=seed)
        if variant.startswith('availability_'):
            # Zero derived values after scaling; retain their missingness flags.
            train['X'].loc[:, [c for c in train['X'] if c not in RAW_COLUMNS
                and not c.endswith('__missing') and '__category_' not in c
                and not c.endswith('__unknown')]] = 0.0
        if variant.startswith('duplicate_'):
            original_X = train['X'].xs('observed', level='view')
            train['X'] = pd.concat({'observed': original_X, 'duplicate': original_X}, names=['view','ecg_id'])
            train['y'] = pd.concat({'observed': y, 'duplicate': y}, names=['view','ecg_id'])
            w = train['sample_weight'].xs('observed', level='view') / 2
            train['sample_weight'] = pd.concat({'observed': w, 'duplicate': w}, names=['view','ecg_id'])
        train['variant'] = variant
    elif variant == 'ridge':
        base = construct_training_views(raw, y, enhanced=False, seed=seed)
        filler = fit_imputer(raw, method='ridge', alpha=1., max_predictors=8, min_observed=20)
        filled = impute_measurements(raw, filler)
        fitted = fit_preprocessor(filled['raw'])
        X = transform_features(filled['raw'], fitted, missing_mask=filled['missing_mask'])
        train = {'X': X, 'y': y, 'sample_weight': base['sample_weight'].to_numpy(),
                 'preprocessor': fitted, 'perturb_iqr': base['perturb_iqr'],
                 'filler': filler, 'filling_training_audit': filled['audit'],
                 'augmentation_audit': [], 'variant': variant}
    else:
        raise ValueError('Unknown variant')
    train['train_ids'] = train_ids
    return train


def transform_heldout(raw, pipeline):
    variant = pipeline['variant']
    if variant == 'ridge':
        filled = impute_measurements(raw, pipeline['filler'])
        return transform_features(filled['raw'], pipeline['preprocessor'], missing_mask=filled['missing_mask'])
    enhanced = variant in ('B', 'D', 'duplicate_enhanced') or variant.startswith('availability_')
    X = transform_features(build_features(raw, enhanced), pipeline['preprocessor'])
    if variant.startswith('availability_'):
        derived = [c for c in X if c not in RAW_COLUMNS and not c.endswith('__missing')
                   and '__category_' not in c and not c.endswith('__unknown')]
        X.loc[:, derived] = 0.0
    return X


def select_C(scores):
    # Break exact ties in favour of stronger regularization.
    return min(scores, key=lambda C: (-scores[C], C))


def refinement_grid(grid, best):
    grid = sorted(grid)
    if best == grid[0]: return [grid[0] / 10, grid[0], grid[1]]
    if best == grid[-1]: return [grid[-2], grid[-1], grid[-1] * 10]
    return None


def tune(package, outer, variant, grid, refine=True):
    cache = []
    fits, rows, predicted = [], [], {}
    for inner in range(3):
        train_ids, valid_ids = split_ids(package, outer, inner)
        pipeline = training_pipeline(package, train_ids, variant)
        X = np.asarray(pipeline['X'], dtype=np.float32, order='C')
        V = np.asarray(transform_heldout(package['raw'].loc[valid_ids], pipeline), dtype=np.float32, order='C')
        cache.append((X, np.asarray(pipeline['y']), np.asarray(pipeline['sample_weight']),
                      V, package['targets'].loc[valid_ids, 'target_afib'].to_numpy(), valid_ids))
        del pipeline
    warm_starts = [None] * 3
    def evaluate(values, stage):
        for C in sorted(values):
            if C in predicted: continue
            predicted[C] = []
            for inner, (X, y, weights, V, truth, ids) in enumerate(cache):
                estimator = LogisticRegression(C).fit(X, y, weights, initial=warm_starts[inner])
                warm_starts[inner] = estimator.theta_.copy()
                score = estimator.predict_proba(V)
                metric = binary_metrics(truth, score)['macro_f1']
                rows.append({'outer_fold': outer, 'variant': variant, 'stage': stage,
                             'inner_fold': inner, 'C': C, 'macro_f1': metric})
                fits.append({'phase': 'inner', 'inner_fold': inner, **estimator.audit_})
                predicted[C].append(pd.DataFrame({'ecg_id': ids, 'inner_fold': inner,
                    'patient_id': package['folds'].loc[ids, 'patient_id'].to_numpy(),
                    'y_true': truth, 'score': score}))
                print(f'  {variant} outer={outer} inner={inner} C={C:g} F1={metric:.4f}', flush=True)
    evaluate(grid, 'initial')
    def means(values):
        return {C: float(np.mean([row['macro_f1'] for row in rows if row['C'] == C])) for C in values}
    best = select_C(means(grid))
    final_grid = refinement_grid(grid, best) if refine else None
    if final_grid is not None:
        evaluate(final_grid, 'refinement')
        best = select_C(means(final_grid))
    else:
        final_grid = sorted(grid)
    oof = pd.concat(predicted[best], ignore_index=True).sort_values('ecg_id')
    expected = package['folds'].index[package['folds'].outer_fold.ne(outer)]
    if set(oof.ecg_id) != set(expected) or oof.ecg_id.duplicated().any():
        raise ValueError('Selected inner OOF predictions do not cover outer training records exactly once.')
    chosen = {'C': best, 'initial_grid': grid, 'final_grid': final_grid,
              'refined': final_grid != sorted(grid), 'selected_middle': best == sorted(final_grid)[1],
              'inner_mean_macro_f1': means(final_grid)[best]}
    del cache, predicted
    gc.collect()
    return chosen, oof, rows, fits


def patient_weighted_metrics(y, probability, patients, threshold):
    # Give each patient total weight one, regardless of their ECG count.
    ids = pd.Series(patients)
    w = ids.map(1 / ids.value_counts()).to_numpy(dtype=float)
    y = np.asarray(y)
    pred = np.asarray(probability) >= threshold
    tp, tn = w[pred & (y == 1)].sum(), w[~pred & (y == 0)].sum()
    fp, fn = w[pred & (y == 0)].sum(), w[~pred & (y == 1)].sum()
    div = lambda a,b: float(a/b) if b else float('nan')
    return {'n_patients': len(set(patients)), 'weighted_tp': tp, 'weighted_tn': tn,
            'weighted_fp': fp, 'weighted_fn': fn,
            'macro_f1': .5 * (div(2*tp, 2*tp+fp+fn) + div(2*tn, 2*tn+fp+fn)),
            'sensitivity': div(tp, tp+fn), 'specificity': div(tn, tn+fp),
            'balanced_accuracy': .5*(div(tp,tp+fn)+div(tn,tn+fp)),
            'accuracy': div(tp+tn, w.sum()), 'brier': float(w @ ((probability-y)**2)/w.sum())}


def reliability_bins(y, scores, identity_row):
    bins = np.minimum((np.asarray(scores) * 10).astype(int), 9)
    rows = []
    for index in range(10):
        selected = bins == index
        rows.append({**identity_row, 'bin': index, 'n': int(selected.sum()),
                     'mean_probability': float(np.mean(scores[selected])) if selected.any() else np.nan,
                     'fraction_positive': float(np.mean(y[selected])) if selected.any() else np.nan})
    return rows


def recovery_experiment(package, train_ids, test_ids, outer):
    train, truth = package['raw'].loc[train_ids], package['raw'].loc[test_ids]
    methods = {name: fit_imputer(train, method=name) for name in ['median', 'ridge']}
    rows, detail, audits, groups = [], [], [], []
    quality = source_quality_groups(package['metadata'].reset_index()).loc[test_ids]
    labels = package['targets'].loc[test_ids, 'target_afib']
    for condition in ['cells_0.10', 'missing_1', 'missing_3', 'missing_6']:
        for repeat in range(3):
            options = {'fraction': .1} if condition == 'cells_0.10' else {'missing_leads': int(condition[-1])}
            masked = mask_observed_measurements(truth, repeat=repeat, seed=package['manifest']['seed'], **options)
            for method, fit in methods.items():
                filled = impute_measurements(masked['raw'], fit)
                scored = score_recovery(truth, filled['raw'], masked['scoring_mask'], fit)
                key = {'outer_fold': outer, 'method': method, 'scenario': condition, 'repeat': repeat}
                rows.append({**key, **scored['summary']})
                detail.append(scored['per_feature'].reset_index().assign(**key))
                audits.append({**key, 'mask': masked['audit'], 'filling': filled['audit']})
                memberships = {'AFIB': labels.eq(1), 'without_AFIB': labels.eq(0),
                               'recorded_quality_flag': quality.eq('recorded_quality_flag'),
                               'no_recorded_quality_flag': quality.eq('no_recorded_quality_flag')}
                for name, mask in memberships.items():
                    ids = test_ids[mask.to_numpy()]
                    sub = score_recovery(truth.loc[ids], filled['raw'].loc[ids], masked['scoring_mask'].loc[ids], fit)
                    groups.append({**key, 'subgroup': name, 'n_ecgs': len(ids), **sub['summary']})
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True), audits, pd.DataFrame(groups)


def run_variant(package, outer, variant, args, output):
    location = output / 'output_data' / 'folds' / f'{outer:02d}_{variant}'
    location.mkdir(parents=True, exist_ok=True)
    if (location / 'complete.json').exists():
        print(f'Resume: fold {outer} {variant} already complete', flush=True)
        return
    started = time.perf_counter()
    chosen, oof, tuning, fits = tune(package, outer, variant, args.c_grid, not args.no_refine)
    key = identity(variant, outer)
    oof = oof.assign(**key)
    threshold = choose_sensitivity_threshold(oof.y_true, oof.score, .90)
    if threshold['status'] != 'selected_on_inner_oof': raise ValueError('Threshold unavailable')
    train_ids, test_ids = split_ids(package, outer)
    pipeline = training_pipeline(package, train_ids, variant)
    model = LogisticRegression(chosen['C']).fit(pipeline['X'], pipeline['y'], pipeline['sample_weight'])
    fits.append({'phase': 'outer_refit', **model.audit_})
    # Save the fitted preprocessing and model together for this outer fold.
    with (location / 'pipeline.pkl').open('wb') as handle:
        pickle.dump({'model': model, 'preprocessor': pipeline['preprocessor'], 'variant': variant,
                     'perturb_iqr': pipeline['perturb_iqr'], 'filler': pipeline.get('filler'),
                     'train_ids': train_ids.to_numpy(), 'threshold': threshold}, handle)
    quality = source_quality_groups(package['metadata'].reset_index()).loc[test_ids]
    y = package['targets'].loc[test_ids, 'target_afib'].to_numpy()
    patients = package['folds'].loc[test_ids, 'patient_id'].to_numpy()
    raw = package['raw'].loc[test_ids]
    metrics, subgroup, patient_rows, calibration, corruption = [], [], [], [], []
    scenarios = list(SCENARIOS) if variant != 'ridge' else ['observed', 'missing_3']
    predictions_path = location / 'outer_predictions.csv.gz'
    # Replace partial predictions when resuming an incomplete fold.
    import gzip
    with gzip.open(predictions_path, 'wt', encoding='utf-8', newline='') as handle:
        first = True
        for scenario in scenarios:
            for repeat in ([0] if scenario == 'observed' else range(args.repeats)):
                changed, audit = perturb_measurements(raw, pipeline['perturb_iqr'],
                    **SCENARIOS[scenario], repeat=repeat, seed=package['manifest']['seed'], return_audit=True)
                scores = model.predict_proba(transform_heldout(changed, pipeline))
                condition = {**key, 'scenario': scenario, 'repeat': repeat}
                pd.DataFrame({'ecg_id': test_ids, 'patient_id': patients, 'y_true': y,
                    'score': scores, 'quality_group': quality.to_numpy()}).assign(**condition).to_csv(handle, index=False, header=first)
                first = False
                corruption.append({**condition, **audit})
                for policy, decision in [('fixed_0.50', .5), ('inner_sensitivity_0.90', threshold['threshold'])]:
                    meta = {**condition, 'threshold_policy': policy}
                    metrics.append({**meta, **binary_metrics(y, scores, decision)})
                    patient_rows.append({**meta, **patient_weighted_metrics(y, scores, patients, decision)})
                    subgroup.extend({**meta, **row} for row in quality_subgroup_metrics(y, scores, quality, decision).to_dict('records'))
                calibration.extend(reliability_bins(y, scores, condition))
                print(f'  Tested {variant} fold={outer} {scenario} repeat={repeat}', flush=True)
    pd.DataFrame(metrics).to_csv(location / 'metrics.csv', index=False)
    pd.DataFrame(patient_rows).to_csv(location / 'patient_weighted_metrics.csv', index=False)
    pd.DataFrame(subgroup).to_csv(location / 'quality_subgroups.csv', index=False)
    pd.DataFrame(calibration).to_csv(location / 'reliability.csv', index=False)
    oof.to_csv(location / 'inner_oof_predictions.csv.gz', index=False)
    pd.DataFrame(tuning).to_csv(location / 'tuning.csv', index=False)
    json_write(location / 'fit_audit.json', fits)
    json_write(location / 'perturbation_audit.json', {'training': pipeline['augmentation_audit'], 'test': corruption})
    json_write(location / 'selection.json', {**key, **chosen, 'secondary_threshold': threshold,
                                            'seconds': time.perf_counter()-started})
    if variant == 'ridge':
        recovery, feature_recovery, audits, recovery_groups = recovery_experiment(package, train_ids, test_ids, outer)
        recovery.to_csv(location / 'recovery.csv', index=False)
        feature_recovery.to_csv(location / 'recovery_by_feature.csv', index=False)
        recovery_groups.to_csv(location / 'recovery_by_subgroup.csv', index=False)
        json_write(location / 'recovery_audit.json', audits)
    json_write(location / 'complete.json', {'completed': True})
    print(f'COMPLETE fold={outer} variant={variant} C={chosen["C"]} seconds={time.perf_counter()-started:.1f}', flush=True)
    del pipeline, model
    gc.collect()


def prediction_export_directory(output, args):
    official = PROJECT / 'results/logistic_regression'
    complete = (set(args.outer_folds) == set(range(10)) and args.repeats == 3
                and set(args.variants) == set(VARIANTS))
    if Path(output).resolve() == official.resolve() and complete:
        return official / 'predictions'
    if Path(output).resolve() == official.resolve():
        return Path(output) / 'pilots/incomplete_run/predictions'
    return Path(output) / 'predictions'


def export_predictions(locations, export):
    export.mkdir(parents=True, exist_ok=True)
    import gzip
    for name in ['inner_oof_predictions', 'outer_predictions']:
        with gzip.open(export / f'{name}.csv.gz', 'wt', encoding='utf-8', newline='') as dest:
            first = True
            for path in locations:
                if path.name.split('_', 1)[1] not in ARMS:
                    continue
                for chunk in pd.read_csv(path / f'{name}.csv.gz', chunksize=50000):
                    chunk.to_csv(dest, index=False, header=first)
                    first = False


def summarise(output, args, package):
    (output / 'output_data').mkdir(parents=True, exist_ok=True)
    (output / 'reports').mkdir(parents=True, exist_ok=True)
    locations = [output/'output_data'/'folds'/f'{fold:02d}_{variant}' for fold in args.outer_folds for variant in args.variants]
    selections = pd.DataFrame([json.loads((p/'selection.json').read_text()) for p in locations])
    selections.to_csv(output/'output_data'/'selected_hyperparameters.csv', index=False)
    combined = {}
    for name in ['metrics','patient_weighted_metrics','quality_subgroups','reliability','tuning']:
        combined[name] = pd.concat([pd.read_csv(p/f'{name}.csv') for p in locations], ignore_index=True)
        combined[name].to_csv(output/'output_data'/f'{name}.csv', index=False)
    metrics = combined['metrics']
    for (variant, fold), group in metrics.groupby(['variant','outer_fold']):
        expected = SCENARIOS if variant != 'ridge' else ['observed','missing_3']
        validate_scenario_repeats(group.drop_duplicates(['scenario','repeat']), expected, repeats=range(args.repeats))
    summaries = []
    # Average stress repeats within folds, then compute SD across folds.
    fold_means = metrics.groupby(['variant','outer_fold','scenario','threshold_policy'])[METRICS].mean().reset_index()
    fold_means.to_csv(output/'output_data'/'fold_mean_metrics.csv', index=False)
    for keys, group in fold_means.groupby(['variant','scenario','threshold_policy']):
        for metric in METRICS:
            values = group[metric].to_numpy()
            summaries.append(dict(zip(['variant','scenario','threshold_policy'],keys)) | {
                'metric':metric,'mean':float(np.mean(values)), 'sd':float(np.std(values,ddof=1)) if len(values)>1 else np.nan,
                'n_folds':len(values),'n_defined':int(np.isfinite(values).sum())})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'output_data'/'summary.csv', index=False)
    if len(args.outer_folds) == 10:
        drops = []
        for (variant, policy), group in metrics.groupby(['variant','threshold_policy']):
            for metric in METRICS:
                _, result = summarise_paired_drops(group, metric, higher_is_better=metric!='brier')
                drops.append(result.assign(variant=variant,threshold_policy=policy))
        pd.concat(drops,ignore_index=True).to_csv(output/'output_data'/'paired_stress_drops.csv',index=False)
    comparisons=[]
    for left,right in [('B','A'),('D','C'),('C','A'),('D','B'),
                       ('B','availability_observed'),('D','availability_augmented'),
                       ('duplicate_original','A'),('duplicate_enhanced','B'),('ridge','A')]:
        if left not in args.variants or right not in args.variants: continue
        for scenario in set(fold_means.loc[fold_means.variant.eq(left),'scenario']):
            for metric in METRICS:
                a=fold_means[(fold_means.variant.eq(left)) & fold_means.scenario.eq(scenario)
                             & fold_means.threshold_policy.eq('fixed_0.50')].set_index('outer_fold')[metric]
                b=fold_means[(fold_means.variant.eq(right)) & fold_means.scenario.eq(scenario)
                             & fold_means.threshold_policy.eq('fixed_0.50')].set_index('outer_fold')[metric]
                delta=(a-b).to_numpy()
                comparisons.append({'comparison':f'{left} minus {right}','scenario':scenario,'metric':metric,
                    'mean_difference':float(np.mean(delta)),'sd_difference':float(np.std(delta,ddof=1)) if len(delta)>1 else np.nan,
                    'n_folds':len(delta),'direction':'lower better' if metric=='brier' else 'higher better'})
    pd.DataFrame(comparisons).to_csv(output/'output_data'/'paired_method_differences.csv',index=False)
    baseline=[]
    for outer in args.outer_folds:
        train,test=split_ids(package,outer)
        baseline.append({'outer_fold':outer,**majority_baseline(package['targets'].loc[train,'target_afib'],package['targets'].loc[test,'target_afib'])})
    pd.DataFrame(baseline).to_csv(output/'output_data'/'majority_baseline.csv',index=False)
    for kind in ['patient_weighted_metrics','quality_subgroups']:
        frame=combined[kind]
        labels=['variant','scenario','threshold_policy']+(['subgroup'] if kind=='quality_subgroups' else [])
        values=[m for m in METRICS if m in frame]
        per_fold=frame.groupby(labels+['outer_fold'])[values].mean().reset_index()
        result=per_fold.groupby(labels)[values].agg(['mean','std','count'])
        result.columns=['_'.join(c) for c in result.columns]
        result.to_csv(output/'output_data'/f'{kind}_summary.csv')
    # Export A-D predictions for the team's three-model evaluation notebook.
    export_predictions(locations, prediction_export_directory(output, args))
    if 'ridge' in args.variants:
        paths=[p for p in locations if p.name.endswith('_ridge')]
        for name in ['recovery','recovery_by_feature','recovery_by_subgroup']:
            pd.concat([pd.read_csv(p/f'{name}.csv') for p in paths],ignore_index=True).to_csv(output/'output_data'/f'{name}.csv',index=False)
        recovery=pd.read_csv(output/'output_data'/'recovery.csv')
        means=recovery.groupby(['method','scenario','outer_fold']).cell_mean_mae_in_training_iqr.mean().reset_index()
        means.groupby(['method','scenario']).cell_mean_mae_in_training_iqr.agg(['mean','std','count']).to_csv(output/'output_data'/'recovery_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=summary[(summary.threshold_policy=='fixed_0.50') & summary.variant.isin(ARMS)]
    for metric in ['macro_f1','sensitivity','brier']:
        fig,axis=plt.subplots(figsize=(12,5))
        scenarios=list(SCENARIOS)
        for offset,arm in enumerate(['A','B','C','D']):
            if arm not in args.variants:continue
            frame=selected[(selected.metric==metric)&(selected.variant==arm)].set_index('scenario').reindex(scenarios)
            axis.errorbar(np.arange(len(scenarios))+(offset-1.5)*.06,frame['mean'],yerr=frame['sd'],marker='o',capsize=3,label=arm)
        axis.set_xticks(np.arange(len(scenarios)),scenarios,rotation=35,ha='right')
        axis.set_ylabel(metric); axis.set_title('Logistic regression: outer-fold mean +/- sample SD')
        axis.legend(title='Arm');axis.grid(alpha=.2);fig.tight_layout()
        fig.savefig(output/'reports'/f'{metric}.png',dpi=160);plt.close(fig)
    audits=[row for p in locations for row in json.loads((p/'fit_audit.json').read_text())]
    tuning_status=selections.groupby('variant').selected_middle.agg(['sum','count']).to_dict('index')
    full=set(args.outer_folds)==set(range(10)) and args.repeats==3 and set(args.variants)==set(VARIANTS)
    json_write(output/'reports'/'completion.json',{'complete_logistic_plan':full,'classifiers_completed':['logistic_regression'],
        'other_classifiers_pending':True,'n_fits':len(audits),'all_fits_converged':all(x['converged'] for x in audits),
        'max_gradient':max(x['max_abs_gradient'] for x in audits),'middle_selection_counts':tuning_status,
        'all_variants':args.variants,'outer_folds':args.outer_folds,'stress_repeats':args.repeats,
        'error_bars':'sample SD of outer-fold means; descriptive, not clinical confidence intervals',
        'data_unchanged':load_processed_data(PROJECT/'processed_data')['manifest']==package['manifest']})
    print('Observed primary results:',flush=True)
    print(summary[(summary.scenario=='observed')&(summary.threshold_policy=='fixed_0.50')
                  & summary.metric.isin(['macro_f1','sensitivity','specificity'])].to_string(index=False),flush=True)


def run_fold(outer, args):
    package = load_processed_data(PROJECT/'processed_data')
    for variant in args.variants:
        run_variant(package, outer, variant, args, args.output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    parser.add_argument('--outer-folds',nargs='+',type=int,default=list(range(10)))
    parser.add_argument('--c-grid',nargs=3,type=float,default=[.01,.1,1.])
    parser.add_argument('--no-refine',action='store_true')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--workers',type=int,default=1,choices=[1,2])
    parser.add_argument('--output',type=Path,default=PROJECT/'results/logistic_regression')
    args=parser.parse_args()
    if (len(set(args.outer_folds))!=len(args.outer_folds) or not set(args.outer_folds).issubset(range(10))
        or len(set(args.variants))!=len(args.variants) or args.repeats<1 or args.repeats>3
        or not all(np.isfinite(c) and c>0 for c in args.c_grid) or len(set(args.c_grid))!=3):
        parser.error('Choose unique valid folds/variants, 1..3 repeats and three distinct positive C values.')
    args.c_grid=sorted(args.c_grid)
    args.output.mkdir(parents=True,exist_ok=True)
    for var,folder in [('MPLCONFIGDIR','matplotlib'),('IPYTHONDIR','ipython')]:
        os.environ[var]=str(PROJECT/'.runtime'/folder)
    package=load_processed_data(PROJECT/'processed_data')
    signature={'variants':args.variants,'outer_folds':args.outer_folds,'c_grid':args.c_grid,
        'refine':not args.no_refine,'repeats':args.repeats,'manifest':package['manifest'],
        'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('model.py'),PROJECT/'src/afib/preprocessing.py',
             PROJECT/'src/afib/imputation.py',PROJECT/'src/afib/evaluation.py']},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    config=args.output/'run_config.json'
    if config.exists() and json.loads(config.read_text())!=signature:
        raise ValueError('Existing output has different code/data/configuration; choose a new --output folder.')
    json_write(config,signature)
    if args.workers == 1:
        for fold in args.outer_folds:
            for variant in args.variants:
                run_variant(package,fold,variant,args,args.output)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_fold,fold,args) for fold in args.outer_folds]
            for future in futures: future.result()
    summarise(args.output,args,package)


if __name__=='__main__':
    main()

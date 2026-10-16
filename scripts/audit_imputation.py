"""Check imputation on one existing inner partition, without classifier claims.

This is a real-data engineering check. It does not run the course's complete
nested model experiment and never evaluates an outer-test fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.preprocessing import RAW_COLUMNS, CATEGORY_COLUMNS, transform_features, fit_preprocessor
from afib.data_preparation import load_processed_data
from afib.paths import processed_directory
from afib.evaluation import assert_patient_disjoint, source_quality_groups
from afib.imputation import (
    DEFAULT_IMPUTATION_TARGETS, fit_imputer, impute_measurements,
    mask_observed_measurements, score_recovery,
)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    def finite(obj):
        if isinstance(obj, dict):
            return {str(k): finite(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [finite(v) for v in obj]
        if isinstance(obj, np.generic):
            return finite(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False), encoding='utf-8')


def validate_folds(folds):
    if not folds.index.is_unique or folds.index.hasnans or folds.patient_id.isna().any():
        raise ValueError('Unique ECGs with patient IDs are required.')
    if set(folds.outer_fold) != set(range(10)):
        raise ValueError('The stored protocol requires ten outer folds.')
    if not folds.groupby('patient_id').outer_fold.nunique().eq(1).all():
        raise ValueError('A patient crosses outer folds.')
    for outer in range(10):
        train = folds.loc[folds.outer_fold.ne(outer)]
        test = folds.loc[folds.outer_fold.eq(outer)]
        assert_patient_disjoint(train.patient_id, test.patient_id)
        column = f'inner_fold_when_outer_{outer}'
        if set(train[column]) != {0, 1, 2} or not test[column].eq(-1).all():
            raise ValueError('Three inner folds must be restricted to outer-training patients.')
        if not train.groupby('patient_id')[column].nunique().eq(1).all():
            raise ValueError('A patient crosses inner folds.')
    return {'outer_folds': 10, 'inner_folds_per_outer': 3, 'patient_separation': 'passed'}


def run_audit(output_dir=None, validation_limit=256, *, processed_dir=None):
    if int(validation_limit) != validation_limit or validation_limit < 1:
        raise ValueError('Choose a positive number of diagnostic validation ECGs.')
    output = Path(output_dir or PROJECT / 'results' / 'imputation').resolve()
    prepared = Path(processed_dir or processed_directory(PROJECT)).resolve()
    package = load_processed_data(prepared)
    package_hashes = {name: digest(prepared / name)
                      for name in [*package['manifest']['files'], 'manifest.json']}
    output.mkdir(parents=True, exist_ok=True)
    raw, folds = package['raw'], package['folds']
    quality = source_quality_groups(package['metadata'].reset_index())
    fold_check = validate_folds(folds)
    outer_train = folds.loc[folds.outer_fold.ne(0)]
    inner_column = 'inner_fold_when_outer_0'
    train_ids = outer_train.index[outer_train[inner_column].ne(0)]
    valid_ids = outer_train.index[outer_train[inner_column].eq(0)][:int(validation_limit)]
    assert_patient_disjoint(folds.loc[train_ids, 'patient_id'], folds.loc[valid_ids, 'patient_id'])
    train_raw, valid_raw = raw.loc[train_ids], raw.loc[valid_ids]
    groups = {'AFIB': folds.loc[valid_ids, 'target_afib'].eq(1),
              'without_AFIB': folds.loc[valid_ids, 'target_afib'].eq(0),
              'recorded_quality_flag': quality.loc[valid_ids].eq('recorded_quality_flag'),
              'no_recorded_quality_flag': quality.loc[valid_ids].eq('no_recorded_quality_flag')}
    observed_original = valid_raw.copy(deep=True)
    numerical = [c for c in RAW_COLUMNS if c not in CATEGORY_COLUMNS]
    methods, training_reports = {}, []
    for name in ('median', 'ridge'):
        print(f'Fitting {name} on {len(train_raw):,} inner-training ECGs...', flush=True)
        fit = fit_imputer(train_raw, method=name)
        train_fill = impute_measurements(train_raw, fit)
        preprocessing = fit_preprocessor(train_fill['raw'])
        matrix = transform_features(train_fill['raw'], preprocessing, missing_mask=train_fill['missing_mask'])
        np.testing.assert_array_equal(matrix[[c + '__missing' for c in numerical]], train_raw[numerical].isna())
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError('Nonfinite training matrix.')
        methods[name] = (fit, preprocessing)
        training_reports.append({'method': name, **train_fill['audit'], 'encoded_columns': len(matrix.columns)})
    scenarios = [('observed', {}, [0]), ('cells_0.10', {'fraction': .10}, range(3))]
    scenarios += [(f'missing_{k}', {'missing_leads': k}, range(3)) for k in (1, 3, 6)]
    rows, summaries, transformation_audits, subgroup_rows, mask_audits = [], [], [], [], []
    for scenario, options, repeats in scenarios:
        for repeat in repeats:
            if scenario == 'observed':
                changed = valid_raw.copy()
                mask = pd.DataFrame(False, index=valid_raw.index, columns=DEFAULT_IMPUTATION_TARGETS)
            else:
                masked = mask_observed_measurements(valid_raw, repeat=repeat, **options)
                changed, mask = masked['raw'], masked['scoring_mask']
                mask_audits.append({'scenario': scenario, **masked['audit']})
            for name, (fit, preprocessing) in methods.items():
                fill = impute_measurements(changed, fit)
                matrix = transform_features(fill['raw'], preprocessing, missing_mask=fill['missing_mask'])
                if matrix.columns.tolist() != preprocessing['columns'] or not np.isfinite(matrix.to_numpy()).all():
                    raise ValueError('Invalid held-out matrix.')
                np.testing.assert_array_equal(matrix[[c + '__missing' for c in numerical]], changed[numerical].isna())
                pd.testing.assert_frame_equal(fill['raw'].where(changed.notna()), changed)
                transformation_audits.append({'method': name, 'scenario': scenario, 'repeat': repeat, **fill['audit']})
                if scenario != 'observed':
                    scored = score_recovery(valid_raw, fill['raw'], mask, fit)
                    frame = scored['per_feature'].reset_index()
                    for key, value in [('method', name), ('scenario', scenario), ('repeat', repeat)]:
                        frame[key] = value
                    rows.append(frame)
                    summaries.append({'method': name, 'scenario': scenario, 'repeat': repeat, **scored['summary']})
                    # Labels and quality flags enter reporting only, never the filler.
                    for group_name, member in groups.items():
                        ids = valid_ids[member.to_numpy()]
                        subgroup = score_recovery(valid_raw.loc[ids], fill['raw'].loc[ids], mask.loc[ids], fit)
                        subgroup_rows.append({'method': name, 'scenario': scenario, 'repeat': repeat,
                                              'subgroup': group_name, 'n_ecgs': len(ids),
                                              **subgroup['summary']})
            print(f'Checked {scenario}, repeat {repeat}', flush=True)
    pd.testing.assert_frame_equal(valid_raw, observed_original)
    if any(digest(prepared / name) != expected for name, expected in package_hashes.items()):
        raise ValueError('The processed-data package changed during the check.')
    pd.concat(rows, ignore_index=True).to_csv(output / 'recovery_by_feature.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'recovery_summary.csv', index=False)
    pd.DataFrame(transformation_audits).to_csv(output / 'transformation_audits.csv', index=False)
    pd.DataFrame(subgroup_rows).to_csv(output / 'recovery_by_subgroup.csv', index=False)
    pd.DataFrame(mask_audits).to_csv(output / 'mask_audits.csv', index=False)
    result = {
        'scope': 'One inner-partition engineering check; not final research results or clinical validation',
        'input_folder': 'processed_data', 'processed_data_sha256': package_hashes,
        'sources': package['manifest']['sources'],
        'cohort_ecgs': len(raw), 'patients': int(folds.patient_id.nunique()),
        'train_ecgs': len(train_raw), 'validation_ecgs': len(valid_raw),
        'validation_subgroups': {name: int(member.sum()) for name, member in groups.items()},
        'outer_test_ecgs_used': 0, 'fold_checks': fold_check,
        'eligible_continuous_targets': len(DEFAULT_IMPUTATION_TARGETS),
        'protected_P_families': ['PQ_Int', 'PR_Int', 'P_Amp', 'P_DurFull'],
        'learned_configuration': {'alpha': 1.0, 'max_predictors': 8, 'min_observed': 20, 'min_joint': 20},
        'training_audits': training_reports, 'n_method_scenario_repeat_checks': len(transformation_audits),
        'recovery_summary': summaries,
        'processed_data_preservation': 'passed', 'original_missing_flags': 'preserved in both model matrices',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'course_status': {'dataset_size': len(raw) >= 10000, 'original_feature_count': len(RAW_COLUMNS) >= 100,
                          'fold_assignment_checks': 'passed', 'complete_nested_classifier_runs': 'pending',
                          'middle_setting_selection_evidence': 'pending', 'three_final_results_with_error_bars': 'pending'},
        'classifiers_trained': False, 'clinical_benefit_established': False,
    }
    write_json(output / 'audit.json', result)
    print(f'Saved engineering evidence to {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--processed-dir', type=Path, default=processed_directory(PROJECT))
    parser.add_argument('--validation-limit', type=int, default=256)
    args = parser.parse_args()
    run_audit(args.output_dir, args.validation_limit, processed_dir=args.processed_dir)

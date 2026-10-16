"""One portable data package and a shared, leakage-safe model-input loader.

The notebook writes only ``processed_data/``. Learned preparation is delayed
until a training partition is chosen; all three classifiers use the same loader.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import (
    ARMS, CATEGORY_COLUMNS, RAW_COLUMNS, SCENARIOS, SEED, build_features,
    construct_training_views, perturb_measurements, transform_features,
)

QUALITY_COLUMNS = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems']
PACKAGE_FILES = ('lead_features.npz', 'targets.csv', 'folds.csv', 'metadata.csv')
SCHEMA_VERSION = 1


def _sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _integer_ids(values, name):
    values = pd.to_numeric(values, errors='raise')
    if (values.isna().any() or not np.isfinite(values).all()
            or not values.eq(np.floor(values)).all() or values.lt(0).any()):
        raise ValueError(f'{name} must contain nonnegative integer identifiers.')
    return values.astype('int64')


def _codes(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (dict, list)):
        raise ValueError('SCP annotations must be a dictionary or a list of pairs.')
    return dict(parsed)


def load_source_data(project):
    """Read and validate local sources without writing or downloading files."""
    project = Path(project).resolve()
    feature_paths = list((project / 'data').glob('**/features/ecgdeli_features.csv'))
    if len(feature_paths) != 1:
        raise FileNotFoundError('Keep exactly one features/ecgdeli_features.csv under data/.')
    paths = {
        'features': feature_paths[0],
        'labels': feature_paths[0].parent.parent / 'labels/ptbxl_statements.csv',
        'metadata': project / 'data/reference/ptbxl_database.csv',
    }
    features = pd.read_csv(paths['features'])
    labels = pd.read_csv(paths['labels'], usecols=['ecg_id', 'scp_codes'])
    metadata = pd.read_csv(paths['metadata'])
    for frame in (features, labels, metadata):
        frame['ecg_id'] = _integer_ids(frame.ecg_id, 'ecg_id')
        if not frame.ecg_id.is_unique:
            raise ValueError('Each source must contain exactly one row per ECG ID.')
    if not (set(features.ecg_id) == set(labels.ecg_id) == set(metadata.ecg_id)):
        raise ValueError('Feature, label and metadata sources must cover identical ECG IDs.')
    features = features.set_index('ecg_id').sort_index()
    labels = labels.set_index('ecg_id').reindex(features.index)
    metadata = metadata.set_index('ecg_id').reindex(features.index)
    metadata['patient_id'] = _integer_ids(metadata.patient_id, 'patient_id')
    target = metadata.scp_codes.map(_codes).map(lambda codes: int('AFIB' in codes))
    plus_target = labels.scp_codes.map(_codes).map(lambda codes: int('AFIB' in codes))
    if not target.equals(plus_target):
        raise ValueError('AFIB membership disagrees between PTB-XL and PTB-XL+ annotations.')
    if not set(RAW_COLUMNS).issubset(features.columns):
        raise ValueError('The source lacks required ECGDeli lead fields.')
    extra = [column for column in features if column not in RAW_COLUMNS]
    if len(features.columns) != 531 or len(extra) != 27 or any('_Global' not in c for c in extra):
        raise ValueError('Expected 504 lead fields and 27 Global fields in the source.')
    raw = features.loc[:, RAW_COLUMNS].apply(pd.to_numeric, errors='raise')
    nonfinite_count = int(np.isinf(raw.to_numpy()).sum())
    raw = raw.replace([np.inf, -np.inf], np.nan)
    targets = pd.DataFrame({'patient_id': metadata.patient_id, 'target_afib': target.astype('int8')})
    quality = metadata.loc[:, QUALITY_COLUMNS].copy()
    sources = {name: {'path': str(path.relative_to(project)).replace('\\', '/'),
                      'sha256': _sha256(path)} for name, path in paths.items()}
    return {'raw': raw, 'targets': targets, 'metadata': quality, 'sources': sources,
            'checks': {'source_fields': len(features.columns), 'excluded_global_fields': len(extra),
                       'nonfinite_cells_converted_to_missing': nonfinite_count,
                       'rows_removed': 0}}


def _group_folds(targets, n_folds, seed):
    patient_target = targets.groupby('patient_id').target_afib.max().sort_index()
    if set(patient_target.unique()) != {0, 1} or patient_target.value_counts().min() < n_folds:
        raise ValueError('Each patient-level class needs at least one patient per fold.')
    assignments = pd.Series(-1, index=patient_target.index, dtype='int64')
    generator = np.random.default_rng(seed)
    for label in (0, 1):
        patients = generator.permutation(patient_target.index[patient_target.eq(label)].to_numpy())
        assignments.loc[patients] = np.arange(len(patients)) % n_folds
    return targets.patient_id.map(assignments)


def make_patient_folds(targets, n_outer=10, n_inner=3, seed=SEED):
    """Manually stratify patient groups, never individual ECG recordings."""
    if n_outer < 2 or n_inner < 2:
        raise ValueError('At least two outer and inner folds are required.')
    if (not targets.index.is_unique or targets[['patient_id', 'target_afib']].isna().any().any()
            or not targets.target_afib.isin([0, 1]).all()):
        raise ValueError('Unique ECG IDs, patient IDs and binary targets are required.')
    folds = targets[['patient_id', 'target_afib']].copy()
    folds['outer_fold'] = _group_folds(targets, n_outer, seed + 1)
    for outer in range(n_outer):
        train = folds.outer_fold.ne(outer)
        column = f'inner_fold_when_outer_{outer}'
        folds[column] = -1
        folds.loc[train, column] = _group_folds(targets.loc[train], n_inner, seed + 100 + outer)
    _validate_folds(folds, n_outer, n_inner)
    return folds


def _validate_folds(folds, n_outer, n_inner):
    if set(folds.outer_fold) != set(range(n_outer)):
        raise ValueError('Outer folds do not match the configured protocol.')
    if not folds.groupby('patient_id').outer_fold.nunique().eq(1).all():
        raise ValueError('A patient crosses outer folds.')
    for outer in range(n_outer):
        train = folds.outer_fold.ne(outer)
        column = f'inner_fold_when_outer_{outer}'
        if (not folds.loc[~train, column].eq(-1).all()
                or set(folds.loc[train, column]) != set(range(n_inner))
                or not folds.loc[train].groupby('patient_id')[column].nunique().eq(1).all()):
            raise ValueError(f'Invalid patient grouping in inner folds for outer fold {outer}.')


def _write_package(destination, source, folds, n_outer, n_inner, seed):
    """Write data first and its checked manifest last, all inside one folder."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw = source['raw']
    with (destination / 'lead_features.npz').open('wb') as handle:
        np.savez_compressed(handle, values=raw.to_numpy(dtype='float64'),
                            ecg_id=raw.index.to_numpy(dtype='int64'),
                            columns=np.asarray(raw.columns, dtype=str))
    source['targets'].to_csv(destination / 'targets.csv', index_label='ecg_id')
    folds.to_csv(destination / 'folds.csv', index_label='ecg_id')
    source['metadata'].to_csv(destination / 'metadata.csv', index_label='ecg_id')
    manifest = {
        'schema_version': SCHEMA_VERSION, 'seed': int(seed),
        'rows': len(raw), 'patients': int(source['targets'].patient_id.nunique()),
        'target': 'target_afib', 'positive_definition': 'AFIB is present in source SCP annotations, including likelihood zero',
        'lead_fields': len(RAW_COLUMNS), 'categorical_fields': CATEGORY_COLUMNS,
        'enhanced_additional_fields': 63, 'outer_folds': n_outer, 'inner_folds': n_inner,
        'preparation': 'load_prepared_split fits only current training patients; no global imputation, encoding or scaling',
        'sources': source['sources'], 'checks': source['checks'],
        'files': {name: {'sha256': _sha256(destination / name)} for name in PACKAGE_FILES},
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def prepare_dataset(project):
    """Create the notebook's sole output folder: ``project/processed_data``."""
    project = Path(project).resolve()
    source = load_source_data(project)
    folds = make_patient_folds(source['targets'])
    manifest = _write_package(project / 'processed_data', source, folds, 10, 3, SEED)
    return manifest


def load_processed_data(processed_dir):
    """Read and verify the package; raw values retain their original gaps.

    Returns raw, targets, folds, metadata and manifest. These are preparation
    inputs; use load_prepared_split for finite matrices passed to classifiers.
    """
    directory = Path(processed_dir)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported processed-data schema. Rerun preprocessing.')
    if set(manifest['files']) != set(PACKAGE_FILES):
        raise ValueError('Incomplete processed-data manifest. Rerun preprocessing.')
    for name in PACKAGE_FILES:
        if _sha256(directory / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'{name} does not match the manifest. Rerun preprocessing.')
    with np.load(directory / 'lead_features.npz', allow_pickle=False) as saved:
        raw = pd.DataFrame(saved['values'], index=pd.Index(saved['ecg_id'], name='ecg_id'),
                           columns=saved['columns'].tolist())
    frames = {name: pd.read_csv(directory / f'{name}.csv', index_col='ecg_id')
              for name in ('targets', 'folds', 'metadata')}
    if (len(raw) != manifest['rows'] or raw.columns.tolist() != RAW_COLUMNS
            or not raw.index.is_unique or np.isinf(raw.to_numpy()).any()):
        raise ValueError('Invalid predictor matrix in processed_data.')
    for name, frame in frames.items():
        if not frame.index.equals(raw.index) or not frame.index.is_unique:
            raise ValueError(f'{name} records do not align with predictor records.')
    if (not frames['targets'].equals(frames['folds'][['patient_id', 'target_afib']])
            or frames['targets'].isna().any().any()
            or not frames['targets'].target_afib.isin([0, 1]).all()):
        raise ValueError('Invalid or inconsistent target and patient information.')
    _validate_folds(frames['folds'], manifest['outer_folds'], manifest['inner_folds'])
    return {'raw': raw, **frames, 'manifest': manifest}


def load_prepared_split(processed_dir, *, outer_fold, inner_fold=None, arm='A',
                        scenario='observed', repeat=0):
    """Return ready-to-fit numerical matrices without seeing held-out statistics.

    inner_fold=None prepares outer training/test data; 0..2 prepares inner
    training/validation data and excludes every outer-test record. Arms A/B
    use observed training records; C/D add one weighted corrupted copy. B/D
    add 63 constructed features. Scenario changes held-out inputs only.
    """
    package = load_processed_data(processed_dir)
    manifest, folds = package['manifest'], package['folds']
    if (isinstance(outer_fold, bool) or not isinstance(outer_fold, (int, np.integer))
            or outer_fold not in range(manifest['outer_folds'])):
        raise ValueError('Choose a valid outer-fold integer.')
    if inner_fold is not None and (isinstance(inner_fold, bool)
            or not isinstance(inner_fold, (int, np.integer))
            or inner_fold not in range(manifest['inner_folds'])):
        raise ValueError('Choose a valid inner-fold integer or None.')
    if arm not in ARMS or scenario not in SCENARIOS:
        raise ValueError('Unknown preparation arm or held-out scenario.')
    train_mask = folds.outer_fold.ne(outer_fold)
    valid_mask = ~train_mask
    if inner_fold is not None:
        inner = folds[f'inner_fold_when_outer_{outer_fold}']
        valid_mask = train_mask & inner.eq(inner_fold)
        train_mask = train_mask & inner.ne(inner_fold)
    train_ids, valid_ids = folds.index[train_mask], folds.index[valid_mask]
    if not set(folds.loc[train_ids, 'patient_id']).isdisjoint(folds.loc[valid_ids, 'patient_id']):
        raise ValueError('Training and held-out patients overlap.')
    training = construct_training_views(package['raw'].loc[train_ids],
                                        package['targets'].loc[train_ids, 'target_afib'],
                                        **ARMS[arm], seed=manifest['seed'])
    held_out = perturb_measurements(package['raw'].loc[valid_ids], training['perturb_iqr'],
                                    **SCENARIOS[scenario], repeat=repeat,
                                    stream='evaluation', seed=manifest['seed'])
    X_valid = transform_features(build_features(held_out, ARMS[arm]['enhanced']),
                                 training['preprocessor'])
    train_metadata = training['metadata'].copy()
    train_metadata['patient_id'] = train_metadata.source_ecg_id.map(folds.patient_id)
    valid_metadata = package['targets'].loc[valid_ids].join(package['metadata'].loc[valid_ids])
    return {'X_train': training['X'], 'X_valid': X_valid,
            'y_train': training['y'], 'y_valid': package['targets'].loc[valid_ids, 'target_afib'],
            'sample_weight': training['sample_weight'], 'train_metadata': train_metadata,
            'valid_metadata': valid_metadata, 'preprocessor': training['preprocessor'],
            'perturb_iqr': training['perturb_iqr']}

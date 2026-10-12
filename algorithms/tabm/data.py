"""Training-partition preparation and manual summaries for the TabM study.

All learned transformations use the shared AFIB implementation. Metadata is
used for alignment and reports only, never appended to the model matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afib.evaluation import assert_patient_disjoint
from afib.preprocessing import (
    ARMS, RAW_COLUMNS, build_features, construct_training_views,
    transform_features,
)

VARIANTS = [
    'A', 'B', 'C', 'D', 'availability_observed', 'availability_augmented',
    'duplicate_original', 'duplicate_enhanced',
]
METRICS = [
    'macro_f1', 'sensitivity', 'specificity', 'precision', 'npv',
    'balanced_accuracy', 'accuracy', 'auroc', 'average_precision', 'brier',
]


def split_ids(package, outer, inner=None):
    folds = package['folds']
    if outer not in range(10) or (inner is not None and inner not in range(3)):
        raise ValueError('Use outer folds 0..9 and inner folds 0..2.')
    training = folds.outer_fold.ne(outer)
    validation = ~training
    if inner is not None:
        assignments = folds[f'inner_fold_when_outer_{outer}']
        validation = training & assignments.eq(inner)
        training &= assignments.ne(inner)
    train_ids, valid_ids = folds.index[training], folds.index[validation]
    assert_patient_disjoint(folds.loc[train_ids, 'patient_id'],
                            folds.loc[valid_ids, 'patient_id'])
    if inner is not None and folds.loc[train_ids.union(valid_ids), 'outer_fold'].eq(outer).any():
        raise ValueError('Inner partitions contain outer test patients.')
    return train_ids, valid_ids


def identity(variant, outer):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    enhanced = variant in ('B', 'D', 'duplicate_enhanced') or variant.startswith('availability_')
    augmented = variant in ('C', 'D', 'availability_augmented')
    return {
        'model': 'tabm', 'variant': variant,
        'feature_set': 'enhanced' if enhanced else 'original',
        'training_arm': 'augmented' if augmented else 'observed',
        'outer_fold': int(outer),
    }


def _zero_derived_values(X):
    derived = [column for column in X if column not in RAW_COLUMNS
               and not column.endswith('__missing')
               and '__category_' not in column and not column.endswith('__unknown')]
    X.loc[:, derived] = 0.0
    return X


def training_pipeline(package, train_ids, variant):
    """Refit filling, encoding, scaling and weights on observed training ECGs."""
    row = identity(variant, 0)
    raw = package['raw'].loc[train_ids]
    labels = package['targets'].loc[train_ids, 'target_afib']
    pipeline = construct_training_views(
        raw, labels, enhanced=row['feature_set'] == 'enhanced',
        augment=row['training_arm'] == 'augmented', seed=package['manifest']['seed'],
    )
    if variant.startswith('availability_'):
        # Preserve every constructed missingness flag, zeroing numerical values.
        _zero_derived_values(pipeline['X'])
    if variant.startswith('duplicate_'):
        original = pipeline['X'].xs('observed', level='view')
        original_weights = pipeline['sample_weight'].xs('observed', level='view') / 2
        pipeline['X'] = pd.concat({'observed': original, 'duplicate': original},
                                  names=['view', 'ecg_id'])
        pipeline['y'] = pd.concat({'observed': labels, 'duplicate': labels},
                                  names=['view', 'ecg_id'])
        pipeline['sample_weight'] = pd.concat(
            {'observed': original_weights, 'duplicate': original_weights},
            names=['view', 'ecg_id'],
        )
    pipeline['variant'] = variant
    pipeline['train_ids'] = train_ids.to_numpy()
    if not (pipeline['X'].index.equals(pipeline['y'].index)
            and pipeline['X'].index.equals(pipeline['sample_weight'].index)):
        raise ValueError('Training values, labels and weights are misaligned.')
    return pipeline


def transform_heldout(raw, pipeline):
    enhanced = identity(pipeline['variant'], 0)['feature_set'] == 'enhanced'
    X = transform_features(build_features(raw, enhanced), pipeline['preprocessor'])
    return _zero_derived_values(X) if pipeline['variant'].startswith('availability_') else X


def pipeline_state(pipeline):
    """Compact fitted preparation state, without copies of input data."""
    return {key: pipeline[key] for key in (
        'variant', 'train_ids', 'preprocessor', 'perturb_iqr', 'augmentation_audit',
    )}


def patient_weighted_metrics(y, probability, patients, threshold):
    """Each patient has total weight one, independent of their ECG count."""
    ids = pd.Series(patients)
    weights = ids.map(1 / ids.value_counts()).to_numpy(dtype=float)
    truth = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    positive = probability >= threshold
    tp, tn = weights[positive & (truth == 1)].sum(), weights[~positive & (truth == 0)].sum()
    fp, fn = weights[positive & (truth == 0)].sum(), weights[~positive & (truth == 1)].sum()
    def divide(a, b):
        return float(a / b) if b else float('nan')
    return {
        'n_patients': int(ids.nunique()), 'weighted_tp': tp, 'weighted_tn': tn,
        'weighted_fp': fp, 'weighted_fn': fn,
        'macro_f1': .5 * (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)),
        'sensitivity': divide(tp, tp + fn), 'specificity': divide(tn, tn + fp),
        'balanced_accuracy': .5 * (divide(tp, tp + fn) + divide(tn, tn + fp)),
        'accuracy': divide(tp + tn, weights.sum()),
        'brier': float(weights @ ((probability - truth) ** 2) / weights.sum()),
    }


def reliability_bins(y, scores, identity_row):
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y)
    bins = np.minimum((scores * 10).astype(int), 9)
    rows = []
    for index in range(10):
        selected = bins == index
        rows.append({**identity_row, 'bin': index, 'n': int(selected.sum()),
                     'mean_probability': float(np.mean(scores[selected])) if selected.any() else np.nan,
                     'fraction_positive': float(np.mean(y[selected])) if selected.any() else np.nan})
    return rows

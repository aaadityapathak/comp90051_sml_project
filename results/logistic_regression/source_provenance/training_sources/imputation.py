"""Fold-local numerical imputation and controlled recovery experiments.

These generated measurements are estimates, not established patient values.
Default targets exclude P-wave-related fields, whose absence can itself carry
AFIB information. Observed values are never changed. Counts and morphology are
left for the existing preprocessing policy. Fit only on the current training
partition; recovery errors alone must not select a clinical prediction method.
"""

import numpy as np
import pandas as pd

from .preprocessing import (
    NOISE_COLUMNS, NONNEGATIVE_COLUMNS, PARTS, RAW_COLUMNS, SEED,
    _generator, _lead_view, _qt_qrs_check, perturb_measurements,
)


PROTECTED_P_FAMILIES = frozenset({'PQ_Int', 'PR_Int', 'P_Amp', 'P_DurFull'})
DEFAULT_IMPUTATION_TARGETS = [c for c in NOISE_COLUMNS if PARTS[c][0] not in PROTECTED_P_FAMILIES]


def _checked_raw(raw):
    """Retain canonical inputs only; reject infinity rather than hide it."""
    result = _lead_view(raw)
    try:
        values = result[NOISE_COLUMNS].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError('Continuous lead measurements must be numeric or missing.') from error
    if np.isinf(values).any():
        raise ValueError('Continuous lead measurements cannot contain infinity.')
    return result


def _targets(targets):
    chosen = list(DEFAULT_IMPUTATION_TARGETS if targets is None else targets)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(NOISE_COLUMNS):
        raise ValueError('Choose unique continuous lead targets; counts, categories and labels are forbidden.')
    return [c for c in NOISE_COLUMNS if c in chosen]


def fit_imputer(train_raw, method='median', targets=None, alpha=1.0,
                max_predictors=8, min_observed=20, min_joint=20):
    """Fit independent training-only median or ridge models for selected gaps.

    The ridge method selects at most max_predictors (eight by default) by absolute pairwise
    correlation on observed training pairs. Each fit uses only rows with an
    observed target, training medians for predictor gaps, standardized numeric
    inputs/targets and centered predictor-missing flags. A fixed positive ridge
    penalty is used; any tuning must occur inside the inner training folds.
    Sparse/constant targets and unsupported predictors fall back to a median;
    entirely missing training targets use zero and are explicitly unsupported.
    """
    if method not in {'median', 'ridge'}:
        raise ValueError('method must be median or ridge.')
    for value, lower, name in [(max_predictors, 1, 'max_predictors'),
                               (min_observed, 2, 'min_observed'), (min_joint, 2, 'min_joint')]:
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}.')
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    selected = _targets(targets)
    raw = _checked_raw(train_raw)
    if raw.empty:
        raise ValueError('A nonempty training partition is required.')
    continuous = raw[NOISE_COLUMNS].astype(float)
    medians = continuous.median().fillna(0.0)
    scale = continuous.quantile(.75) - continuous.quantile(.25)
    counts = continuous.count()
    if not np.isfinite(medians.to_numpy()).all() or not np.isfinite(scale.fillna(0.0).to_numpy()).all():
        raise ValueError('Nonfinite training summaries; inspect measurement magnitudes.')
    # Pairwise correlations use no labels, IDs, counts, categories or global fields.
    correlations = continuous.corr(min_periods=int(min_joint)) if method == 'ridge' else None
    models = {}
    for target in selected:
        observed = continuous[target].notna()
        target_values = continuous.loc[observed, target].to_numpy()
        n = int(observed.sum())
        model = {'kind': 'median', 'n_observed': n, 'predictors': [],
                 'median': float(medians[target]), 'supported': bool(n),
                 'fallback_reason': 'median_baseline' if n else 'no_observed_training_target'}
        if method == 'ridge' and n:
            target_mean = float(target_values.mean())
            target_scale = float(target_values.std())
            if not np.isfinite([target_mean, target_scale]).all():
                raise ValueError('Nonfinite target standardization; inspect training measurement magnitudes.')
            if n < min_observed:
                model['fallback_reason'] = 'insufficient_observed_target'
            elif target_scale <= np.finfo(float).eps:
                model['fallback_reason'] = 'constant_target'
            else:
                eligible = [c for c in NOISE_COLUMNS if c != target and np.isfinite(correlations.at[c, target])]
                # Python's stable sort retains the canonical order for ties.
                predictors = sorted(eligible, key=lambda c: -abs(correlations.at[c, target]))[:int(max_predictors)]
                if not predictors:
                    model['fallback_reason'] = 'no_supported_predictors'
                else:
                    block = continuous.loc[observed, predictors]
                    filled = block.fillna(medians[predictors]).to_numpy()
                    means = filled.mean(axis=0)
                    scales = filled.std(axis=0)
                    scales = np.where(scales <= np.finfo(float).eps, 1.0, scales)
                    missing = block.isna().to_numpy(dtype=float)
                    missing_means = missing.mean(axis=0)
                    design = np.concatenate([(filled - means) / scales, missing - missing_means], axis=1)
                    if not np.isfinite(design).all() or not np.isfinite(means).all() or not np.isfinite(scales).all():
                        raise ValueError('Nonfinite predictor standardization; inspect training measurement magnitudes.')
                    y = (target_values - target_mean) / target_scale
                    coefficients = np.linalg.solve(design.T @ design + alpha * np.eye(design.shape[1]), design.T @ y)
                    if not np.isfinite(coefficients).all():
                        raise ValueError('Nonfinite ridge coefficients; inspect training measurement magnitudes.')
                    model.update(kind='ridge', predictors=predictors, coefficients=coefficients,
                                 predictor_medians=medians[predictors].to_numpy(), predictor_means=means,
                                 predictor_scales=scales, missing_means=missing_means,
                                 target_mean=target_mean, target_scale=target_scale, fallback_reason=None)
        models[target] = model
    return {'method': method, 'targets': selected, 'alpha': float(alpha),
            'max_predictors': int(max_predictors), 'min_observed': int(min_observed),
            'min_joint': int(min_joint), 'models': models,
            'training_iqr': scale[selected].fillna(0.0), 'training_support': counts[selected],
            'n_training_ecgs': len(raw)}


def impute_measurements(raw, fitted):
    """Fill selected gaps once from original inputs and retain their provenance.

    Never use one generated target as another model's predictor. Only generated
    durations and IQRs are clipped at zero; this bound does not establish clinical
    plausibility. When every selected predictor is missing, use the training
    median instead of a conditional prediction without observed evidence.
    Return the original 504-field missing mask for downstream flags,
    plus generation/unsupported masks over the 312 continuous fields. Downstream
    summaries should be constructed from available values before filling; replace
    selected raw feature values and pass the original missing mask to scaling.
    """
    original = _checked_raw(raw)
    targets = _targets(fitted['targets'])
    result = original.copy()
    generated = pd.DataFrame(False, index=original.index, columns=NOISE_COLUMNS)
    unsupported = generated.copy()
    clipped, eligible = 0, 0
    method_counts = {'ridge': 0, 'median': 0}
    fallback_counts = {}
    for target in targets:
        model = fitted['models'][target]
        hidden = original[target].isna()
        n = int(hidden.sum())
        if not n:
            continue
        if model['kind'] == 'ridge':
            block = original.loc[hidden, model['predictors']].astype(float)
            inputs = block.to_numpy()
            missing = np.isnan(inputs)
            filled = np.where(missing, model['predictor_medians'], inputs)
            design = np.concatenate([(filled - model['predictor_means']) / model['predictor_scales'],
                                     missing.astype(float) - model['missing_means']], axis=1)
            estimates = model['target_mean'] + model['target_scale'] * (design @ model['coefficients'])
            without_observed_predictors = missing.all(axis=1)
            estimates[without_observed_predictors] = model['median']
            n_fallback = int(without_observed_predictors.sum())
            if n_fallback:
                reason = 'all_predictors_missing_at_transform'
                fallback_counts[reason] = fallback_counts.get(reason, 0) + n_fallback
            method_counts['ridge'] += n - n_fallback
            method_counts['median'] += n_fallback
        elif model['kind'] == 'median':
            estimates = np.full(n, model['median'], dtype=float)
            method_counts['median'] += n
            reason = model['fallback_reason']
            if reason != 'median_baseline':
                fallback_counts[reason] = fallback_counts.get(reason, 0) + n
        else:
            raise ValueError('Unknown fitted imputation model.')
        if not np.isfinite(estimates).all():
            raise ValueError('Generated measurements must be finite; inspect fitted models and inputs.')
        if target in NONNEGATIVE_COLUMNS:
            eligible += n
            clipped += int((estimates < 0).sum())
            estimates = np.maximum(estimates, 0.0)
        result.loc[hidden, target] = estimates
        generated.loc[hidden, target] = True
        unsupported.loc[hidden, target] = not model['supported']
    audit = {'n_ecgs': len(original), 'n_target_fields': len(targets),
             'generated_cells': int(generated.to_numpy().sum()),
             'unsupported_generated_cells': int(unsupported.to_numpy().sum()),
             'generated_cells_by_method': method_counts,
             'median_fallback_cells_by_reason': fallback_counts,
             'clip_eligible_generated_cells': eligible, 'clipped_generated_cells': clipped,
             'clipping_fraction': clipped / eligible if eligible else 0.0,
             'qt_qrs_before': _qt_qrs_check(original),
             'qt_qrs_after': _qt_qrs_check(result)}
    return {'raw': result, 'missing_mask': original.isna(), 'generated_mask': generated,
            'unsupported_mask': unsupported, 'audit': audit, 'models': fitted['models']}


def mask_observed_measurements(raw, fraction=.10, missing_leads=0, repeat=0, seed=SEED):
    """Hide available measurements with ECG-keyed evaluation draws.

    With missing_leads > 0, hide all 42 fields of each selected lead, exactly as
    existing missing-lead stress tests do; fraction is then unused. Otherwise
    independently hide each observed continuous cell with probability fraction
    (nominal 10%, not exactly 10% in every sample). The scoring mask marks only
    originally observed continuous values. Natural gaps never acquire a truth.
    """
    if (not np.isfinite([missing_leads, fraction, repeat, seed]).all()
            or isinstance(missing_leads, (bool, np.bool_)) or int(missing_leads) != missing_leads
            or missing_leads not in {0, 1, 3, 6} or not np.isfinite(fraction) or not 0 <= fraction <= 1
            or isinstance(repeat, (bool, np.bool_)) or int(repeat) != repeat or repeat < 0
            or isinstance(seed, (bool, np.bool_)) or int(seed) != seed or seed < 0):
        raise ValueError('Use a fraction in [0,1], 0/1/3/6 hidden leads and a nonnegative integer repeat.')
    original = _checked_raw(raw)
    if missing_leads:
        result = perturb_measurements(original, pd.Series(0., index=NOISE_COLUMNS),
                                      missing_leads=missing_leads, repeat=repeat, stream='evaluation', seed=seed)
    else:
        result = original.copy()
        draws = np.stack([_generator(ecg, repeat, 'evaluation', seed, 50).random(len(NOISE_COLUMNS))
                          for ecg in original.index]) if len(original) else np.empty((0, len(NOISE_COLUMNS)))
        result[NOISE_COLUMNS] = result[NOISE_COLUMNS].mask(draws < fraction)
    natural = original.isna()
    artificial = result.isna() & ~natural
    scoring = artificial[NOISE_COLUMNS].copy()
    n_observed_continuous = int(original[NOISE_COLUMNS].notna().to_numpy().sum())
    n_hidden_continuous = int(scoring.to_numpy().sum())
    return {'raw': result, 'scoring_mask': scoring, 'natural_missing_mask': natural,
            'artificial_missing_mask': artificial,
            'audit': {'missing_leads': int(missing_leads),
                      'fraction': float(fraction) if not missing_leads else None,
                      'repeat': int(repeat), 'natural_missing_cells': int(natural.to_numpy().sum()),
                      'artificial_missing_cells': int(artificial.to_numpy().sum()),
                      'scorable_continuous_cells': n_hidden_continuous,
                      'observed_continuous_cells': n_observed_continuous,
                      'achieved_continuous_mask_fraction': n_hidden_continuous / n_observed_continuous
                      if n_observed_continuous else None}}


def score_recovery(truth_raw, filled_raw, scoring_mask, fitted):
    """Score artificial hidden truth only; expose unresolved coverage failures.

    Accept the full continuous masking mask or a selected-target subset. Score
    only fitted targets and report excluded mask cells. Absolute errors are in
    each feature's original units; normalize only by positive training IQRs.
    Finite unsupported zero fallbacks remain in errors and are flagged by their
    training_support, so scores cannot improve by quietly omitting them.
    """
    truth, filled = _checked_raw(truth_raw), _checked_raw(filled_raw)
    if not truth.index.equals(filled.index):
        raise ValueError('Truth and filled measurements must have identical ordered ECG indices.')
    if (not isinstance(scoring_mask, pd.DataFrame) or not scoring_mask.index.is_unique
            or not scoring_mask.columns.is_unique or not scoring_mask.index.equals(truth.index)
            or not set(scoring_mask.columns).issubset(NOISE_COLUMNS)
            or not all(dtype == bool for dtype in scoring_mask.dtypes)):
        raise ValueError('Provide an aligned boolean scoring mask with unique continuous field columns.')
    if (scoring_mask & truth[scoring_mask.columns].isna()).to_numpy().any():
        raise ValueError('Naturally missing truth cannot be used in recovery scoring.')
    targets = _targets(fitted['targets'])
    mask = scoring_mask.reindex(columns=targets, fill_value=False)
    excluded = int(scoring_mask.loc[:, ~scoring_mask.columns.isin(targets)].to_numpy().sum())
    rows = []
    normalized_error_sum, n_normalized = 0., 0
    for target in targets:
        selected = mask[target]
        available = selected & filled[target].notna()
        n_targets, n_scored = int(selected.sum()), int(available.sum())
        errors = (filled.loc[available, target] - truth.loc[available, target]).abs()
        if not np.isfinite(errors.to_numpy()).all():
            raise ValueError('Recovery errors must be finite; inspect measurement magnitudes.')
        mae = float(errors.mean()) if n_scored else np.nan
        iqr = float(fitted['training_iqr'][target])
        if not np.isfinite(iqr) or iqr < 0:
            raise ValueError('Training IQRs must be finite and nonnegative.')
        if iqr > 0 and n_scored:
            normalized_error_sum += float(errors.sum()) / iqr
            n_normalized += n_scored
        rows.append({'feature': target, 'n_targets': n_targets, 'n_scored': n_scored,
                     'n_unresolved': n_targets - n_scored,
                     'coverage': n_scored / n_targets if n_targets else np.nan,
                     'mae': mae, 'training_iqr': iqr,
                     'mae_in_training_iqr': mae / iqr if iqr > 0 else np.nan,
                     'training_support': int(fitted['training_support'][target]),
                     'model_kind': fitted['models'][target]['kind']})
    per_feature = pd.DataFrame(rows).set_index('feature')
    n_targets, n_scored = int(per_feature.n_targets.sum()), int(per_feature.n_scored.sum())
    return {'per_feature': per_feature,
            'summary': {'n_targets': n_targets, 'n_scored': n_scored,
                        'n_unresolved': n_targets - n_scored,
                        'coverage': n_scored / n_targets if n_targets else None,
                        'n_excluded_targets': excluded, 'n_normalized_scored': n_normalized,
                        'cell_mean_mae_in_training_iqr': normalized_error_sum / n_normalized if n_normalized else None}}

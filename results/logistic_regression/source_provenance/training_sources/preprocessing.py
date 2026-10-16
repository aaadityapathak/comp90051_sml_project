"""Feature construction and reproducible measurement-space perturbations.

These are controlled tabular stress tests, not calibrated ECG waveform artefacts.
Only observed training records may be passed to either fit function. Validation
draws use a separate stream from augmentation, and all draws are keyed by ECG ID.
"""

import numpy as np
import pandas as pd

SEED = 20261008
LEADS = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
FAMILIES = ['PQ_Int', 'PR_Int', 'P_Amp', 'P_DurFull', 'Q_Amp', 'QRS_Dur',
            'QT_IntCorr', 'QT_Int', 'R_Amp', 'S_Amp', 'ST_Elev', 'T_Amp', 'T_DurFull']
DURATIONS = ['PQ_Int', 'PR_Int', 'P_DurFull', 'QRS_Dur', 'QT_IntCorr', 'QT_Int', 'T_DurFull']
AMPLITUDES = ['P_Amp', 'Q_Amp', 'R_Amp', 'S_Amp', 'ST_Elev', 'T_Amp']
CHEST = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LIMB = [lead for lead in LEADS if lead not in CHEST]
RAW_COLUMNS = [f'{family}_{lead}{suffix}' for family in FAMILIES[:4] + ['P_Morph'] + FAMILIES[4:]
               for lead in LEADS for suffix in ['', '_iqr', '_count']]
CATEGORY_COLUMNS = [f'P_Morph_{lead}{suffix}' for lead in LEADS for suffix in ['', '_iqr']]
SCENARIOS = {'observed': {}}
SCENARIOS.update({f'missing_{k}': {'missing_leads': k} for k in [1, 3, 6]})
SCENARIOS.update({f'noise_{s:.2f}': {'noise_sigma': s} for s in [.05, .10, .20]})
SCENARIOS.update({f'gain_{s:.2f}': {'gain_sigma': s} for s in [.05, .10, .20]})
SCENARIOS['combined_3lead_noise0.10'] = {'missing_leads': 3, 'noise_sigma': .10}
AUGMENTATION_SCENARIOS = ['missing_3', 'noise_0.10', 'gain_0.10', 'combined_3lead_noise0.10']
ARMS = {
    'A': {'enhanced': False, 'augment': False},
    'B': {'enhanced': True, 'augment': False},
    'C': {'enhanced': False, 'augment': True},
    'D': {'enhanced': True, 'augment': True},
}
STREAMS = {'evaluation': 1, 'augmentation': 2}


def column_parts(column):
    stem = column.removesuffix('_iqr').removesuffix('_count')
    family, lead = stem.rsplit('_', 1)
    kind = 'iqr' if column.endswith('_iqr') else 'count' if column.endswith('_count') else 'value'
    if lead not in LEADS:
        raise ValueError(f'Unexpected lead in {column}')
    return family, lead, kind


PARTS = {c: column_parts(c) for c in RAW_COLUMNS}
LEAD_COLUMNS = {lead: [c for c in RAW_COLUMNS if PARTS[c][1] == lead] for lead in LEADS}
NOISE_COLUMNS = [c for c in RAW_COLUMNS if PARTS[c][0] in FAMILIES and PARTS[c][2] != 'count']
NONNEGATIVE_COLUMNS = [c for c in NOISE_COLUMNS if PARTS[c][2] == 'iqr' or PARTS[c][0] in DURATIONS]
GAIN_COLUMNS = {lead: [c for c in LEAD_COLUMNS[lead] if PARTS[c][0] in AMPLITUDES
                      and PARTS[c][2] != 'count'] for lead in LEADS}


def _lead_view(raw):
    if not raw.index.is_unique or not raw.columns.is_unique:
        raise ValueError('One unique row per ECG and one column per field are required.')
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing {len(missing)} required lead fields.')
    # Canonical source order also keeps random values stable after column reordering.
    return raw.loc[:, RAW_COLUMNS].copy()


def build_features(raw, enhanced=True):
    raw = _lead_view(raw)
    if not enhanced:
        return raw
    derived = {}
    for family in FAMILIES:
        block = raw[[f'{family}_{lead}' for lead in LEADS]]
        n = block.count(axis=1)
        derived[f'{family}__median'] = block.median(axis=1)
        derived[f'{family}__lead_iqr'] = (block.quantile(.75, axis=1) - block.quantile(.25, axis=1)).where(n.ge(2))
        derived[f'{family}__range'] = (block.max(axis=1) - block.min(axis=1)).where(n.ge(2))
        derived[f'{family}__chest_minus_limb'] = raw[[f'{family}_{l}' for l in CHEST]].median(axis=1) - raw[[f'{family}_{l}' for l in LIMB]].median(axis=1)
    for family in DURATIONS:
        values = raw[[f'{family}_{lead}' for lead in LEADS]].copy()
        variability = raw[[f'{family}_{lead}_iqr' for lead in LEADS]].copy()
        values.columns = variability.columns = LEADS
        derived[f'{family}__relative_beat_iqr'] = (variability / values.where(values.gt(1.0))).median(axis=1)
    derived['QRS__squared'] = derived['QRS_Dur__median'] ** 2
    derived['QT__squared'] = derived['QT_Int__median'] ** 2
    derived['QRS_x_QT'] = derived['QRS_Dur__median'] * derived['QT_Int__median']
    derived['P_amp_x_duration'] = derived['P_Amp__median'] * derived['P_DurFull__median']
    assert len(derived) == 63
    return pd.concat([raw, pd.DataFrame(derived, index=raw.index)], axis=1).replace([np.inf, -np.inf], np.nan)


def fit_perturber(train_raw):
    """Training IQR only; zero/all-missing IQR fields receive no additive noise."""
    raw = _lead_view(train_raw)
    scale = raw[NOISE_COLUMNS].quantile(.75) - raw[NOISE_COLUMNS].quantile(.25)
    return scale.fillna(0).clip(lower=0)


def _generator(ecg_id, repeat, stream, seed, operation):
    if isinstance(ecg_id, (bool, np.bool_)) or int(ecg_id) != ecg_id or int(ecg_id) < 0:
        raise ValueError('Nonnegative integer ECG IDs are required for stable random draws.')
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAMS[stream], int(ecg_id), int(repeat), operation]))


def _qt_qrs_check(raw):
    qt = raw[[f'QT_Int_{lead}' for lead in LEADS]].to_numpy(dtype=float)
    qrs = raw[[f'QRS_Dur_{lead}' for lead in LEADS]].to_numpy(dtype=float)
    valid = np.isfinite(qt) & np.isfinite(qrs)
    n = int(valid.sum())
    violations = int(((qt < qrs) & valid).sum())
    return {'observed_pairs': n, 'qt_below_qrs_pairs': violations,
            'fraction': violations / n if n else None}


def perturb_measurements(raw, fitted_iqr, missing_leads=0, noise_sigma=0., gain_sigma=0.,
                         repeat=0, stream='evaluation', seed=SEED, return_audit=False):
    """Return deterministic changed lead fields, optionally with dose diagnostics.

    The same operation draws are reused across doses and masks are nested within
    a repeat. Separate operation/stream seeds make paired comparisons independent
    of call order. Gain exp(sigma*z) is positive with median one, not mean one.
    """
    if (int(missing_leads) != missing_leads or not 0 <= missing_leads <= len(LEADS)
            or not np.isfinite([noise_sigma, gain_sigma]).all() or min(noise_sigma, gain_sigma) < 0
            or int(repeat) != repeat or repeat < 0 or stream not in STREAMS):
        raise ValueError('Invalid perturbation setting.')
    original = _lead_view(raw)
    result = original.copy()
    scale = fitted_iqr.reindex(NOISE_COLUMNS)
    if scale.isna().any() or not np.isfinite(scale).all() or scale.lt(0).any():
        raise ValueError('Provide finite nonnegative training IQRs for every continuous field.')
    active = scale.index[scale.gt(0)].tolist()
    nonnegative = [c for c in NONNEGATIVE_COLUMNS if c in active]
    clip_eligible, clipped = 0, 0
    if noise_sigma and len(result):
        noise = np.stack([_generator(ecg, repeat, stream, seed, 10).standard_normal(len(NOISE_COLUMNS)) for ecg in result.index])
        result[NOISE_COLUMNS] = result[NOISE_COLUMNS] + noise_sigma * noise * scale.to_numpy()
        # Leave natural values in zero-IQR fields unchanged, including unusual values.
        clip_eligible = int(original[nonnegative].notna().sum().sum())
        clipped = int(result[nonnegative].lt(0).sum().sum())
        result[nonnegative] = result[nonnegative].clip(lower=0)
    if gain_sigma and len(result):
        gains = np.stack([_generator(ecg, repeat, stream, seed, 20).standard_normal(len(LEADS)) for ecg in result.index])
        for j, lead in enumerate(LEADS):
            result[GAIN_COLUMNS[lead]] = result[GAIN_COLUMNS[lead]].mul(np.exp(gain_sigma * gains[:, j]), axis=0)
    # Audit numerical dose before masking, then also report the dose retained after masking.
    numerical_result = result.copy() if return_audit else None
    if missing_leads and len(result):
        masks = np.stack([_generator(ecg, repeat, stream, seed, 30).permutation(len(LEADS)) for ecg in result.index])
        for j, lead in enumerate(LEADS):
            hidden = (masks[:, :missing_leads] == j).any(axis=1)
            result.loc[result.index[hidden], LEAD_COLUMNS[lead]] = np.nan
    if not return_audit:
        return result
    def scaled_rms(changed):
        delta = (changed[active] - original[active]).div(scale[active], axis=1).to_numpy(dtype=float)
        finite = np.isfinite(delta)
        return {'n_comparable_cells': int(finite.sum()),
                'rms_in_training_iqr': float(np.sqrt(np.mean(delta[finite] ** 2))) if finite.any() else None}
    audit = {
        'n_ecgs': len(result), 'stream': stream, 'repeat': int(repeat),
        'missing_leads': int(missing_leads), 'noise_sigma': float(noise_sigma), 'gain_sigma': float(gain_sigma),
        'natural_missing_cells': int(original.isna().sum().sum()),
        'newly_masked_observed_cells': int((result.isna() & original.notna()).sum().sum()),
        'zero_iqr_fields_without_additive_noise': scale.index[scale.eq(0)].tolist(),
        'clip_eligible_observed_nonnegative_cells': clip_eligible,
        'clipped_nonnegative_cells': clipped,
        'clipping_fraction': clipped / clip_eligible if clip_eligible else 0.0,
        'numeric_dose_before_mask': scaled_rms(numerical_result),
        'numeric_dose_after_mask': scaled_rms(result),
        'qt_qrs_before': _qt_qrs_check(original), 'qt_qrs_after': _qt_qrs_check(result),
    }
    return result, audit


def fit_preprocessor(train_features):
    clean = train_features.replace([np.inf, -np.inf], np.nan)
    categorical = [c for c in CATEGORY_COLUMNS if c in clean]
    numeric = [c for c in clean if c not in categorical]
    medians = clean[numeric].median().fillna(0.0)
    filled = clean[numeric].fillna(medians)
    means, scales = filled.mean(), filled.std(ddof=0)
    scales = scales.mask(scales.le(np.finfo(float).eps) | scales.isna(), 1.0)
    vocabulary = {c: sorted(clean[c].dropna().unique().tolist()) for c in categorical}
    fitted = {'numeric': numeric, 'categorical': categorical, 'median': medians,
              'mean': means, 'scale': scales, 'vocabulary': vocabulary,
              'all_missing_training': clean.columns[clean.count().eq(0)].tolist(),
              'constant_training': clean.columns[clean.nunique().eq(1)].tolist(), 'removed': []}
    fitted['columns'] = transform_features(train_features.iloc[:1], fitted).columns.tolist()
    return fitted


def transform_features(features_to_transform, fitted, *, missing_mask=None):
    """Apply a training fit, optionally retaining gaps from before numerical filling.

    The optional mask must describe the same feature matrix before imputation.
    It is provenance, not a label distinguishing natural from artificial gaps.
    Category encoding continues to use the unchanged categorical observations.
    """
    clean = features_to_transform.replace([np.inf, -np.inf], np.nan)
    if missing_mask is not None:
        if (not isinstance(missing_mask, pd.DataFrame)
                or not missing_mask.index.equals(clean.index)
                or not missing_mask.columns.equals(clean.columns)
                or missing_mask.isna().any().any()
                or not all(pd.api.types.is_bool_dtype(dtype) for dtype in missing_mask.dtypes)):
            raise ValueError('Provide an aligned boolean missingness mask for every input field.')
        if (clean.isna() & ~missing_mask).any().any():
            raise ValueError('The supplied mask must include every remaining missing input.')
        categorical = fitted['categorical']
        if not missing_mask[categorical].equals(clean[categorical].isna()):
            raise ValueError('Categorical gaps cannot be replaced by numerical imputation.')
    numeric = clean[fitted['numeric']]
    scaled = (numeric.fillna(fitted['median']) - fitted['mean']) / fitted['scale']
    original_gaps = numeric.isna() if missing_mask is None else missing_mask[fitted['numeric']]
    flags = original_gaps.astype(float).add_suffix('__missing')
    categories = {}
    for column in fitted['categorical']:
        values, vocabulary = clean[column], fitted['vocabulary'][column]
        for i, value in enumerate(vocabulary):
            categories[f'{column}__category_{i}'] = values.eq(value).astype(float)
        categories[f'{column}__missing'] = values.isna().astype(float)
        categories[f'{column}__unknown'] = (values.notna() & ~values.isin(vocabulary)).astype(float)
    result = pd.concat([scaled, flags, pd.DataFrame(categories, index=clean.index)], axis=1).astype('float32')
    if 'columns' in fitted:
        assert result.columns.tolist() == fitted['columns']
    assert result.columns.is_unique and np.isfinite(result.to_numpy()).all()
    return result


def augmentation_assignments(ids, repeat=0, seed=SEED):
    """Balanced deterministic mixture; assignments depend only on this training set."""
    ids = pd.Index(ids)
    if not ids.is_unique:
        raise ValueError('Augmentation requires one row per ECG.')
    keyed = [(float(_generator(ecg, repeat, 'augmentation', seed, 40).random()), int(ecg)) for ecg in ids]
    ordered = [ecg for _, ecg in sorted(keyed)]
    assignments = pd.Series({ecg: AUGMENTATION_SCENARIOS[i % 4] for i, ecg in enumerate(ordered)}, name='scenario')
    return assignments.reindex(ids)


def construct_training_views(train_raw, train_y, enhanced=True, augment=False, repeat=0, seed=SEED):
    """Fit on observed training data and return weighted observed/corrupt views.

    Each original ECG receives n/(2*n_class) total weight. With augmentation its
    two views each receive half that weight. Pass sample_weight to the estimator;
    do not also enable estimator class_weight. No labels are changed.
    """
    raw = _lead_view(train_raw)
    if not isinstance(train_y, pd.Series) or not train_y.index.is_unique:
        raise ValueError('Supply one target Series value per ECG.')
    labels = train_y.reindex(raw.index)
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError('Both binary target classes are required in each training partition.')
    scale = fit_perturber(raw)
    observed_features = build_features(raw, enhanced)
    fitted = fit_preprocessor(observed_features)
    matrices = {'observed': transform_features(observed_features, fitted)}
    assignments = pd.Series('observed', index=raw.index, name='scenario')
    audits = []
    if augment:
        assignments = augmentation_assignments(raw.index, repeat, seed)
        corrupted = raw.copy()
        for scenario in AUGMENTATION_SCENARIOS:
            ids = assignments.index[assignments.eq(scenario)]
            changed, audit = perturb_measurements(raw.loc[ids], scale, repeat=repeat, stream='augmentation',
                                                 seed=seed, return_audit=True, **SCENARIOS[scenario])
            corrupted.loc[ids] = changed
            audits.append({'scenario': scenario, **audit})
        matrices['corrupted'] = transform_features(build_features(corrupted, enhanced), fitted)
    class_counts = labels.value_counts()
    weights = labels.map({label: len(labels) / (2 * count) for label, count in class_counts.items()})
    X = pd.concat(matrices, names=['view', 'ecg_id'])
    targets = pd.concat({view: labels for view in matrices}, names=['view', 'ecg_id']).rename('target_afib')
    sample_weight = pd.concat({view: weights / len(matrices) for view in matrices}, names=['view', 'ecg_id']).rename('sample_weight')
    metadata = pd.DataFrame({'scenario': np.concatenate([np.repeat('observed', len(raw))] +
                              ([assignments.to_numpy()] if augment else [])),
                             'source_ecg_id': X.index.get_level_values('ecg_id')}, index=X.index)
    return {'X': X, 'y': targets, 'sample_weight': sample_weight, 'preprocessor': fitted,
            'perturb_iqr': scale, 'metadata': metadata, 'augmentation_audit': audits}

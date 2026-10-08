"""Leakage, provenance and honest recovery checks for numerical imputation."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from afib.imputation import (
    DEFAULT_IMPUTATION_TARGETS, fit_imputer, impute_measurements,
    mask_observed_measurements, score_recovery,
)
from afib.preprocessing import (
    CATEGORY_COLUMNS, LEAD_COLUMNS, NOISE_COLUMNS, RAW_COLUMNS,
    build_features, fit_preprocessor, perturb_measurements, transform_features,
)


def sample_data(n=60):
    rng = np.random.default_rng(190)
    raw = pd.DataFrame(rng.uniform(1., 5., (n, len(RAW_COLUMNS))),
                       index=pd.Index(np.arange(100, 100+n), name='ecg_id'), columns=RAW_COLUMNS)
    raw[CATEGORY_COLUMNS] = 1.
    for column in RAW_COLUMNS:
        if column.endswith('_count'):
            raw[column] = 10.
    raw['R_Amp_I'] = np.linspace(-3., 3., n)
    raw['R_Amp_II'] = 2. * raw['R_Amp_I'] + 3.
    raw['QRS_Dur_I'] = 100. + 2. * raw['R_Amp_I']
    raw.loc[raw.index[:4], 'S_Amp_I'] = np.nan
    return raw


class ImputationTests(unittest.TestCase):
    def setUp(self):
        self.raw = sample_data()

    def test_generated_duration_relationships_are_reported_not_silently_repaired(self):
        training = self.raw.copy()
        training['QT_Int_I'] = 200.
        fitted = fit_imputer(training, targets=['QT_Int_I'])
        validation = training.iloc[:2].copy()
        validation['QT_Int_I'] = np.nan
        validation['QRS_Dur_I'] = 300.
        output = impute_measurements(validation, fitted)
        self.assertTrue(output['raw']['QT_Int_I'].eq(200.).all())
        self.assertEqual(output['audit']['qt_qrs_after']['observed_pairs'],
                         output['audit']['qt_qrs_before']['observed_pairs'] + 2)
        self.assertEqual(output['audit']['qt_qrs_after']['qt_below_qrs_pairs'],
                         output['audit']['qt_qrs_before']['qt_below_qrs_pairs'] + 2)

    def test_default_scope_and_forbidden_targets(self):
        self.assertEqual(len(DEFAULT_IMPUTATION_TARGETS), 216)
        self.assertNotIn('P_Amp_I', DEFAULT_IMPUTATION_TARGETS)
        fitted = fit_imputer(self.raw, method='ridge')
        self.assertEqual(fitted['targets'], DEFAULT_IMPUTATION_TARGETS)
        for target in ['ecg_id', 'target_afib', 'R_Amp_I_count', 'P_Morph_I']:
            with self.assertRaises(ValueError):
                fit_imputer(self.raw, targets=[target])

    def test_leakage_exclusions_and_training_only_fit(self):
        fitted = fit_imputer(self.raw, 'ridge', targets=['R_Amp_II'], max_predictors=1)
        injected = self.raw.copy()
        injected['target_afib'] = self.raw['R_Amp_II']
        injected['global_interval'] = self.raw['R_Amp_II']
        other = fit_imputer(injected.iloc[:, ::-1], 'ridge', targets=['R_Amp_II'], max_predictors=1)
        a, b = fitted['models']['R_Amp_II'], other['models']['R_Amp_II']
        self.assertEqual(a['predictors'], b['predictors'])
        self.assertNotIn('R_Amp_II', a['predictors'])
        self.assertIn(a['predictors'][0], ['R_Amp_I', 'QRS_Dur_I'])
        np.testing.assert_array_equal(a['coefficients'], b['coefficients'])
        saved = a['coefficients'].copy()
        validation = self.raw.iloc[:3].copy()
        validation['R_Amp_II'] = np.nan
        validation['target_afib'] = 1e10
        output = impute_measurements(validation, fitted)
        self.assertLess((output['raw']['R_Amp_II'] - self.raw.iloc[:3]['R_Amp_II']).abs().max(), .2)
        np.testing.assert_array_equal(saved, a['coefficients'])

    def test_independent_predictions_no_generation_feedback_or_mutation(self):
        fitted = fit_imputer(self.raw, 'ridge', targets=['R_Amp_I', 'R_Amp_II'], max_predictors=1)
        hidden = self.raw.iloc[:3].copy()
        hidden[['R_Amp_I', 'R_Amp_II']] = np.nan
        before = hidden.copy(deep=True)
        both = impute_measurements(hidden, fitted)
        subset = {**fitted, 'targets': ['R_Amp_II']}
        alone = impute_measurements(hidden, subset)
        pd.testing.assert_series_equal(both['raw']['R_Amp_II'], alone['raw']['R_Amp_II'])
        pd.testing.assert_frame_equal(hidden, before)
        untouched = [c for c in RAW_COLUMNS if c not in fitted['targets']]
        pd.testing.assert_frame_equal(both['raw'][untouched], hidden[untouched])
        observed = self.raw.copy()
        observed.loc[observed.index[0], 'QRS_Dur_I'] = -5.
        pd.testing.assert_frame_equal(impute_measurements(observed, fitted)['raw'], observed)

    def test_sparse_and_unsupported_fallbacks_and_generated_clipping(self):
        raw = self.raw.copy()
        raw['S_Amp_II'] = np.nan
        raw['T_Amp_II'] = np.nan
        raw.loc[raw.index[0], 'T_Amp_II'] = 7.
        raw['QRS_Dur_III'] = -5.
        fitted = fit_imputer(raw, 'ridge', targets=['S_Amp_II', 'T_Amp_II', 'QRS_Dur_III'])
        changed = raw.iloc[:2].copy()
        changed.loc[changed.index[0], 'QRS_Dur_III'] = np.nan
        result = impute_measurements(changed, fitted)
        self.assertEqual(fitted['models']['T_Amp_II']['fallback_reason'], 'insufficient_observed_target')
        self.assertTrue(result['raw']['S_Amp_II'].eq(0).all())
        self.assertTrue(result['unsupported_mask']['S_Amp_II'].all())
        self.assertEqual(result['raw'].iloc[1]['T_Amp_II'], 7.)
        self.assertEqual(result['raw'].iloc[0]['QRS_Dur_III'], 0.)
        self.assertEqual(result['raw'].iloc[1]['QRS_Dur_III'], -5.)
        self.assertEqual(result['audit']['clipped_generated_cells'], 1)

    def test_no_fitted_or_available_predictors_falls_back_with_reason(self):
        constant_predictors = self.raw.copy()
        constant_predictors[NOISE_COLUMNS] = 1.
        constant_predictors['R_Amp_II'] = self.raw['R_Amp_II']
        no_fit = fit_imputer(constant_predictors, 'ridge', targets=['R_Amp_II'])
        self.assertEqual(no_fit['models']['R_Amp_II']['fallback_reason'], 'no_supported_predictors')
        train = self.raw.copy()
        train['R_Amp_II'] = np.exp(train['R_Amp_I'])
        fitted = fit_imputer(train, 'ridge', targets=['R_Amp_II'], max_predictors=1)
        model = fitted['models']['R_Amp_II']
        hidden = train.iloc[:2].copy()
        hidden['R_Amp_II'] = np.nan
        hidden.loc[hidden.index[0], model['predictors']] = np.nan
        result = impute_measurements(hidden, fitted)
        self.assertEqual(result['raw'].iloc[0]['R_Amp_II'], train['R_Amp_II'].median())
        self.assertEqual(result['audit']['generated_cells_by_method'], {'ridge': 1, 'median': 1})
        self.assertEqual(result['audit']['median_fallback_cells_by_reason'], {'all_predictors_missing_at_transform': 1})

    def test_mask_provenance_and_stable_subsets(self):
        a = mask_observed_measurements(self.raw)
        b = mask_observed_measurements(self.raw.iloc[::-1, ::-1])
        pd.testing.assert_frame_equal(a['raw'], b['raw'].loc[self.raw.index])
        c = mask_observed_measurements(self.raw.iloc[:5])
        pd.testing.assert_frame_equal(a['raw'].iloc[:5], c['raw'])
        self.assertFalse((a['scoring_mask'] & self.raw[NOISE_COLUMNS].isna()).to_numpy().any())
        self.assertFalse((a['natural_missing_mask'] & a['artificial_missing_mask']).to_numpy().any())
        pd.testing.assert_frame_equal(a['natural_missing_mask'] | a['artificial_missing_mask'], a['raw'].isna())
        protected = [c for c in RAW_COLUMNS if c not in NOISE_COLUMNS]
        pd.testing.assert_frame_equal(a['raw'][protected], self.raw[protected])
        self.assertFalse(a['raw'].equals(mask_observed_measurements(self.raw, repeat=1)['raw']))
        full = mask_observed_measurements(self.raw, fraction=1)['audit']
        empty = mask_observed_measurements(self.raw, fraction=0)['audit']
        self.assertEqual(full['observed_continuous_cells'], len(self.raw) * len(NOISE_COLUMNS) - 4)
        self.assertEqual(full['achieved_continuous_mask_fraction'], 1.)
        self.assertEqual(empty['achieved_continuous_mask_fraction'], 0.)
        no_values = self.raw.copy()
        no_values[NOISE_COLUMNS] = np.nan
        self.assertIsNone(mask_observed_measurements(no_values)['audit']['achieved_continuous_mask_fraction'])

    def test_whole_lead_masks_match_existing_evaluation(self):
        scale = pd.Series(0., index=NOISE_COLUMNS)
        for count in [1, 3, 6]:
            masked = mask_observed_measurements(self.raw, missing_leads=count, repeat=2)
            expected = perturb_measurements(self.raw, scale, missing_leads=count, repeat=2)
            pd.testing.assert_frame_equal(masked['raw'], expected)
            for ecg in self.raw.index:
                self.assertEqual(sum(masked['raw'].loc[ecg, columns].isna().all()
                                     for columns in LEAD_COLUMNS.values()), count)

    def test_recovery_units_normalization_missing_predictions_and_scope(self):
        fitted = fit_imputer(self.raw, targets=['R_Amp_I', 'QRS_Dur_I'])
        filled = self.raw.copy()
        scoring = pd.DataFrame(False, index=self.raw.index, columns=NOISE_COLUMNS)
        ids = self.raw.index[:3]
        scoring.loc[ids, ['R_Amp_I', 'QRS_Dur_I', 'P_Amp_I']] = True
        filled.loc[ids, 'R_Amp_I'] += 2.
        filled.loc[ids[:2], 'QRS_Dur_I'] += 4.
        filled.loc[ids[2], 'QRS_Dur_I'] = np.nan
        result = score_recovery(self.raw, filled, scoring, fitted)
        self.assertEqual(result['summary']['n_targets'], 6)
        self.assertEqual(result['summary']['n_scored'], 5)
        self.assertEqual(result['summary']['n_unresolved'], 1)
        self.assertEqual(result['summary']['n_excluded_targets'], 3)
        self.assertAlmostEqual(result['summary']['coverage'], 5/6)
        self.assertAlmostEqual(result['per_feature'].loc['R_Amp_I', 'mae'], 2.)
        self.assertAlmostEqual(result['per_feature'].loc['R_Amp_I', 'mae_in_training_iqr'], 2/3)
        self.assertAlmostEqual(result['per_feature'].loc['QRS_Dur_I', 'mae_in_training_iqr'], 2/3)
        self.assertAlmostEqual(result['summary']['cell_mean_mae_in_training_iqr'], 2/3)
        with self.assertRaises(ValueError):
            score_recovery(self.raw, filled, scoring.iloc[::-1], fitted)
        scoring.loc[self.raw.index[0], 'S_Amp_I'] = True
        with self.assertRaises(ValueError):
            score_recovery(self.raw, filled, scoring, fitted)

    def test_unsupported_values_remain_in_error_zero_iqr_not_normalized(self):
        train = self.raw.copy()
        train['R_Amp_I'] = np.nan
        fitted = fit_imputer(train, targets=['R_Amp_I'])
        masked = mask_observed_measurements(self.raw, fraction=1)
        predictions = impute_measurements(masked['raw'], fitted)
        result = score_recovery(self.raw, predictions['raw'], masked['scoring_mask'], fitted)
        self.assertEqual(result['summary']['n_scored'], len(self.raw))
        self.assertEqual(result['summary']['n_normalized_scored'], 0)
        self.assertGreater(result['per_feature'].loc['R_Amp_I', 'mae'], 0.)
        self.assertEqual(result['per_feature'].loc['R_Amp_I', 'training_support'], 0)

    def test_original_flags_and_observed_only_summaries_survive_integration(self):
        raw = self.raw.copy()
        raw.loc[raw.index[0], 'R_Amp_I'] = np.nan
        fitted = fit_imputer(self.raw, targets=['R_Amp_I'])
        imputed = impute_measurements(raw, fitted)
        features = build_features(raw)
        original_missing = features.isna()
        summary = features['R_Amp__median'].copy()
        features[fitted['targets']] = imputed['raw'][fitted['targets']]
        processor = fit_preprocessor(features)
        matrix = transform_features(features, processor, missing_mask=original_missing)
        self.assertEqual(matrix.iloc[0]['R_Amp_I__missing'], 1.)
        self.assertEqual(matrix.iloc[1]['R_Amp_I__missing'], 0.)
        pd.testing.assert_series_equal(features['R_Amp__median'], summary)
        self.assertTrue(np.isfinite(matrix.to_numpy()).all())

    def test_preprocessor_rejects_forged_or_unaligned_gap_masks(self):
        raw = self.raw.copy()
        raw.loc[raw.index[0], CATEGORY_COLUMNS[0]] = np.nan
        processor = fit_preprocessor(raw)
        mask = raw.isna()
        malformed = [mask.iloc[::-1], mask.iloc[:, ::-1], mask.astype(float), mask.iloc[:, 1:]]
        suppress_gap = mask.copy()
        suppress_gap['S_Amp_I'] = False
        malformed.append(suppress_gap)
        for invalid in malformed:
            with self.assertRaises(ValueError):
                transform_features(raw, processor, missing_mask=invalid)
        filled_category = raw.copy()
        filled_category.loc[raw.index[0], CATEGORY_COLUMNS[0]] = 1.
        with self.assertRaises(ValueError):
            transform_features(filled_category, processor, missing_mask=mask)

    def test_rejects_duplicate_nonfinite_and_invalid_config(self):
        duplicate = self.raw.copy()
        duplicate.index = [1] * len(duplicate)
        with self.assertRaises(ValueError):
            fit_imputer(duplicate)
        invalid = self.raw.copy()
        invalid.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            fit_imputer(invalid)
        for kwargs in [{'alpha': 0}, {'min_joint': 1}, {'max_predictors': 1.5}, {'targets': ['R_Amp_I'] * 2}]:
            with self.assertRaises(ValueError):
                fit_imputer(self.raw, **kwargs)
        with self.assertRaises(ValueError):
            mask_observed_measurements(self.raw, missing_leads=2)


if __name__ == '__main__':
    unittest.main()

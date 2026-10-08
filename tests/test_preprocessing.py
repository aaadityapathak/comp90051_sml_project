"""Focused correctness checks for fold-local augmentation and stress pairing."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from afib.preprocessing import (
    ARMS, CATEGORY_COLUMNS, LEADS, LEAD_COLUMNS, NOISE_COLUMNS, PARTS, RAW_COLUMNS,
    augmentation_assignments, build_features, construct_training_views, fit_perturber,
    fit_preprocessor, perturb_measurements, transform_features,
)


def sample_data(n=24):
    generator = np.random.default_rng(53)
    raw = pd.DataFrame(generator.uniform(5, 20, size=(n, 504)),
                       index=pd.Index(np.arange(100, 100+n), name='ecg_id'), columns=RAW_COLUMNS)
    raw.loc[:, CATEGORY_COLUMNS] = generator.integers(0, 3, size=(n, 24))
    raw.loc[:, [c for c in RAW_COLUMNS if PARTS[c][2] == 'count']] = 10
    raw.loc[raw.index[:3], 'P_Amp_I'] = np.nan
    raw['QRS_Dur_I'] = 5.0  # Degenerate-IQR numerical field.
    raw['P_DurFull_II'] = np.nan  # Entirely unobserved numerical field.
    labels = pd.Series([int(i % 4 == 0) for i in range(n)], index=raw.index)
    return raw, labels


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.y = sample_data()
        self.scale = fit_perturber(self.raw)

    def test_identity_natural_missingness_and_no_mutation(self):
        original = self.raw.copy(deep=True)
        pd.testing.assert_frame_equal(perturb_measurements(self.raw, self.scale), original)
        changed = perturb_measurements(self.raw, self.scale, missing_leads=3, noise_sigma=.1, gain_sigma=.1)
        self.assertTrue(changed.where(original.isna()).isna().all().all())
        pd.testing.assert_frame_equal(self.raw, original)

    def test_row_column_subset_and_call_order_invariance(self):
        kwargs = dict(missing_leads=3, noise_sigma=.1, gain_sigma=.1, repeat=2)
        a = perturb_measurements(self.raw, self.scale, **kwargs)
        _ = perturb_measurements(self.raw, self.scale, noise_sigma=.2, repeat=1)
        b = perturb_measurements(self.raw.iloc[::-1, ::-1], self.scale, **kwargs)
        pd.testing.assert_frame_equal(a, b.loc[a.index])
        c = perturb_measurements(self.raw.iloc[:5], self.scale, **kwargs)
        pd.testing.assert_frame_equal(c, a.iloc[:5])

    def test_masks_are_whole_lead_and_nested(self):
        one = perturb_measurements(self.raw, self.scale, missing_leads=1)
        three = perturb_measurements(self.raw, self.scale, missing_leads=3)
        self.assertTrue((one.notna() | three.isna()).all().all())
        for ecg in self.raw.index:
            hidden = [lead for lead in LEADS if three.loc[ecg, LEAD_COLUMNS[lead]].isna().all()]
            self.assertEqual(len(hidden), 3)
            for lead in set(LEADS) - set(hidden):
                pd.testing.assert_series_equal(three.loc[ecg, LEAD_COLUMNS[lead]], self.raw.loc[ecg, LEAD_COLUMNS[lead]])

    def test_discrete_and_zero_iqr_fields_unchanged(self):
        changed = perturb_measurements(self.raw, self.scale, noise_sigma=.2, gain_sigma=.2)
        protected = [c for c in RAW_COLUMNS if PARTS[c][0] == 'P_Morph' or PARTS[c][2] == 'count']
        pd.testing.assert_frame_equal(changed[protected], self.raw[protected])
        pd.testing.assert_series_equal(changed['QRS_Dur_I'], self.raw['QRS_Dur_I'])
        self.assertTrue(changed['P_DurFull_II'].isna().all())
        self.assertEqual(self.scale['P_DurFull_II'], 0)

    def test_clipping_audit_and_nonnegative_domain(self):
        raw = self.raw.copy()
        raw['QT_Int_I'] = 0.0
        scale = self.scale.copy()
        scale['QT_Int_I'] = 1000.0
        changed, audit = perturb_measurements(raw, scale, noise_sigma=.2, return_audit=True)
        self.assertTrue(changed['QT_Int_I'].ge(0).all())
        self.assertGreater(audit['clipped_nonnegative_cells'], 0)
        self.assertAlmostEqual(audit['clipping_fraction'], audit['clipped_nonnegative_cells'] / audit['clip_eligible_observed_nonnegative_cells'])
        self.assertGreater(audit['numeric_dose_before_mask']['rms_in_training_iqr'], 0)
        self.assertIn('qt_below_qrs_pairs', audit['qt_qrs_after'])

    def test_separate_rng_streams_and_paired_gaussian_doses(self):
        a = perturb_measurements(self.raw, self.scale, noise_sigma=.05)
        b = perturb_measurements(self.raw, self.scale, noise_sigma=.10)
        columns = [c for c in NOISE_COLUMNS if PARTS[c][0] == 'P_Amp' and PARTS[c][2] == 'value']
        np.testing.assert_allclose((b[columns]-self.raw[columns]).to_numpy(),
                                   2*(a[columns]-self.raw[columns]).to_numpy(), atol=1e-12)
        c = perturb_measurements(self.raw, self.scale, noise_sigma=.05, stream='augmentation')
        self.assertFalse(a.equals(c))

    def test_positive_gain_is_shared_within_lead_preserves_amplitude_sign(self):
        raw = self.raw.copy()
        raw['S_Amp_I'] = -raw['S_Amp_I']
        changed = perturb_measurements(raw, self.scale, gain_sigma=.2)
        np.testing.assert_allclose(changed['R_Amp_I']/raw['R_Amp_I'], changed['S_Amp_I']/raw['S_Amp_I'])
        self.assertTrue(changed['S_Amp_I'].lt(0).all())

    def test_augmented_weights_labels_pairing_and_training_only_statistics(self):
        original = construct_training_views(self.raw, self.y, **ARMS['A'])
        augmented = construct_training_views(self.raw, self.y, **ARMS['C'])
        enhanced = construct_training_views(self.raw, self.y, **ARMS['D'])
        self.assertEqual(len(augmented['X']), 2 * len(self.raw))
        pd.testing.assert_series_equal(augmented['sample_weight'].groupby('ecg_id').sum(),
                                       original['sample_weight'].droplevel('view'), check_names=False)
        self.assertAlmostEqual(augmented['sample_weight'][augmented['y'].eq(1)].sum(), len(self.raw)/2)
        pd.testing.assert_series_equal(augmented['y'].xs('corrupted'), self.y, check_names=False)
        pd.testing.assert_series_equal(augmented['preprocessor']['median'], original['preprocessor']['median'])
        pd.testing.assert_frame_equal(augmented['metadata'], enhanced['metadata'])
        self.assertEqual(augmented['augmentation_audit'], enhanced['augmentation_audit'])
        pd.testing.assert_frame_equal(augmented['X'], enhanced['X'][augmented['X'].columns])

    def test_balanced_augmentation_and_row_invariance(self):
        assignments = augmentation_assignments(self.raw.index[:23])
        self.assertLessEqual(assignments.value_counts().max()-assignments.value_counts().min(), 1)
        pd.testing.assert_series_equal(assignments, augmentation_assignments(self.raw.index[:23][::-1]).reindex(assignments.index))

    def test_features_recomputed_after_masks_and_preprocessor_edge_cases(self):
        changed = perturb_measurements(self.raw, self.scale, missing_leads=6)
        features = build_features(changed)
        self.assertEqual(features.shape[1], 567)
        pd.testing.assert_series_equal(features['QRS_Dur__median'], changed[[f'QRS_Dur_{lead}' for lead in LEADS]].median(axis=1), check_names=False)
        features['forced_empty'] = np.nan
        features['forced_constant'] = 7.
        fitted = fit_preprocessor(features)
        before = fitted['median'].copy()
        validation = features.iloc[:2].copy()
        validation.iloc[0, 0] = 1e12
        validation.loc[validation.index[0], CATEGORY_COLUMNS[0]] = 987654
        validation.loc[validation.index[1], CATEGORY_COLUMNS[0]] = np.nan
        out = transform_features(validation, fitted)
        self.assertEqual(out.iloc[0][CATEGORY_COLUMNS[0]+'__unknown'], 1)
        self.assertEqual(out.iloc[1][CATEGORY_COLUMNS[0]+'__missing'], 1)
        self.assertTrue(out['forced_empty__missing'].eq(1).all())
        self.assertTrue(out['forced_constant'].eq(0).all())
        pd.testing.assert_series_equal(fitted['median'], before)
        self.assertTrue(np.isfinite(out).all().all())

    def test_rejects_duplicate_ids_and_missing_scales(self):
        duplicated = self.raw.copy()
        duplicated.index = [1] * len(duplicated)
        with self.assertRaises(ValueError):
            perturb_measurements(duplicated, self.scale)
        with self.assertRaises(ValueError):
            perturb_measurements(self.raw, self.scale.iloc[1:])


if __name__ == '__main__':
    unittest.main()

"""Package integrity, patient isolation and ready-to-fit model input checks."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from afib.data_preparation import (
    QUALITY_COLUMNS, _write_package, load_prepared_split, load_processed_data,
    load_source_data, make_patient_folds, prepare_dataset,
)
from afib.preprocessing import CATEGORY_COLUMNS, RAW_COLUMNS, SEED, build_features, fit_preprocessor
from afib.paths import data_directory, processed_directory, source_path


def source_fixture(n=120):
    generator = np.random.default_rng(124)
    ids = pd.Index(range(1, n + 1), name='ecg_id')
    raw = pd.DataFrame(generator.uniform(10, 30, size=(n, 504)), index=ids, columns=RAW_COLUMNS)
    raw[CATEGORY_COLUMNS] = generator.integers(0, 3, size=(n, 24))
    raw.loc[ids[::5], 'P_Amp_I'] = np.nan
    raw['P_DurFull_II'] = np.nan
    raw['QRS_Dur_I'] = 10.0
    patients = np.arange(n) // 2 + 1000
    targets = pd.DataFrame({'patient_id': patients, 'target_afib': (patients % 2 == 0).astype('int64')}, index=ids)
    metadata = pd.DataFrame('', index=ids, columns=QUALITY_COLUMNS)
    return {'raw': raw, 'targets': targets, 'metadata': metadata, 'sources': {}, 'checks': {}}


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.source = source_fixture()
        self.folds = make_patient_folds(self.source['targets'])
        self.directory = self.root / 'processed_data'
        _write_package(self.directory, self.source, self.folds, 10, 3, SEED)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(Path(__file__).resolve().parent))
        self.temp.cleanup()

    def test_package_roundtrip_has_only_five_outputs(self):
        package = load_processed_data(self.directory)
        self.assertEqual(set(path.name for path in self.directory.iterdir()),
                         {'lead_features.npz', 'targets.csv', 'folds.csv', 'metadata.csv', 'manifest.json'})
        pd.testing.assert_frame_equal(package['raw'], self.source['raw'], check_dtype=False)
        pd.testing.assert_frame_equal(package['targets'], self.source['targets'])
        self.assertEqual(package['manifest']['rows'], 120)

    def test_patient_folds_are_reproducible_row_order_independent_and_disjoint(self):
        reordered = make_patient_folds(self.source['targets'].iloc[::-1]).sort_index()
        pd.testing.assert_frame_equal(self.folds, reordered)
        for outer in range(10):
            outer_test = self.folds.outer_fold.eq(outer)
            for inner in range(3):
                inner_valid = ~outer_test & self.folds[f'inner_fold_when_outer_{outer}'].eq(inner)
                inner_train = ~outer_test & ~inner_valid
                groups = [set(self.folds.loc[mask, 'patient_id']) for mask in [outer_test, inner_valid, inner_train]]
                self.assertTrue(groups[0].isdisjoint(groups[1] | groups[2]))
                self.assertTrue(groups[1].isdisjoint(groups[2]))

    def test_inner_loader_excludes_outer_test_and_fits_current_training_only(self):
        split = load_prepared_split(self.directory, outer_fold=0, inner_fold=1, arm='B')
        ids = split['X_train'].index.get_level_values('ecg_id')
        forbidden = self.folds.index[self.folds.outer_fold.eq(0)]
        self.assertTrue(set(ids).isdisjoint(forbidden))
        self.assertTrue(set(split['X_valid'].index).isdisjoint(forbidden))
        expected = fit_preprocessor(build_features(self.source['raw'].loc[ids]))
        pd.testing.assert_series_equal(split['preprocessor']['median'], expected['median'])
        pd.testing.assert_series_equal(split['preprocessor']['mean'], expected['mean'])
        self.assertEqual(split['X_train'].columns.tolist(), split['X_valid'].columns.tolist())
        self.assertTrue(np.isfinite(split['X_train']).all().all())
        self.assertTrue(np.isfinite(split['X_valid']).all().all())
        self.assertNotIn('patient_id', split['X_train'].columns)
        self.assertNotIn('target_afib', split['X_train'].columns)

    def test_held_out_changes_cannot_change_training_transform(self):
        before = load_prepared_split(self.directory, outer_fold=0)
        held_ids = self.folds.index[self.folds.outer_fold.eq(0)]
        self.source['raw'].loc[held_ids, 'R_Amp_I'] = 1e9
        self.source['raw'].loc[held_ids, CATEGORY_COLUMNS[0]] = 9876
        _write_package(self.directory, self.source, self.folds, 10, 3, SEED)
        after = load_prepared_split(self.directory, outer_fold=0)
        pd.testing.assert_frame_equal(before['X_train'], after['X_train'])
        self.assertTrue(after['X_valid'][CATEGORY_COLUMNS[0] + '__unknown'].eq(1).all())

    def test_augmentation_keeps_patient_metadata_targets_and_total_weights(self):
        observed = load_prepared_split(self.directory, outer_fold=1, arm='A')
        augmented = load_prepared_split(self.directory, outer_fold=1, arm='C')
        self.assertEqual(len(augmented['X_train']), 2 * len(observed['X_train']))
        pd.testing.assert_series_equal(augmented['sample_weight'].groupby('ecg_id').sum(),
                                       observed['sample_weight'].droplevel('view'), check_names=False)
        self.assertTrue(augmented['train_metadata'].index.equals(augmented['X_train'].index))
        self.assertTrue(augmented['train_metadata'].patient_id.notna().all())
        pd.testing.assert_frame_equal(observed['X_valid'], augmented['X_valid'])

    def test_stress_changes_held_out_only_and_unknown_choices_fail(self):
        observed = load_prepared_split(self.directory, outer_fold=0)
        stressed = load_prepared_split(self.directory, outer_fold=0, scenario='missing_3', repeat=1)
        pd.testing.assert_frame_equal(observed['X_train'], stressed['X_train'])
        self.assertFalse(observed['X_valid'].equals(stressed['X_valid']))
        for kwargs in ({'outer_fold': -1}, {'outer_fold': 0.0}, {'outer_fold': 0, 'inner_fold': 3},
                       {'outer_fold': 0, 'arm': 'E'}, {'outer_fold': 0, 'scenario': 'bad'}):
            with self.assertRaises(ValueError):
                load_prepared_split(self.directory, **kwargs)

    def test_file_tampering_is_rejected(self):
        with (self.directory / 'targets.csv').open('a', encoding='utf-8') as handle:
            handle.write('999,999,1\n')
        with self.assertRaisesRegex(ValueError, 'does not match the manifest'):
            load_processed_data(self.directory)

    def _write_sources(self, project=None):
        project = self.root if project is None else project
        dataset = project / 'data/example'
        (dataset / 'features').mkdir(parents=True)
        (dataset / 'labels').mkdir()
        (project / 'data/reference').mkdir()
        features = self.source['raw'].copy()
        for index in range(27):
            features[f'Test_Global_{index}'] = index
        features.to_csv(dataset / 'features/ecgdeli_features.csv', index_label='ecg_id')
        labels = self.source['targets'].copy()
        labels['scp_codes'] = labels.target_afib.map({0: "{'SR': 100}", 1: "{'AFIB': 0}"})
        labels[['scp_codes']].to_csv(dataset / 'labels/ptbxl_statements.csv', index_label='ecg_id')
        metadata = labels[['patient_id', 'scp_codes']].join(self.source['metadata'])
        metadata.to_csv(project / 'data/reference/ptbxl_database.csv', index_label='ecg_id')
        return dataset

    def test_end_to_end_source_join_and_output_boundary(self):
        project = self.root / 'fresh_project'
        self._write_sources(project)
        before = {path.relative_to(project): path.read_bytes() for path in project.rglob('*') if path.is_file()}
        destination = self.root / 'external_processed'
        manifest = prepare_dataset(project, data_dir=project / 'data', processed_dir=destination)
        after = {path.relative_to(project) for path in project.rglob('*') if path.is_file()}
        self.assertEqual(after, set(before))
        self.assertEqual({path.name for path in destination.iterdir()},
                         {'lead_features.npz', 'targets.csv', 'folds.csv', 'metadata.csv', 'manifest.json'})
        for path, original in before.items():
            self.assertEqual((project / path).read_bytes(), original)
        self.assertEqual(manifest['checks']['source_fields'], 531)
        self.assertEqual(manifest['checks']['rows_removed'], 0)
        pd.testing.assert_frame_equal(load_processed_data(destination)['targets'], self.source['targets'])

    def test_source_annotation_disagreement_fails(self):
        dataset = self._write_sources()
        path = dataset / 'labels/ptbxl_statements.csv'
        labels = pd.read_csv(path)
        labels.loc[0, 'scp_codes'] = "{'SR': 100}"
        labels.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            load_source_data(self.root, data_dir=self.root / 'data')

    def test_dataset_locations_are_external_and_overrides_are_stable(self):
        project = self.root / 'repository'
        with patch.dict('os.environ', {'AFIB_DATA_DIR': '../custom_raw',
                                      'AFIB_PROCESSED_DIR': '../custom_prepared'}):
            self.assertEqual(data_directory(project), self.root / 'custom_raw')
            self.assertEqual(processed_directory(project), self.root / 'custom_prepared')
            self.assertEqual(source_path(project, 'data/reference/ptbxl_database.csv'),
                             self.root / 'custom_raw/reference/ptbxl_database.csv')
        with self.assertRaisesRegex(ValueError, 'outside the repository'):
            prepare_dataset(project, processed_dir=project / 'processed_data')
        with self.assertRaisesRegex(ValueError, 'relative data/'):
            source_path(project, 'data/../private.csv')


if __name__ == '__main__':
    unittest.main()


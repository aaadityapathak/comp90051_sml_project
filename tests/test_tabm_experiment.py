"""Leakage, control, tuning and resumability checks without expensive training."""

from pathlib import Path
from types import SimpleNamespace
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
from models.tabm.data import (
    VARIANTS, identity, patient_weighted_metrics, split_ids,
    training_pipeline, transform_heldout,
)
from models.tabm.run_experiment import (
    checked_completion, fit_seed, inner_fit, json_write, main, output_lock,
    refinement_grid, run_signature, run_variant, select_blocks, source_sha256,
)
from afib.preprocessing import CATEGORY_COLUMNS, RAW_COLUMNS
from models.tabm.source_verification import RUNNER_SOURCE, active_source_name, compatible_signature


def layout_source_fixture(project):
    """A runner-shaped fixture with observable training and signature guards."""
    original = """import json
CANONICAL_OUTPUT = PROJECT / 'models/tabm'
def fit_model(weight):
    return weight * 2
def main():
    parser.error('Reduced runs require a separate --output, for example models/tabm/pilots/smoke.')
    if json.loads(config.read_text(encoding='utf-8')) != signature:
        raise ValueError('Different experiment')
"""
    current = original.replace('models/tabm', 'results/tabm').replace(
        'import json\n', 'import json\nfrom models.tabm.source_verification import compatible_signature\n').replace(
        "json.loads(config.read_text(encoding='utf-8')) != signature",
        "not compatible_signature(PROJECT, args.output, json.loads(config.read_text(encoding='utf-8')), signature)")
    path = Path(project) / active_source_name(RUNNER_SOURCE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(current, encoding='utf-8', newline='\n')
    output = Path(project) / 'results/tabm'
    snapshot = output / 'source_provenance/layout_v1' / RUNNER_SOURCE
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(original, encoding='utf-8', newline='\n')
    saved = {'epochs': 100, 'versions': {'torch': 'test'}, 'source_hash_normalization': 'LF',
             'code': {RUNNER_SOURCE: hashlib.sha256(original.encode()).hexdigest(),
                      'algorithms/tabm/model.py': 'unchanged-model-hash'}}
    signature = json.loads(json.dumps(saved))
    signature['code'] = {active_source_name(name): digest for name, digest in saved['code'].items()}
    signature['code'][active_source_name(RUNNER_SOURCE)] = source_sha256(path)
    return output, path, snapshot, saved, signature


def package_fixture():
    generator = np.random.default_rng(981)
    ids = pd.Index(range(100, 148), name='ecg_id')
    raw = pd.DataFrame(generator.uniform(5, 25, size=(len(ids), len(RAW_COLUMNS))),
                       index=ids, columns=RAW_COLUMNS)
    raw[CATEGORY_COLUMNS] = generator.integers(0, 3, size=(len(ids), len(CATEGORY_COLUMNS)))
    raw.loc[ids[::3], 'P_Amp_I'] = np.nan
    patients = np.arange(len(ids)) // 2
    labels = np.arange(len(ids)) % 2
    targets = pd.DataFrame({'patient_id': patients, 'target_afib': labels}, index=ids)
    folds = targets.assign(outer_fold=patients % 10)
    for outer in range(10):
        folds[f'inner_fold_when_outer_{outer}'] = np.where(patients % 10 == outer, -1, patients % 3)
    return {'raw': raw, 'targets': targets, 'folds': folds, 'manifest': {'seed': 20261008}}


class FakeClassifier:
    fits = 0

    def __init__(self, n_blocks, epochs, seed, device, batch_size):
        self.blocks, self.epochs = n_blocks, epochs

    def fit(self, X, y, weights):
        type(self).fits += 1
        self.audit_ = {'n_blocks': self.blocks, 'epochs': self.epochs, 'epochs_completed': self.epochs}
        self.history_ = [{'epoch': i + 1, 'weighted_bce': .5, 'seconds': .01} for i in range(self.epochs)]
        return self

    def predict_proba(self, X):
        return np.linspace(.15, .85, len(X)).astype(np.float32)

    def save(self, path):
        Path(path).write_bytes(b'synthetic checkpoint for runner test')

    @classmethod
    def load(cls, path, device='cpu'):
        return cls(n_blocks=2, epochs=2, seed=1, device=device, batch_size=256)


class TabMExperimentTests(unittest.TestCase):
    def setUp(self):
        self.package = package_fixture()
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / 'tests')
        self.output = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, **changes):
        values = {'output': self.output, 'epochs': 2, 'device': 'cpu', 'repeats': 1,
                  'outer_folds': [0], 'variants': ['A'], 'no_refine': False,
                  'phase': 'run', 'blocks': None}
        return SimpleNamespace(**(values | changes))

    def test_inner_partitions_preserve_patient_isolation(self):
        folds = self.package['folds']
        for inner in range(3):
            train, valid = split_ids(self.package, 0, inner)
            self.assertFalse(folds.loc[train.union(valid), 'outer_fold'].eq(0).any())
            self.assertTrue(set(folds.loc[train, 'patient_id']).isdisjoint(folds.loc[valid, 'patient_id']))
            outer_train, outer_test = split_ids(self.package, 0)
            self.assertEqual(set(train) | set(valid), set(outer_train))
            self.assertTrue(set(train).isdisjoint(outer_test))

    def test_duplicate_and_augmented_views_preserve_per_ecg_weights(self):
        train, valid = split_ids(self.package, 0)
        original = training_pipeline(self.package, train, 'A')
        expected_weights = original['sample_weight'].groupby('ecg_id').sum()
        for variant in ['C', 'duplicate_original']:
            pipeline = training_pipeline(self.package, train, variant)
            self.assertEqual(len(pipeline['X']), 2 * len(original['X']))
            pd.testing.assert_series_equal(pipeline['sample_weight'].groupby('ecg_id').sum(), expected_weights)
            pd.testing.assert_frame_equal(transform_heldout(self.package['raw'].loc[valid], pipeline),
                                          transform_heldout(self.package['raw'].loc[valid], original))
        duplicate = training_pipeline(self.package, train, 'duplicate_original')
        pd.testing.assert_frame_equal(duplicate['X'].xs('observed'), duplicate['X'].xs('duplicate'))

    def test_availability_control_retains_missingness_and_removes_only_derived_values(self):
        train, valid = split_ids(self.package, 0)
        enhanced = training_pipeline(self.package, train, 'B')
        control = training_pipeline(self.package, train, 'availability_observed')
        derived = [name for name in enhanced['X'] if name not in RAW_COLUMNS and not name.endswith('__missing')
                   and '__category_' not in name and not name.endswith('__unknown')]
        self.assertEqual(len(derived), 63)
        self.assertTrue(control['X'][derived].eq(0).all().all())
        pd.testing.assert_frame_equal(control['X'].drop(columns=derived), enhanced['X'].drop(columns=derived))
        heldout = transform_heldout(self.package['raw'].loc[valid], control)
        self.assertTrue(heldout[derived].eq(0).all().all())

    def test_heldout_values_cannot_change_training_preparation(self):
        train, valid = split_ids(self.package, 0, 1)
        before = training_pipeline(self.package, train, 'D')
        self.package['raw'].loc[valid, 'R_Amp_I'] = 1e12
        self.package['raw'].loc[valid, CATEGORY_COLUMNS[0]] = 999
        after = training_pipeline(self.package, train, 'D')
        pd.testing.assert_frame_equal(before['X'], after['X'])
        pd.testing.assert_series_equal(before['preprocessor']['median'], after['preprocessor']['median'])

    def test_manual_selection_ties_refines_once_and_uses_matched_seeds(self):
        self.assertEqual(select_blocks({2: .7, 3: .7, 4: .6}), 2)
        self.assertEqual(refinement_grid([2, 3, 4], 2), [1, 2, 3])
        self.assertEqual(refinement_grid([2, 3, 4], 4), [3, 4, 5])
        self.assertIsNone(refinement_grid([2, 3, 4], 3))
        with self.assertRaises(ValueError):
            refinement_grid([1, 2, 3], 1)
        with self.assertRaises(ValueError):
            select_blocks({2: np.nan, 3: .8})
        seeds = [fit_seed(20261008, outer, inner) for outer in range(10) for inner in [0, 1, 2, None]]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_patient_weights_give_each_patient_one_vote(self):
        result = patient_weighted_metrics([1, 1, 0], [.8, .8, .8], [7, 7, 9], .5)
        self.assertEqual(result['weighted_tp'], 1)
        self.assertEqual(result['weighted_fp'], 1)
        self.assertEqual(result['accuracy'], .5)

    def test_completed_fit_is_reused_and_tampered_predictions_are_rejected(self):
        FakeClassifier.fits = 0
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'models.tabm.model': SimpleNamespace(TabMClassifier=FakeClassifier),
                                      'torch': fake_torch}):
            first = inner_fit(self.package, 0, 'A', 2, 0, self.args(), {})
            second = inner_fit(self.package, 0, 'A', 2, 0, self.args(), {})
        self.assertEqual(FakeClassifier.fits, 1)
        self.assertEqual(first, second)
        location = self.output / 'output_data/tuning/00_A/blocks_2/inner_0'
        self.assertEqual(len(pd.read_csv(location / 'history.csv')), 2)
        self.assertTrue(first['checkpoints'])
        for relative, expected in first['checkpoints'].items():
            self.assertTrue((self.output / relative).exists())
        (location / 'predictions.csv.gz').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'recorded hash'):
            checked_completion(location)

    def test_parameter_scripts_share_signature_but_training_changes_do_not(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': fake_torch}), patch(
                'models.tabm.run_experiment.importlib.metadata.version', return_value='test-version'):
            left = run_signature(self.args(phase='tune', blocks=2), self.package)
            right = run_signature(self.args(phase='tune', blocks=4), self.package)
            changed = run_signature(self.args(epochs=3), self.package)
        self.assertEqual(left, right)
        self.assertNotEqual(left, changed)
        self.assertIn('models/tabm/model.py', left['code'])
        self.assertFalse(left['protocol_complete_requested'])

    def test_source_signature_ignores_git_line_endings_but_not_code_changes(self):
        first, second = self.output / 'first.py', self.output / 'second.py'
        first.write_bytes(b'def identity(x):\n    return x\n')
        second.write_bytes(b'def identity(x):\r\n    return x\r\n')
        self.assertEqual(source_sha256(first), source_sha256(second))
        second.write_bytes(b'def identity(x):\r\n    return x + 1\r\n')
        self.assertNotEqual(source_sha256(first), source_sha256(second))

    def test_device_aliases_share_signature_only_for_the_same_actual_device(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {'torch': fake_torch}), patch(
                'models.tabm.run_experiment.importlib.metadata.version', return_value='test-version'):
            automatic = run_signature(self.args(device='auto'), self.package)
            gpu = run_signature(self.args(device='cuda'), self.package)
            cpu = run_signature(self.args(device='cpu'), self.package)
        self.assertEqual(automatic, gpu)
        self.assertNotEqual(automatic, cpu)
        self.assertEqual(automatic['device'], 'cuda')
        self.assertEqual(cpu['resolved_device'], 'cpu')

    def test_outer_evaluation_resumes_completed_fit_and_exports_frozen_predictions(self):
        FakeClassifier.fits = 0
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        package = self.package
        package['metadata'] = pd.DataFrame('', index=package['raw'].index,
            columns=['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems'])
        args = self.args(no_refine=True)
        with patch.dict(sys.modules, {'models.tabm.model': SimpleNamespace(TabMClassifier=FakeClassifier),
                                      'torch': fake_torch}):
            with patch('models.tabm.run_experiment.importlib.metadata.version', return_value='test-version'):
                json_write(self.output / 'run_config.json', run_signature(args, package))
            run_variant(package, 0, 'A', args)
            location = self.output / 'output_data/folds/00_A'
            self.assertTrue(checked_completion(location)['completed'])
            self.assertEqual(FakeClassifier.fits, 10)
            predictions = pd.read_csv(location / 'outer_predictions.csv.gz')
            self.assertEqual(len(predictions.scenario.unique()), 11)
            self.assertEqual(set(predictions.patient_id), set(package['folds'].loc[
                package['folds'].outer_fold.eq(0), 'patient_id']))
            metrics = pd.read_csv(location / 'metrics.csv')
            self.assertEqual(len(metrics), 22)
            for _, condition in metrics.groupby('threshold_policy'):
                self.assertEqual(condition.threshold.nunique(), 1)
            # Simulate a crash after the expensive outer fit: rerun evaluation
            # and restore its model instead of repeating any training.
            (location / 'complete.json').unlink()
            run_variant(package, 0, 'A', args)
            self.assertEqual(FakeClassifier.fits, 10)
            pd.testing.assert_frame_equal(predictions, pd.read_csv(location / 'outer_predictions.csv.gz'))
        from models.tabm.check_results import check
        checked = check(self.output, package=package)
        self.assertTrue(checked['requested_scope_complete'])
        self.assertEqual(checked['completed_inner_checkpoints_verified'], 9)
        self.assertFalse(checked['complete_tabm_plan'])

    def test_full_plan_claim_requires_all_folds_controls_repeats_and_epochs(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': fake_torch}), patch(
                'models.tabm.run_experiment.importlib.metadata.version', return_value='test-version'):
            full = self.args(epochs=100, repeats=3, outer_folds=list(range(10)), variants=VARIANTS)
            self.assertTrue(run_signature(full, self.package)['protocol_complete_requested'])
            full.variants = ['A', 'B', 'C', 'D']
            self.assertFalse(run_signature(full, self.package)['protocol_complete_requested'])

    def test_pilot_cannot_write_canonical_output(self):
        with self.assertRaises(SystemExit):
            main(['--epochs', '1'])

    def test_layout_migration_preserves_signature_and_supports_nested_pilot(self):
        output, path, snapshot, saved, signature = layout_source_fixture(self.output)
        original_saved = json.dumps(saved)
        self.assertTrue(compatible_signature(self.output, output, saved, signature))
        self.assertTrue(compatible_signature(self.output, output / 'pilots/benchmark', saved, signature))
        self.assertEqual(json.dumps(saved), original_saved)
        snapshot.write_bytes(b'tampered source evidence')
        self.assertFalse(compatible_signature(self.output, output, saved, signature))

    def test_layout_migration_rejects_training_configuration_and_other_source_changes(self):
        output, path, snapshot, saved, signature = layout_source_fixture(self.output)
        for key, value in [('epochs', 1), ('versions', {'torch': 'different'})]:
            changed = signature | {key: value}
            self.assertFalse(compatible_signature(self.output, output, saved, changed))
        changed = json.loads(json.dumps(signature))
        changed['code']['models/tabm/model.py'] = 'changed-model-hash'
        self.assertFalse(compatible_signature(self.output, output, saved, changed))
        path.write_text(path.read_text().replace('weight * 2', 'weight * 3'), encoding='utf-8')
        signature['code'][active_source_name(RUNNER_SOURCE)] = source_sha256(path)
        self.assertFalse(compatible_signature(self.output, output, saved, signature))

    def test_layout_migration_requires_original_evidence_and_exact_guard_call(self):
        output, path, snapshot, saved, signature = layout_source_fixture(self.output)
        path.write_text(path.read_text().replace('if not compatible_signature(', 'if compatible_signature('),
                        encoding='utf-8')
        signature['code'][active_source_name(RUNNER_SOURCE)] = source_sha256(path)
        self.assertFalse(compatible_signature(self.output, output, saved, signature))
        output, path, snapshot, saved, signature = layout_source_fixture(self.output)
        snapshot.unlink()
        self.assertFalse(compatible_signature(self.output, output, saved, signature))

    def test_os_lock_prevents_second_process_then_allows_resume(self):
        source = (
            'from pathlib import Path; '
            'from models.tabm.run_experiment import output_lock; '
            'import sys; '
            'context=output_lock(Path(sys.argv[1])); '
            'context.__enter__(); context.__exit__(None,None,None)'
        )
        with output_lock(self.output):
            result = subprocess.run([sys.executable, '-c', source, str(self.output)],
                                    cwd=PROJECT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Another process owns', result.stderr)
        with output_lock(self.output):
            pass


if __name__ == '__main__':
    unittest.main()

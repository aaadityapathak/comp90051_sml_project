"""Reporting and independent-output checks using small saved predictions only."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))
from algorithms.tabm import reporting
from algorithms.tabm import check_results
from afib.evaluation import binary_metrics, choose_sensitivity_threshold, majority_baseline, quality_subgroup_metrics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def mark_complete(location, files):
    write_json(location / 'complete.json', {'completed': True,
               'files': {name: hashlib.sha256((location / name).read_bytes()).hexdigest() for name in files}})


def fixture(output, complete=False, repeats=3):
    labels = [0, 1, 0, 1] * 4
    folds = pd.DataFrame({'patient_id': range(101, 117), 'target_afib': labels,
                          'outer_fold': [0] * 4 + [1] * 12,
                          'inner_fold_when_outer_0': [-1] * 4 + [0] * 4 + [1] * 4 + [2] * 4},
                         index=pd.Index(range(1, 17), name='ecg_id'))
    package = {'manifest': {'synthetic': True}, 'folds': folds,
               'targets': folds[['patient_id', 'target_afib']]}
    config = {'model': 'tabm', 'variants': ['A'], 'outer_folds': [0], 'n_inner_folds': 3,
              'initial_grid': [2, 3, 4], 'refine': True, 'epochs': 2, 'repeats': repeats,
              'seed': 20261008, 'batch_size': 256, 'device': 'cpu',
              'architecture': {'width': 128, 'members': 16, 'dropout': .1, 'numerical_embeddings': False},
              'optimizer': {'name': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001},
              'versions': {'synthetic': 'test fixture'}, 'manifest': package['manifest'], 'code': {}}
    write_json(output / 'run_config.json', config)
    if not complete:
        return package
    history = pd.DataFrame({'epoch': [1, 2], 'weighted_bce': [.7, .6], 'seconds': [.01, .01]})
    fits, tuning, selected_rows = [], [], []
    for blocks in [2, 3, 4]:
        for inner in range(3):
            location = output / 'output_data' / 'tuning' / '00_A' / f'blocks_{blocks}' / f'inner_{inner}'
            location.mkdir(parents=True)
            reference = folds[folds.inner_fold_when_outer_0.eq(inner)]
            scores = [.8, .8, .3, .9] if blocks == 2 else [.1, .8, .3, .9]
            rows = pd.DataFrame({'ecg_id': reference.index, 'patient_id': reference.patient_id,
                                 'y_true': reference.target_afib, 'score': scores, 'inner_fold': inner})
            rows.to_csv(location / 'predictions.csv.gz', index=False)
            history.to_csv(location / 'history.csv', index=False)
            audit = {'epochs': 2, 'epochs_completed': 2, 'n_blocks': blocks, 'phase': 'inner',
                     'inner_fold': inner, 'macro_f1': binary_metrics(rows.y_true, rows.score)['macro_f1']}
            write_json(location / 'audit.json', audit)
            mark_complete(location, ['predictions.csv.gz', 'history.csv', 'audit.json'])
            fits.append(audit)
            tuning.append({'outer_fold': 0, 'variant': 'A', 'n_blocks': blocks, 'inner_fold': inner,
                           'stage': 'initial', 'macro_f1': audit['macro_f1']})
            if blocks == 3:
                selected_rows.append(rows)
    outer_path = output / 'output_data' / 'folds' / '00_A'
    outer_path.mkdir(parents=True)
    oof = pd.concat(selected_rows, ignore_index=True)
    oof.to_csv(outer_path / 'inner_oof_predictions.csv.gz', index=False)
    threshold = choose_sensitivity_threshold(oof.y_true, oof.score)
    write_json(outer_path / 'selection.json', {'n_blocks': 3, 'initial_grid': [2, 3, 4],
               'final_grid': [2, 3, 4], 'selected_middle': True, 'refined': False,
               'secondary_threshold': threshold, 'seconds': .2, 'variant': 'A', 'outer_fold': 0})
    outer_audit = {'epochs': 2, 'epochs_completed': 2, 'n_blocks': 3, 'phase': 'outer_refit'}
    write_json(outer_path / 'outer_fit_audit.json', outer_audit)
    write_json(outer_path / 'fit_audit.json', fits + [outer_audit])
    write_json(outer_path / 'perturbation_audit.json', {'training': {}, 'test': []})
    history.to_csv(outer_path / 'training_history.csv', index=False)
    pd.DataFrame(tuning).to_csv(outer_path / 'tuning.csv', index=False)
    predictions, metrics, patients, subgroups, calibration = [], [], [], [], []
    for scenario, repeat in sorted(reporting.expected_conditions(repeats)):
        meta = {'model': 'tabm', 'variant': 'A', 'outer_fold': 0, 'scenario': scenario, 'repeat': repeat}
        rows = pd.DataFrame({'ecg_id': range(1, 5), 'patient_id': range(101, 105),
                             'y_true': labels[:4], 'score': [.1, .8, .3, .9],
                             'quality_group': ['no_recorded_quality_flag'] * 4}).assign(**meta)
        predictions.append(rows)
        for policy, decision in [('fixed_0.50', .5), ('inner_sensitivity_0.90', threshold['threshold'])]:
            key = {**meta, 'threshold_policy': policy}
            metrics.append({**key, **binary_metrics(rows.y_true, rows.score, decision)})
            patients.append({**key, **check_results._weighted_metrics(rows, decision)})
            subgroups.extend({**key, **record} for record in
                             quality_subgroup_metrics(rows.y_true, rows.score, rows.quality_group, decision).to_dict('records'))
        for bucket in range(10):
            selected = rows[np.minimum((rows.score * 10).astype(int), 9).eq(bucket)]
            calibration.append({**meta, 'bin': bucket, 'n': len(selected),
                                'mean_probability': selected.score.mean(), 'fraction_positive': selected.y_true.mean()})
    pd.concat(predictions).to_csv(outer_path / 'outer_predictions.csv.gz', index=False)
    for name, records in [('metrics', metrics), ('patient_weighted_metrics', patients),
                          ('quality_subgroups', subgroups), ('reliability', calibration)]:
        pd.DataFrame(records).to_csv(outer_path / f'{name}.csv', index=False)
    write_json(outer_path / 'majority_baseline.json',
               majority_baseline(folds.loc[folds.outer_fold.ne(0), 'target_afib'], labels[:4]))
    mark_complete(outer_path, reporting.FOLD_FILES)
    return package


class ReportingTests(unittest.TestCase):
    def test_fold_aggregation_propagates_undefined_and_uses_sample_sd(self):
        rows = []
        for outer, values in [(0, [.2, .4, .6]), (1, [.6, .8, 1.])]:
            for repeat, value in enumerate(values):
                rows.append({'variant': 'A', 'scenario': 'noise_0.10', 'threshold_policy': 'fixed_0.50',
                             'outer_fold': outer, 'repeat': repeat, 'macro_f1': value,
                             'precision': np.nan if outer == 0 and repeat == 0 else value})
        folds, summary = reporting.fold_summaries(pd.DataFrame(rows))
        result = summary.set_index('metric')
        self.assertAlmostEqual(result.loc['macro_f1', 'mean'], .6)
        self.assertAlmostEqual(result.loc['macro_f1', 'sd'], np.std([.4, .8], ddof=1))
        self.assertTrue(np.isnan(result.loc['precision', 'mean']))
        self.assertEqual(result.loc['precision', 'n_defined'], 1)

    def test_paired_comparison_uses_shared_folds_and_brier_deterioration_sign(self):
        rows = []
        for variant, folds in [('A', [0, 1]), ('B', [1, 2])]:
            for fold in folds:
                for scenario, brier in [('observed', .1), ('missing_3', .3)]:
                    rows.append({'variant': variant, 'outer_fold': fold, 'scenario': scenario,
                                 'threshold_policy': 'fixed_0.50', 'brier': brier,
                                 'macro_f1': .8 + (.1 if variant == 'B' else 0)})
        drops, comparisons = reporting.paired_effects(pd.DataFrame(rows))
        self.assertTrue(np.allclose(drops[drops.metric.eq('brier')].mean_deterioration, .2))
        selected = comparisons[comparisons.metric.eq('macro_f1')]
        self.assertTrue(selected.n_folds.eq(1).all())
        self.assertTrue(np.allclose(selected.mean_difference, .1))
        self.assertTrue(selected.sd_difference.isna().all())

    def test_empty_report_is_incomplete_and_strict_checker_rejects(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            package = fixture(output)
            (output / 'output_data').mkdir()
            pd.DataFrame({'macro_f1': [1.0]}).to_csv(output / 'output_data' / 'metrics.csv', index=False)
            with patch.object(reporting, 'load_processed_data', return_value=package):
                status = reporting.generate_reports(output)
            self.assertFalse(status['complete_tabm_plan'])
            self.assertEqual(status['n_fits'], 0)
            self.assertEqual(status['completed_outer_fold_variants'], 0)
            self.assertTrue(pd.read_csv(output / 'output_data' / 'metrics.csv').empty)
            self.assertEqual({path.name for path in (output / 'reports').glob('*.png')},
                             {'progress.png', 'robustness_macro_f1.png', 'robustness_sensitivity.png',
                              'robustness_brier.png', 'calibration.png', 'training_loss.png'})
            self.assertIn('INCOMPLETE TABM PLAN', (output / 'reports' / 'report.md').read_text(encoding='utf-8'))
            with self.assertRaisesRegex(ValueError, 'Run incomplete'):
                check_results.check(output, package=package)
            result = check_results.check(output, allow_partial=True, package=package)
            self.assertEqual(result['status'], 'passed_available_outputs_run_incomplete')

    def test_completed_reduced_run_verifies_31_conditions_without_full_claim(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            package = fixture(output, complete=True)
            with patch.object(reporting, 'load_processed_data', return_value=package):
                status = reporting.generate_reports(output)
            self.assertTrue(status['requested_scope_complete'])
            self.assertFalse(status['complete_tabm_plan'])
            self.assertEqual(status['n_fits'], 10)
            self.assertEqual(status['completed_tuning_candidates'], 3)
            result = check_results.check(output, package=package)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['outer_conditions_verified'], 31)
            self.assertTrue(result['fold_mean_and_sd_verified'])
            # Even replacing the completion hashes cannot hide wrong numeric metrics.
            location = output / 'output_data' / 'folds' / '00_A'
            metrics = pd.read_csv(location / 'metrics.csv')
            metrics.loc[0, 'accuracy'] = .123
            metrics.to_csv(location / 'metrics.csv', index=False)
            mark_complete(location, reporting.FOLD_FILES)
            with self.assertRaises(AssertionError):
                check_results.check(output, package=package)

    def test_full_settings_require_epochs_controls_and_exact_architecture(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            fixture(output)
            config = reporting.read_json(output / 'run_config.json')
            config.update(variants=list(reporting.VARIANTS), outer_folds=list(range(10)), epochs=100)
            self.assertTrue(reporting.full_protocol_settings(config))
            config['epochs'] = 2
            self.assertFalse(reporting.full_protocol_settings(config))
            config['epochs'] = 100
            config['architecture']['members'] = 2
            self.assertFalse(reporting.full_protocol_settings(config))

    def test_patient_overlap_is_rejected_before_metrics(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            package = fixture(output)
            package['folds'].loc[5, 'patient_id'] = package['folds'].loc[1, 'patient_id']
            with self.assertRaisesRegex(ValueError, 'Patient leakage'):
                check_results.check(output, allow_partial=True, package=package)

    def test_source_normalization_accepts_checkout_line_endings_but_rejects_code_change(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            package = fixture(output)
            source = output / 'source.py'
            source.write_bytes(b'value = 1\r\n')
            config = reporting.read_json(output / 'run_config.json')
            config.update(source_hash_normalization='LF',
                          code={'source.py': hashlib.sha256(b'value = 1\n').hexdigest()})
            write_json(output / 'run_config.json', config)
            with patch.object(check_results, 'PROJECT', output):
                result = check_results.check(output, allow_partial=True, package=package)
                self.assertEqual(result['source_files_verified'], 1)
                source.write_bytes(b'value = 2\r\n')
                with self.assertRaisesRegex(ValueError, 'Training source changed'):
                    check_results.check(output, allow_partial=True, package=package)

    def test_checker_reports_preserved_layout_sources_and_rejects_training_changes(self):
        from test_tabm_experiment import layout_source_fixture
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            project = Path(temp)
            output, source, snapshot, saved, signature = layout_source_fixture(project)
            package = fixture(output)
            config = reporting.read_json(output / 'run_config.json')
            config.update(source_hash_normalization='LF', code={
                'algorithms/tabm/run_experiment.py': saved['code']['algorithms/tabm/run_experiment.py']})
            write_json(output / 'run_config.json', config)
            with patch.object(check_results, 'PROJECT', project):
                result = check_results.check(output, allow_partial=True, package=package)
                self.assertEqual(result['source_verification']['mode'], 'preserved_layout_v1_sources')
                self.assertEqual(result['source_files_verified'], 1)
                source.write_text(source.read_text().replace('weight * 2', 'weight * 3'), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Training source changed'):
                    check_results.check(output, allow_partial=True, package=package)


if __name__ == '__main__':
    unittest.main()


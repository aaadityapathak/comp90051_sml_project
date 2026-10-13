"""Completed-result refresh must never launch training or publishing by default."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from algorithms.tabm import run_full_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT / 'tests')
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.output = self.project / 'results/tabm'
        (self.output / 'reports').mkdir(parents=True)
        (self.output / 'run_config.json').write_text('{}', encoding='utf-8')
        (self.output / 'reports/completion.json').write_text(
            json.dumps({'complete_tabm_plan': True}), encoding='utf-8')

    def run_refresh(self, flags=(), side_effect=None):
        with patch.object(pipeline, 'PROJECT', self.project), \
                patch.object(pipeline, 'command', side_effect=side_effect) as commands:
            pipeline.main(['--report-only', *flags])
        return commands

    def test_fast_refresh_has_no_training_git_or_notebook_execution(self):
        calls = self.run_refresh(['--skip-notebook']).call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0][2:], [
            'algorithms/tabm/run_experiment.py', '--phase', 'report', '--device', 'auto'])
        self.assertEqual(calls[1].args[0][1:3], ['algorithms/tabm/check_results.py', '--save'])
        status = json.loads((self.output / 'pipeline_status.json').read_text())
        self.assertEqual(status['state'], 'completed_and_verified')
        self.assertFalse(status['publish_on_completion'])
        self.assertTrue(status['report_only'])

    def test_optional_notebook_refresh_explicitly_disables_training(self):
        calls = self.run_refresh().call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1].args[0][1], 'scripts/execute_notebook.py')
        self.assertEqual(calls[-1].kwargs['env']['TABM_RUN_TRAINING'], '0')

    def test_refresh_requires_an_existing_run(self):
        (self.output / 'run_config.json').unlink()
        with patch.object(pipeline, 'PROJECT', self.project), patch.object(pipeline, 'command') as commands:
            with self.assertRaisesRegex(ValueError, 'requires an existing'):
                pipeline.main(['--report-only', '--skip-notebook'])
        commands.assert_not_called()

    def test_failed_validation_stops_before_notebook_and_records_failure(self):
        error = subprocess.CalledProcessError(1, ['check_results.py'])
        with patch.object(pipeline, 'PROJECT', self.project), \
                patch.object(pipeline, 'command', side_effect=[None, error]) as commands:
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.main(['--report-only'])
        self.assertEqual(commands.call_count, 2)
        status = json.loads((self.output / 'pipeline_status.json').read_text())
        self.assertEqual(status['state'], 'failed')

    def test_partial_run_cannot_be_marked_complete(self):
        (self.output / 'reports/completion.json').write_text(
            json.dumps({'complete_tabm_plan': False}), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'completion was not verified'):
            self.run_refresh(['--skip-notebook'])
        self.assertEqual(json.loads((self.output / 'pipeline_status.json').read_text())['state'], 'failed')


if __name__ == '__main__':
    unittest.main()

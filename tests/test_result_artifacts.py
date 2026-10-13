"""Bundle integrity checks must detect tampering, missing files and unsafe paths."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_result_artifacts import FOLD_REQUIRED, check_manifest, check_git_index, compare_git_index


class ResultArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.fold = self.output / 'output_data' / 'folds' / '00_A'
        self.fold.mkdir(parents=True)
        (self.output / 'reports').mkdir()
        self.write('run_config.json', {'outer_folds': [0], 'variants': ['A']})
        self.write('reports/completion.json', {'complete_logistic_plan': True, 'n_fits': 1})
        self.write('reports/validation.json', {'status': 'passed'})
        for name in FOLD_REQUIRED:
            (self.fold / name).write_text('{}', encoding='utf-8')
        self.write('output_data/folds/00_A/complete.json', {'completed': True})

    def write(self, relative, value):
        (self.output / relative).write_text(json.dumps(value), encoding='utf-8')

    def test_integrity_is_portable_and_ignores_local_state(self):
        check_manifest(self.output, 'logistic_regression', write=True)
        (self.output / 'status.json').write_text('running')
        result = check_manifest(self.output, 'logistic_regression')
        self.assertEqual(result['status'], 'passed')
        manifest = json.loads((self.output / 'reports/artifact_manifest.json').read_text())
        self.assertTrue(all('\\' not in name and not Path(name).is_absolute() for name in manifest['files']))

    def test_changed_or_added_artifact_is_rejected(self):
        check_manifest(self.output, 'logistic_regression', write=True)
        path = self.fold / 'metrics.csv'
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact integrity failed'):
            check_manifest(self.output, 'logistic_regression')
        path.write_text('{}')
        (self.fold / 'unexpected.csv').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'artifact integrity failed'):
            check_manifest(self.output, 'logistic_regression')

    def test_missing_fold_and_failed_numerical_validation_are_rejected(self):
        self.write('run_config.json', {'outer_folds': [0, 1], 'variants': ['A']})
        with self.assertRaisesRegex(ValueError, 'fold coverage differs'):
            check_manifest(self.output, 'logistic_regression', write=True)
        self.write('reports/validation.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'independent validation has not passed'):
            check_manifest(self.output, 'logistic_regression', write=True)

    def test_tabm_original_hash_prevents_accepting_corruption(self):
        self.write('reports/completion.json', {'complete_tabm_plan': True, 'n_fits': 1})
        self.write('output_data/folds/00_A/complete.json', {
            'completed': True, 'files': {'metrics.csv': hashlib.sha256(b'{}').hexdigest()}})
        check_manifest(self.output, 'tabm', write=True)
        (self.fold / 'metrics.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'original artifact hash changed'):
            check_manifest(self.output, 'tabm', write=True)

    def test_tabm_marker_cannot_reference_a_parent_directory(self):
        self.write('reports/completion.json', {'complete_tabm_plan': True, 'n_fits': 1})
        self.write('output_data/folds/00_A/complete.json', {
            'completed': True, 'files': {'../00_A/metrics.csv': hashlib.sha256(b'{}').hexdigest()}})
        with self.assertRaisesRegex(ValueError, 'Unsafe artifact name'):
            check_manifest(self.output, 'tabm', write=True)

    def test_staged_check_detects_missing_and_line_ending_changes(self):
        path = self.output / 'staged.csv'
        path.write_bytes(b'a,b\n1,2\n')
        blob = b'blob 8\0a,b\n1,2\n'
        index = {'staged.csv': hashlib.sha1(blob).hexdigest()}
        compare_git_index(self.output.resolve(), [path], index, 'sha1')
        path.write_bytes(b'a,b\r\n1,2\r\n')
        with self.assertRaisesRegex(ValueError, 'different_bytes'):
            compare_git_index(self.output.resolve(), [path], index, 'sha1')
        with self.assertRaisesRegex(ValueError, 'unstaged/unmerged'):
            compare_git_index(self.output.resolve(), [path], {}, 'sha1')

    def test_index_inventory_uses_repository_root(self):
        # git ls-files from results/ would omit tracked data outside that folder
        # and return result paths without the results/ prefix.
        attributes = b'/results/** -text\n'
        (self.output / '.gitattributes').write_bytes(attributes)
        object_id = hashlib.sha1(f'blob {len(attributes)}\0'.encode() + attributes).hexdigest()
        outputs = [SimpleNamespace(stdout=str(self.output.resolve())),
                   SimpleNamespace(stdout='sha1'),
                   SimpleNamespace(stdout=f'100644 {object_id} 0\t.gitattributes\0')]
        with patch('scripts.verify_result_artifacts.subprocess.run', side_effect=outputs) as run:
            check_git_index(self.output / 'results', [])
        self.assertEqual(run.call_args_list[2].args[0][2], str(self.output.resolve()))


if __name__ == '__main__':
    unittest.main()

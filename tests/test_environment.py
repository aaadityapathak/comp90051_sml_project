"""Historical environment checks must not disguise a different numerical stack."""
from importlib import metadata
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_environment import compare_versions


class EnvironmentTests(unittest.TestCase):
    def test_python_patch_and_cuda_wheel_suffix_are_significant(self):
        rows = compare_versions({'python': '3.12.15', 'torch': '2.7.1+cu128'},
                                python_version='3.12.10', package_version=lambda _: '2.7.1')
        self.assertTrue(all(not row['matches'] for row in rows))

    def test_missing_package_is_reported_without_importing_it(self):
        def missing(name):
            raise metadata.PackageNotFoundError(name)
        rows = compare_versions({'tabm': '0.0.3'}, package_version=missing)
        self.assertIsNone(rows[0]['installed'])
        self.assertFalse(rows[0]['matches'])

    def test_matching_recorded_versions(self):
        rows = compare_versions({'python': '3.12.10', 'tabm': '0.0.3'},
                                python_version='3.12.10', package_version=lambda _: '0.0.3')
        self.assertTrue(all(row['matches'] for row in rows))


if __name__ == '__main__':
    unittest.main()

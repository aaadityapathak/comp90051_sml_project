"""Rebuild preprocessing in a temporary folder and compare with saved run inputs.

Existing source data, processed_data, predictions and trained models are untouched.
Requires the original three local CSV files, but performs no model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.data_preparation import _write_package, load_source_data, make_patient_folds
from afib.preprocessing import SEED


def verify(project=PROJECT):
    project = Path(project).resolve()
    source = load_source_data(project)
    folds = make_patient_folds(source['targets'])
    runtime = project / '.runtime'
    runtime.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='preprocessing-check-', dir=runtime)
    destination = Path(temporary.name).resolve()
    if destination.parent != runtime.resolve():
        raise RuntimeError('Temporary preprocessing output escaped the runtime directory.')
    try:
        rebuilt = _write_package(destination, source, folds, 10, 3, SEED)
        comparisons = {}
        expected_manifests = {
            'processed_data': json.loads((project / 'processed_data/manifest.json').read_text(encoding='utf-8')),
        }
        for model in ('logistic_regression', 'tabm'):
            config = json.loads((project / 'results' / model / 'run_config.json').read_text(encoding='utf-8'))
            expected_manifests[model] = config['manifest']
        for name, expected in expected_manifests.items():
            comparisons[name] = {
                'matches': rebuilt == expected,
                'different_manifest_fields': sorted(
                    field for field in set(rebuilt) | set(expected)
                    if rebuilt.get(field) != expected.get(field)),
            }
        return {'status': 'passed' if all(row['matches'] for row in comparisons.values()) else 'failed',
                'model_training_performed': False, 'existing_processed_data_modified': False,
                'rows': rebuilt['rows'], 'patients': rebuilt['patients'],
                'source_files': rebuilt['sources'], 'rebuilt_files': rebuilt['files'],
                'comparisons': comparisons}
    finally:
        temporary.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path, help='Optional JSON verification report.')
    args = parser.parse_args(argv)
    result = verify()
    text = json.dumps(result, indent=2) + '\n'
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Compare installed versions with a saved run without importing model libraries.

This checks the versions the run recorded; it does not establish equivalence of
hardware, drivers, unrecorded dependencies, source code, or training outputs.
Saved-prediction verification can still succeed in a different environment.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform

PROJECT = Path(__file__).resolve().parents[1]


def compare_versions(recorded, python_version=None, package_version=None):
    """Return exact comparisons, including CUDA wheel suffixes and Python patch."""
    python_version = python_version or platform.python_version()
    package_version = package_version or metadata.version
    rows = []
    for name, expected in sorted(recorded.items()):
        try:
            actual = python_version if name == 'python' else package_version(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append({'package': name, 'recorded': expected, 'installed': actual,
                     'matches': actual == expected})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['tabm', 'logistic_regression'], default='tabm')
    parser.add_argument('--run', type=Path, help='Custom run directory, overrides --model.')
    args = parser.parse_args(argv)
    run = args.run or PROJECT / 'results' / args.model
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    recorded = config['versions']
    if not recorded or 'python' not in recorded:
        raise ValueError('Run configuration lacks recorded Python/package versions.')
    rows = compare_versions(recorded)
    matches = all(row['matches'] for row in rows)
    print(json.dumps({'status': 'matched' if matches else 'different',
                      'run': str(run), 'versions': rows,
                      'scope': 'Recorded versions only; no training or GPU initialization.'}, indent=2))
    return 0 if matches else 1


if __name__ == '__main__':
    raise SystemExit(main())

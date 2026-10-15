"""Check that published observation documents reproduce byte-for-byte.

Requires requirements-docs.txt. Uses a temporary directory; does not train,
modify existing documents, or load model checkpoints. This checks contents and
deterministic generation, not Word page rendering.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STEMS = ('Logistic_Regression_Observations', 'Histogram_Gradient_Boosting_Observations',
         'TabM_Observations', 'Model_Comparison')


def check():
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='observations-check-', dir=runtime) as directory:
        destination = Path(directory).resolve()
        if destination.parent != runtime.resolve():
            raise RuntimeError('Temporary directory escaped project runtime.')
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_model_observations.py'),
                        '--output-dir', str(destination / 'docs'),
                        '--comparison-output', str(destination / 'comparison')], check=True,
                       capture_output=True, text=True)
        files = [Path('docs') / f'{stem}.{extension}' for stem in STEMS for extension in ('docx', 'md')]
        files += [Path('comparison') / p.name for p in (destination / 'comparison').iterdir()]
        checks = []
        for relative in sorted(files):
            current = ROOT / relative if relative.parts[0] == 'docs' else ROOT / 'results' / relative.parent / 'output_data' / relative.name
            expected, actual = current.read_bytes(), (destination / relative).read_bytes()
            if actual != expected:
                raise ValueError(f'Generated artifact differs: {current}')
            checks.append({'file': current.relative_to(ROOT).as_posix(), 'bytes': len(actual),
                           'sha256': hashlib.sha256(actual).hexdigest()})
    return {'status': 'passed', 'byte_identical_files': len(checks), 'files': checks,
            'models_trained': False, 'word_page_rendering_verified': False,
            'scope': 'Deterministic document/table regeneration; model checks are separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    result = check()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()

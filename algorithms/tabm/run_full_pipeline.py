"""Run or refresh the experiment, verify it, optionally refresh the notebook/publish.

Publishing is opt-in and stages only generated TabM results and its notebook.
It never force-pushes, switches branches, or commits unrelated working changes.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]


def command(arguments, **kwargs):
    return subprocess.run(arguments, cwd=PROJECT, check=True, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--report-only', action='store_true',
                        help='Refresh and verify saved results without loading or training models.')
    parser.add_argument('--skip-notebook', action='store_true',
                        help='Skip notebook execution for a faster reports-and-validation pass.')
    parser.add_argument('--publish', action='store_true', help='Commit verified generated results and push the current branch.')
    args = parser.parse_args(argv)
    output = PROJECT / 'results/tabm'
    if args.report_only and not (output / 'run_config.json').is_file():
        raise ValueError('--report-only requires an existing results/tabm/run_config.json.')
    output.mkdir(parents=True, exist_ok=True)
    branch = (command(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()
              if args.publish else None)
    if args.publish and not branch:
        raise RuntimeError('Publishing requires a named branch.')

    def status(state, **extra):
        value = {'state': state, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(),
                 'publish_on_completion': args.publish, 'branch': branch,
                 'report_only': args.report_only, 'notebook_refresh_requested': not args.skip_notebook, **extra}
        (output / 'pipeline_status.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    status('running')
    # Hold a process-scoped sleep inhibitor during this long computation.
    # Restore normal idle-sleep behaviour when the worker exits.
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        command([sys.executable, '-u', 'algorithms/tabm/run_experiment.py',
                 '--phase', 'report' if args.report_only else 'run', '--device', args.device])
        command([sys.executable, 'algorithms/tabm/check_results.py',
                 '--save', 'results/tabm/reports/validation.json'])
        environment = dict(os.environ, TABM_RUN_TRAINING='0',
                           MPLCONFIGDIR=str(PROJECT / '.runtime/matplotlib'),
                           IPYTHONDIR=str(PROJECT / '.runtime/ipython'))
        notebook = 'algorithms/tabm/notebooks/05_tabm_tuning.ipynb'
        if not args.skip_notebook:
            command([sys.executable, 'scripts/execute_notebook.py', notebook, notebook], env=environment)
        completion = json.loads((output / 'reports/completion.json').read_text(encoding='utf-8'))
        if not completion.get('complete_tabm_plan'):
            raise RuntimeError('Full TabM completion was not verified; results will not be published as complete.')
        status('completed_and_verified')
        if args.publish:
            current = command(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()
            if current != branch:
                raise RuntimeError('Git branch changed during training; publish the verified results manually.')
            paths = ['results/tabm/output_data', 'results/tabm/reports',
                     'results/tabm/run_config.json', 'results/tabm/status.json', notebook]
            command(['git', 'add', '--', *paths])
            changes = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=PROJECT)
            if changes.returncode == 1:
                command(['git', 'commit', '--only', '-m', 'Add verified complete TabM experiment results', '--', *paths])
            elif changes.returncode != 0:
                raise RuntimeError('Could not inspect staged TabM results.')
            command(['git', 'push', 'origin', branch])
            revision = command(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
            status('published', commit=revision)
    except BaseException as error:
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

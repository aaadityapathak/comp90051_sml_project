"""Run future TabM experiments with a bounded pool of independent processes.

The serial training implementation remains the source of the scientific protocol.
Use --dry-run to inspect a plan without loading data, touching CUDA, or writing
results. Parallel runs use their own output directory and resume via this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from models.tabm import run_experiment as serial
from models.tabm.parallel_runtime import execute_task, initialize_worker

DEFAULT_OUTPUT = PROJECT / 'results/tabm_parallel'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2, help='Total processes, never multiplied by tuning jobs (default: 2).')
    parser.add_argument('--parallel-level', choices=['outer', 'inner'], default='outer',
                        help='outer: concurrent folds/variants; inner: concurrent split groups sharing preparation, then outer jobs.')
    device = parser.add_mutually_exclusive_group()
    device.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N (default: auto).')
    device.add_argument('--devices', nargs='+', help='Distinct GPUs, e.g. cuda:0 cuda:1; workers are assigned round-robin.')
    parser.add_argument('--threads-per-worker', type=int, default=1, help='CPU math threads per worker (default: 1).')
    parser.add_argument('--gpu-memory-fraction', type=float,
                        help='PyTorch allocator limit per worker; default: 0.8 / workers sharing its GPU.')
    parser.add_argument('--phase', choices=['run', 'tune', 'report'], default='run')
    parser.add_argument('--blocks', type=int, choices=range(1, 6), help='One candidate for --phase tune.')
    parser.add_argument('--variants', nargs='+', choices=serial.VARIANTS, default=list(serial.VARIANTS))
    parser.add_argument('--outer-folds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run', action='store_true', help='Print scheduling bounds only; no models, data, CUDA, workers, or output writes.')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.workers < 1 or (os.name == 'nt' and args.workers > 61):
        parser.error('--workers must be positive (at most 61 on Windows).')
    if args.threads_per_worker < 1:
        parser.error('--threads-per-worker must be positive.')
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        parser.error('--gpu-memory-fraction must be in (0, 1].')
    requested = args.devices or [args.device]
    if any(not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', value) for value in requested):
        parser.error('Devices must be auto, cpu, cuda, or cuda:N.')
    requested = [('cuda:0' if value == 'cuda' else f'cuda:{int(value.split(":")[1])}')
                 if value.startswith('cuda') else value for value in requested]
    if len(set(requested)) != len(requested):
        parser.error('List each device only once; --workers controls sharing.')
    if len(requested) > 1 and any(value in ('cpu', 'auto') for value in requested):
        parser.error('--devices supports either one CPU/auto device or a list of CUDA GPUs.')
    if args.workers < len(requested):
        parser.error('--workers must be at least the number of devices.')
    if args.gpu_memory_fraction is not None:
        sharing = (args.workers + len(requested) - 1) // len(requested)
        if args.gpu_memory_fraction * sharing > 1:
            parser.error('Per-worker GPU memory fractions must sum to at most 1 on each GPU.')
        if requested == ['cpu']:
            parser.error('--gpu-memory-fraction requires a CUDA device.')
    args.devices = requested
    if (not args.outer_folds or len(set(args.outer_folds)) != len(args.outer_folds)
            or not set(args.outer_folds).issubset(range(10))
            or len(set(args.variants)) != len(args.variants)
            or args.epochs < 1 or args.repeats not in range(1, 4)):
        parser.error('Use unique valid folds/variants, positive epochs, and 1..3 repeats.')
    if args.blocks is not None and args.phase != 'tune':
        parser.error('--blocks only applies to --phase tune.')
    args.outer_folds = sorted(args.outer_folds)
    args.variants = [variant for variant in serial.VARIANTS if variant in args.variants]
    if args.output == serial.CANONICAL_OUTPUT.resolve():
        parser.error('Use a separate output directory for parallel runs; results/tabm is reserved for the existing serial run.')
    if args.output == DEFAULT_OUTPUT.resolve() and not serial._full_protocol(args) and args.phase != 'report':
        parser.error('Reduced runs require a separate --output, for example results/tabm_parallel/pilots/smoke.')
    return args


def dry_run_plan(args):
    pairs = len(args.outer_folds) * len(args.variants)
    candidates = 1 if args.blocks is not None else len(serial.INITIAL_GRID)
    initial = 0 if args.phase == 'report' else pairs * candidates * 3
    refinements = pairs * 3 if args.phase == 'run' and not args.no_refine else 0
    outer = pairs if args.phase == 'run' else 0
    return {'dry_run': True, 'models_trained': 0, 'output': str(args.output),
            'workers': args.workers, 'parallel_level': args.parallel_level,
            'requested_devices': args.devices, 'threads_per_worker': args.threads_per_worker,
            'fold_variant_pairs': pairs, 'initial_inner_fits': initial,
            'maximum_refinement_fits': refinements, 'outer_refits': outer,
            'maximum_new_fits_before_resume': initial + refinements + outer,
            'note': 'Upper bounds before checkpoint reuse; hardware availability is checked only for a real run.'}


def resolve_devices(requested):
    import torch
    devices = [('cuda:0' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value
               for value in requested]
    for device in devices:
        if device.startswith('cuda:'):
            if not torch.cuda.is_available() or int(device.split(':')[1]) >= torch.cuda.device_count():
                raise ValueError(f'Requested device is unavailable: {device}')
    return devices


def parallel_signature(args, package):
    signature = serial.run_signature(args, package)
    for name in ('run_parallel.py', 'parallel_runtime.py'):
        path = Path(__file__).with_name(name)
        signature['code'][path.relative_to(PROJECT).as_posix()] = serial.source_sha256(path)
    signature['execution_semantics'] = {
        'scheduler': 'spawn-v2-split-groups', 'devices': args.devices,
        'threads_per_worker': args.threads_per_worker,
    }
    return signature


def ensure_signature(args, signature):
    path = args.output / 'run_config.json'
    if path.exists():
        if not serial.compatible_signature(PROJECT, args.output,
                                           json.loads(path.read_text(encoding='utf-8')), signature):
            raise ValueError('Existing output has different code/data/configuration. Use a new --output folder. '
                             'Serial outputs cannot be resumed by the parallel runner.')
    else:
        if any((args.output / name).exists() for name in ('output_data', 'checkpoints')):
            raise ValueError('Output contains artifacts without run_config.json; use an empty --output folder.')
        serial.json_write(path, signature)


def fold_location(args, outer, variant):
    return args.output / 'output_data/folds' / f'{outer:02d}_{variant}'


def inner_location(args, outer, variant, blocks, inner):
    return args.output / 'output_data/tuning' / f'{outer:02d}_{variant}' / f'blocks_{blocks}' / f'inner_{inner}'


def pending_pairs(args):
    return [(outer, variant) for outer in args.outer_folds for variant in args.variants
            if serial.checked_completion(fold_location(args, outer, variant)) is None]


def inner_tasks(args, pairs, candidates):
    tasks = []
    for outer, variant in pairs:
        for inner in range(3):
            for blocks in candidates:
                if serial.checked_completion(inner_location(args, outer, variant, blocks, inner)) is None:
                    tasks.append({'kind': 'inner', 'outer': outer, 'variant': variant,
                                  'blocks': blocks, 'inner': inner})
    return tasks


def refinement_tasks(args, pairs):
    if args.no_refine:
        return []
    tasks = []
    for outer, variant in pairs:
        scores = {}
        for blocks in serial.INITIAL_GRID:
            markers = [serial.checked_completion(inner_location(args, outer, variant, blocks, inner))
                       for inner in range(3)]
            if any(marker is None for marker in markers):
                raise ValueError(f'Initial tuning is incomplete for outer={outer} variant={variant}.')
            scores[blocks] = float(serial.np.mean([marker['macro_f1'] for marker in markers]))
        grid = serial.refinement_grid(serial.INITIAL_GRID, serial.select_blocks(scores))
        if grid is not None:
            tasks.extend(inner_tasks(args, [(outer, variant)],
                                     [blocks for blocks in grid if blocks not in serial.INITIAL_GRID]))
    return tasks


def group_inner_tasks(tasks):
    """Keep a split's candidates in one worker so preparation is reused.

    A process pool cannot guarantee that adjacent submissions reach the same
    worker. Bundling by partition makes reuse explicit without retaining more
    than one prepared split per inner-stage worker. Completion remains atomic
    per candidate: interruption only restarts the unfinished fit.
    """
    grouped = {}
    for task in tasks:
        key = (task['outer'], task['variant'], task['inner'])
        if key not in grouped:
            grouped[key] = {'kind': 'inner_group', 'outer': key[0],
                            'variant': key[1], 'inner': key[2], 'blocks': []}
        grouped[key]['blocks'].append(task['blocks'])
    return list(grouped.values())


@contextmanager
def worker_environment(threads):
    """Set math library limits before spawned Python imports NumPy/PyTorch."""
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    try:
        for name in THREAD_VARIABLES:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_tasks(executor, tasks, args, phase, worker=execute_task):
    """Submit at most one task per slot; drain workers before releasing the lock."""
    pending, completed = {}, 0
    source = iter(tasks)

    def submit_one():
        task = next(source, None)
        if task is not None:
            pending[executor.submit(worker, task)] = task

    def status():
        serial.progress(args.output, 'running', phase, workers=args.workers,
                        completed_jobs=completed, total_jobs=len(tasks), active_jobs=list(pending.values()))

    try:
        for _ in range(min(args.workers, len(tasks))):
            submit_one()
        status()
        while pending:
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            # Observe every error in this batch before queuing more work.
            for future in done:
                future.result()
            for future in done:
                pending.pop(future)
                completed += 1
                submit_one()
            status()
    except BaseException:
        for future in pending:
            future.cancel()
        raise


def run_stages(executor, args):
    pairs = pending_pairs(args) if args.phase == 'run' else [
        (outer, variant) for outer in args.outer_folds for variant in args.variants]
    if args.parallel_level == 'outer':
        kind = 'outer' if args.phase == 'run' else 'tune'
        tasks = [{'kind': kind, 'outer': outer, 'variant': variant} for outer, variant in pairs]
        if kind == 'tune' and args.blocks is not None:
            for task in tasks:
                task['blocks'] = args.blocks
        run_tasks(executor, tasks, args, kind)
    else:
        candidates = [args.blocks] if args.blocks is not None else serial.INITIAL_GRID
        run_tasks(executor, group_inner_tasks(inner_tasks(args, pairs, candidates)), args, 'initial_tuning')
        if args.phase == 'run':
            run_tasks(executor, group_inner_tasks(refinement_tasks(args, pairs)), args, 'refinement_tuning')
            run_tasks(executor, [{'kind': 'outer', 'outer': outer, 'variant': variant}
                                 for outer, variant in pairs], args, 'outer_evaluation')


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        result = dry_run_plan(args)
        print(json.dumps(result, indent=2))
        return result
    for variable, folder in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython')]:
        os.environ.setdefault(variable, str(PROJECT / '.runtime' / folder))
    # Own the output for the entire worker lifetime, including error shutdown.
    with serial.output_lock(args.output):
        if args.phase == 'report':
            from models.tabm.reporting import generate_reports
            return generate_reports(args.output)
        args.devices = resolve_devices(args.devices)
        args.device = 'cpu' if args.devices == ['cpu'] else 'cuda'
        if args.device == 'cpu' and args.gpu_memory_fraction is not None:
            raise ValueError('--gpu-memory-fraction requires a CUDA device.')
        package = serial.load_processed_data(PROJECT / 'processed_data')
        ensure_signature(args, parallel_signature(args, package))
        options = vars(args).copy()
        options['output'] = str(args.output)
        execution = {**options, 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}
        serial.json_write(args.output / 'execution.json', execution)
        serial.json_write(args.output / 'logs' / f'execution-{os.getpid()}-{time_stamp()}.json', execution)
        serial.progress(args.output, 'running', 'starting', workers=args.workers)
        sleep_inhibited = False
        try:
            if os.name == 'nt':
                import ctypes
                sleep_inhibited = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            context = multiprocessing.get_context('spawn')
            slots = context.Queue()
            try:
                for slot in range(args.workers):
                    slots.put(slot)
                with worker_environment(args.threads_per_worker):
                    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                             initializer=initialize_worker, initargs=(options, slots)) as executor:
                        run_stages(executor, args)
            finally:
                slots.close()
                slots.join_thread()
            serial.progress(args.output, 'running', 'reporting', workers=args.workers)
            from models.tabm.reporting import generate_reports
            generate_reports(args.output)
            serial.progress(args.output, 'running', 'validating', workers=args.workers)
            from models.tabm.check_results import check
            validation = check(args.output, allow_partial=args.phase == 'tune', package=package)
            serial.json_write(args.output / 'reports/validation.json', validation)
            serial.progress(args.output, 'completed', args.phase, workers=args.workers,
                            complete_tabm_plan=validation['complete_tabm_plan'],
                            requested_scope_complete=validation['requested_scope_complete'])
            return validation
        except BaseException as error:
            serial.progress(args.output, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            args.phase, error=f'{type(error).__name__}: {error}', workers=args.workers)
            raise
        finally:
            if sleep_inhibited:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def time_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

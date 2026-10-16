"""Spawn-safe workers for the optional parallel TabM experiment runner.

Importing this module performs no data loading, training, or device discovery.
The parent owns the output lock and all shared reports. Each worker writes only
its assigned, non-overlapping fit outputs plus its own log and status file.
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import traceback


_worker = None


def _record(state, phase, **current):
    """Publish progress without touching the parent process's status.json."""
    if _worker is None:
        return
    record = {
        'state': state,
        'phase': phase,
        'pid': os.getpid(),
        'slot': _worker['slot'],
        'device': _worker['device'],
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'task': _worker.get('task'),
        **current,
    }
    path = _worker['status_path']
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    print(json.dumps(record, default=str), flush=True)


def _worker_progress(output, state, phase, **current):
    # Keep the serial runner's call signature while isolating worker updates.
    _record(state, phase, **current)


def _clear_prepared():
    if _worker is not None:
        _worker['cache'].clear()
        _worker['cache_key'] = None
    gc.collect()


def _release_device_cache():
    """Release unused allocations only for this worker's assigned CUDA device."""
    if _worker is not None and _worker['device'].startswith('cuda:'):
        torch = _worker.get('torch')
        if torch is not None:
            torch.cuda.empty_cache()


def _close_log():
    if _worker is not None:
        log = _worker.get('log')
        if log is not None and not log.closed:
            log.flush()


def initialize_worker(options: dict, slot_queue):
    """Load data once in a spawned process and assign its bounded resources.

    ``slot_queue`` contains one zero-based slot per worker. ``devices`` must
    already contain normalized CPU or indexed CUDA names. The parent must set
    numerical-library thread environment variables before spawning the pool.
    """
    global _worker
    if _worker is not None:
        raise RuntimeError('A parallel worker may only be initialized once.')
    slot = slot_queue.get(timeout=30)
    workers = int(options['workers'])
    devices = list(options['devices'])
    if not 0 <= slot < workers or not devices:
        raise ValueError('Worker slots and normalized devices must be supplied by the parent.')
    device = devices[slot % len(devices)]
    if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError(f'Worker device is not normalized: {device!r}')
    output = Path(options['output'])
    log_path = output / 'logs' / f'worker-{slot}.log'
    status_path = output / 'worker_status' / f'worker-{slot}.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open('a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    _worker = {
        'slot': slot, 'device': device, 'status_path': status_path,
        'log': log, 'cache': {}, 'cache_key': None, 'task': None,
    }
    atexit.register(_close_log)
    try:
        _record('starting', 'initialize')
        # The serial model also sets this, but CUDA is configured before that
        # module is imported in a worker.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch

        _worker['torch'] = torch
        threads = int(options['threads_per_worker'])
        if threads < 1:
            raise ValueError('threads_per_worker must be positive.')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        memory_fraction = None
        if device.startswith('cuda:'):
            torch.cuda.set_device(device)
            sharing = sum(devices[index % len(devices)] == device for index in range(workers))
            memory_fraction = options.get('gpu_memory_fraction')
            if memory_fraction is None:
                memory_fraction = .8 / sharing
            memory_fraction = float(memory_fraction)
            if not 0 < memory_fraction <= 1:
                raise ValueError('gpu_memory_fraction must be greater than zero and at most one.')
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
        from models.tabm import run_experiment as serial

        serial.progress = _worker_progress
        args = argparse.Namespace(**options)
        args.output = output
        args.device = device
        _worker.update(serial=serial, args=args)
        _worker['package'] = serial.load_processed_data(serial.processed_directory(serial.PROJECT))
        _record('ready', 'idle', threads_per_worker=threads,
                gpu_memory_fraction=memory_fraction)
    except BaseException as error:
        _record('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                'initialize', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        raise


def _run_inner(task):
    outer, variant, blocks, inner = (task[name] for name in ('outer', 'variant', 'blocks', 'inner'))
    key = (outer, variant, inner)
    if _worker['cache_key'] != key:
        _clear_prepared()
        _worker['cache_key'] = key
    args = _worker['args']
    location = (args.output / 'output_data/tuning' / f'{outer:02d}_{variant}'
                / f'blocks_{blocks}' / f'inner_{inner}')
    existed = (location / 'complete.json').exists()
    # inner_fit verifies existing completion hashes before deciding to resume.
    _worker['serial'].inner_fit(_worker['package'], outer, variant, blocks,
                                inner, args, _worker['cache'])
    return not existed


def execute_task(task: dict):
    """Execute a disjoint inner fit/group, outer evaluation, or tuning group.

    A tune task supplies ``outer`` and ``variant`` and optionally ``blocks``
    (one integer or a list); omitted blocks use the initial prespecified grid.
    An inner_group task supplies one ``inner`` split and a list of ``blocks``
    candidates, all sharing the same prepared arrays in this process.
    All scientific settings, seeds and checkpoint checks remain serial code.
    """
    if _worker is None or 'package' not in _worker:
        raise RuntimeError('initialize_worker must finish before execute_task.')
    task = dict(task)
    _worker['task'] = task
    kind = task.get('kind')
    try:
        _record('running', 'task_started')
        if kind == 'inner':
            changed = _run_inner(task)
        elif kind == 'inner_group':
            changed = False
            for candidate in task['blocks']:
                changed = _run_inner({**task, 'blocks': candidate}) or changed
            _clear_prepared()
        elif kind == 'outer':
            _clear_prepared()
            changed = _worker['serial'].run_variant(
                _worker['package'], task['outer'], task['variant'], _worker['args'])
        elif kind == 'tune':
            _clear_prepared()
            blocks = task.get('blocks')
            if blocks is None:
                blocks = _worker['serial'].INITIAL_GRID
            elif isinstance(blocks, int):
                blocks = [blocks]
            changed = False
            # Prepare only one split at a time and share it across candidates.
            for inner in range(3):
                for candidate in blocks:
                    changed = _run_inner({**task, 'blocks': candidate, 'inner': inner}) or changed
            _clear_prepared()
        else:
            raise ValueError(f'Unknown parallel task kind: {kind!r}')
        _release_device_cache()
        _record('ready', 'task_completed', changed=bool(changed))
        return {'task': task, 'changed': bool(changed)}
    except BaseException as error:
        advice = None
        if _worker['device'].startswith('cuda:') and 'out of memory' in str(error).lower():
            advice = ('Resume with fewer workers per GPU or more GPU memory. '
                      'Completed checkpoints are retained; no automatic retry changes the experiment.')
            if hasattr(error, 'add_note'):
                error.add_note(advice)
        _record('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                'task_failed', error=f'{type(error).__name__}: {error}', advice=advice)
        traceback.print_exc()
        _clear_prepared()
        raise
    finally:
        _worker['task'] = None

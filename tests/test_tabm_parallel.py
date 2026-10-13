"""Parallel scheduling checks using synthetic markers and mocked fit functions.

No test loads the study data, imports a real model, or initializes CUDA. The
only real process pool executes the built-in arithmetic function ``abs``.
"""
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
from algorithms.tabm import parallel_runtime as runtime
from algorithms.tabm import run_parallel as parallel


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT / 'tests')
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'output'

    def args(self, **changes):
        values = dict(output=self.output, workers=2, parallel_level='inner',
                      devices=['cpu'], device='cpu', threads_per_worker=1,
                      gpu_memory_fraction=None, outer_folds=[0], variants=['A'],
                      epochs=2, repeats=1, no_refine=False, phase='run',
                      blocks=None, dry_run=False)
        return SimpleNamespace(**(values | changes))

    def marker(self, location, score=.5):
        location.mkdir(parents=True, exist_ok=True)
        (location / 'synthetic.txt').write_text('Synthetic checkpoint.', encoding='utf-8')
        parallel.serial.complete_files(location, ['synthetic.txt'], macro_f1=score)

    def initial_markers(self, args, scores):
        for outer in args.outer_folds:
            for variant in args.variants:
                for blocks, score in scores.items():
                    for inner in range(3):
                        self.marker(parallel.inner_location(args, outer, variant, blocks, inner), score)

    def test_cli_rejects_invalid_resources_and_protocol(self):
        cases = [
            ['--workers', '0'], ['--threads-per-worker', '0'],
            ['--gpu-memory-fraction', '0'], ['--gpu-memory-fraction', '1.1'],
            ['--device', 'cuda:-1'], ['--devices', 'cuda', 'cuda:0'],
            ['--workers', '2', '--devices', 'cuda:0', 'cuda:00'],
            ['--devices', 'cpu', 'cuda:0'], ['--devices', 'auto', 'cuda:0'],
            ['--workers', '1', '--devices', 'cuda:0', 'cuda:1'],
            ['--workers', '2', '--device', 'cuda', '--gpu-memory-fraction', '.6'],
            ['--device', 'cpu', '--gpu-memory-fraction', '.3'],
            ['--outer-folds', '0', '0'], ['--outer-folds', '10'],
            ['--variants', 'A', 'A'], ['--epochs', '0'], ['--repeats', '4'],
            ['--blocks', '2'],
        ]
        if os.name == 'nt':
            cases.append(['--workers', '62'])
        for flags in cases:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parallel.parse_args(['--output', str(self.output), *flags])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_separates_active_serial_output_and_reduced_runs(self):
        for flags in (['--output', str(parallel.serial.CANONICAL_OUTPUT)],
                      ['--epochs', '2']):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parallel.parse_args(flags)
        args = parallel.parse_args(['--output', str(self.output), '--outer-folds', '2', '0',
                                    '--variants', 'B', 'A', '--device', 'cuda'])
        self.assertEqual(args.outer_folds, [0, 2])
        self.assertEqual(args.variants, ['A', 'B'])
        self.assertEqual(args.devices, ['cuda:0'])
        self.assertFalse(self.output.exists())

    def test_dry_run_fresh_process_does_not_import_torch_or_write(self):
        script = """import importlib.abc, json, pathlib, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Dry-run attempted to import torch')
sys.meta_path.insert(0, NoTorch())
from algorithms.tabm.run_parallel import main
main(['--dry-run', '--output', sys.argv[1]])
assert 'torch' not in sys.modules
assert 'algorithms.tabm.model' not in sys.modules
assert not pathlib.Path(sys.argv[1]).exists()
"""
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment.update({name: '1' for name in parallel.THREAD_VARIABLES})
        result = subprocess.run([sys.executable, '-B', '-c', script, str(self.output)],
                                cwd=PROJECT, env=environment, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['models_trained'], 0)
        self.assertEqual(plan['initial_inner_fits'], 720)
        self.assertEqual(plan['maximum_refinement_fits'], 240)
        self.assertEqual(plan['outer_refits'], 80)
        self.assertEqual(plan['maximum_new_fits_before_resume'], 1040)
        self.assertFalse(self.output.exists())

    def test_dry_run_bypasses_loading_lock_and_workers(self):
        with patch.object(parallel.serial, 'load_processed_data') as load, \
                patch.object(parallel.serial, 'output_lock') as lock, \
                patch.object(parallel, 'resolve_devices') as devices, \
                patch.object(parallel, 'ProcessPoolExecutor') as pool, \
                redirect_stdout(io.StringIO()):
            plan = parallel.main(['--dry-run', '--output', str(self.output)])
        for action in (load, lock, devices, pool):
            action.assert_not_called()
        self.assertTrue(plan['dry_run'])
        self.assertFalse(self.output.exists())

    def test_plan_counts_for_tune_report_and_no_refinement(self):
        cases = [
            (dict(), (9, 3, 1)),
            (dict(no_refine=True), (9, 0, 1)),
            (dict(phase='tune'), (9, 0, 0)),
            (dict(phase='tune', blocks=5), (3, 0, 0)),
            (dict(phase='report'), (0, 0, 0)),
        ]
        for options, counts in cases:
            with self.subTest(options=options):
                result = parallel.dry_run_plan(self.args(**options))
                self.assertEqual(tuple(result[key] for key in (
                    'initial_inner_fits', 'maximum_refinement_fits', 'outer_refits')), counts)
                self.assertEqual(result['maximum_new_fits_before_resume'], sum(counts))

    def test_initial_jobs_are_unique_and_skip_verified_checkpoints(self):
        args = self.args(outer_folds=[0, 1], variants=['A', 'B'])
        pairs = parallel.pending_pairs(args)
        jobs = parallel.inner_tasks(args, pairs, parallel.serial.INITIAL_GRID)
        identities = {(job['outer'], job['variant'], job['blocks'], job['inner']) for job in jobs}
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(identities), len(jobs))
        location = parallel.inner_location(args, 0, 'A', 2, 0)
        self.marker(location)
        self.assertEqual(len(parallel.inner_tasks(args, pairs, [2, 3, 4])), 35)
        self.marker(parallel.fold_location(args, 1, 'B'))
        self.assertEqual(parallel.pending_pairs(args), [(0, 'A'), (0, 'B'), (1, 'A')])
        (location / 'synthetic.txt').write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash'):
            parallel.inner_tasks(args, pairs, [2, 3, 4])

    def test_completed_outer_checkpoint_tampering_is_rejected(self):
        args = self.args()
        location = parallel.fold_location(args, 0, 'A')
        self.marker(location)
        (location / 'synthetic.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'hash'):
            parallel.pending_pairs(args)

    def test_refinement_endpoint_middle_and_tie_rules(self):
        cases = [({2: .8, 3: .7, 4: .6}, 1),
                 ({2: .6, 3: .7, 4: .8}, 5),
                 ({2: .6, 3: .8, 4: .7}, None),
                 ({2: .8, 3: .8, 4: .8}, 1)]
        for index, (scores, expected) in enumerate(cases):
            with self.subTest(scores=scores):
                args = self.args(output=self.output / str(index))
                self.initial_markers(args, scores)
                jobs = parallel.refinement_tasks(args, [(0, 'A')])
                self.assertEqual({job['blocks'] for job in jobs},
                                 set() if expected is None else {expected})
                self.assertEqual(len(jobs), 0 if expected is None else 3)
                if expected is not None:
                    self.marker(parallel.inner_location(args, 0, 'A', expected, 0))
                    self.assertEqual(len(parallel.refinement_tasks(args, [(0, 'A')])), 2)

    def test_refinement_requires_all_initial_scores(self):
        args = self.args()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            parallel.refinement_tasks(args, [(0, 'A')])
        self.assertEqual(parallel.refinement_tasks(self.args(no_refine=True), [(0, 'A')]), [])

    def test_split_groups_preserve_only_missing_candidates_and_share_one_job(self):
        args = self.args(outer_folds=[0, 1], variants=['A', 'B'])
        self.marker(parallel.inner_location(args, 0, 'A', 3, 0))
        missing = parallel.inner_tasks(args, parallel.pending_pairs(args), [2, 3, 4])
        groups = parallel.group_inner_tasks(missing)
        self.assertEqual(len(groups), 12)
        self.assertEqual(groups[0], dict(kind='inner_group', outer=0, variant='A', inner=0, blocks=[2, 4]))
        expanded = [{**group, 'kind': 'inner', 'blocks': block}
                    for group in groups for block in group['blocks']]
        self.assertEqual(expanded, missing)
        self.assertEqual(sum(len(group['blocks']) for group in groups), 35)

    def test_inner_stages_finish_tuning_before_outer_without_duplicate_fits(self):
        args = self.args(outer_folds=[0, 1], variants=['A', 'B'])
        self.marker(parallel.fold_location(args, 1, 'B'))
        phases, fits = [], []

        def stage(executor, tasks, received, phase):
            self.assertIs(received, args)
            phases.append(phase)
            for task in tasks:
                if task['kind'] == 'inner_group':
                    for blocks in task['blocks']:
                        key = (task['outer'], task['variant'], blocks, task['inner'])
                        self.assertNotIn(key, fits)
                        fits.append(key)
                        self.marker(parallel.inner_location(args, *key), .9 if blocks == 2 else .5)
                else:
                    self.assertEqual(phases, ['initial_tuning', 'refinement_tuning', 'outer_evaluation'])
                    for inner in range(3):
                        self.assertIsNotNone(parallel.serial.checked_completion(
                            parallel.inner_location(args, task['outer'], task['variant'], 1, inner)))

        with patch.object(parallel, 'run_tasks', side_effect=stage) as schedule:
            parallel.run_stages(object(), args)
        self.assertEqual(len(fits), 36)
        outer_tasks = schedule.call_args.args[1]
        self.assertEqual(len(outer_tasks), 3)
        self.assertNotIn({'kind': 'outer', 'outer': 1, 'variant': 'B'}, outer_tasks)

    def test_outer_mode_and_single_candidate_tune_scope(self):
        for level in ('outer', 'inner'):
            with self.subTest(level=level):
                args = self.args(phase='tune', blocks=5, parallel_level=level)
                with patch.object(parallel, 'run_tasks') as schedule:
                    parallel.run_stages(object(), args)
                schedule.assert_called_once()
                jobs = schedule.call_args.args[1]
                self.assertEqual(len(jobs), 1 if level == 'outer' else 3)
                self.assertTrue(all(job['blocks'] == (5 if level == 'outer' else [5]) for job in jobs))
                self.assertTrue(all(job['kind'] != 'outer' for job in jobs))
        args = self.args(parallel_level='outer', outer_folds=[0, 1])
        self.marker(parallel.fold_location(args, 0, 'A'))
        with patch.object(parallel, 'run_tasks') as schedule:
            parallel.run_stages(object(), args)
        self.assertEqual(schedule.call_args.args[1], [{'kind': 'outer', 'outer': 1, 'variant': 'A'}])

    def test_bounded_submission_and_complete_synthetic_work(self):
        args = self.args(workers=3)
        submitted, sizes = [], []

        def submit(worker, task):
            future = Future()
            future.task = task
            submitted.append(future)
            return future

        def complete_one(pending, **options):
            sizes.append(len(pending))
            self.assertEqual(options['return_when'], parallel.FIRST_COMPLETED)
            future = min(pending, key=lambda item: item.task)
            future.set_result(abs(future.task))
            return {future}, set(pending) - {future}

        executor = Mock(submit=Mock(side_effect=submit))
        with patch.object(parallel, 'wait', side_effect=complete_one), \
                patch.object(parallel.serial, 'progress') as progress:
            parallel.run_tasks(executor, list(range(9)), args, 'synthetic', worker=abs)
        self.assertEqual(len(submitted), 9)
        self.assertEqual(max(sizes), args.workers)
        self.assertEqual([future.result() for future in submitted], list(range(9)))
        self.assertEqual(progress.call_args.kwargs['completed_jobs'], 9)
        self.assertEqual(progress.call_args.kwargs['active_jobs'], [])

    def test_error_batch_is_observed_before_new_submissions(self):
        submitted = []

        def submit(worker, task):
            future = Future()
            submitted.append(future)
            return future

        def failed_batch(pending, **options):
            submitted[0].set_result(1)
            submitted[1].set_exception(RuntimeError('synthetic failure'))
            return set(submitted), set()

        executor = Mock(submit=Mock(side_effect=submit))
        with patch.object(parallel, 'wait', side_effect=failed_batch), \
                patch.object(parallel.serial, 'progress'):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                parallel.run_tasks(executor, list(range(20)), self.args(), 'synthetic', worker=abs)
        self.assertEqual(executor.submit.call_count, 2)

    def test_submission_failure_cancels_existing_pending_future(self):
        pending = Future()
        executor = Mock(submit=Mock(side_effect=[pending, RuntimeError('cannot submit')]))
        with patch.object(parallel.serial, 'progress'):
            with self.assertRaisesRegex(RuntimeError, 'cannot submit'):
                parallel.run_tasks(executor, [1, 2, 3], self.args(), 'synthetic', worker=abs)
        self.assertTrue(pending.cancelled())
        self.assertEqual(executor.submit.call_count, 2)

    def test_real_spawn_pool_only_runs_arithmetic(self):
        submitted = []
        with parallel.worker_environment(1):
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
                def submit(worker, task):
                    future = pool.submit(worker, task)
                    submitted.append(future)
                    return future
                with patch.object(parallel.serial, 'progress'):
                    parallel.run_tasks(SimpleNamespace(submit=submit), [-3, -1, -8, -4],
                                       self.args(), 'arithmetic', worker=abs)
        self.assertEqual([future.result() for future in submitted], [3, 1, 8, 4])
        self.assertFalse(self.output.exists())

    def test_thread_environment_restores_on_error(self):
        name = parallel.THREAD_VARIABLES[0]
        with patch.dict(os.environ, {name: '7'}):
            original = {key: os.environ.get(key) for key in parallel.THREAD_VARIABLES}
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                with parallel.worker_environment(2):
                    self.assertTrue(all(os.environ[key] == '2' for key in parallel.THREAD_VARIABLES))
                    raise RuntimeError('stop')
            self.assertEqual({key: os.environ.get(key) for key in parallel.THREAD_VARIABLES}, original)

    def signature(self, args):
        # Avoid serial.run_signature importing torch: only extension semantics
        # are under test here; the scientific signature has separate coverage.
        with patch.object(parallel.serial, 'run_signature', side_effect=lambda *a: {
                'model': 'tabm', 'code': {'algorithms/tabm/model.py': 'fixture-hash'}}):
            return parallel.parallel_signature(args, {'manifest': {'seed': 1}})

    def test_signature_tracks_scheduler_sources_devices_and_threads(self):
        original = self.signature(self.args())
        for name in ('run_parallel.py', 'parallel_runtime.py'):
            source = PROJECT / 'algorithms/tabm' / name
            self.assertEqual(original['code'][f'algorithms/tabm/{name}'], parallel.serial.source_sha256(source))
        changed_resources = self.signature(self.args(workers=1, parallel_level='outer', gpu_memory_fraction=.2))
        self.assertEqual(original, changed_resources)
        self.assertNotEqual(original, self.signature(self.args(threads_per_worker=2)))
        self.assertNotEqual(original, self.signature(self.args(devices=['cuda:0'])))

    def test_resume_signature_rejects_serial_or_modified_code_and_orphan_output(self):
        args = self.args()
        signature = self.signature(args)
        parallel.ensure_signature(args, signature)
        parallel.ensure_signature(args, self.signature(self.args(workers=1)))
        for changed in (dict(signature, execution_semantics={'scheduler': 'other'}),
                        {key: value for key, value in signature.items() if key != 'execution_semantics'}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'different'):
                parallel.ensure_signature(args, changed)
        changed = copy.deepcopy(signature)
        changed['code']['algorithms/tabm/parallel_runtime.py'] = 'modified-source'
        with self.assertRaisesRegex(ValueError, 'different'):
            parallel.ensure_signature(args, changed)
        orphan = self.args(output=self.output / 'orphan')
        (orphan.output / 'checkpoints').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'without run_config'):
            parallel.ensure_signature(orphan, signature)

    def mock_worker(self, **changes):
        serial = SimpleNamespace(INITIAL_GRID=[2, 3, 4], inner_fit=Mock(), run_variant=Mock(return_value=True))
        worker = dict(slot=0, device='cpu', status_path=self.output / 'worker_status/worker-0.json',
                      cache={}, cache_key=None, task=None, args=self.args(), package=object(), serial=serial)
        worker.update(changes)
        return worker

    def test_worker_caches_only_one_partition_and_shares_candidates(self):
        worker = self.mock_worker()
        observed = []

        def inner_fit(package, outer, variant, blocks, inner, args, cache):
            observed.append((outer, variant, blocks, inner, dict(cache)))
            cache[inner] = f'{outer}-{variant}-{inner}'

        worker['serial'].inner_fit.side_effect = inner_fit
        task = dict(kind='inner', outer=0, variant='A', blocks=2, inner=0)
        with patch.object(runtime, '_worker', worker), patch.object(runtime, '_record'), \
                patch.object(runtime.gc, 'collect'):
            runtime.execute_task(task)
            runtime.execute_task(dict(task, blocks=3))
            runtime.execute_task(dict(task, inner=1))
            runtime.execute_task(dict(task, variant='B', inner=1))
        self.assertEqual([call[4] for call in observed], [{}, {0: '0-A-0'}, {}, {}])
        self.assertIsNone(worker['task'])
        self.assertEqual(worker['cache_key'], (0, 'B', 1))

    def test_worker_tune_and_outer_dispatch_clear_prepared_cache(self):
        worker = self.mock_worker(cache={'old': object()}, cache_key=(9, 'D', 2))
        with patch.object(runtime, '_worker', worker), patch.object(runtime, '_record'), \
                patch.object(runtime.gc, 'collect'):
            runtime.execute_task(dict(kind='tune', outer=1, variant='B', blocks=5))
            self.assertEqual(worker['serial'].inner_fit.call_count, 3)
            self.assertTrue(all(call.args[3] == 5 for call in worker['serial'].inner_fit.call_args_list))
            self.assertEqual(worker['cache'], {})
            runtime.execute_task(dict(kind='outer', outer=1, variant='B'))
        worker['serial'].run_variant.assert_called_once_with(worker['package'], 1, 'B', worker['args'])
        self.assertIsNone(worker['cache_key'])

    def test_worker_split_group_prepares_once_and_restarts_only_unfinished_candidate(self):
        worker = self.mock_worker()
        builds, fits = [], []

        def inner_fit(package, outer, variant, blocks, inner, args, cache):
            location = parallel.inner_location(args, outer, variant, blocks, inner)
            if parallel.serial.checked_completion(location) is not None:
                return
            if inner not in cache:
                builds.append((outer, variant, inner))
                cache[inner] = 'prepared split'
            fits.append(blocks)
            if blocks == 3 and fits == [2, 3]:
                raise RuntimeError('interrupted candidate')
            self.marker(location)

        worker['serial'].inner_fit.side_effect = inner_fit
        task = dict(kind='inner_group', outer=0, variant='A', inner=0, blocks=[2, 3, 4])
        with patch.object(runtime, '_worker', worker), patch.object(runtime, '_record'), \
                patch.object(runtime.gc, 'collect'), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'interrupted candidate'):
                runtime.execute_task(task)
            self.assertEqual(worker['cache'], {})
            result = runtime.execute_task(task)
        self.assertTrue(result['changed'])
        self.assertEqual(fits, [2, 3, 3, 4])
        self.assertEqual(builds, [(0, 'A', 0), (0, 'A', 0)])
        self.assertEqual(worker['cache'], {})
        self.assertIsNone(worker['cache_key'])

    def test_worker_progress_is_isolated_from_coordinator_status(self):
        self.output.mkdir()
        coordinator = self.output / 'status.json'
        coordinator.write_text('coordinator sentinel', encoding='utf-8')
        for slot in (0, 1):
            worker = self.mock_worker(slot=slot, status_path=self.output / f'worker_status/worker-{slot}.json')
            worker['status_path'].parent.mkdir(exist_ok=True)
            with patch.object(runtime, '_worker', worker), redirect_stdout(io.StringIO()):
                runtime._worker_progress(self.output, 'running', 'mock_fit', outer=slot)
            record = json.loads(worker['status_path'].read_text(encoding='utf-8'))
            self.assertEqual(record['slot'], slot)
            self.assertEqual(record['outer'], slot)
            self.assertFalse(worker['status_path'].with_name(worker['status_path'].name + '.tmp').exists())
        self.assertEqual(coordinator.read_text(encoding='utf-8'), 'coordinator sentinel')
        self.assertEqual(len(list((self.output / 'worker_status').glob('*.json'))), 2)

    def test_worker_initialization_assigns_resources_without_real_torch(self):
        options = vars(self.args(workers=3, devices=['cuda:0', 'cuda:1'], threads_per_worker=2))
        options['output'] = str(self.output)
        fake_torch = Mock()
        queue = Mock(get=Mock(return_value=2))
        with patch.object(runtime, '_worker', None), \
                patch.dict(sys.modules, {'torch': fake_torch}), \
                patch.object(parallel.serial, 'load_processed_data', return_value={'synthetic': True}) as load, \
                patch.object(parallel.serial, 'progress'), \
                patch.object(runtime.atexit, 'register'), patch.dict(os.environ), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            runtime.initialize_worker(options, queue)
            worker = runtime._worker
            worker['log'].close()
        self.assertEqual(worker['device'], 'cuda:0')
        self.assertEqual(worker['args'].device, 'cuda:0')
        self.assertEqual(worker['package'], {'synthetic': True})
        self.assertEqual(worker['cache'], {})
        load.assert_called_once()
        fake_torch.set_num_threads.assert_called_once_with(2)
        fake_torch.set_num_interop_threads.assert_called_once_with(1)
        fake_torch.cuda.set_device.assert_called_once_with('cuda:0')
        fake_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(.4, device='cuda:0')
        self.assertFalse((self.output / 'status.json').exists())
        self.assertTrue((self.output / 'logs/worker-2.log').is_file())
        self.assertEqual(json.loads((self.output / 'worker_status/worker-2.json').read_text())['state'], 'ready')

    def test_worker_failure_is_recorded_without_retry_and_clears_cache(self):
        worker = self.mock_worker(cache={1: 'prepared'})
        worker['serial'].run_variant.side_effect = RuntimeError('synthetic failure')
        with patch.object(runtime, '_worker', worker), patch.object(runtime, '_record') as record, \
                patch.object(runtime.gc, 'collect'), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                runtime.execute_task(dict(kind='outer', outer=0, variant='A'))
        worker['serial'].run_variant.assert_called_once()
        self.assertEqual(record.call_args.args, ('failed', 'task_failed'))
        self.assertEqual(worker['cache'], {})
        self.assertIsNone(worker['task'])


if __name__ == '__main__':
    unittest.main()

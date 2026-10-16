"""Run and resume the prespecified TabM patient-grouped nested experiment.

Individual block scripts populate the same immutable inner-fit cache. The
default run finishes tuning, freezes choices, refits and evaluates all stresses.
Short runs require a separate output folder and cannot become full-plan results.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
for directory in (PROJECT, PROJECT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from models.tabm.data import (
    METRICS, VARIANTS, identity, patient_weighted_metrics, pipeline_state,
    reliability_bins, split_ids, training_pipeline, transform_heldout,
)
from afib.data_preparation import load_processed_data
from afib.paths import processed_directory
from afib.evaluation import (
    binary_metrics, choose_sensitivity_threshold, majority_baseline,
    quality_subgroup_metrics, source_quality_groups, validate_scenario_repeats,
)
from afib.preprocessing import SCENARIOS
from models.tabm.source_verification import compatible_signature

INITIAL_GRID = [2, 3, 4]
CANONICAL_OUTPUT = PROJECT / 'results/tabm'


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Index)):
        return str(value) if isinstance(value, Path) else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=_json_default) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def csv_write(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    compression = 'gzip' if path.name.endswith('.gz') else None
    pd.DataFrame(frame).to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def source_sha256(path):
    """Git checkout line endings must not invalidate equivalent source code."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def complete_files(location, filenames, **metadata):
    marker = {'completed': True, **metadata,
              'files': {name: sha256(location / name) for name in filenames}}
    json_write(location / 'complete.json', marker)
    return marker


def checked_completion(location):
    marker_path = location / 'complete.json'
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    if not marker.get('completed') or not marker.get('files'):
        raise ValueError(f'Invalid checkpoint completion marker: {marker_path}')
    for name, expected in marker['files'].items():
        path = (location / name).resolve()
        if not path.is_relative_to(location.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Checkpoint file does not match its recorded hash: {path}')
    return marker


@contextmanager
def output_lock(output):
    """An OS lock releases automatically after a crash; stale files are harmless."""
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a+b')
    lock.seek(0, os.SEEK_END)
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError(f'Another process owns {output}. Wait for it to finish.') from None
    try:
        yield
    finally:
        lock.seek(0)
        if os.name == 'nt':
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def progress(output, state, phase, **current):
    inner = list((output / 'output_data/tuning').glob('*/blocks_*/inner_*/complete.json'))
    folds = list((output / 'output_data/folds').glob('*/complete.json'))
    record = {
        'state': state, 'phase': phase, 'pid': os.getpid(),
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'completed_inner_fits': len(inner), 'completed_outer_evaluations': len(folds),
        **current,
    }
    json_write(output / 'status.json', record)
    print(json.dumps(record, default=_json_default), flush=True)


def select_blocks(scores):
    if not scores or not all(np.isfinite(value) for value in scores.values()):
        raise ValueError('Every block candidate needs finite mean inner macro-F1.')
    return min(scores, key=lambda blocks: (-scores[blocks], blocks))


def refinement_grid(grid, best):
    grid = sorted(grid)
    if grid != INITIAL_GRID:
        raise ValueError('The prespecified initial block grid is [2, 3, 4].')
    if best == grid[0]:
        return [1, 2, 3]
    if best == grid[-1]:
        return [3, 4, 5]
    return None


def fit_seed(seed, outer, inner=None):
    # Match initialization and minibatch random streams across settings/arms.
    # The model reinitializes from scratch for every training partition.
    return int(seed + 100 * outer + (99 if inner is None else inner))


def _save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _save_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    model.save(temporary)
    os.replace(temporary, path)


def inner_fit(package, outer, variant, blocks, inner, args, cache):
    """A completed candidate/inner split is never retrained on resume."""
    location = args.output / 'output_data/tuning' / f'{outer:02d}_{variant}' / f'blocks_{blocks}' / f'inner_{inner}'
    marker = checked_completion(location)
    if marker is not None:
        return marker
    from models.tabm.model import TabMClassifier
    if inner not in cache:
        train_ids, valid_ids = split_ids(package, outer, inner)
        pipeline = training_pipeline(package, train_ids, variant)
        cache[inner] = {
            'X': pipeline['X'].to_numpy(dtype=np.float32),
            'y': pipeline['y'].to_numpy(),
            'weights': pipeline['sample_weight'].to_numpy(dtype=np.float32),
            'V': transform_heldout(package['raw'].loc[valid_ids], pipeline).to_numpy(dtype=np.float32),
            'valid_ids': valid_ids,
            'truth': package['targets'].loc[valid_ids, 'target_afib'].to_numpy(),
            'pipeline': pipeline_state(pipeline),
        }
        del pipeline
    prepared = cache[inner]
    seed = fit_seed(package['manifest']['seed'], outer, inner)
    progress(args.output, 'running', 'inner_fit', outer_fold=outer, variant=variant,
             n_blocks=blocks, inner_fold=inner, epochs=args.epochs)
    model = TabMClassifier(n_blocks=blocks, epochs=args.epochs, seed=seed,
                          device=args.device, batch_size=256)
    model.fit(prepared['X'], prepared['y'], prepared['weights'])
    # Promote float32 model output before CSV serialization so independently
    # recomputed metrics use exactly the same probabilities after round-trip.
    scores = np.asarray(model.predict_proba(prepared['V']), dtype=np.float64)
    macro_f1 = binary_metrics(prepared['truth'], scores, .5)['macro_f1']
    location.mkdir(parents=True, exist_ok=True)
    csv_write(location / 'predictions.csv.gz', pd.DataFrame({
        'ecg_id': prepared['valid_ids'],
        'patient_id': package['folds'].loc[prepared['valid_ids'], 'patient_id'].to_numpy(),
        'y_true': prepared['truth'], 'score': scores,
    }).assign(**identity(variant, outer), inner_fold=inner, n_blocks=blocks))
    csv_write(location / 'history.csv', model.history_)
    audit = {**model.audit_, 'phase': 'inner', 'outer_fold': outer, 'variant': variant,
             'inner_fold': inner, 'n_blocks': blocks, 'macro_f1': macro_f1}
    json_write(location / 'audit.json', audit)
    checkpoint = args.output / 'checkpoints/tuning' / f'{outer:02d}_{variant}' / f'blocks_{blocks}' / f'inner_{inner}'
    _save_model(checkpoint / 'model.pt', model)
    _save_pickle(checkpoint / 'preprocessing.pkl', prepared['pipeline'])
    marker = complete_files(
        location, ['predictions.csv.gz', 'history.csv', 'audit.json'],
        outer_fold=outer, variant=variant, inner_fold=inner, n_blocks=blocks,
        macro_f1=macro_f1, seed=seed,
        checkpoints={str((checkpoint / name).relative_to(args.output)): sha256(checkpoint / name)
                     for name in ('model.pt', 'preprocessing.pkl')},
    )
    del model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    progress(args.output, 'running', 'inner_fit_complete', outer_fold=outer,
             variant=variant, n_blocks=blocks, inner_fold=inner, macro_f1=macro_f1)
    return marker


def tune(package, outer, variant, args):
    cache, results, stage_for = {}, {}, {}
    def evaluate(values, stage):
        for blocks in values:
            if blocks in results:
                continue
            results[blocks] = [inner_fit(package, outer, variant, blocks, inner, args, cache)
                               for inner in range(3)]
            stage_for[blocks] = stage
    def means(values):
        return {blocks: float(np.mean([row['macro_f1'] for row in results[blocks]])) for blocks in values}
    evaluate(INITIAL_GRID, 'initial')
    best = select_blocks(means(INITIAL_GRID))
    final_grid = refinement_grid(INITIAL_GRID, best) if not args.no_refine else None
    if final_grid is not None:
        evaluate(final_grid, 'refinement')
        best = select_blocks(means(final_grid))
    else:
        final_grid = list(INITIAL_GRID)
    chosen = {
        'n_blocks': best, 'initial_grid': INITIAL_GRID, 'final_grid': final_grid,
        'refined': final_grid != INITIAL_GRID, 'selected_middle': best == final_grid[1],
        'inner_mean_macro_f1': means(final_grid)[best],
    }
    rows, audits, oof_parts = [], [], []
    for blocks in results:
        for inner in range(3):
            location = args.output / 'output_data/tuning' / f'{outer:02d}_{variant}' / f'blocks_{blocks}' / f'inner_{inner}'
            rows.append({'outer_fold': outer, 'variant': variant, 'stage': stage_for[blocks],
                         'inner_fold': inner, 'n_blocks': blocks,
                         'macro_f1': results[blocks][inner]['macro_f1']})
            audits.append(json.loads((location / 'audit.json').read_text(encoding='utf-8')))
            if blocks == best:
                oof_parts.append(pd.read_csv(location / 'predictions.csv.gz'))
    oof = pd.concat(oof_parts, ignore_index=True).sort_values('ecg_id')
    expected, _ = split_ids(package, outer)
    if set(oof.ecg_id) != set(expected) or oof.ecg_id.duplicated().any():
        raise ValueError('Selected inner predictions must cover outer training exactly once.')
    del cache
    gc.collect()
    return chosen, oof, rows, audits


def outer_fit(package, outer, variant, chosen, args, location):
    from models.tabm.model import TabMClassifier
    checkpoint = args.output / 'checkpoints/folds' / f'{outer:02d}_{variant}'
    marker = checked_completion(checkpoint)
    train_ids, test_ids = split_ids(package, outer)
    if marker is not None:
        if marker['n_blocks'] != chosen['n_blocks']:
            raise ValueError('Saved outer model does not match selected blocks.')
        with (checkpoint / 'preprocessing.pkl').open('rb') as handle:
            pipeline = pickle.load(handle)
        if not np.array_equal(pipeline['train_ids'], train_ids.to_numpy()):
            raise ValueError('Saved preparation uses different outer training records.')
        model = TabMClassifier.load(checkpoint / 'model.pt', device=args.device)
        return model, pipeline, test_ids
    progress(args.output, 'running', 'outer_refit', outer_fold=outer, variant=variant,
             n_blocks=chosen['n_blocks'], epochs=args.epochs)
    pipeline = training_pipeline(package, train_ids, variant)
    model = TabMClassifier(n_blocks=chosen['n_blocks'], epochs=args.epochs,
                          seed=fit_seed(package['manifest']['seed'], outer),
                          device=args.device, batch_size=256)
    model.fit(pipeline['X'], pipeline['y'], pipeline['sample_weight'])
    fitted = pipeline_state(pipeline)
    _save_model(checkpoint / 'model.pt', model)
    _save_pickle(checkpoint / 'preprocessing.pkl', fitted)
    csv_write(location / 'training_history.csv', model.history_)
    json_write(location / 'outer_fit_audit.json', {
        **model.audit_, 'phase': 'outer_refit', **identity(variant, outer),
        'n_blocks': chosen['n_blocks'],
    })
    # Output evidence is duplicated beside the checkpoint marker so a crash
    # before evaluation can resume without performing another 100-epoch fit.
    csv_write(checkpoint / 'training_history.csv', model.history_)
    json_write(checkpoint / 'outer_fit_audit.json', json.loads((location / 'outer_fit_audit.json').read_text()))
    complete_files(checkpoint, ['model.pt', 'preprocessing.pkl', 'training_history.csv', 'outer_fit_audit.json'],
                   n_blocks=chosen['n_blocks'], outer_fold=outer, variant=variant)
    del pipeline
    gc.collect()
    progress(args.output, 'running', 'outer_refit_complete', outer_fold=outer,
             variant=variant, n_blocks=chosen['n_blocks'])
    return model, fitted, test_ids


def run_variant(package, outer, variant, args):
    location = args.output / 'output_data/folds' / f'{outer:02d}_{variant}'
    if checked_completion(location) is not None:
        print(f'Resume: outer={outer} variant={variant} already complete', flush=True)
        return False
    started = time.perf_counter()
    chosen, oof, tuning, audits = tune(package, outer, variant, args)
    threshold = choose_sensitivity_threshold(oof.y_true, oof.score, .90)
    if threshold['status'] != 'selected_on_inner_oof':
        raise ValueError('A training-side sensitivity threshold could not be selected.')
    location.mkdir(parents=True, exist_ok=True)
    model, pipeline, test_ids = outer_fit(package, outer, variant, chosen, args, location)
    checkpoint = args.output / 'checkpoints/folds' / f'{outer:02d}_{variant}'
    # Restore small report evidence when resuming a finished outer fit.
    for name in ['training_history.csv', 'outer_fit_audit.json']:
        (location / name).write_bytes((checkpoint / name).read_bytes())
    audits.append(json.loads((location / 'outer_fit_audit.json').read_text()))
    key = identity(variant, outer)
    quality = source_quality_groups(package['metadata'].reset_index()).loc[test_ids]
    truth = package['targets'].loc[test_ids, 'target_afib'].to_numpy()
    patients = package['folds'].loc[test_ids, 'patient_id'].to_numpy()
    raw = package['raw'].loc[test_ids]
    metrics, patient_rows, subgroup, reliability, corruption = [], [], [], [], []
    from afib.preprocessing import perturb_measurements
    predictions = location / 'outer_predictions.csv.gz'
    temporary = predictions.with_name(predictions.name + '.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', newline='') as handle:
        first = True
        for scenario, options in SCENARIOS.items():
            for repeat in ([0] if scenario == 'observed' else range(args.repeats)):
                changed, audit = perturb_measurements(
                    raw, pipeline['perturb_iqr'], **options, repeat=repeat,
                    seed=package['manifest']['seed'], stream='evaluation', return_audit=True,
                )
                scores = np.asarray(model.predict_proba(transform_heldout(changed, pipeline)), dtype=np.float64)
                condition = {**key, 'scenario': scenario, 'repeat': repeat}
                pd.DataFrame({'ecg_id': test_ids, 'patient_id': patients, 'y_true': truth,
                              'score': scores, 'quality_group': quality.to_numpy()}).assign(
                                  **condition).to_csv(handle, index=False, header=first)
                first = False
                corruption.append({**condition, **audit})
                for policy, decision in [('fixed_0.50', .5), ('inner_sensitivity_0.90', threshold['threshold'])]:
                    meta = {**condition, 'threshold_policy': policy}
                    metrics.append({**meta, **binary_metrics(truth, scores, decision)})
                    patient_rows.append({**meta, **patient_weighted_metrics(truth, scores, patients, decision)})
                    subgroup.extend({**meta, **row} for row in
                                    quality_subgroup_metrics(truth, scores, quality, decision).to_dict('records'))
                reliability.extend(reliability_bins(truth, scores, condition))
                print(f'Tested outer={outer} variant={variant} scenario={scenario} repeat={repeat}', flush=True)
    os.replace(temporary, predictions)
    validate_scenario_repeats(pd.DataFrame(metrics).drop_duplicates(['scenario', 'repeat']),
                             SCENARIOS, repeats=range(args.repeats))
    for name, frame in [
        ('metrics.csv', metrics), ('patient_weighted_metrics.csv', patient_rows),
        ('quality_subgroups.csv', subgroup), ('reliability.csv', reliability),
        ('inner_oof_predictions.csv.gz', oof), ('tuning.csv', tuning),
    ]:
        csv_write(location / name, frame)
    json_write(location / 'fit_audit.json', audits)
    json_write(location / 'perturbation_audit.json', {'training': pipeline['augmentation_audit'], 'test': corruption})
    json_write(location / 'selection.json', {
        **key, **chosen, 'secondary_threshold': threshold,
        'seconds': time.perf_counter() - started,
    })
    train_ids, _ = split_ids(package, outer)
    json_write(location / 'majority_baseline.json', {
        'outer_fold': outer,
        **majority_baseline(package['targets'].loc[train_ids, 'target_afib'], truth),
    })
    filenames = [
        'metrics.csv', 'patient_weighted_metrics.csv', 'quality_subgroups.csv',
        'reliability.csv', 'inner_oof_predictions.csv.gz', 'outer_predictions.csv.gz',
        'tuning.csv', 'fit_audit.json', 'perturbation_audit.json', 'selection.json',
        'training_history.csv', 'outer_fit_audit.json', 'majority_baseline.json',
    ]
    complete_files(location, filenames, **key, n_blocks=chosen['n_blocks'],
                   checkpoints={str((checkpoint / name).relative_to(args.output)): sha256(checkpoint / name)
                                for name in ('model.pt', 'preprocessing.pkl')})
    del model, pipeline
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    progress(args.output, 'running', 'outer_evaluation_complete', outer_fold=outer,
             variant=variant, n_blocks=chosen['n_blocks'])
    return True


def _full_protocol(args):
    return (set(args.outer_folds) == set(range(10)) and set(args.variants) == set(VARIANTS)
            and args.repeats == 3 and args.epochs == 100 and not args.no_refine)


def run_signature(args, package):
    import torch
    resolved = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if resolved == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch cannot access a CUDA GPU.')
    sources = [Path(__file__), Path(__file__).with_name('data.py'), Path(__file__).with_name('model.py'),
               PROJECT / 'src/afib/preprocessing.py', PROJECT / 'src/afib/data_preparation.py',
               PROJECT / 'src/afib/evaluation.py']
    versions = {'python': platform.python_version()}
    for package_name in ['numpy', 'pandas', 'torch', 'tabm']:
        versions[package_name] = importlib.metadata.version(package_name)
    return {
        'model': 'tabm', 'variants': args.variants, 'outer_folds': args.outer_folds,
        'n_inner_folds': 3, 'initial_grid': INITIAL_GRID, 'refine': not args.no_refine,
        'epochs': args.epochs, 'repeats': args.repeats, 'seed': package['manifest']['seed'],
        'batch_size': 256, 'device': resolved, 'resolved_device': resolved,
        'architecture': {'width': 128, 'members': 16, 'dropout': .1, 'numerical_embeddings': False},
        'optimizer': {'name': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001},
        'selection': 'mean observed inner macro-F1 at 0.5; exact ties use fewer blocks',
        'thresholds': {'primary': .5, 'secondary_inner_sensitivity': .9},
        'manifest': package['manifest'],
        'code': {str(path.relative_to(PROJECT)).replace('\\', '/'): source_sha256(path) for path in sources},
        'source_hash_normalization': 'LF',
        'versions': versions, 'protocol_complete_requested': _full_protocol(args),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['tune', 'run', 'report'], default='run')
    parser.add_argument('--blocks', type=int, choices=range(1, 6), help='One candidate for --phase tune.')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--outer-folds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--output', type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase == 'report':
        from models.tabm.reporting import generate_reports
        return generate_reports(args.output)
    if (not args.outer_folds or len(set(args.outer_folds)) != len(args.outer_folds)
            or not set(args.outer_folds).issubset(range(10))
            or len(set(args.variants)) != len(args.variants)
            or args.epochs < 1 or args.repeats not in range(1, 4)):
        parser.error('Use unique valid folds/variants, at least one epoch and 1..3 repeats.')
    if args.blocks is not None and args.phase != 'tune':
        parser.error('--blocks only applies to --phase tune; the full run selects from the planned grid.')
    args.outer_folds = sorted(args.outer_folds)
    args.variants = [variant for variant in VARIANTS if variant in args.variants]
    if args.output == CANONICAL_OUTPUT.resolve() and not _full_protocol(args):
        parser.error('Reduced runs require a separate --output, for example results/tabm/pilots/smoke.')
    for variable, folder in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython')]:
        os.environ.setdefault(variable, str(PROJECT / '.runtime' / folder))
    with output_lock(args.output):
        try:
            package = load_processed_data(processed_directory(PROJECT))
            signature = run_signature(args, package)
            config = args.output / 'run_config.json'
            if config.exists():
                if not compatible_signature(PROJECT, args.output, json.loads(config.read_text(encoding='utf-8')), signature):
                    raise ValueError('Existing output has different code/data/configuration. Use a new --output folder.')
            else:
                json_write(config, signature)
            progress(args.output, 'running', args.phase)
            for outer in args.outer_folds:
                for variant in args.variants:
                    if args.phase == 'tune':
                        cache = {}
                        for blocks in ([args.blocks] if args.blocks is not None else INITIAL_GRID):
                            for inner in range(3):
                                inner_fit(package, outer, variant, blocks, inner, args, cache)
                        del cache
                        gc.collect()
                    else:
                        if run_variant(package, outer, variant, args):
                            from models.tabm.reporting import generate_reports
                            generate_reports(args.output)
            if args.phase == 'run':
                from models.tabm.reporting import generate_reports
                generate_reports(args.output)
            progress(args.output, 'completed', args.phase,
                     complete_tabm_plan=args.phase == 'run' and _full_protocol(args))
        except BaseException as error:
            progress(args.output, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     args.phase, error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()

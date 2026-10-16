"""Verify the saved result bundle without datasets, model imports or training.

Use --write-manifests once after independently validating deliberately refreshed
results. The default verifies the recorded inventory and SHA-256 file bytes.
This is an integrity check, not a substitute for either model's check_results.py.
Submission copies may omit LR fitted pipelines and TabM's local course-coverage
audit; both exclusions are reported explicitly against the unchanged manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
MODELS = ('logistic_regression', 'tabm')
MANIFEST_NAME = 'artifact_manifest.json'
FOLD_REQUIRED = ('complete.json', 'selection.json', 'metrics.csv', 'tuning.csv',
                 'fit_audit.json', 'perturbation_audit.json',
                 'inner_oof_predictions.csv.gz', 'outer_predictions.csv.gz')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_digest(path, git_object_format=None):
    digest = hashlib.new(git_object_format or 'sha256')
    if git_object_format:
        digest.update(f'blob {path.stat().st_size}\0'.encode('ascii'))
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(output):
    """Canonical evidence only; exclude pilots, local TabM models and live state."""
    candidates = [output / 'run_config.json']
    for name in ('output_data', 'predictions', 'reports', 'source_provenance'):
        directory = output / name
        if directory.exists():
            candidates.extend(path for path in directory.rglob('*') if path.is_file())
    for path in sorted(candidates):
        if path.name == MANIFEST_NAME or path.name.endswith('.tmp'):
            continue
        require(path.resolve().is_relative_to(output.resolve()), f'Artifact escapes output: {path}')
        yield path


def build_manifest(output, model):
    """Hash evidence and check that completed fold directories match the request."""
    config = read_json(output / 'run_config.json')
    completion = read_json(output / 'reports' / 'completion.json')
    validation = read_json(output / 'reports' / 'validation.json')
    require(completion.get(f'complete_{"logistic" if model == "logistic_regression" else "tabm"}_plan') is True,
            f'{model}: full experiment is not marked complete')
    require(validation.get('status') == 'passed', f'{model}: independent validation has not passed')
    expected = {f'{fold:02d}_{variant}' for fold in config['outer_folds'] for variant in config['variants']}
    folds = output / 'output_data' / 'folds'
    actual = {path.name for path in folds.iterdir() if path.is_dir()}
    require(actual == expected, f'{model}: fold coverage differs; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}')
    paths = list(artifact_paths(output))
    files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': file_digest(path)}
             for path in paths}
    for name in sorted(expected):
        location = folds / name
        require(all((location / file).is_file() for file in FOLD_REQUIRED), f'{model}/{name}: required fold evidence is missing')
        require(read_json(location / 'complete.json').get('completed') is True, f'{model}/{name}: incomplete fold')
    # TabM's original commit markers provide an independent checksum baseline.
    # Logistic regression predates those hashes; its new manifest records the
    # independently rechecked bundle without inventing historical checksums.
    markers_verified = 0
    if model == 'tabm':
        markers = sorted((output / 'output_data').rglob('complete.json'))
        for marker_path in markers:
            marker = read_json(marker_path)
            require(marker.get('completed') is True, f'Incomplete marker: {marker_path}')
            require(bool(marker.get('files')), f'Missing recorded artifact hashes: {marker_path}')
            for name, digest in marker['files'].items():
                require(name not in ('', '.', '..') and '/' not in name and '\\' not in name,
                        f'Unsafe artifact name in marker: {name}')
                relative = (marker_path.parent / name).relative_to(output).as_posix()
                require(relative in files, f'Unknown artifact in marker: {relative}')
                require(files[relative]['sha256'] == digest, f'TabM original artifact hash changed: {relative}')
            markers_verified += 1
        require(markers_verified == completion['n_fits'], 'TabM fit marker count differs from completion report')
    return {
        'schema_version': 1,
        'model': model,
        'hash_algorithm': 'sha256',
        'hash_scope': 'raw file bytes; paths relative to this model result directory',
        'excluded': ['README.md', 'pilots/', 'checkpoints/', 'logs/', 'live status and locks', MANIFEST_NAME, '*.tmp'],
        'validation_scope': 'Artifact integrity and declared coverage only; numerical and data checks are recorded in reports/validation.json.',
        'completed_outer_fold_variants': len(expected),
        'fits_recorded': completion['n_fits'],
        'original_tabm_commit_markers_verified': markers_verified,
        'file_count': len(files),
        'total_bytes': sum(item['bytes'] for item in files.values()),
        'files': files,
    }


def check_manifest(output, model, write=False):
    current = build_manifest(output, model)
    target = output / 'reports' / MANIFEST_NAME
    if write:
        target.write_text(json.dumps(current, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    else:
        require(target.is_file(), f'Missing manifest: {target}; create it with --write-manifests after numerical validation')
        saved = read_json(target)
        old_files, new_files = saved.get('files', {}), current['files']
        changed = sorted(name for name in old_files.keys() & new_files.keys() if old_files[name] != new_files[name])
        # Fitted LR pipelines contain learned arrays and are unnecessary for
        # checking saved predictions. A submission may omit these local resume
        # files while retaining the original manifest as the historical record.
        optional_models = {name for name in old_files if model == 'logistic_regression'
                           and len(Path(name).parts) == 4
                           and Path(name).parts[:2] == ('output_data', 'folds')
                           and Path(name).name == 'pipeline.pkl'}
        omitted_models = sorted((old_files.keys() - new_files.keys()) & optional_models)
        # This syllabus keyword audit records personal course-file locations;
        # it is unrelated to model predictions, metrics or experiment coverage.
        optional_ancillary = {'reports/course_coverage.json'} if model == 'tabm' else set()
        omitted_ancillary = sorted((old_files.keys() - new_files.keys()) & optional_ancillary)
        missing = sorted(old_files.keys() - new_files.keys() - optional_models - optional_ancillary)
        extra = sorted(new_files.keys() - old_files.keys())
        require(not (changed or missing or extra), f'{model}: artifact integrity failed; changed={changed[:10]}, missing={missing[:10]}, extra={extra[:10]}')
        comparable = dict(saved)
        omitted = set(omitted_models + omitted_ancillary)
        if omitted:
            comparable['files'] = {name: record for name, record in old_files.items() if name not in omitted}
            comparable['file_count'] = len(comparable['files'])
            comparable['total_bytes'] = sum(record['bytes'] for record in comparable['files'].values())
        require(comparable == current, f'{model}: manifest metadata differs')
    return {'model': model, 'status': 'manifest_written' if write else 'passed',
            'files': current['file_count'], 'bytes': current['total_bytes'],
            'outer_fold_variants': current['completed_outer_fold_variants'],
            'fits': current['fits_recorded'], 'datasets_required': False,
            'model_training_performed': False,
            'omitted_local_fitted_models': 0 if write else len(omitted_models),
            'omitted_ancillary_audits': [] if write else omitted_ancillary,
            'original_manifest_modified': bool(write)}


def compare_git_index(project, paths, index, object_format):
    """Require staged blob bytes to match the already verified working files."""
    missing, changed = [], []
    for path in sorted(set(paths)):
        relative = path.resolve().relative_to(project).as_posix()
        staged = index.get(relative)
        if staged is None:
            missing.append(relative)
        elif staged != file_digest(path, git_object_format=object_format):
            changed.append(relative)
    require(not (missing or changed), f'Git index differs from verified files; unstaged/unmerged={missing[:10]}, different_bytes={changed[:10]}. Stage the intended files first; no files were staged by this checker.')


def check_git_index(results, models):
    def git(*arguments, directory=results):
        return subprocess.run(['git', '-C', str(directory), *arguments], check=True,
                              capture_output=True, text=True, encoding='utf-8').stdout

    project = Path(git('rev-parse', '--show-toplevel').strip()).resolve()
    object_format = git('rev-parse', '--show-object-format').strip()
    require(object_format in ('sha1', 'sha256'), f'Unsupported Git object format: {object_format}')
    index = {}
    for row in git('ls-files', '--stage', '-z', directory=project).split('\0'):
        if row:
            metadata, name = row.split('\t', 1)
            _, object_id, stage = metadata.split()
            if stage == '0':
                index[name] = object_id
    # The staged attributes must preserve these bytes on a future checkout too.
    paths, data_names = [project / '.gitattributes'], set()
    for model in models:
        output = (results / model).resolve()
        paths.extend(artifact_paths(output))
        paths.append(output / 'reports' / MANIFEST_NAME)
        config = read_json(output / 'run_config.json')
        manifest = config.get('manifest', {})
        names = [source['path'] for source in manifest.get('sources', {}).values()]
        names.extend(f'processed_data/{name}' for name in manifest.get('files', {}))
        names.append('processed_data/manifest.json')
        # .gitignore alone does not remove files that were tracked previously.
        # A submission index must exclude all raw and prepared dataset files.
        data_names.update(name for name in names if name in index)
    data_names.update(name for name in index if name.startswith(('data/', 'processed_data/')))
    require(not data_names, f'Dataset files remain in the Git index: {sorted(data_names)[:10]}. '
            'Stage their removal before submitting; keep local datasets outside the repository.')
    compare_git_index(project, paths, index, object_format)
    return {'git_index': 'passed', 'staged_files_checked': len(set(paths)),
            'tracked_input_files_checked': len(data_names), 'files_staged_by_checker': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PROJECT / 'results')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write-manifests', action='store_true', help='Record deliberately validated outputs; changes the checksum baseline.')
    mode.add_argument('--check-git-index', action='store_true', help='After staging, check artifact bytes and ensure datasets are absent; never stages files.')
    args = parser.parse_args()
    try:
        reports = [check_manifest(args.results / model, model, write=args.write_manifests) for model in args.models]
        if args.check_git_index:
            reports.append(check_git_index(args.results, args.models))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Result integrity check failed: {error}\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()

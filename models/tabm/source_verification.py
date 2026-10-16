"""Verify documented directory changes against saved TabM training sources.

Historical run configurations remain immutable. A preserved runner must match
the recorded source hash before its syntax tree can authorize the path changes.
Reviewed input routing preserves data identities and calculations. Runtime
settings and all remaining experiment options still have to match exactly.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


RUNNER_SOURCE = 'algorithms/tabm/run_experiment.py'
_COMPATIBILITY_IMPORT = 'from algorithms.tabm.source_verification import compatible_signature'
_OLD_GUARD = "json.loads(config.read_text(encoding='utf-8')) != signature"
_NEW_GUARD = "not compatible_signature(PROJECT, args.output, json.loads(config.read_text(encoding='utf-8')), signature)"


def _tree(source):
    return ast.parse(source)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def active_source_name(name):
    """Resolve historical source keys without modifying saved configurations."""
    prefix = 'algorithms/tabm/'
    return 'models/tabm/' + name[len(prefix):] if name.startswith(prefix) else name


def _historical_source_name(name):
    prefix = 'models/tabm/'
    return 'algorithms/tabm/' + name[len(prefix):] if name.startswith(prefix) else name


def _layout_v1_tree(source):
    """Reverse routing, compatibility and package imports; retain training syntax."""
    tree = _tree(source)
    class RestorePackage(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'models.tabm' or (node.module or '').startswith('models.tabm.'):
                node.module = 'algorithms.tabm' + node.module[len('models.tabm'):]
            return node

    tree = RestorePackage().visit(tree)
    compatibility_import = _dump(_tree(_COMPATIBILITY_IMPORT).body[0])
    path_import = _dump(_tree('from afib.paths import processed_directory').body[0])
    tree.body = [node for node in tree.body if _dump(node) not in (compatibility_import, path_import)]
    replacements = {
        _dump(_tree("CANONICAL_OUTPUT = PROJECT / 'results/tabm'").body[0]):
            _tree("CANONICAL_OUTPUT = PROJECT / 'models/tabm'").body[0],
        _dump(_tree("parser.error('Reduced runs require a separate --output, for example results/tabm/pilots/smoke.')").body[0]):
            _tree("parser.error('Reduced runs require a separate --output, for example models/tabm/pilots/smoke.')").body[0],
        _dump(ast.parse(_NEW_GUARD, mode='eval').body):
            ast.parse(_OLD_GUARD, mode='eval').body,
        _dump(ast.parse('processed_directory(PROJECT)', mode='eval').body):
            ast.parse("PROJECT / 'processed_data'", mode='eval').body,
    }

    class RestoreLayout(ast.NodeTransformer):
        def generic_visit(self, node):
            replacement = replacements.get(_dump(node))
            return replacement if replacement is not None else super().generic_visit(node)

    return _dump(RestoreLayout().visit(tree))


def _digest(source, normalization):
    if normalization not in ('raw', 'LF'):
        raise ValueError('Unsupported source hash normalization')
    if normalization == 'LF':
        source = source.replace(b'\r\n', b'\n')
    return hashlib.sha256(source).hexdigest()


def verify_runner_migration(project, output, expected, normalization):
    """Require original evidence and unchanged training syntax; return provenance."""
    project, output = Path(project).resolve(), Path(output).resolve()
    relative = Path('source_provenance/layout_v1') / RUNNER_SOURCE
    candidates = [output / relative, project / 'results/tabm' / relative]
    snapshot = next((path for path in candidates if path.is_file()), None)
    if snapshot is None:
        raise ValueError('Training source changed: missing preserved TabM layout source')
    original = snapshot.read_bytes()
    if _digest(original, normalization) != expected:
        raise ValueError('Training source changed: preserved TabM runner hash differs from saved run')
    current = (project / active_source_name(RUNNER_SOURCE)).read_text(encoding='utf-8')
    if _layout_v1_tree(current) != _dump(_tree(original.decode('utf-8'))):
        raise ValueError('Training source changed: TabM runner differs beyond approved paths and imports')
    return str(snapshot)


def _without_docstrings(tree):
    """Remove explanatory text while retaining every executable statement."""
    class StripDocstrings(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:]
            return node
    return StripDocstrings().visit(tree)


def verify_source_migration(project, output, name, expected, normalization):
    """Verify the original hash and the exact reviewed external-data routing.

    Only two source-loading functions changed to accept external directories.
    Their complete ASTs are pinned below; preprocessing, folds and labels still
    have to match the original source statement for statement.
    """
    if _historical_source_name(name) == RUNNER_SOURCE:
        return verify_runner_migration(project, output, expected, normalization)
    if _historical_source_name(name) == 'algorithms/tabm/model.py':
        # The estimator changed only this obsolete requirements-path reference.
        source = Path(project) / active_source_name(name)
        restored = source.read_bytes().replace(
            b'pinned in models/tabm/requirements.txt',
            b'pinned in algorithms/tabm/requirements.txt')
        if _digest(restored, normalization) != expected:
            raise ValueError('Training source changed: TabM estimator differs beyond its documentation path')
        return str(source)
    if name != 'src/afib/data_preparation.py':
        raise ValueError(f'Training source changed since saved run: {name}')
    project = Path(project)
    snapshot = project / 'models/tabm/historical_sources' / name
    if not snapshot.is_file() or _digest(snapshot.read_bytes(), normalization) != expected:
        raise ValueError('Training source changed: preserved data-preparation hash differs from saved run')
    original = _without_docstrings(_tree(snapshot.read_text(encoding='utf-8')))
    current = _without_docstrings(_tree((project / name).read_text(encoding='utf-8')))
    approved_functions = {
        'load_source_data': '53637052c4e4c261ffecfba6697171de903527014e4a03130057ebcf96e0f891',
        'prepare_dataset': '3933f69ba309be4e297b8ed15e4e3668f50b7dfe64921ac364b771f85a1ecbb9',
    }
    old_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
    path_import = _dump(_tree('from .paths import data_directory, processed_directory').body[0])
    restored = []
    for node in current.body:
        if _dump(node) == path_import:
            continue
        if isinstance(node, ast.FunctionDef) and node.name in approved_functions:
            digest = hashlib.sha256(_dump(node).encode()).hexdigest()
            if digest != approved_functions[node.name]:
                raise ValueError(f'Training source changed: unreviewed data routing in {node.name}')
            node = old_functions[node.name]
        restored.append(node)
    current.body = restored
    if _dump(current) != _dump(original):
        raise ValueError('Training source changed: data preparation differs beyond reviewed directory routing')
    return str(snapshot)


def compatible_signature(project, output, saved, current):
    """Accept exact signatures or a proven path-only migration of the runner."""
    if saved == current:
        return True
    if {key: value for key, value in saved.items() if key != 'code'} != {
            key: value for key, value in current.items() if key != 'code'}:
        return False
    saved_code = {_historical_source_name(name): digest for name, digest in saved.get('code', {}).items()}
    current_code = {_historical_source_name(name): digest for name, digest in current.get('code', {}).items()}
    if len(saved_code) != len(saved.get('code', {})) or len(current_code) != len(current.get('code', {})):
        return False
    if set(saved_code) != set(current_code) or RUNNER_SOURCE not in saved_code:
        return False
    # Every changed source must match the supplied current hash and its
    # preserved original; new paths cannot hide a training calculation edit.
    normalization = saved.get('source_hash_normalization', 'raw')
    try:
        for name, expected in saved_code.items():
            if expected == current_code[name]:
                continue
            source = Path(project) / active_source_name(name)
            if _digest(source.read_bytes(), normalization) != current_code[name]:
                return False
            verify_source_migration(project, output, name, expected, normalization)
    except (OSError, SyntaxError, ValueError):
        return False
    return True

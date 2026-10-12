"""Recognize only the documented output-path migration in saved TabM runs.

Historical run configurations remain immutable. A preserved runner must match
the recorded source hash before its syntax tree can authorize the path changes.
Every other source, runtime setting and experiment option still matches exactly.
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


def _layout_v1_tree(source):
    """Reverse four exact routing/compatibility nodes; retain all other syntax."""
    tree = _tree(source)
    compatibility_import = _dump(_tree(_COMPATIBILITY_IMPORT).body[0])
    tree.body = [node for node in tree.body if _dump(node) != compatibility_import]
    replacements = {
        _dump(_tree("CANONICAL_OUTPUT = PROJECT / 'results/tabm'").body[0]):
            _tree("CANONICAL_OUTPUT = PROJECT / 'models/tabm'").body[0],
        _dump(_tree("parser.error('Reduced runs require a separate --output, for example results/tabm/pilots/smoke.')").body[0]):
            _tree("parser.error('Reduced runs require a separate --output, for example models/tabm/pilots/smoke.')").body[0],
        _dump(ast.parse(_NEW_GUARD, mode='eval').body):
            ast.parse(_OLD_GUARD, mode='eval').body,
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
    current = (project / RUNNER_SOURCE).read_text(encoding='utf-8')
    if _layout_v1_tree(current) != _dump(_tree(original.decode('utf-8'))):
        raise ValueError('Training source changed: TabM runner differs beyond approved output paths')
    return str(snapshot)


def compatible_signature(project, output, saved, current):
    """Accept exact signatures or a proven path-only migration of the runner."""
    if saved == current:
        return True
    if {key: value for key, value in saved.items() if key != 'code'} != {
            key: value for key, value in current.items() if key != 'code'}:
        return False
    saved_code, current_code = saved.get('code', {}), current.get('code', {})
    if set(saved_code) != set(current_code) or RUNNER_SOURCE not in saved_code:
        return False
    if any(digest != current_code[name] for name, digest in saved_code.items()
           if name != RUNNER_SOURCE):
        return False
    # Do not trust a caller-supplied digest that differs from the actual runner.
    normalization = saved.get('source_hash_normalization', 'raw')
    if _digest((Path(project) / RUNNER_SOURCE).read_bytes(), normalization) != current_code[RUNNER_SOURCE]:
        return False
    try:
        verify_runner_migration(project, output, saved_code[RUNNER_SOURCE], normalization)
    except (OSError, SyntaxError, ValueError):
        return False
    return True

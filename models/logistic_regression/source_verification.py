"""Verify saved training sources and the limited changes made after the run."""
import ast
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _syntax(node):
    return ast.dump(node, include_attributes=False)


def _normalise_runner_paths(tree):
    """Undo only the reviewed relocation expressions, leaving calculations intact."""
    paths = {
        'run_variant': {"output / 'output_data' / 'folds'": "output / 'folds'"},
        'summarise': {
            "output / 'output_data' / " + tail: 'output / ' + tail
            for tail in (
                "'folds'", "'selected_hyperparameters.csv'", "f'{name}.csv'",
                "'fold_mean_metrics.csv'", "'summary.csv'", "'paired_stress_drops.csv'",
                "'paired_method_differences.csv'", "'majority_baseline.csv'",
                "f'{kind}_summary.csv'", "'recovery.csv'", "'recovery_summary.csv'",
            )
        },
        'main': {"PROJECT / 'results/logistic_regression'": "PROJECT / 'reports/logistic_regression'"},
    }
    paths['summarise'].update({
        "output / 'reports' / f'{metric}.png'": "output / f'{metric}.png'",
        "output / 'reports' / 'completion.json'": "output / 'completion.json'",
    })
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in paths:
            continue
        replacements = {
            _syntax(ast.parse(new, mode='eval').body): old
            for new, old in paths[function.name].items()
        }

        class RelocatedPaths(ast.NodeTransformer):
            def visit_BinOp(self, node):
                replacement = replacements.get(_syntax(node))
                if replacement is not None:
                    return ast.parse(replacement, mode='eval').body
                return self.generic_visit(node)

        RelocatedPaths().visit(function)
        if function.name == 'summarise':
            directory_setup = {
                _syntax(ast.parse(statement).body[0]) for statement in (
                    "(output / 'output_data').mkdir(parents=True, exist_ok=True)",
                    "(output / 'reports').mkdir(parents=True, exist_ok=True)",
                )
            }
            function.body = [statement for statement in function.body
                             if _syntax(statement) not in directory_setup]
    return tree


def _normalise_reviewed_exports(tree):
    """Accept exact reviewed export code, never arbitrary code sharing its name."""
    # These digests cover complete, documentation-free ASTs. Any change to the
    # export loops or routing must be reviewed explicitly before it is allowed.
    functions = {
        'prediction_export_directory': '4389f5515aae6506710590f0c37080dc56a91c32fe3ec1fe50fe42bbe45bdc44',
        'export_predictions': '5350da9c00d565d5ef869359034b2202de36e0c0365313ffce16779508fa3cde',
    }
    legacy_export = 'e5b72a107d86b41ddc4279325755751bc8899aab16a1107cee8a24528a00bb80'
    export_call = _syntax(ast.parse(
        'export_predictions(locations, prediction_export_directory(output, args))').body[0])
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            assert sha256(_syntax(node).encode()) == functions[node.name], node.name + ': export code changed'
            continue
        if isinstance(node, ast.FunctionDef) and node.name == 'summarise':
            statements = []
            index = 0
            while index < len(node.body):
                block = ast.Module(body=node.body[index:index + 4], type_ignores=[])
                if sha256(_syntax(block).encode()) == legacy_export:
                    index += 4
                    continue
                statement = node.body[index]
                if _syntax(statement) != export_call:
                    statements.append(statement)
                index += 1
            node.body = statements
        body.append(node)
    tree.body = body
    return tree


def calculation_tree(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    class StripDocumentation(ast.NodeTransformer):
        def visit_Constant(self, node):
            return node

        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]
            return node
    tree = StripDocumentation().visit(tree)
    if Path(path).name == 'run_experiment.py':
        tree = _normalise_reviewed_exports(tree)
        tree = _normalise_runner_paths(tree)
    return _syntax(tree)


def verify_sources(project, output, config):
    evidence = output / 'source_provenance'
    if not evidence.exists():
        for name, expected in config['code'].items():
            directory = project / ('models/logistic_regression' if name in ('model.py', 'run_experiment.py') else 'src/afib')
            assert sha256((directory / name).read_bytes()) == expected, name
        return {'mode': 'exact_saved_hashes'}
    original = json.loads((evidence / 'training_run_config.json').read_text())
    reviewed = json.loads((evidence / 'reviewed_run_config.json').read_text())
    assert config == reviewed, 'Saved configuration differs from the preserved reviewed configuration'
    assert {k: v for k, v in original.items() if k != 'code'} == {k: v for k, v in config.items() if k != 'code'}
    checked = []
    for name, expected in original['code'].items():
        baseline = evidence / 'training_sources' / name
        raw = baseline.read_bytes()
        assert expected in (sha256(raw), sha256(raw.replace(b'\r\n', b'\n'))), name + ': training snapshot hash'
        directory = project / ('models/logistic_regression' if name in ('model.py', 'run_experiment.py') else 'src/afib')
        assert calculation_tree(baseline) == calculation_tree(directory / name), name + ': calculation changed'
        checked.append(name)
    return {'mode': 'preserved_training_sources', 'files_checked': checked,
            'allowed_changes': 'documentation, exact reviewed prediction export code, and explicit output path relocations only',
            'training_and_metric_calculations_unchanged': True}

"""Verify saved training sources and the limited changes made after the run."""
import ast
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


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
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and
                     n.name in ('prediction_export_directory', 'export_predictions'))]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'summarise':
                def export_statement(statement):
                    if isinstance(statement, ast.Assign):
                        return any(isinstance(t, ast.Name) and t.id == 'export' for t in statement.targets)
                    if isinstance(statement, ast.Import):
                        return [a.name for a in statement.names] == ['gzip']
                    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                        function = statement.value.func
                        return ((isinstance(function, ast.Name) and function.id == 'export_predictions') or
                                (isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                                 and function.value.id == 'export' and function.attr == 'mkdir'))
                    if isinstance(statement, ast.For):
                        return (isinstance(statement.target, ast.Name) and statement.target.id == 'name'
                                and isinstance(statement.iter, ast.List)
                                and [getattr(x, 'value', None) for x in statement.iter.elts]
                                == ['inner_oof_predictions', 'outer_predictions'])
                    return False
                node.body = [s for s in node.body if not export_statement(s)]
    return ast.dump(tree, include_attributes=False)


def verify_sources(project, output, config):
    evidence = output / 'source_provenance'
    if not evidence.exists():
        for name, expected in config['code'].items():
            directory = project / ('algorithms/logistic_regression' if name in ('model.py', 'run_experiment.py') else 'src/afib')
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
        directory = project / ('algorithms/logistic_regression' if name in ('model.py', 'run_experiment.py') else 'src/afib')
        assert calculation_tree(baseline) == calculation_tree(directory / name), name + ': calculation changed'
        checked.append(name)
    return {'mode': 'preserved_training_sources', 'files_checked': checked,
            'allowed_changes': 'documentation and prediction export routing only',
            'training_and_metric_calculations_unchanged': True}

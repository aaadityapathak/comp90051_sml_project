"""Small numerical and split checks for the logistic regression implementation."""
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'algorithms/logistic_regression'))
from model import LogisticRegression, objective_gradient
from run_experiment import select_C, refinement_grid, patient_weighted_metrics, split_ids
from run_experiment import prediction_export_directory, export_predictions, VARIANTS
from source_verification import calculation_tree

PROJECT = Path(__file__).absolute().parents[1]


class LogisticTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[-2.,1.],[-1.,0.],[1.,0.],[2.,1.]])
        self.y=np.array([0,0,1,1]);self.w=np.array([1.,2.,2.,1.])

    def test_analytic_gradient_matches_finite_differences(self):
        theta=np.array([.3,-.2,.1]);eps=1e-6
        _,gradient=objective_gradient(theta,self.X,self.y,self.w,.1)
        numerical=[]
        for j in range(3):
            delta=np.zeros(3);delta[j]=eps
            numerical.append((objective_gradient(theta+delta,self.X,self.y,self.w,.1)[0]
                              -objective_gradient(theta-delta,self.X,self.y,self.w,.1)[0])/(2*eps))
        np.testing.assert_allclose(gradient,numerical,rtol=1e-6,atol=1e-8)

    def test_duplicate_half_weights_preserve_solution(self):
        a=LogisticRegression(.1).fit(self.X,self.y,self.w)
        b=LogisticRegression(.1).fit(np.tile(self.X,(2,1)),np.tile(self.y,2),np.tile(self.w/2,2))
        np.testing.assert_allclose(a.theta_,b.theta_,atol=1e-8)

    def test_regularization_and_separation(self):
        strong=LogisticRegression(.01).fit(self.X,self.y)
        weak=LogisticRegression(10).fit(self.X,self.y)
        self.assertGreater(np.linalg.norm(weak.theta_[:-1]),np.linalg.norm(strong.theta_[:-1]))
        self.assertTrue(np.array_equal(weak.predict_proba(self.X)>=.5,self.y))

    def test_intercept_unpenalized_and_stable_extreme_logits(self):
        X=np.zeros((4,1)); y=np.array([0,0,0,1])
        model=LogisticRegression(.001).fit(X,y)
        np.testing.assert_allclose(model.predict_proba(X),.25,atol=1e-5)
        value,gradient=objective_gradient(np.array([1000.,0.,0.]),self.X,self.y,self.w,.1)
        self.assertTrue(np.isfinite(value) and np.isfinite(gradient).all())

    def test_invalid_inputs_and_nonconvergence_fail(self):
        with self.assertRaises(ValueError):LogisticRegression(-1)
        with self.assertRaises(ValueError):LogisticRegression().fit(self.X,self.y,[0,1,1,1])
        with self.assertRaises(RuntimeError):LogisticRegression(max_iter=1).fit(self.X,self.y)

    def test_search_ties_and_one_step_endpoint_refinement(self):
        self.assertEqual(select_C({.01:.8,.1:.8,1.:.7}),.01)
        self.assertEqual(refinement_grid([.01,.1,1.],.01),[.001,.01,.1])
        self.assertEqual(refinement_grid([.01,.1,1.],1.),[.1,1.,10.])
        self.assertIsNone(refinement_grid([.01,.1,1.],.1))

    def test_patient_weights_not_record_weights(self):
        result=patient_weighted_metrics(np.array([1,1,0]),np.array([.8,.8,.8]),[10,10,20],.5)
        self.assertAlmostEqual(result['weighted_tp'],1.)
        self.assertAlmostEqual(result['weighted_fp'],1.)
        self.assertAlmostEqual(result['accuracy'],.5)

    def test_inner_split_excludes_outer_patients(self):
        frame=pd.DataFrame({'patient_id':[1,1,2,3,4], 'outer_fold':[0,0,1,2,2],
                            'inner_fold_when_outer_0':[-1,-1,0,1,2]},index=[10,11,12,13,14])
        training,validation=split_ids({'folds':frame},0,0)
        self.assertEqual(set(training),{13,14});self.assertEqual(set(validation),{12})

    def test_public_exports_require_complete_canonical_run(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            project = Path(temp)
            full = SimpleNamespace(outer_folds=list(range(10)), repeats=3, variants=VARIANTS)
            pilot = SimpleNamespace(outer_folds=[0], repeats=1, variants=['A'])
            canonical = project / 'results/logistic_regression'
            with patch('run_experiment.PROJECT', project):
                self.assertEqual(prediction_export_directory(canonical, full), canonical / 'predictions')
                self.assertEqual(prediction_export_directory(canonical, pilot), canonical / 'pilots/incomplete_run/predictions')
                custom = canonical / 'pilots/pilot'
                self.assertEqual(prediction_export_directory(custom, full), custom / 'predictions')
                self.assertEqual(prediction_export_directory(custom, pilot), custom / 'predictions')

    def test_pilot_export_preserves_public_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            project = Path(temp)
            public = project / 'results/logistic_regression/predictions'
            public.mkdir(parents=True)
            names = ['inner_oof_predictions', 'outer_predictions']
            for name in names:
                (public / f'{name}.csv.gz').write_bytes(b'official prediction sentinel')
            output = project / 'results/logistic_regression/pilots/pilot'
            location = output / 'output_data/folds/00_A'
            location.mkdir(parents=True)
            for name in names:
                pd.DataFrame({'ecg_id': [1], 'probability': [.7]}).to_csv(location / f'{name}.csv.gz', index=False)
            args = SimpleNamespace(outer_folds=[0], repeats=1, variants=['A'])
            with patch('run_experiment.PROJECT', project):
                export_predictions([location], prediction_export_directory(output, args))
            for name in names:
                self.assertEqual((public / f'{name}.csv.gz').read_bytes(), b'official prediction sentinel')
                self.assertEqual(len(pd.read_csv(output / f'predictions/{name}.csv.gz')), 1)

    def test_source_comparison_rejects_calculation_changes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            first, second = Path(temp) / 'first.py', Path(temp) / 'second.py'
            first.write_text('"""Before."""\ndef f(x):\n    return x + 1\n')
            second.write_text('"""After."""\n# Comment\ndef f(x):\n    return x + 1\n')
            self.assertEqual(calculation_tree(first), calculation_tree(second))
            second.write_text('def f(x):\n    return x + 2\n')
            self.assertNotEqual(calculation_tree(first), calculation_tree(second))

    def test_source_comparison_accepts_only_known_output_relocations(self):
        source = PROJECT / 'algorithms/logistic_regression/run_experiment.py'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            relocated = Path(temp) / 'run_experiment.py'
            text = source.read_text(encoding='utf-8')
            relocated.write_text(text.replace("output/'output_data'/'summary.csv'", "output/'summary.csv'"), encoding='utf-8')
            self.assertEqual(calculation_tree(source), calculation_tree(relocated))
            relocated.write_text(text.replace("output/'output_data'/'summary.csv'", "output/'elsewhere'/'summary.csv'"), encoding='utf-8')
            self.assertNotEqual(calculation_tree(source), calculation_tree(relocated))
            relocated.write_text(text.replace("'mean':float(np.mean(values))", "'mean':float(np.median(values))"), encoding='utf-8')
            self.assertNotEqual(calculation_tree(source), calculation_tree(relocated))

    def test_source_comparison_checks_export_logic_instead_of_ignoring_it(self):
        source = PROJECT / 'algorithms/logistic_regression/run_experiment.py'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            changed = Path(temp) / 'run_experiment.py'
            text = source.read_text(encoding='utf-8')
            changed.write_text(text.replace('chunksize=50000', 'chunksize=1'), encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'export_predictions: export code changed'):
                calculation_tree(changed)
            changed.write_text(text.replace('and complete:', 'and not complete:'), encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'prediction_export_directory: export code changed'):
                calculation_tree(changed)


if __name__=='__main__':unittest.main()

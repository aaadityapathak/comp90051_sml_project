"""Small numerical and split checks for the logistic regression implementation."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'algorithms/logistic_regression'))
from model import LogisticRegression, objective_gradient
from run_experiment import select_C, refinement_grid, patient_weighted_metrics, split_ids


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


if __name__=='__main__':unittest.main()

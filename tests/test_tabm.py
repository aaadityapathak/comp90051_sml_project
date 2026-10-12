"""Numerical protocol checks for the official TabM classifier wrapper."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.tabm.model import TabMClassifier, memberwise_weighted_loss


class TabMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.X = np.array([[-2., 1.], [-1., 0.], [1., 0.], [2., 1.]], dtype=np.float32)
        self.y = np.array([0., 0., 1., 1.], dtype=np.float32)
        self.weights = np.array([1., 3., 3., 1.])

    def classifier(self, **kwargs):
        return TabMClassifier(epochs=2, device='cpu', batch_size=3, verbose=False, **kwargs)

    def test_member_loss_matches_analytic_weighted_bce_and_gradient(self):
        values = np.array([[-3., 2.], [1., 4.]], dtype=np.float64)
        target = np.array([0., 1.])
        weight = np.array([0.5, 1.5])
        logits = torch.tensor(values[:, :, None], requires_grad=True)
        loss = memberwise_weighted_loss(logits, torch.tensor(target), torch.tensor(weight))
        expected = np.mean(weight * (np.logaddexp(0, values) - target[:, None] * values).mean(axis=1))
        self.assertAlmostEqual(loss.item(), expected, places=12)
        loss.backward()
        gradient = weight[:, None] * (1 / (1 + np.exp(-values)) - target[:, None]) / values.size
        np.testing.assert_allclose(logits.grad.numpy().squeeze(-1), gradient, rtol=1e-12)
        # Averaging logits before BCE changes the objective and must not occur.
        wrong = np.mean(weight * (np.logaddexp(0, values.mean(axis=1)) - target * values.mean(axis=1)))
        self.assertGreater(abs(loss.item() - wrong), 0.1)

    def test_dataset_weight_normalization_not_minibatch_ratio(self):
        logits = torch.tensor([[[0.]], [[2.]], [[-1.]], [[0.5]]])
        target = torch.tensor(self.y)
        weights = torch.tensor(self.weights / self.weights.mean(), dtype=torch.float32)
        full = memberwise_weighted_loss(logits, target, weights)
        # Unequal batch sizes still reconstruct the full objective exactly.
        first = memberwise_weighted_loss(logits[:1], target[:1], weights[:1])
        rest = memberwise_weighted_loss(logits[1:], target[1:], weights[1:])
        self.assertAlmostEqual(full.item(), ((first + 3 * rest) / 4).item(), places=6)
        # A one-row batch must retain its class/view weight.
        self.assertAlmostEqual(first.item(), np.log(2) * weights[0].item(), places=6)

    def test_probability_average_is_not_logit_average(self):
        class FixedMembers(torch.nn.Module):
            def forward(self, x_num, x_cat=None):
                return torch.tensor([0., 4.]).reshape(1, 2, 1).expand(len(x_num), -1, -1)
        fitted = self.classifier()
        fitted.model_, fitted.device_, fitted.n_features_in_ = FixedMembers(), torch.device('cpu'), 2
        actual = fitted.predict_proba(self.X)
        expected = (0.5 + 1 / (1 + np.exp(-4))) / 2
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        self.assertGreater(abs(actual[0] - 1 / (1 + np.exp(-2))), 0.1)

    def test_same_seed_reproduces_fit_and_global_weight_scaling(self):
        first = self.classifier(seed=7).fit(self.X, self.y, self.weights)
        second = self.classifier(seed=7).fit(self.X, self.y, self.weights * 8)
        np.testing.assert_array_equal(first.predict_proba(self.X), second.predict_proba(self.X))
        for name, parameter in first.model_.state_dict().items():
            torch.testing.assert_close(parameter, second.model_.state_dict()[name], rtol=0, atol=0)
        self.assertEqual([row['weighted_bce'] for row in first.history_],
                         [row['weighted_bce'] for row in second.history_])

    def test_fit_configuration_history_and_finite_probabilities(self):
        fitted = self.classifier(n_blocks=3).fit(self.X, self.y, self.weights)
        architecture = fitted.audit_['architecture']
        self.assertEqual({key: architecture[key] for key in ('n_blocks', 'd_block', 'k', 'dropout')},
                         {'n_blocks': 3, 'd_block': 128, 'k': 16, 'dropout': 0.1})
        self.assertIsNone(fitted.model_.num_module)
        self.assertEqual(fitted.model_.k, 16)
        self.assertEqual(fitted.n_features_in_, 2)
        self.assertEqual([row['epoch'] for row in fitted.history_], [1, 2])
        self.assertFalse(fitted.audit_['early_stopping'])
        self.assertEqual(fitted.audit_['optimizer'], {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4})
        self.assertEqual(fitted.audit_['weight_sum'], 8)
        result = fitted.predict_proba(self.X)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.isfinite(result).all() and ((result > 0) & (result < 1)).all())

    def test_checkpoint_roundtrip_is_exact(self):
        fitted = self.classifier().fit(self.X, self.y, self.weights)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'model.pt'
            fitted.save(path)
            restored = TabMClassifier.load(path, device='cpu')
            np.testing.assert_array_equal(fitted.predict_proba(self.X), restored.predict_proba(self.X))
            self.assertEqual(fitted.audit_, restored.audit_)
            self.assertEqual(fitted.history_, restored.history_)
            self.assertFalse(path.with_name('model.pt.tmp').exists())

    def test_invalid_inputs_fail(self):
        for kwargs in ({'epochs': 0}, {'n_blocks': 1.5}, {'batch_size': -1}, {'seed': -1}):
            with self.assertRaises(ValueError):
                TabMClassifier(**kwargs)
        for X, y, weight in [(self.X, self.y, [0, 1, 1, 1]),
                             (self.X, [0, 0, 0, 0], self.weights),
                             (self.X, [0, 1], self.weights),
                             (self.X * np.nan, self.y, self.weights)]:
            with self.assertRaises(ValueError):
                self.classifier().fit(X, y, weight)
        with self.assertRaises(RuntimeError):
            self.classifier().predict_proba(self.X)


if __name__ == '__main__':
    unittest.main()

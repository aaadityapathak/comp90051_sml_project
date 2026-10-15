"""L2 logistic regression for the AFIB experiments.

The intercept is unpenalized. The weighted objective is
(sum_i w_i * (log(1 + exp(z_i)) - y_i*z_i) + ||beta||^2/(2*C)) / sum_i w_i.
Using the same total weight for original and augmented data keeps C comparable.
SciPy supplies the L-BFGS optimizer; the loss and gradient are defined below.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def objective_gradient(theta, X, y, weight, C):
    # logaddexp avoids overflow for large positive logits.
    z = X @ theta[:-1] + theta[-1]
    total = weight.sum()
    loss = (weight @ (np.logaddexp(0, z) - y * z)
            + theta[:-1] @ theta[:-1] / (2 * C)) / total
    residual = weight * (expit(z) - y) / total
    gradient = np.empty_like(theta)
    gradient[:-1] = X.T @ residual + theta[:-1] / (C * total)
    gradient[-1] = residual.sum()
    return float(loss), gradient


class LogisticRegression:
    def __init__(self, C=0.1, max_iter=2000, tolerance=1e-5):
        if not np.isfinite(C) or C <= 0:
            raise ValueError('C must be finite and positive.')
        self.C, self.max_iter, self.tolerance = float(C), int(max_iter), float(tolerance)

    def fit(self, X, y, sample_weight=None, initial=None):
        X = np.asarray(X, dtype=np.float64, order='C')
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if (X.ndim != 2 or y.shape != (len(X),) or w.shape != y.shape
                or not X.shape[1] or not np.isfinite(X).all()
                or not np.isfinite(w).all() or (w <= 0).any()
                or set(np.unique(y)) != {0, 1}):
            raise ValueError('Finite 2D predictors, both binary classes, aligned positive weights required.')
        start = np.zeros(X.shape[1] + 1) if initial is None else np.asarray(initial).copy()
        if start.shape != (X.shape[1] + 1,) or not np.isfinite(start).all():
            raise ValueError('Initial coefficients do not match predictors.')
        # No bounds are passed, so this is an unconstrained L-BFGS fit.
        result = minimize(objective_gradient, start, args=(X, y, w, self.C),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': self.max_iter, 'maxfun': self.max_iter * 5,
                                   'gtol': self.tolerance, 'ftol': 1e-12, 'maxls': 50, 'maxcor': 20})
        gradient_max = float(np.max(np.abs(result.jac)))
        if not result.success or not np.isfinite(result.x).all() or gradient_max > 1e-4:
            raise RuntimeError(f'Unresolved convergence: {result.message}; max gradient={gradient_max:.3g}')
        self.theta_ = result.x
        self.audit_ = {'C': self.C, 'iterations': int(result.nit), 'objective': float(result.fun),
                       'max_abs_gradient': gradient_max, 'converged': bool(result.success),
                       'termination': str(result.message), 'n_train_views': len(y),
                       'n_features': X.shape[1], 'weight_sum': float(w.sum())}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.theta_) - 1 or not np.isfinite(X).all():
            raise ValueError('Prediction matrix must match finite fitted predictors.')
        return expit(X @ self.theta_[:-1] + self.theta_[-1])

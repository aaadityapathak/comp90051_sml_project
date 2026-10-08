# L2 logistic regression

**Purpose:** establish a regularised linear baseline for AFIB classification.

**Status:** selected; training and results are pending.

| Setting | Planned value |
| --- | --- |
| Tuned parameter | C = 0.01, 0.1, 1 |
| Fixed settings | L2 penalty, intercept, L-BFGS, maximum 2,000 iterations |
| Prediction identifier | `logistic_regression` |

Load each training/validation partition from `processed_data/` through the
[shared input workflow](../README.md). Use returned sample weights and resolve
convergence warnings before comparing scores. Fit all four A-D arms.

The separate median-versus-ridge-filling comparison also uses this classifier,
with original features and observed-only training. Keep it separate from the
main A-D comparison and follow the project plan.

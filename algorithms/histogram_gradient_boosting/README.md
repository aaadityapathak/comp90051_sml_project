# Shallow histogram gradient boosting

**Purpose:** test nonlinear relationships and interactions with shallow trees.

**Status:** selected; training and results are pending.

| Setting | Planned value |
| --- | --- |
| Tuned parameter | Maximum depth = 2, 3, 4 |
| Fixed settings | 150 boosting iterations; learning rate 0.05; at most 16 leaves; at least 30 rows per leaf; L2 = 1 |
| Early stopping | Disabled; use the supplied patient folds |
| Prediction identifier | `hist_gradient_boosting` |

Load each training/validation partition from `processed_data/` through the
[shared input workflow](../README.md). Use its finite encoded matrices and
sample weights for all four A-D arms. Keep the prediction identifier above
when saving outputs so the evaluation code can recognise this model.

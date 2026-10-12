# Shallow histogram gradient boosting

**Purpose:** test nonlinear relationships and interactions with shallow trees.

**Status:** selected; training and results are pending.

Follow the same layout as the other algorithms: keep Python source and any
algorithm-specific requirements directly in `algorithms/histogram_gradient_boosting/`,
and place algorithm-specific notebooks in its `notebooks/` subfolder when added.
Shared data preparation and cross-model evaluation notebooks remain in `notebooks/`
at the repository root.

Reserve `results/histogram_gradient_boosting/` for this algorithm's results:

| Subfolder | Contents when generated |
| --- | --- |
| `output_data/` | Fold outputs, metrics, tuning tables and audits |
| `reports/` | Figures, written reports and completion summaries |
| `checkpoints/` | Local model and resume checkpoints |
| `predictions/` | Published prediction tables for comparison |
| `pilots/` | Separate pilot and diagnostic runs |

Run configuration and source provenance belong at the algorithm's results root.
These locations are reserved; no training outputs have been generated yet.

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

# L2 logistic regression

This folder contains the linear baseline and its patient-grouped experiments.

The completed local run is recorded in `reports/logistic_regression/completion.json`.

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

## Run

From the project root using its Python 3.12 environment:

```powershell
python -m unittest discover -s tests -v
python algorithms/logistic_regression/run_experiment.py --workers 2
```

Default: all ten outer patient folds, all three inner folds, four A-D arms,
31 outer conditions (observed once and ten stress scenarios three times),
availability controls for B/D, duplicate-observed controls for A/B, and a
separate ridge-versus-median experiment. Ridge downstream comparisons use
observed and three-missing-lead conditions; recovery masks 10% of observed
continuous cells or 1/3/6 leads with three repeats in each outer-held-out fold.

For a small pilot, use a separate output directory:

```powershell
python algorithms/logistic_regression/run_experiment.py --variants A --outer-folds 0 --repeats 1 --output reports/logistic_pilot_new
```

Pilot scores are not final ten-fold results. The full run can take substantial
time. Completed fold/variant checkpoints are reused only if code, input hashes,
software versions and configuration match. Incomplete checkpoints are rerun.
Each process uses one BLAS thread; `--workers 1` uses less RAM. Do not run two
invocations against the same output directory simultaneously.

## Method and safeguards

`model.py` implements stable weighted binary cross-entropy, its analytic gradient,
an unpenalized intercept and L2 regularization. Unconstrained L-BFGS is supplied
by SciPy via `minimize(method='L-BFGS-B')` with no bounds. Maximum 2,000 iterations,
gradient tolerance 1e-5, and a hard failure for unsuccessful termination or a
remaining gradient above 1e-4. Coefficients and optimizer audits are saved.

The loss is `(sum(weight * BCE) + ||beta||^2/(2*C)) / sum(weight)`.
Class weights come from the shared training loader, without a second balancing
factor. Half-weight duplicates preserve exactly the same objective. C tuning is
manual: choose mean inner observed macro-F1 at threshold 0.5; exact ties choose
smaller C. One endpoint-directed three-value grid refinement is permitted, with
cached predictions for previously tried settings. Warm starts never cross an
inner split or use held-out labels. Every candidate converges independently.

The runner loads/checks the shared package once per process, then uses the same
`construct_training_views`, `build_features`, `perturb_measurements` and
`transform_features` functions as `load_prepared_split`. This lets the fitted training pipeline be reused across test stresses. Fold
membership comes from the existing manual patient fold file. All training
statistics are refitted inside each inner fold. Outer-test patients are excluded
from model selection, feature fitting, filler fitting and threshold selection.

Availability controls retain original predictors plus the 63 constructed
missingness indicators, while setting all constructed numerical values to zero.
Duplicate controls copy observed rows with half weights. The ridge filler uses
only training measurements and preserves original missingness indicators;
observed values remain unchanged. Its median reference is main arm A downstream,
and the shared median filler in the separate recovery experiment. No probability
recalibration is performed. The secondary 90%-sensitivity threshold is chosen
only from selected inner OOF predictions and frozen for all outer stresses.

## Results

`reports/logistic_regression/` contains:

- `summary.csv`: outer-fold mean and sample SD for every metric/condition.
- `metrics.csv`: confusion counts and metrics for each fold/repeat/threshold.
- `paired_stress_drops.csv`, `paired_method_differences.csv`: paired comparisons.
- `patient_weighted_metrics*.csv`: each patient contributes total weight one.
- `quality_subgroups*.csv`: source-annotation groups and their sizes.
- `reliability.csv`: ten probability bins per fold/condition, for calibration plots.
- `selected_hyperparameters.csv`, `tuning.csv`: grid searches and middle selections.
- `recovery*.csv`: eligible-field MAE, training-IQR-scaled errors, coverage and groups.
- `folds/`: inner/outer predictions, model/preprocessor pickles and audits.
- `macro_f1.png`, `sensitivity.png`, `brier.png`: means with outer-fold SD bars.

Main A-D predictions are exported separately into
`models/predictions/logistic_regression/*_predictions.csv.gz` with the group
notebook's required columns. Merge them with the other two classifiers before
running `02_noise_evaluation.ipynb`: that notebook correctly requires all three
models. Control/filler predictions remain in their own variant checkpoints.
Never load pickle files from untrusted sources.

Error bars use sample SD across ten outer-fold means. Stress repeats are averaged
within folds; these bars are not clinical confidence intervals. Record-level metrics are
primary; patient-weighted summaries are secondary. Results cannot establish
clinical safety or diagnostic benefit. Record the selected C and any grid refinement alongside the scores.

SciPy optimizer reference: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

## Check saved results without training

```powershell
python algorithms/logistic_regression/check_results.py
```

This command checks saved probabilities, patient separation, tuning choices,
thresholds, metrics and fold summaries. It does not fit a classifier. To view the
results, open `notebooks/04_logistic_regression_results.ipynb` and run its cells.
The bilingual handover report is in the parent `outputs` folder.

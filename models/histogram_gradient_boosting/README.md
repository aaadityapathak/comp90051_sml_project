
# Histogram Gradient Boosting (HGB) for AFIB Classification

## 1. Purpose

This model investigates nonlinear relationships and feature interactions for binary atrial fibrillation (AFIB) classification using Histogram Gradient Boosting (HGB).

We evaluate model performance under observed ECG conditions and under several types of signal corruption, including missing leads, noise, gain changes, and combined corruption.

## 2. Model Implementation

**Estimator:** `HistGradientBoostingClassifier`

The original model implementation uses the following fixed hyperparameters:

| Hyperparameter | Value |
|---|---|
| `max_iter` | 150 |
| `learning_rate` | 0.05 |
| `max_leaf_nodes` | 16 |
| `min_samples_leaf` | 30 |
| `l2_regularization` | 1.0 |
| `early_stopping` | False |
| `random_state` | 42 |

**Tuned hyperparameter:** `max_depth`

**Selection metric:** Mean inner-validation Macro-F1

**Prediction identifier:** `hist_gradient_boosting`

## 3. Experimental Arms

Four experimental arms are evaluated:

| Arm | Feature Set | Training Augmentation |
|---|---|---|
| A | Original ECG features | No |
| B | Original and engineered ECG features | No |
| C | Original ECG features | Yes |
| D | Original and engineered ECG features | Yes |

All arms follow the shared patient-grouped cross-validation splits and fold-specific preprocessing workflow.

For augmented arms, training-data corruption is applied to additional training copies. The held-out test folds are not used during model fitting or hyperparameter selection.

## 4. Cross-Validation and Hyperparameter Tuning

The experiments use:

- 10 outer patient-grouped folds for evaluation
- 3 inner folds per outer fold for hyperparameter selection
- Macro-F1 as the primary selection and evaluation metric
- Fold-specific preprocessing to avoid using held-out statistics during training

### 4.1 Initial Hyperparameter Search

The original HGB experiment used:

- Initial `max_depth` grid: `[2, 3, 4]`
- Additional boundary refinement: depth `5`

The earlier tuning results frequently selected the upper boundary of the search range, motivating an expanded search.

The original experiments and previously reported results are retained as historical records.

### 4.2 Retuned Hyperparameter Search

The updated experiments use:

`max_depth = [7, 8, 9]`

For each outer fold, the best maximum depth is selected based on the mean Macro-F1 across the three corresponding inner-validation folds.

The selected depth is then used to fit the model on the outer-training data and evaluate it on the held-out outer fold.

The updated experiments have been completed for all four arms (A, B, C, and D).

The retuned results are available at:

[Retuned HGB Results](../../results/histogram_gradient_boosting/retuned_7_8_9/)

## 5. Evaluation Scenarios

Each model is evaluated under 11 scenarios:

| Scenario | Description |
|---|---|
| `observed` | Original observed ECG data |
| `missing_1` | One missing ECG lead |
| `missing_3` | Three missing ECG leads |
| `missing_6` | Six missing ECG leads |
| `noise_0.05` | Noise level 0.05 |
| `noise_0.10` | Noise level 0.10 |
| `noise_0.20` | Noise level 0.20 |
| `gain_0.05` | Gain variation 0.05 |
| `gain_0.10` | Gain variation 0.10 |
| `gain_0.20` | Gain variation 0.20 |
| `combined_3lead_noise0.10` | Three missing leads with noise level 0.10 |

The observed scenario is evaluated once per outer fold.

Each corruption scenario is evaluated with three repeats.

This produces:

- 31 evaluation records per outer fold
- 310 evaluation records per experimental arm
- 1,240 evaluation records across all four arms

## 6. Final Retuned Results

### 6.1 Observed ECG Performance

The following results are reported as the mean and sample standard deviation of Macro-F1 across the 10 outer folds.

| Arm | Mean Macro-F1 | Standard Deviation |
|---|---:|---:|
| A | 0.924192 | 0.013804 |
| B | **0.924355** | 0.012939 |
| C | 0.918543 | 0.013281 |
| D | 0.917408 | 0.013316 |

Arm B achieves the highest mean Macro-F1 under observed conditions, although its improvement over Arm A is very small.

The augmented models (Arms C and D) have slightly lower observed-data performance than their corresponding non-augmented models.

### 6.2 Performance Under Six Missing Leads

| Arm | Observed Macro-F1 | Missing-6 Macro-F1 | Performance Drop |
|---|---:|---:|---:|
| A | 0.924192 | 0.587080 | 0.337112 |
| B | 0.924355 | 0.730876 | 0.193479 |
| C | 0.918543 | **0.895531** | **0.023012** |
| D | 0.917408 | 0.882679 | 0.034729 |

The augmented models (Arms C and D) retain substantially higher Macro-F1 scores when six ECG leads are missing.

In contrast, Arms A and B show larger performance reductions.

This suggests that training augmentation improves robustness to severe missing-lead corruption, with a small trade-off in observed-data performance.

These values describe the experimental results and do not independently establish statistical significance.

## 7. Results Aggregation

For each corrupted scenario, the three repeated evaluations are first averaged within each outer fold.

The final scenario mean and sample standard deviation are then calculated across the 10 outer folds.

Performance drops are computed relative to the observed Macro-F1 from the same outer fold.

Paired method comparisons use matching outer folds and evaluation scenarios.

## 8. Retuned Result Files

The completed main-experiment results are stored under:

`results/histogram_gradient_boosting/retuned_7_8_9/`

| File | Description |
|---|---|
| `A_selected_depths.csv` | Selected depths for Arm A |
| `B_selected_depths.csv` | Selected depths for Arm B |
| `C_selected_depths.csv` | Selected depths for Arm C |
| `D_selected_depths.csv` | Selected depths for Arm D |
| `HGB_ABCD_all_metrics.csv` | All 1,240 evaluation records |
| `HGB_ABCD_fold_scenario_means.csv` | Mean metrics per outer fold and scenario |
| `HGB_ABCD_scenario_summary.csv` | Across-fold means and standard deviations |
| `HGB_ABCD_stress_drops.csv` | Performance changes relative to observed conditions |
| `HGB_ABCD_paired_differences.csv` | Descriptive paired differences between experimental arms |

The individual outer-fold outputs and predictions were generated and checked before aggregation.

The consolidated metrics listed above are the primary published artifacts for the retuned main experiments.

## 9. Earlier Reported Results

The earlier HGB experiment reported the following observed-data results:

| Arm | Earlier Mean Macro-F1 | Earlier SD |
|---|---:|---:|
| A | 0.9213 | 0.0158 |
| B | 0.9230 | 0.0143 |
| C | 0.9156 | 0.0122 |
| D | 0.9179 | 0.0126 |

These are historical reported values from the earlier maximum-depth search configuration.

The original checkout did not contain a complete raw result bundle sufficient to independently verify all earlier scores.

The earlier HGB report is retained here:

[Original HGB Report](../../results/histogram_gradient_boosting/source_provenance/reported_summary.docx)

The earlier reported results should not be mixed with the retuned `[7, 8, 9]` results.

## 10. Execution and Reproducibility

### Original Repository Workflow

Install the HGB requirements from the project root:

```bash
python -m pip install -r models/histogram_gradient_boosting/requirements.txt
```

Run the original HGB experiment script:

```bash
python models/histogram_gradient_boosting/run_experiment.py --arms A B C D --outer-folds 0 1 2 3 4 5 6 7 8 9
```

The original script expects the shared `processed_data` input package.

**Important:** This is the original repository execution workflow. The existing runner must be checked or updated before claiming that this command reproduces the retuned `[7, 8, 9]` experiments.

### Retuned Experiment Reproducibility

The retuned results use:

- The shared processed ECG feature package
- The same patient-grouped outer-fold assignments
- Three inner folds for selecting the best depth
- `max_depth` grid `[7, 8, 9]`
- The same 11 evaluation scenarios
- Three corruption repeats per scenario
- scikit-learn version 1.3.0

The updated experiments were completed across multiple computing environments and consolidated into a single result set.

The following consistency checks were applied to the retuned main experiments:

- All 10 outer folds present for each arm
- Correct selected maximum depths
- Exactly 31 scenario evaluations per outer fold
- No duplicate arm/fold/scenario/repeat metric records
- Complete consolidated metrics for all four arms

The original `check_results.py` applies to the existing repository workflow and has not been confirmed to operate directly on the separate retuned result directory.

The HGB results notebook is available at:

[HGB Results Notebook](../../notebooks/06_histogram_gradient_boosting_results.ipynb)

## 11. Notes

The retuned `[7, 8, 9]` experiments represent the updated HGB main-experiment results.

The earlier maximum-depth search and its reported results remain documented for transparency and traceability.

The experiment results should be interpreted with attention to the trade-off between observed-data performance and robustness under corrupted ECG conditions.

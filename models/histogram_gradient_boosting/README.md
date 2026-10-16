
# Histogram Gradient Boosting (HGB) for AFIB Classification

## 1. Purpose

This project investigates nonlinear relationships and feature interactions for binary atrial fibrillation (AFIB) classification using Histogram Gradient Boosting (HGB).

We evaluate classification performance under observed ECG conditions and robustness to missing leads, noise, gain changes, and combined corruption.

The updated main experiments use a retuned maximum-depth search grid of `[7, 8, 9]`.

## 2. Model Implementation

**Estimator:** `HistGradientBoostingClassifier`

The HGB implementation uses the following fixed hyperparameters:

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

**Classification threshold:** 0.50

**Prediction identifier:** `hist_gradient_boosting`

The same model implementation is used across the four experimental arms, with maximum depth selected separately for each outer fold.

## 3. Experimental Arms

Four experimental arms are evaluated:

| Arm | Feature Set | Training Augmentation |
|---|---|---|
| A | Original ECG features | No |
| B | Original and engineered ECG features | No |
| C | Original ECG features | Yes |
| D | Original and engineered ECG features | Yes |

The original feature representation contains 504 lead-related features.

The enhanced representation adds 63 engineered features.

For Arms C and D, one corrupted training copy is added per original training recording, with the total training weight preserved across the two views.

All arms use the same patient-grouped outer-fold assignments and fold-specific preprocessing.

The held-out outer folds are never used for fitting the model or selecting hyperparameters.

## 4. Cross-Validation and Hyperparameter Tuning

The experimental design uses:

- 10 outer patient-grouped folds for evaluation
- 3 inner patient-grouped folds for hyperparameter selection
- Macro-F1 as the primary selection and evaluation metric
- Training-fold-specific preprocessing
- Fold-specific hyperparameter selection

### 4.1 Earlier HGB Experiments

The initial HGB search used:

- Initial `max_depth` grid: `[2, 3, 4]`
- Additional boundary refinement that could extend the upper depth to `5`

In the earlier experiment, depth 5 was reported as the selected value in all 40 main arm-fold runs.

This boundary-selection pattern motivated further investigation of deeper trees.

The earlier experiments and their reported results are retained as historical records.

### 4.2 Retuned HGB Experiments

The completed retuned experiments use the fixed candidate grid:

`max_depth = [7, 8, 9]`

For each outer fold:

1. Each candidate depth is evaluated on the three inner-validation folds.
2. Mean inner-validation Macro-F1 is calculated for each candidate.
3. The candidate with the highest mean Macro-F1 is selected.
4. Exact ties favour the smaller maximum depth.
5. The selected model is fitted on the full outer-training partition.
6. The fitted model is evaluated on the held-out outer fold.

Across the 40 main arm-fold runs, the selected-depth frequencies were:

| Maximum Depth | Selection Count |
|---|---:|
| 7 | 10 |
| 8 | 15 |
| 9 | 15 |
| **Total** | **40** |

The middle candidate, depth 8, was tied with depth 9 for the highest selection frequency.

All four arms completed 10 outer-fold evaluations using the retuned configuration.

The completed results are stored at:

[Retuned HGB Results](../../results/histogram_gradient_boosting/retuned7_8_9/)

## 5. Evaluation Scenarios

Each outer-fold model is evaluated under 11 scenarios.

| Scenario | Description |
|---|---|
| `observed` | Observed ECG data |
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

Therefore:

- 31 evaluation records are generated per outer fold.
- 310 evaluation records are generated per arm.
- 1,240 evaluation records are generated across the four arms.

All corruption scenarios are evaluated on the held-out outer-fold data after the model has been fitted.

## 6. Final Retuned Results

### 6.1 Observed ECG Performance

The following values are the mean and sample standard deviation of Macro-F1 across 10 outer folds.

| Arm | Mean Macro-F1 | Standard Deviation |
|---|---:|---:|
| A | 0.924192 | 0.013804 |
| B | **0.924355** | 0.012939 |
| C | 0.918543 | 0.013281 |
| D | 0.917408 | 0.013316 |

Arm B achieved the highest mean Macro-F1 under observed ECG conditions.

However, the difference between Arms A and B was very small.

Arms C and D had slightly lower observed-data performance than their corresponding non-augmented arms.

### 6.2 Performance Under Six Missing Leads

| Arm | Observed Macro-F1 | Missing-6 Macro-F1 | Performance Drop |
|---|---:|---:|---:|
| A | 0.924192 | 0.587080 | 0.337112 |
| B | 0.924355 | 0.730876 | 0.193479 |
| C | 0.918543 | **0.895531** | **0.023012** |
| D | 0.917408 | 0.882679 | 0.034729 |

Arms C and D retained considerably higher Macro-F1 when six leads were missing.

Arm C achieved the highest mean Macro-F1 under this severe missing-lead condition.

The results suggest that training augmentation substantially improves robustness to severe missing-lead corruption, while introducing a small trade-off in observed-data performance.

These are descriptive findings based on the evaluated outer folds. The summary means alone do not establish statistical significance.

### 6.3 General Robustness Findings

The experiments also considered lower levels of lead missingness, additive noise, gain variation, and combined corruption.

All 11 scenario results are provided in:

`HGB_ABCD_scenario_summary.csv`

The corresponding performance changes relative to observed conditions are provided in:

`HGB_ABCD_stress_drops.csv`

Different corruption scenarios are reported separately rather than combined into a single overall robustness score.

## 7. Results Aggregation

The same aggregation procedure is applied to all four arms.

First, the three repeats of each corruption scenario are averaged within the same outer fold.

Second, the resulting fold-level scores are summarized across the 10 outer folds.

The final results report:

- Mean Macro-F1
- Sample standard deviation across outer folds
- Scenario-specific performance changes
- Paired differences between arms using matching outer folds

The observed scenario contains one evaluation per outer fold and does not require repeat averaging.

Standard deviations describe variation across the evaluated folds. They are not confidence intervals.

## 8. Retuned Result Files

The main results are stored under:

`results/histogram_gradient_boosting/retuned7_8_9/`

### 8.1 Main Evaluation Results

| File | Description |
|---|---|
| `A_selected_depths.csv` | Selected maximum depth for each Arm A outer fold |
| `B_selected_depths.csv` | Selected maximum depth for each Arm B outer fold |
| `C_selected_depths.csv` | Selected maximum depth for each Arm C outer fold |
| `D_selected_depths.csv` | Selected maximum depth for each Arm D outer fold |
| `HGB_ABCD_all_metrics.csv` | All 1,240 evaluation records |
| `HGB_ABCD_fold_scenario_means.csv` | Fold-level scenario means |
| `HGB_ABCD_scenario_summary.csv` | Scenario means and sample standard deviations |
| `HGB_ABCD_stress_drops.csv` | Performance changes relative to observed data |
| `HGB_ABCD_paired_differences.csv` | Descriptive paired differences between arms |

### 8.2 Complete Inner-Cross-Validation Records

The retuned experiments produced 360 inner-validation tuning records:

`4 arms × 10 outer folds × 3 candidate depths × 3 inner folds = 360 records`

The consolidated tuning file is:

`HGB_ABCD_all_inner_tuning.csv`

The complete table was reconstructed from retained fold-level tuning files and previously saved inner-CV records.

All 40 selected maximum depths were checked against the mean inner-validation Macro-F1 values.

This file provides the complete candidate-score evidence supporting the selected depths.

### 8.3 Per-Sample Predictions

The complete per-sample prediction records for all four HGB arms are provided in the retuned results directory:

- `HGB_retuned_A_predictions.zip`
- `HGB_retuned_B_predictions.zip`
- `HGB_retuned_C_predictions.zip`
- `HGB_retuned_D_predictions.zip`

Each ZIP archive contains the corresponding arm's `all_predictions.csv`.

Each prediction record includes:

- Experimental arm
- Outer fold
- Evaluation scenario
- Corruption repeat
- ECG identifier
- True classification label
- Predicted AFIB probability

Each arm contains 675,769 prediction records, covering all 10 outer folds and 31 evaluation settings per fold.

These prediction archives correspond to the retuned maximum-depth search grid `[7, 8, 9]`.

The earlier HGB aggregate tables are kept at the HGB results root. The four
ZIP files above contain predictions only; they are experiment outputs, not
source ECG measurements.

## 9. Earlier Reported Results

The earlier HGB experiments reported:

| Arm | Earlier Mean Macro-F1 | Earlier SD |
|---|---:|---:|
| A | 0.9213 | 0.0158 |
| B | 0.9230 | 0.0143 |
| C | 0.9156 | 0.0122 |
| D | 0.9179 | 0.0126 |

These values are retained as historical reported results from the earlier hyperparameter-search configuration.

The original repository checkout did not contain a complete raw result bundle sufficient to independently recalculate all earlier metrics.

The historical report remains available at:

[Original HGB Report](../../results/histogram_gradient_boosting/source_provenance/reported_summary.docx)

The earlier reported values are not mixed with the retuned `[7, 8, 9]` evaluation results.

## 10. Execution and Reproducibility

### 10.1 Original Repository Workflow

The original experiment runner is retained:

`models/histogram_gradient_boosting/run_experiment.py`

The original runner uses the earlier depth-search procedure and should not be assumed to reproduce the retuned experiments.

### 10.2 Retuned Experiment Runner

A separate retuned experiment runner is available:

`models/histogram_gradient_boosting/run_experiment_retuned.py`

It uses the fixed maximum-depth grid `[7, 8, 9]`.

Prepare the external input package first using the preprocessing notebook.
The runner reads `AFIB_PROCESSED_DIR` or `../afib_data/processed`; use
`--processed <directory>` to choose a package explicitly.

Use Python 3.11 for this repository runner and its pinned retuned dependencies.
Install them from the project root:

```bash
python -m pip install -r models/histogram_gradient_boosting/requirements_retuned.txt
```

Run the retuned experiment:

```bash
python models/histogram_gradient_boosting/run_experiment_retuned.py --arms A B C D --outer-folds 0 1 2 3 4 5 6 7 8 9
```

The runner saves outputs in a separate location:

`results/histogram_gradient_boosting/retuned_7_8_9_runs/`

This command trains the requested models and performs inner cross-validation again. It can require substantial computing time.

The completed experimental summaries are already published separately under:

`results/histogram_gradient_boosting/retuned7_8_9/`

The new runner was added to document the retuned execution procedure. An independent full end-to-end reproduction of all 40 arm-fold runs through this repository script has not yet been performed.

### 10.3 Retuned Experiment Environment

The completed retuned HGB experiments used the following principal package versions:

| Software | Version |
|---|---|
| scikit-learn | 1.3.0 |
| NumPy | 1.24.3 |
| Pandas | 2.0.3 |
| SciPy | 1.11.4 |
| Python | 3.10 or 3.11, depending on the computing environment |

The package versions for the retuned experiments are documented in:

`models/histogram_gradient_boosting/requirements_retuned.txt`

The original requirements file is retained for the historical workflow.

The retuned experiments were completed across multiple computing environments, and their results were subsequently consolidated into a single set of evaluation files.

The documented package versions are the primary dependencies used for the completed runs, rather than a full historical environment snapshot.

Exact numerical reproduction across different hardware, operating systems, and Python versions has not been independently established.

### 10.4 Result Validation

The following consistency checks were completed:

- All four experimental arms were present.
- All 10 outer folds were present for each arm.
- Each outer fold contained 31 evaluation records.
- All selected depths belonged to the grid `[7, 8, 9]`.
- No duplicated arm/fold/scenario/repeat metric records were found.
- All 1,240 main evaluation records were consolidated.
- All 360 inner-validation tuning records were recovered and validated.
- The selected depth for every arm-fold combination matched its inner-CV results.

Run the independent checker without data or retraining:

```powershell
python models/histogram_gradient_boosting/check_results.py
```

It recomputes every saved outer metric from the four prediction archives and
checks all 360 tuning scores and 40 selected depths. Pass `--output <folder>`
to check a fresh runner output containing per-fold files.

The original HGB results notebook is available at:

[HGB Results Notebook](../../notebooks/06_histogram_gradient_boosting_results.ipynb)

The notebook compares original HGB and retuned HGB results side by side.

## 11. Limitations and Future Improvements

### Hyperparameter Search

The retuned experiments evaluated only maximum depths `[7, 8, 9]`.

Other parameters, including learning rate, maximum leaf nodes, and number of boosting iterations, were held fixed.

Further tuning could investigate a broader model-capacity and regularization search under a clearly defined computational budget.

### Augmentation Trade-off

Training augmentation improved robustness to severe missing-lead corruption but slightly reduced observed-data Macro-F1 in the evaluated experiments.

Future work could explore augmentation mixtures and corruption strengths selected exclusively using inner-validation data.

### Statistical Inference

The reported results are descriptive summaries of the evaluated folds.

Although paired differences are available, formal statistical significance claims require additional analysis and consideration of dependence between cross-validation folds.

### Generalization

Patient-grouped cross-validation reduces leakage between records from the same patient.

However, it does not establish external clinical validity or robustness to every possible ECG measurement failure.

## 12. Conclusion

The retuned HGB experiments demonstrate a trade-off between observed-data classification performance and robustness under missing-lead corruption.

Arm B achieved the highest observed-data mean Macro-F1 of 0.924355.

Arm C achieved the highest mean Macro-F1 under six missing leads, reaching 0.895531.

The augmented models maintained considerably stronger performance under severe lead missingness, whereas the non-augmented models achieved slightly better observed-data results.

The updated repository records the selected maximum depths, complete inner-validation tuning scores, consolidated metrics, and the documented experimental configuration.

The historical HGB results are retained separately for transparency and traceability.

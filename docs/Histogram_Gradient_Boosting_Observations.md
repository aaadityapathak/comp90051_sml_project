
# Histogram Gradient Boosting Observations

## 1. Overview

This report presents the updated Histogram Gradient Boosting (HGB) results for binary atrial fibrillation (AFIB) classification.

The original HGB experiments used an initial maximum-depth grid of `[2, 3, 4]`, with a boundary refinement that could extend the search to depth 5.

Following the initial experiments, an additional hyperparameter search was conducted using `[7, 8, 9]`. The results presented below correspond to this retuned configuration.

The earlier results are retained as historical records and should not be confused with the updated results.

## 2. Dataset and Experimental Design

The dataset contains 21,799 ECG recordings from 18,869 patients, with approximately 6.95% AFIB-labelled recordings.

The experiments use:

- 504 original ECG lead features
- 63 engineered features
- 10 patient-grouped outer folds
- 3 inner folds for hyperparameter selection
- Macro-F1 as the primary evaluation metric
- Classification threshold of 0.50
- Training-fold-specific preprocessing

Four experimental arms were evaluated:

| Arm | Feature Set | Training Augmentation |
|---|---|---|
| A | Original ECG features | No |
| B | Original and engineered ECG features | No |
| C | Original ECG features | Yes |
| D | Original and engineered ECG features | Yes |

For Arms C and D, one corrupted training copy was added per original training recording, with the combined sample weight preserved.

The same outer patient-grouped folds were used across all arms.

## 3. HGB Model and Hyperparameter Tuning

The classifier is `HistGradientBoostingClassifier`.

The fixed model hyperparameters are:

| Parameter | Value |
|---|---|
| `max_iter` | 150 |
| `learning_rate` | 0.05 |
| `max_leaf_nodes` | 16 |
| `min_samples_leaf` | 30 |
| `l2_regularization` | 1.0 |
| `early_stopping` | False |
| `random_state` | 42 |

### 3.1 Initial Experiments

The initial maximum-depth search grid was `[2, 3, 4]`.

An additional boundary refinement was permitted. The earlier report described upper-boundary refinement to depth 5.

The earlier experiments motivated further investigation of deeper trees.

### 3.2 Retuned Experiments

The updated experiments use:

`max_depth = [7, 8, 9]`

For each outer fold, the best depth was selected using the mean Macro-F1 across the three inner-validation folds.

The selected depth was then used to train the HGB classifier on the outer-training data and evaluate it on the held-out outer fold.

All four arms completed 10 outer folds.

## 4. Final Observed-Data Results

The following results represent the mean Macro-F1 and sample standard deviation across the 10 outer folds using the retuned depth grid.

| Arm | Mean Macro-F1 | Standard Deviation |
|---|---:|---:|
| A | 0.924192 | 0.013804 |
| B | **0.924355** | 0.012939 |
| C | 0.918543 | 0.013281 |
| D | 0.917408 | 0.013316 |

Arm B achieved the highest mean observed-data Macro-F1, although the difference between Arms A and B was small.

Compared with their corresponding non-augmented arms, Arms C and D had slightly lower observed-data Macro-F1.

These results indicate that the augmentation strategy did not improve observed-data performance in the main experiments.

## 5. Missing-Lead Robustness

The models were evaluated when 1, 3, or 6 ECG leads were missing.

The following table presents the verified retuned results for the observed and six-missing-lead scenarios.

| Arm | Observed Macro-F1 | Missing 6 Macro-F1 | Performance Drop |
|---|---:|---:|---:|
| A | 0.924192 | 0.587080 | 0.337112 |
| B | 0.924355 | 0.730876 | 0.193479 |
| C | 0.918543 | **0.895531** | **0.023012** |
| D | 0.917408 | 0.882679 | 0.034729 |

### Key Observations

**First, augmentation substantially improved robustness to severe lead missingness.**

Arms C and D maintained considerably higher Macro-F1 scores than Arms A and B when six leads were missing.

**Second, Arm C achieved the highest performance in the six-missing-lead condition.**

Arm C obtained a mean Macro-F1 of 0.895531, compared with 0.882679 for Arm D.

**Third, there was a trade-off between observed-data performance and missing-lead robustness.**

The non-augmented arms had slightly better observed-data performance, while the augmented arms exhibited much smaller performance reductions under severe missingness.

These findings are descriptive comparisons across the shared outer folds. Statistical significance is not established by the mean scores alone.

## 6. Noise, Gain, and Combined Corruption

In addition to missing leads, the models were evaluated under the following test scenarios:

- Noise levels: 0.05, 0.10, and 0.20
- Gain changes: 0.05, 0.10, and 0.20
- Combined three-lead missingness and noise level 0.10

Each corruption scenario was repeated three times per outer fold.

Across the four experimental arms, a total of 11 evaluation scenarios were considered, including observed data.

Each arm produced 31 evaluation records per outer fold and 310 records across 10 folds.

The complete noise, gain, and combined-corruption results are available in the retuned scenario-summary CSV rather than being selectively reproduced in this report.

## 7. Evaluation and Statistical Aggregation

The experiments used a consistent aggregation procedure:

1. For each outer fold and corrupted scenario, the three repeated evaluations were averaged.
2. The resulting fold-level Macro-F1 scores were summarized across 10 outer folds.
3. The mean and sample standard deviation were reported.
4. Stress-performance drops were calculated by comparing each corrupted scenario with the observed condition within the same outer fold.
5. Paired method differences were calculated using matching outer folds and scenarios.

The reported standard deviations describe variation between outer folds and should not be interpreted as confidence intervals.

## 8. Comparison With Earlier HGB Results

The earlier HGB report presented the following observed-data Macro-F1 scores:

| Arm | Earlier Mean Macro-F1 | Earlier SD | Retuned Mean Macro-F1 |
|---|---:|---:|---:|
| A | 0.9213 | 0.0158 | 0.924192 |
| B | 0.9230 | 0.0143 | 0.924355 |
| C | 0.9156 | 0.0122 | 0.918543 |
| D | 0.9179 | 0.0126 | 0.917408 |

The previous report consisted of transcribed results without a complete retained source bundle for independent verification.

The new retuned experiment has complete consolidated metrics and recorded outer-fold depth selections.

Because the two results originate from different hyperparameter search configurations, they are documented separately.

The updated experiments should not be interpreted as proving that the new depth search is statistically superior to the earlier configuration.

## 9. Reproducibility and Result Files

The updated results are stored at:

`results/histogram_gradient_boosting/retuned_7_8_9/`

The directory contains:

- `A_selected_depths.csv`
- `B_selected_depths.csv`
- `C_selected_depths.csv`
- `D_selected_depths.csv`
- `HGB_ABCD_all_metrics.csv`
- `HGB_ABCD_fold_scenario_means.csv`
- `HGB_ABCD_scenario_summary.csv`
- `HGB_ABCD_stress_drops.csv`
- `HGB_ABCD_paired_differences.csv`

The consolidated metric dataset contains 1,240 evaluation records across Arms A, B, C, and D.

All four arms were checked for:

- Complete outer-fold coverage (10 folds per arm)
- 31 evaluation records per outer fold
- Consistent arm and fold identifiers
- No duplicated arm/fold/scenario/repeat metric records

The HGB implementation is available at:

`models/histogram_gradient_boosting/model.py`

The original experiment runner is:

`models/histogram_gradient_boosting/run_experiment.py`

A separate runner for the retuned search grid is:

`models/histogram_gradient_boosting/run_experiment_retuned.py`

The separate retuned runner has been added to the repository, but full end-to-end reproduction through that runner has not yet been independently verified.

## 10. Limitations and Future Improvements

Several limitations remain.

**Hyperparameter search:** Only a limited maximum-depth grid was investigated in the retuned experiments. Other hyperparameters, including learning rate, tree complexity, and number of iterations, may also affect performance.

**Augmentation trade-off:** While training augmentation improved robustness to missing leads, it slightly reduced observed-data performance. Future work could investigate alternative corruption probabilities and augmentation strengths.

**Statistical inference:** Descriptive means and paired differences are provided. Formal uncertainty analysis would be needed before making stronger statistical claims.

**Additional evaluation:** Further investigation could include patient-level subgroup analysis, calibration, and additional robustness scenarios.

## 11. Conclusion

The retuned HGB experiments demonstrate that the choice of training strategy affects the balance between classification accuracy under observed conditions and robustness under missing-lead corruption.

Arm B achieved the highest observed-data mean Macro-F1 of 0.924355.

Arm C achieved the highest mean Macro-F1 under six missing leads, reaching 0.895531.

Overall, augmentation showed its greatest benefit under severe missing-lead conditions, while the non-augmented arms retained slightly higher observed-data performance.

The updated results provide a reproducible record of the selected hyperparameters and consolidated evaluation metrics for subsequent comparison with other classification methods.

## References and Provenance

- Historical reported HGB summary: `results/histogram_gradient_boosting/source_provenance/reported_summary.docx`
- Retuned experiment results: `results/histogram_gradient_boosting/retuned_7_8_9/`
- HGB model and runners: `models/histogram_gradient_boosting/`

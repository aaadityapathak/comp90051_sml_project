# Logistic regression observations

Best primary observed result: D, macro F1 0.9032 (0.0147). It has the highest observed mean across all saved variants. B performs better under severe lead loss; there is no single best arm for every condition.

## Experiment and observed results

The cohort contains 21,799 ECGs from 18,869 patients; 6.95% are AFIB-labelled. This class imbalance makes macro F1 and class-specific errors more informative than accuracy alone.

A: original measurements; B: original plus engineered fields; C: A with augmentation; D: B with augmentation. There are 504 original lead fields and 63 engineered fields. Augmentation adds one corrupted training view per ECG with total weight preserved.

Primary results use threshold 0.50, 10 outer patient folds and 3 inner folds. Filling, scaling and selection use training patients only. Three stress repeats are averaged within each fold, then summarized by mean and sample SD. SD is not a confidence interval.

| Arm | Macro F1 (SD) | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- |
| A | 0.8950 (0.0171) | 0.9570 | 0.9691 | 0.0258 |
| B | 0.9003 (0.0151) | 0.9477 | 0.9721 | 0.0237 |
| C | 0.8984 (0.0181) | 0.9541 | 0.9707 | 0.0248 |
| D | 0.9032 (0.0147) | 0.9530 | 0.9727 | 0.0234 |

Higher macro F1, sensitivity and specificity are better; lower Brier means lower probability error.

B minus A F1 is +0.0053; D minus B is +0.0028. Engineered fields supply relationships a linear decision function cannot construct itself. The results support their usefulness, without identifying which feature family produced the gain.

## Tuning and optimization

Weighted L2 logistic regression used L-BFGS, up to 2,000 iterations, and initial C values 0.01, 0.1 and 1, with one boundary refinement. Larger C weakens regularization. Selection used mean inner observed macro F1 at 0.50.

| Variant | Setting and fold count | Selected inner F1 |
| --- | --- | --- |
| A | 0.01: 1; 0.1: 9 | 0.8939 |
| B | 0.1: 10 | 0.9003 |
| C | 0.1: 7; 1: 3 | 0.8967 |
| D | 0.1: 9; 1: 1 | 0.9015 |
| availability observed | 0.01: 1; 0.1: 9 | 0.8939 |
| availability augmented | 0.1: 7; 1: 3 | 0.8967 |
| duplicate original | 0.01: 1; 0.1: 9 | 0.8940 |
| duplicate enhanced | 0.1: 10 | 0.9004 |
| ridge | 0.01: 1; 0.1: 9 | 0.8943 |

11/90 selections triggered boundary refinement; 90/90 selected the middle of the final grid. Inner F1 averages each fold's selected candidate score; it is selection evidence, not an unbiased generalization estimate. Refined candidates have unequal fold coverage; their aggregate scores cannot establish one globally best setting.

All 933 saved fits converged; maximum absolute gradient was 1e-05. This confirms numerical completion of the specified search, not optimality across other model or feature choices.

## Robustness and additional outputs

| Condition | A | B | C | D |
| --- | --- | --- | --- | --- |
| Observed | 0.8950 | 0.9003 | 0.8984 | 0.9032 |
| 1 missing lead | 0.8999 | 0.8991 | 0.8937 | 0.8976 |
| 3 missing leads | 0.8915 | 0.8919 | 0.8807 | 0.8870 |
| 6 missing leads | 0.7346 | 0.8605 | 0.8339 | 0.8501 |
| Noise 0.05 | 0.8953 | 0.9010 | 0.8991 | 0.9031 |
| Noise 0.10 | 0.8946 | 0.9010 | 0.8973 | 0.9032 |
| Noise 0.20 | 0.8905 | 0.9014 | 0.8915 | 0.9021 |
| Gain 0.05 | 0.8951 | 0.9002 | 0.8984 | 0.9029 |
| Gain 0.10 | 0.8947 | 0.9005 | 0.8975 | 0.9026 |
| Gain 0.20 | 0.8934 | 0.8987 | 0.8973 | 0.9009 |
| 3 leads + noise 0.10 | 0.8906 | 0.8908 | 0.8799 | 0.8878 |

Noise SD is the listed dose times the training-field IQR; gain is exp(dose times a standard-normal draw). Perturbations act on measurement tables, not raw ECG waveforms. Small F1 rises under mild corruption can reflect shifted threshold errors; they do not establish better measurement quality.

With six leads missing, B F1 is 0.8605 versus D 0.8501. D retains sensitivity 0.9206, but specificity falls to 0.9541. Augmentation trades fewer missed positives for more false positives, reducing F1 in this condition.

B exceeds its availability control by 0.0053 observed F1. That control retains original predictors and engineered missingness indicators but zeroes engineered values. Half-weight duplicates reproduce observed F1, consistent with an unchanged logistic objective.

Ridge filling reaches observed F1 0.8947 and three-lead-loss F1 0.8974 (A: 0.8950 and 0.8915). Only these two downstream conditions were tested. Reconstruction error improves over median filling for cell masking and 1, 3 and 6 missing leads; better reconstruction alone does not establish a better classifier.

IQR-normalized reconstruction MAE, median to ridge: 10% cell masking 0.7349 to 0.4311; six missing leads 0.7351 to 0.6969. Only eligible masked continuous cells are scored.

The inner-OOF sensitivity-target rule gives D observed F1 0.9185, sensitivity 0.9153 and specificity 0.9820. The 0.90 sensitivity target was met in 7/10 observed outer folds. This secondary operating point does not replace the primary ranking.

D patient-weighted F1 is 0.9028. F1 is 0.8996 with recorded quality flags and 0.9038 without flags. Flag absence does not establish clean signals; subgroup composition may differ.

## Next improvements and rationale

1. Tune augmentation dose and mixture inside training folds to reduce specificity loss under lead removal. 2. Test feature-family ablations and elastic-net regularization to identify useful predictors and control redundancy. 3. Fit calibration and thresholds inside training folds to compare sensitivity at a fixed false-positive budget. 4. Repeat patient splits and test external data to quantify uncertainty and transfer beyond synthetic corruption.

The one-fold pilot is a workflow check, not additional ten-fold evidence. External validation and statistical superiority are not established.

Evidence: results/logistic_regression/output_data/summary.csv, selected_hyperparameters.csv, paired_method_differences.csv, patient_weighted_metrics_summary.csv, quality_subgroups_summary.csv; run_config.json and reports/completion.json. Rebuild: python scripts/build_model_observations.py.

Ridge reconstruction evidence: output_data/recovery_summary.csv in the same result folder.

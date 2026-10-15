# Histogram gradient boosting observations

Best reported observed arm: B, F1 0.9230 (SD 0.0143). Best reported six-lead-loss arm: C, F1 0.8892. The retained report contains transcribed results; underlying predictions, fold metrics and tuning files are absent, so these figures cannot currently be independently recalculated.

## Experiment and reported results

The cohort contains 21,799 ECGs from 18,869 patients; 6.95% are AFIB-labelled. This class imbalance makes macro F1 and class-specific errors more informative than accuracy alone.

A: original measurements; B: original plus engineered fields; C: A with augmentation; D: B with augmentation. There are 504 original lead fields and 63 engineered fields. Augmentation adds one corrupted training view per ECG with total weight preserved.

Primary results use threshold 0.50, 10 outer patient folds and 3 inner folds. Filling, scaling and selection use training patients only. Three stress repeats are averaged within each fold, then summarized by mean and sample SD. SD is not a confidence interval.

| Arm | Macro F1 (SD) | Sensitivity | Specificity | AUROC |
| --- | --- | --- | --- | --- |
| A | 0.9213 (0.0158) | 0.9514 | 0.9795 | 0.9930 |
| B | 0.9230 (0.0143) | 0.9496 | 0.9803 | 0.9935 |
| C | 0.9156 (0.0122) | 0.9549 | 0.9771 | 0.9931 |
| D | 0.9179 (0.0126) | 0.9583 | 0.9776 | 0.9934 |

B improves observed F1 over A by a reported paired mean 0.0018 (SD 0.0047), with higher specificity and slightly lower sensitivity. Its best F1 reflects this error balance. Trees can capture nonlinear feature interactions; the report does not isolate which interactions or engineered fields produce the gain.

## Tuning and evidence limits

Initial maximum depths were 2, 3 and 4 with one boundary refinement. The current runner permits depth 1 or 5 depending on the boundary winner; the historical report describes upper refinement to 5. Mean inner observed F1 selects depth; ties prefer shallower trees.

Fixed settings: 150 iterations, learning rate 0.05, maximum 16 leaves, minimum 30 samples per leaf, L2 penalty 1, no early stopping and seed 42. These constrain complexity; they do not establish an optimal tuning budget.

The report states 40 completed arm-fold runs and 31 condition/repeat evaluations per run. Candidate scores, selected-depth counts, run versions and source hashes are unavailable. No specific depth can be called the winner from the retained evidence.

## Missing leads and augmentation

| Arm | Observed | 1 missing | 3 missing | 6 missing |
| --- | --- | --- | --- | --- |
| A | 0.9213 | 0.9226 | 0.8524 | 0.5912 |
| B | 0.9230 | 0.9237 | 0.8935 | 0.7356 |
| C | 0.9156 | 0.9119 | 0.9069 | 0.8892 |
| D | 0.9179 | 0.9149 | 0.9035 | 0.8771 |

Augmentation lowers reported observed F1: C minus A is -0.0057; D minus B is -0.0051. With six leads removed, C and D strongly outperform A and B. C exceeds D under three and six missing leads, so enhancement adds no robustness gain in these augmented comparisons.

Under combined three-lead loss and noise 0.10, reported F1 is A 0.8493, B 0.8876, C 0.9062 and D 0.9019. The source describes smaller effects for isolated noise and gain but omits their complete numerical tables. Brier, average precision, availability/duplication controls and subgroup results are not available in the report.

## Next improvements and rationale

1. Retain predictions, selections, data/source hashes and package versions on a future authorized run so scores can be independently checked. 2. Tune augmentation mixtures in inner folds to retain robustness with less observed F1 loss. 3. Study depth, leaves, learning rate and iterations within a declared budget to balance capacity and overfitting. 4. Add the same controls, probability metrics and patient-level comparisons as the other algorithms to complete the comparison.

Source: results/histogram_gradient_boosting/source_provenance/reported_summary.docx. Future output: results/histogram_gradient_boosting/formal/. Runner and checker: models/histogram_gradient_boosting/. This review does not regenerate HGB results.

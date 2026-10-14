# Shallow Histogram Gradient Boosting

## Purpose

Evaluate nonlinear relationships and feature interactions
for binary AFIB classification using histogram gradient boosting.

## Status

Completed: four experimental arms (A-D), with ten outer
patient-grouped folds and three inner folds per outer fold.

## Model

Estimator: HistGradientBoostingClassifier

Fixed hyperparameters:
- max_iter: 150
- learning_rate: 0.05
- max_leaf_nodes: 16
- min_samples_leaf: 30
- l2_regularization: 1.0
- early_stopping: False
- random_state: 42

Tuned parameter:
- Initial max_depth grid: [2, 3, 4]
- Permitted inner-CV boundary refinement: depth 5
- Selection metric: mean inner-validation macro-F1

Prediction identifier: hist_gradient_boosting

## Experimental arms

- A: Original ECG features, no augmentation
- B: Original and engineered features, no augmentation
- C: Original features with training augmentation
- D: Original and engineered features with training augmentation

All arms use shared patient-grouped folds, fold-specific
preprocessing, and shared sample weights.

## Execution

From the project root:

python algorithms/histogram_gradient_boosting/run_experiment.py --arms A B C D --outer-folds 0 1 2 3 4 5 6 7 8 9

The script reads processed_data through the shared input workflow.
See ../README.md for the shared algorithm instructions.

## Results

Completed results are stored under:

reports/histogram_gradient_boosting/formal/

Each completed arm/fold directory contains:
- complete.json
- inner_tuning.csv
- metrics.csv
- predictions.csv

Observed-data macro-F1, mean +/- sample SD across ten outer folds:

| Arm | Macro-F1 |
| --- | --- |
| A | 0.9213 +/- 0.0158 |
| B | 0.9230 +/- 0.0143 |
| C | 0.9156 +/- 0.0122 |
| D | 0.9179 +/- 0.0126 |

Arm B achieved the highest observed-data mean macro-F1.
Training augmentation improved robustness under simulated
missing ECG leads, especially when six leads were hidden.

## Reproducibility

Use the same processed_data package, patient folds, and
software dependencies. Hyperparameters are selected using
inner-validation data only; outer-test results are reserved
for evaluation.

Predictions, per-fold metrics, and tuning decisions are
saved for analysis. Do not include the dataset in the final
code ZIP unless the assignment instructions permit it.

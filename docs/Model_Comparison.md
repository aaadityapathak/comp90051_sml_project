# Algorithm result comparison

TabM D has the best primary observed macro F1 among the main arms and the strongest reported six-lead-loss F1. LR and TabM have saved evidence that can be checked; HGB is represented by a historical summary, so the three-model ranking is descriptive and provisional.

## Best observed main arm per algorithm

| Algorithm and arm | Macro F1 (SD) | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- |
| Logistic D | 0.9032 (0.0147) | 0.9530 | 0.9727 | 0.0234 |
| HGB B reported | 0.9230 (0.0143) | 0.9496 | 0.9803 | Unavailable |
| TabM D | 0.9395 (0.0122) | 0.9166 | 0.9889 | 0.0129 |

Observed data, threshold 0.50, ten outer patient folds; parentheses contain sample SD. Best arm means the highest outer-fold mean among A-D. Selecting an arm after seeing these outcomes does not create a new independent evaluation of that choice.

TabM D exceeds logistic D by 0.0363 F1 and reported HGB B by 0.0165. Its precision is 0.8612 versus logistic D 0.7238. Higher specificity reduces false positives and improves F1, but lower sensitivity means more missed positives.

TabM D average precision is 0.9320 versus logistic D 0.8622. HGB B reported AUROC 0.9935 exceeds TabM D 0.9923; TabM does not win every metric. HGB Brier and average precision are unavailable.

## Robustness changes the best arm

| Algorithm | Best observed arm | Best six-lead-loss arm | F1 under six-lead loss |
| --- | --- | --- | --- |
| Logistic | D | B | 0.8605 |
| HGB reported | B | C | 0.8892 |
| TabM | D | D | 0.9142 |

TabM D combines the strongest main-arm observed and missing-lead F1. Nonlinear modeling and corrupted training views are plausible contributors; different tuning budgets and optimization prevent attributing the full advantage to architecture.

Across all TabM variants, duplicate-enhanced has the largest observed mean (0.9397), only 0.000231 above D, with much weaker missing-lead F1. This does not overturn D's robustness advantage.

## Comparison limits and next steps

LR and TabM have identical prepared-data manifests and 2,480 matching perturbation audits across 80 common fold-variants. Patient folds keep repeat ECGs together. HGB matching cannot be independently verified. Outer training sets overlap; fold SD establishes neither significance nor external validity.

1. Recover the HGB bundle or generate it on a future authorized run before claiming a fully verified three-model comparison. 2. Predeclare contrasts, quantify paired patient-level uncertainty and repeat seeds for small observed gains. 3. Compare at a common sensitivity or false-positive budget, fitting thresholds/calibration inside training folds. 4. Test external data and realistic lead loss before deployment. Do not average stresses without a justified operating distribution.

Evidence: LR/TabM output_data/summary.csv, run_config.json and saved perturbation audits; results/histogram_gradient_boosting/source_provenance/reported_summary.docx. Pilots are excluded from ranking. Rebuild: python scripts/build_model_observations.py.

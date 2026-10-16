# Histogram gradient boosting results

The current completed experiment is in [`retuned7_8_9/`](retuned7_8_9/).
It uses maximum-depth candidates `[7, 8, 9]`, ten patient-separated outer
folds, three inner folds, and arms A-D. See the
[implementation guide](../../models/histogram_gradient_boosting/README.md)
and [current report](../../report/COMP90051_Final_Report.pdf).

| File(s) | Contents |
| --- | --- |
| `HGB_ABCD_all_metrics.csv` | 1,240 fold/scenario/repeat metric rows |
| `HGB_ABCD_all_inner_tuning.csv` | 360 inner-fold scores: four arms x ten outer folds x three depths x three inner folds |
| `A_selected_depths.csv` through `D_selected_depths.csv` | Selected depth for each arm and outer fold |
| `HGB_ABCD_fold_scenario_means.csv` | Stress repeats averaged within each fold |
| `HGB_ABCD_scenario_summary.csv` | Means and sample SD across outer folds |
| `HGB_ABCD_paired_differences.csv` | Within-model arm comparisons |
| `HGB_ABCD_stress_drops.csv` | Within-model change from observed conditions |
| `HGB_retuned_A_predictions.zip` through `HGB_retuned_D_predictions.zip` | Per-record held-out probabilities for all ten folds |

The four top-level `final_*.csv` files retain the earlier HGB experiment.
The report places those scores next to the retuned results instead of replacing
them. `source_provenance/reported_summary.docx` is the earlier written summary;
it predates the newly supplied prediction and tuning evidence. The redundant
`retuned789/` directory contained nine byte-identical copies of current CSVs
and has been removed.

Run `python models/histogram_gradient_boosting/check_results.py` to independently
recompute all 1,240 metric rows from 2,703,076 predictions and verify the 360
inner-fold scores and 40 selected depths. No source dataset is required.
`python report/analyze_results.py` regenerates the report comparisons.
Historical code/evidence in `models/histogram_gradient_boosting/` is retained
for traceability.

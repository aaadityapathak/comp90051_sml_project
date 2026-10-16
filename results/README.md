# Saved experiment results

These folders contain metrics, predictions, tuning records, plots, and audit
records from the completed experiments. Source and prepared datasets are kept
outside the repository. The current combined analysis is in the
[four-page report](../report/COMP90051_Final_Report.pdf).

| Model | Main results | Coverage |
| --- | --- | --- |
| Logistic regression | [output_data/](logistic_regression/output_data/) | 10 outer folds, four arms, four controls, and ridge-imputation control |
| HGB, retuned depth 7/8/9 | [retuned7_8_9/](histogram_gradient_boosting/retuned7_8_9/) | 10 outer folds, four arms, 360 inner tuning scores, and four prediction archives |
| TabM | [output_data/](tabm/output_data/) | 10 outer folds, four arms, and four controls |

## Reading the results

A/B use observed training data with original/enhanced features. C/D use the
corresponding feature sets with corrupted training views. Repeat-level stress
measurements are averaged within each outer fold before calculating means and
sample standard deviations across the ten folds. Sample SD is descriptive,
not a confidence interval.

LR and TabM keep aggregate tables in `output_data/`, per-fold evidence in
`output_data/folds/`, and figures and checks in `reports/`. `run_config.json`
records experiment settings, input hashes, and versions. Source snapshots and
historical paths are retained as evidence of the original executions.

HGB uses the delivered CSV/ZIP layout described in its
[result guide](histogram_gradient_boosting/README.md). Its fixed-0.5 results can
be compared at the fold/scenario level. Do not assume the delivered HGB
corruption draws are identical to LR/TabM draws: the HGB runner uses a different
evaluation stream. Compare means and within-model paired effects accordingly.

## Verify without training

```powershell
python scripts/verify_result_artifacts.py
python models/histogram_gradient_boosting/check_results.py
python report/analyze_results.py
```

The first command uses only Python's standard library. It verifies the LR/TabM
prediction, metric, tuning, and audit bytes against the original manifests,
including TabM's original completion markers. Fitted LR `pipeline.pkl` files
are deliberately omitted from the submission and reported separately; the
original manifests are preserved. An auxiliary local syllabus audit is also
omitted from the code submission and reported separately. The HGB checker requires NumPy/pandas and
verifies the supplied predictions and depth selections. The report command
needs the reporting requirements and recomputes the combined comparisons.

With externally prepared data and the relevant model dependencies, the detailed
model checks are also available:

```powershell
python models/logistic_regression/check_results.py
python models/tabm/check_results.py
```

Use a new output directory for fresh training. Original saved tables,
predictions, and configuration records should not be overwritten. The small
`pilots/` folders preserve preliminary run evidence and are excluded from the
main comparisons. Fitted models, runtime caches, and the duplicate HGB
`retuned789/` mirror are not included.

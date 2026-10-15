# Model comparison

Logistic regression and TabM have completed results on the same ten outer folds.
The [comparison](../../docs/Model_Comparison.md) compares their saved results
with the historical HGB summary ([Word copy](../../docs/Model_Comparison.docx)).
Each algorithm also has a separate observation document under `docs/`.
The main summary CSV contains LR and TabM only; `hgb_reported_reference.json`
keeps the reported HGB figures separate from independently checkable artifacts.

| Artifact | Contents |
| --- | --- |
| [Main summary](output_data/observations_main_summary.csv) | Model, arm, scenario, threshold and metric summaries |
| [Method differences](output_data/observations_method_differences.csv) | Within-model paired arm/control differences used in the observations |
| [Input manifest](output_data/observations_input_manifest.json) | Hashes of the saved inputs used to build the document |
| [Workflow verification](reports/workflow_validation.json) | Earlier saved verification record; see cleanup_validation.json for preservation decisions |
| [Preprocessing reproduction](reports/preprocessing_validation.json) | Byte-identical regeneration of all four prepared-data files against both recorded runs |
| [Stress pairing check](output_data/observations_pairing_validation.json) | Exact comparison of shared test perturbation audits |

Rebuild these tables and both document formats from the repository root with
`python scripts/build_model_observations.py`, after installing
`requirements-docs.txt`. This reads saved evidence and performs no training.

The two models' test perturbation audit records match exactly for all **2,480
shared conditions** (80 common fold/variant combinations, 31 conditions each),
after removing only the classifier name. Both use the same deterministic
evaluation stream and data-manifest seed. Logistic regression additionally has
a ridge-imputation arm with a smaller stress scope; it is not a ninth TabM arm.

The shared three-classifier notebook
[`02_noise_evaluation.ipynb`](../../notebooks/02_noise_evaluation.ipynb) still
requires the missing histogram gradient boosting prediction bundle. Its combined
`predictions/inner_oof_predictions.csv` and `predictions/outer_predictions.csv`
have not been produced. This does not prevent the separate two-model
observations or either model's own complete report. The HGB document reports a
completed historical run, but its raw bundle is not present for verification.

Logistic regression's combined exports are in
[`../logistic_regression/predictions/`](../logistic_regression/predictions/).
TabM's canonical predictions are saved per fold/variant in
[`../tabm/output_data/folds/`](../tabm/output_data/folds/).
Shared legacy evaluation reports remain in `reports/evaluation/` at the
repository root.

# AFIB classification with noisy or missing ECG measurements

COMP90051 Statistical Machine Learning | Aaditya Pathak, Siwen Wang, Jiayi Huang

This project compares logistic regression, histogram gradient boosting (HGB)
and TabM on patient-separated ECG classification. Saved logistic regression and
TabM results cover all ten outer folds. HGB code and previously reported scores
are present, but its raw predictions and tuning bundle are missing from this
checkout; the comparison distinguishes reported scores from verified results.

## Observations

| Document | Contents |
| --- | --- |
| [Logistic regression](docs/Logistic_Regression_Observations.docx) ([Markdown](docs/Logistic_Regression_Observations.md)) | Results, tuning, best variant, limitations and next experiments |
| [Histogram gradient boosting](docs/Histogram_Gradient_Boosting_Observations.docx) ([Markdown](docs/Histogram_Gradient_Boosting_Observations.md)) | Reported results and missing evidence |
| [TabM](docs/TabM_Observations.docx) ([Markdown](docs/TabM_Observations.md)) | Results, tuning, best variant, limitations and next experiments |
| [Model comparison](docs/Model_Comparison.docx) ([Markdown](docs/Model_Comparison.md)) | Matched algorithm comparison and evidence limits |

[Project plan](docs/Project_Plan.docx), [data insights](docs/Data_Insights.docx)
and [preprocessing guide](docs/Preprocessing.docx) describe the study and inputs.

## Repository layout

```text
models/             # model implementations, training runners and result checks
notebooks/          # all active notebooks; archive/ and legacy/ hold historical copies
results/            # saved experiment evidence and generated analysis outputs
reports/            # retained local historical reports and document working files
docs/               # project guides and per-model/comparison observations
src/afib/           # shared leakage-safe preprocessing, imputation and evaluation
scripts/            # notebook execution, reproducibility audits and document generation
tests/              # shared and model-specific checks
data/               # canonical source CSVs
processed_data/     # shared prepared data and manifest
```

`src/afib/` is required by the notebooks and model runners. Keep the source data,
prepared manifest, saved predictions, tuning records and source provenance:
these support independent verification. Existing results and reports are retained.
Generated shared report folders and legacy notebooks stay local through Git
ignore rules; already tracked evidence remains tracked.

All active notebooks are in one folder:

| Notebook | Purpose |
| --- | --- |
| [00_data_insights](notebooks/00_data_insights.ipynb) | Display descriptive findings and plots |
| [01_preprocessing](notebooks/01_preprocessing.ipynb) | Build the shared `processed_data/` package |
| [02_noise_evaluation](notebooks/02_noise_evaluation.ipynb) | Evaluate a complete three-model prediction bundle; otherwise reports missing inputs |
| [03_imputation_reliability](notebooks/03_imputation_reliability.ipynb) | Reproduce the optional imputation engineering audit |
| [04_logistic_regression_results](notebooks/04_logistic_regression_results.ipynb) | Inspect saved logistic regression results |
| [05_tabm_tuning](notebooks/05_tabm_tuning.ipynb) | Inspect TabM; training requires an explicit switch |
| [06_histogram_gradient_boosting_results](notebooks/06_histogram_gradient_boosting_results.ipynb) | Inspect HGB or explicitly run training into a new directory |

## Verify the saved results

Run from this repository with the required model dependencies installed:

```powershell
python scripts/verify_result_artifacts.py
python models/logistic_regression/check_results.py
python models/tabm/check_results.py
python scripts/verify_preprocessing.py
python scripts/check_model_observations.py
python -m unittest discover -s tests -v
```

The artifact check verifies file inventory, hashes and coverage without training
or ML dependencies. Model checkers independently recalculate saved prediction
metrics and check folds, thresholds and source provenance. They do not refit the
historical models. HGB cannot be checked until its raw bundle is restored under
`results/histogram_gradient_boosting/formal/`; its checker accepts `--output` for
another run directory and verifies coverage/structure rather than every metric.

## Reproduce data preparation and training

Use [PTB-XL+ 1.0.1 ECGDeli features](https://physionet.org/content/ptb-xl-plus/1.0.1/)
and [PTB-XL 1.0.3 annotations](https://physionet.org/content/ptb-xl/1.0.3/), with
CC BY 4.0 attribution. Preserve the exact CSV bytes and source paths recorded
in the manifests. Keep only one active PTB-XL+ extraction:

```text
data/
  ptb-xl-plus/
    features/ecgdeli_features.csv
    labels/ptbxl_statements.csv
  reference/ptbxl_database.csv
```

```powershell
python -m pip install -r requirements.txt
python scripts/execute_notebook.py notebooks/01_preprocessing.ipynb
```

The prepared dataset contains 21,799 ECGs from 18,869 patients, with 1,514
AFIB-labelled records. Use 504 lead fields; keep labels, identifiers, quality
flags and Global fields outside predictors. Shared preparation fits imputation,
scaling and category rules on training patients only.

Follow the [model index](models/README.md) and each model's README for training.
Use a **new output directory** for reproduction. The historical logistic
regression run records Python 3.12.15 and packages in
`models/logistic_regression/requirements-recorded.txt`; TabM records Python
3.12.10, CUDA 12.8 and packages in `models/tabm/requirements.txt`. Separate
environments are needed for the differing NumPy/SciPy versions.

```powershell
python scripts/check_environment.py --model logistic_regression
python scripts/check_environment.py --model tabm
```

Exact environment checks report mismatches. Matching versions does not guarantee
bitwise agreement across hardware. `verify_preprocessing.py` rebuilds the data
in a temporary directory and compares bytes and the complete historical manifest.

## Rebuild documents and reports

```powershell
python -m pip install -r requirements-docs.txt
python scripts/build_model_observations.py
python models/tabm/run_full_pipeline.py --report-only --skip-notebook
```

The document generator reads saved evidence without training. For other outputs,
run the corresponding notebook or checker. See [results/README.md](results/README.md)
for result layout, integrity manifests and updating generated outputs.

Document regeneration reproduces all eight Word/Markdown files and five comparison
files byte for byte. Text and tables were checked against saved evidence. Word page
rendering is unverified because this Windows runtime has no LibreOffice renderer;
Markdown copies provide the same reviewed content.

The move from `algorithms/` to `models/` changes active imports and source paths.
Historical configurations and source snapshots retain their original paths and
hashes; source checks explicitly recognize the package relocation while rejecting
training changes. Historical notebook outputs can also contain old paths.

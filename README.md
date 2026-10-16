# AFIB classification with noisy or missing ECG measurements

COMP90051 Statistical Machine Learning, University of Melbourne

We study whether cross-lead feature construction and corrupted training views
improve atrial fibrillation classification when ECG measurements are noisy or
missing. Logistic regression, histogram gradient boosting (HGB), and TabM use
the same manually constructed, patient-separated nested cross-validation folds.

## Team responsibilities

| Teammate | Main responsibility |
| --- | --- |
| Aaditya Pathak | Data preparation, preprocessing, feature construction, and TabM |
| Siwen Wang | Logistic regression |
| Jiayi Huang | Histogram gradient boosting |
| All teammates | Research report, code review, integration, and repository maintenance |

## Start here

- [Four-page research report](report/COMP90051_Final_Report.pdf): methods, results, side-by-side comparisons, and limitations.
- [Readme.txt](Readme.txt): short execution instructions for the code submission.
- [Saved results](results/README.md): original metrics, predictions, tuning records, and validation evidence.
- [Model guides](models/README.md): dependencies and training commands for each classifier.

The latest HGB bundle is `results/histogram_gradient_boosting/retuned7_8_9/`.
It includes all four arms, ten outer folds, the depth grid `[7, 8, 9]`,
inner-fold tuning scores, and compressed held-out predictions. Earlier HGB
results are retained separately so the change in tuning range remains visible.

## Repository layout

```text
models/          classifier implementations, experiment runners, and checks
src/afib/        shared preparation, feature construction, imputation, and metrics
notebooks/       seven numbered preparation and analysis notebooks
scripts/         data preparation, notebook execution, and reproducibility checks
tests/           tests for the shared pipeline and model implementations
results/         saved experiment outputs; no source or prepared datasets
report/          final PDF, editable builder, and comparison tables/figures
docs/            supplementary project notes and earlier detailed observations
```

All original modelling methods and experimental outputs are retained. The
duplicate HGB summary folder, fitted pipeline binaries, local environments,
and caches are excluded from the submission. Saved configurations and source
snapshots keep their historical paths and hashes; active code resolves current
paths independently. Supplementary notes describe stages of the project; the
four-page report is the current combined analysis.

## Inspect the results without downloading data

From the repository root, Python alone can check the recorded LR/TabM result
files against their original checksums:

```powershell
python scripts/verify_result_artifacts.py
python models/histogram_gradient_boosting/check_results.py
```

The checker reports omitted fitted pipeline binaries separately. It still
requires every recorded prediction, metric, tuning record, and audit file.
The HGB check needs NumPy and pandas and independently recomputes metrics from
the supplied prediction archives. These checks do not retrain a model.

## Obtain the data separately

Download only these three CSV files from the public datasets:

| Dataset | File needed | Place under the external raw-data folder |
| --- | --- | --- |
| [PTB-XL+ 1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/) | `features/ecgdeli_features.csv` | `ptb-xl-plus/features/ecgdeli_features.csv` |
| [PTB-XL+ 1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/) | `labels/ptbxl_statements.csv` | `ptb-xl-plus/labels/ptbxl_statements.csv` |
| [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | `ptbxl_database.csv` | `reference/ptbxl_database.csv` |

Credit: Strodthoff et al. (2023), *PTB-XL+*, and Wagner et al. (2020),
*PTB-XL*. The dataset pages provide the CC BY 4.0 licence and full citations.
Raw ECG waveforms are not needed for these experiments.

The default data location is beside the repository:

```text
parent-folder/
  comp90051_sml_project/     # this repository
  afib_data/
    raw/                    # the three source CSVs in the layout above
    processed/              # generated features, labels, folds, and manifest
```

For a different location, set `AFIB_DATA_DIR` and `AFIB_PROCESSED_DIR` to
absolute paths before preparing data or running a model. For example:

```powershell
$env:AFIB_DATA_DIR = "D:\ecg_data\raw"
$env:AFIB_PROCESSED_DIR = "D:\ecg_data\processed"
```

On macOS/Linux, use `export AFIB_DATA_DIR=/path/to/raw` and
`export AFIB_PROCESSED_DIR=/path/to/processed`.

## Prepare and reproduce

Use Python 3.12 for the shared pipeline and LR/TabM. Keep the Python environment
outside this folder, activate it, and run commands from the repository root.

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/verify_preprocessing.py
```

Preparation joins 21,799 ECGs from 18,869 patients, including 1,514 AFIB-labelled
records, and selects 504 lead measurements. Labels, patient/ECG identifiers,
quality flags, and Global measurements are excluded from predictors. Learned
imputation and scaling use each split's training patients only.

Install the selected model's dependencies using its linked guide, then write
a reproduction run to a **new** result directory:

```powershell
python models/logistic_regression/run_experiment.py --output results/logistic_regression_reproduction
python models/histogram_gradient_boosting/run_experiment_retuned.py --arms A B C D --outer-folds 0 1 2 3 4 5 6 7 8 9 --output results/hgb_reproduction
python models/tabm/run_experiment.py --output results/tabm_reproduction --device auto
```

These are full experiments and can take substantial time. The model guides
document smaller runs and optional controls. Recorded LR, HGB, and TabM
environments differ: use a separate environment per model when reproducing
historical versions. See [LR](models/logistic_regression/README.md),
[HGB](models/histogram_gradient_boosting/README.md), and
[TabM](models/tabm/README.md). Seeds, versions, and input hashes are retained;
fresh training can differ across hardware and library versions.

After installing the relevant dependencies, run the existing tests:

```powershell
python -m unittest discover -s tests -v
```

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [00_data_insights](notebooks/00_data_insights.ipynb) | Dataset summaries and plots |
| [01_preprocessing](notebooks/01_preprocessing.ipynb) | Prepare external data and inspect the pipeline |
| [02_noise_evaluation](notebooks/02_noise_evaluation.ipynb) | Shared prediction-bundle evaluation |
| [03_imputation_reliability](notebooks/03_imputation_reliability.ipynb) | Optional imputation audit |
| [04_logistic_regression_results](notebooks/04_logistic_regression_results.ipynb) | Logistic regression results |
| [05_tabm_tuning](notebooks/05_tabm_tuning.ipynb) | TabM tuning and results |
| [06_histogram_gradient_boosting_results](notebooks/06_histogram_gradient_boosting_results.ipynb) | HGB results and optional training |

Start Jupyter from the repository root. Outputs containing dataset rows or
machine-specific execution paths are cleared; useful aggregate displays are
retained. Original experiment results remain in `results/`.

## Report and submission

All required plotting code is included. Regenerate figures from saved results
without training or downloading datasets:

```powershell
python models/logistic_regression/plot_results.py
python models/tabm/plot_results.py
python scripts/execute_notebook.py notebooks/06_histogram_gradient_boosting_results.ipynb
```

The first two commands write fresh PNGs under `results/figures/`, preserving
the recorded experiment files. Notebook 06 contains the earlier/retuned HGB
plotting cells. Dataset-description plots are in notebook 00 and require the
external data. The report's figure and table generation code is documented in
[report/README.md](report/README.md).

The report builder reads saved results and performs no model training:

```powershell
python -m pip install -r report/requirements.txt
python report/analyze_results.py
python report/build_report.py
```

Submit the four-page PDF separately from the code working files. Include
`Readme.txt`; exclude Git metadata (`.git`), environments, caches, and the
external `afib_data/` folder. No source or prepared dataset belongs in the code
submission. Existing compressed prediction files are experimental results.
No submission ZIP or packaging script is generated by this repository.

The teaching team requires at least three team meetings. Include genuine
meeting minutes or logs if the team chooses to supply them.

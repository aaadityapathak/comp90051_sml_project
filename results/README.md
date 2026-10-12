# Experiment results

Each algorithm owns a matching folder here and in `algorithms/`.

```text
results/<algorithm>/
  run_config.json       # saved experiment configuration and source/data hashes
  status.json           # live progress, when supplied by the runner
  output_data/          # metric tables, tuning data and fold artifacts
    folds/
    tuning/             # cached inner fits, when supplied by the runner
  reports/              # figures, written reports and completion checks
  predictions/          # exported predictions for shared evaluation
  checkpoints/          # binary resume models, when stored separately
  source_provenance/    # preserved historical source/configuration evidence
  pilots/<run>/         # smaller runs using the same layout
```

Folders are created as needed. Histogram gradient boosting has no results yet.
The completed logistic regression fold artifacts retain their original pipeline
pickles beside the saved predictions. TabM stores binary checkpoints separately
and keeps them local. Saved run configurations, source snapshots, predictions,
tables and figures retain their original contents.

## Path migration

| Previous location | Current location |
| --- | --- |
| `reports/logistic_regression/*.csv` | `results/logistic_regression/output_data/` |
| `reports/logistic_regression/folds/` | `results/logistic_regression/output_data/folds/` |
| `reports/logistic_regression/*.png` and `completion.json` | `results/logistic_regression/reports/` |
| `reports/logistic_regression/run_config.json` and `source_provenance/` | `results/logistic_regression/` |
| `reports/logistic_pilot/` | `results/logistic_regression/pilots/logistic_pilot/`, with the same table/report separation |
| `models/predictions/logistic_regression/` | `results/logistic_regression/predictions/` |
| `models/tabm/` | `results/tabm/`, preserving its internal folders |
| `models/predictions/` combined evaluation input location | `results/comparison/predictions/` |

The shared evaluation notebook still requires predictions from all three models.
The comparison folder does not imply those results have been produced.

Historical logs, notebook outputs and source snapshots can mention the old
paths: they are retained as evidence of the original runs. Active code and
notebook cells use the current paths. TabM's source compatibility check verifies
the preserved runner snapshot and allows only this layout migration when
resuming existing fits. Changes to training calculations or configuration still
require a new output folder. Logistic regression retains its existing strict
resume guard; inspect its historical results with `check_results.py`.

`data/` and `processed_data/` remain at the repository root so the shared data
manifest and all recorded input hashes remain valid. Shared analysis and legacy
work remain under `reports/`.

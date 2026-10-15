# Experiment results

Each algorithm owns a matching folder here and in `models/`. Logistic
regression and TabM have complete saved bundles. Histogram gradient boosting
has implemented code and reported scores, but its raw result bundle is unavailable
in this checkout.

| Model | Outer fold/variant results | Model fits | Validation |
| --- | ---: | ---: | --- |
| Logistic regression | 90 | 933 | [Passed: 2,520 test conditions](logistic_regression/reports/validation.json) |
| TabM | 80 | 971 | [Passed: 2,480 test conditions](tabm/reports/validation.json) |

The logistic ridge-imputation control accounts for the additional ten
fold/variant results. Both runs contain the same eight principal/control
variants over ten patient-grouped outer folds. See the
[TabM result index](tabm/README.md), [comparison index](comparison/README.md) and
[model comparison](../docs/Model_Comparison.md)
([Word copy](../docs/Model_Comparison.docx)) and per-model observations in `docs/`.

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

Folders are created as needed. Histogram gradient boosting expects its missing
raw bundle at `results/histogram_gradient_boosting/formal/`.
The completed logistic regression fold artifacts retain their original pipeline
pickles beside the saved predictions. TabM stores binary checkpoints separately
and keeps them local. Saved run configurations, source snapshots, predictions,
tables and figures retain their original contents.

## Integrity and numerical reproduction

From the repository root, verify the published bundle without datasets or
training dependencies:

```powershell
python scripts/verify_result_artifacts.py
```

The per-model `reports/artifact_manifest.json` files record relative paths,
byte sizes and SHA-256 checksums for all canonical tables, fold predictions,
reports, source evidence and existing logistic pipeline files. Pilots, local
TabM checkpoints, live status/logs and README navigation are outside this
manifest. TabM's 971 original completion markers are checked as well. Git
attributes preserve recorded result bytes across platform checkouts.

With the original prepared data and dependencies available, independently
recalculate saved prediction metrics, thresholds, patient separation and fold
summaries without retraining:

```powershell
python models/logistic_regression/check_results.py
python models/tabm/check_results.py
```

These checks were rerun successfully after training completed. They establish
consistency of the saved evidence; they do not claim a fresh fit on different
hardware will be bitwise identical. Training requires the source data
and processed data described in the repository README. Some data files were
tracked before ignore rules were added; `.gitignore` does not untrack them.
Original source hashes,
versions and manifests are retained in each `run_config.json` and
`source_provenance/`; do not edit those records to make a changed run appear
compatible.

After intentionally refreshing generated outputs, run both numerical checkers
with their `--save` option, then explicitly record and recheck the new bundle:

```powershell
python scripts/verify_result_artifacts.py --write-manifests
python scripts/verify_result_artifacts.py
```

The canonical evidence totals about 479 MiB across both models, excluding
pilots and the roughly 1.12 GiB of ignored TabM checkpoint files. Keep compressed
prediction files compressed. Individual result files are below 50 MiB. The
user can stage and push the verified outputs; none of these checks publishes
or trains anything.

Before pushing, the repository's existing tracked data and result files need
to retain their recorded byte contents under the current Git attributes. After
reviewing the changes, stage those bytes and the intended new files, then check
the index that will actually be committed:

```powershell
git add --renormalize -- data processed_data results
git add -- .gitattributes .gitignore README.md Readme.txt models notebooks docs scripts tests requirements-docs.txt results
python scripts/verify_result_artifacts.py --check-git-index
```

The optional index check checks `.gitattributes`, both manifests, all inventoried result files
and any already tracked input data against their verified working-tree bytes.
Tracked raw/prepared data must also match the saved run hashes. It reports
missing or different staged files and never stages, commits or pushes anything.
The ordinary integrity check works before staging; the index check is expected
to fail until the intended changes have been staged.

## Path migration

| Previous location | Current location |
| --- | --- |
| `algorithms/` source code | `models/` source code |
| `algorithms/<model>/notebooks/*.ipynb` | `notebooks/` |
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
the preserved runner snapshot and permits only documented output paths and
package imports changed by these migrations when
resuming existing fits. Changes to training calculations or configuration still
require a new output folder. Logistic regression retains its existing strict
resume guard; inspect its historical results with `check_results.py`.

`data/` and `processed_data/` remain at the repository root so the shared data
manifest and all recorded input hashes remain valid. Shared analysis and legacy
work remain under `reports/`.

Generated shared reports (`results/data_insights/`, `results/imputation/`,
`results/comparison/evaluation/`, `results/comparison/reports/`) stay local through
Git ignore rules. Existing tracked artifacts remain tracked. Historical reports
and results were retained; no training or regeneration is needed to read them.

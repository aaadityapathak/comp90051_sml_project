# Completed TabM results

The canonical run is complete: **80 outer fold/variant results, 891 inner fits,
80 outer refits and 297 tuning candidates**. All ten patient-grouped outer folds,
the four main arms and four controls are present. Every fit used 100 epochs.
The independent checker has recalculated all 2,480 saved outer test conditions.

Start with the [experiment report](reports/report.md),
[summary table](output_data/summary.csv),
[completion record](reports/completion.json) and
[numerical validation](reports/validation.json).
The [detailed model observations](../../docs/Model_Observations.md) discuss both
classifiers, limitations and follow-up experiments
([Word copy](../../docs/Model_Observations.docx)).

## Result map

| Location | Purpose |
| --- | --- |
| `run_config.json` | Immutable settings, source hashes, input hashes and recorded versions |
| `reports/` | Written report, figures, environment evidence, completion and verification |
| `output_data/summary.csv` | Mean and sample SD across ten outer folds, after averaging stress repeats within each fold |
| `output_data/fold_mean_metrics.csv` | Fold-level values underlying the summary |
| `output_data/paired_method_differences.csv` | Within-TabM paired arm/control contrasts |
| `output_data/paired_stress_drops.csv` | Paired deterioration from observed conditions |
| `output_data/selected_hyperparameters.csv` | Selected blocks and frozen inner-OOF thresholds for each fold/variant |
| `output_data/tuning_progress.csv` | Completed inner-fit and candidate coverage |
| `output_data/folds/<fold>_<variant>/` | Selected inner OOF predictions, all outer predictions, metrics, audits, histories and commit marker |
| `output_data/tuning/<fold>_<variant>/blocks_<n>/inner_<n>/` | Individual tuning predictions, history, audit and commit marker |
| `source_provenance/layout_v1/` | Original hash-checked runner preserved before the results-folder migration |
| `checkpoints/` | Local model/preprocessor binaries; intentionally excluded from Git |
| `pilots/` | Execution experiments; excluded from canonical summaries and integrity manifest |

`A` is original features/observed training, `B` enhanced/observed, `C`
original/augmented and `D` enhanced/augmented. The other four variants are
availability and duplicate controls. The secondary sensitivity threshold is
chosen using inner OOF predictions; outer labels never choose the threshold.
Sample SD is descriptive and is not a confidence interval.

## Verify and reproduce

Run these commands from the repository root. The first needs only Python's
standard library and the published outputs; the second additionally requires
the prepared data and analysis dependencies, but never trains a model.

```powershell
python scripts/verify_result_artifacts.py --models tabm
python algorithms/tabm/check_results.py
```

To deliberately rebuild derived TabM tables and figures from completed folds:

```powershell
python algorithms/tabm/reporting.py
python algorithms/tabm/check_results.py --save results/tabm/reports/validation.json
python scripts/verify_result_artifacts.py --models tabm --write-manifests
python scripts/verify_result_artifacts.py --models tabm
```

Report rebuilding changes generation timestamps, so update the manifest only
after checking the new outputs. The manifest detects missing, added or changed
evidence and also checks the original TabM commit-marker hashes. It does not
replace numerical validation.

For a new training run, dependency setup and faster execution choices, follow
the [TabM implementation guide](../../algorithms/tabm/README.md) and
[analysis notebook](../../algorithms/tabm/notebooks/05_tabm_tuning.ipynb).
Use a separate output directory for changed settings or new runtime experiments.
Keep this completed run as the historical baseline. Source compatibility checks
allow only the documented output-path migration for its saved training sources.

Seeds, splits and source versions are recorded. Numerical reproducibility is
checked against saved predictions; exact fresh GPU-training bytes are not
guaranteed across hardware, drivers and library versions. A Git clone contains
prediction evidence for validation, not the local TabM model checkpoints needed
to resume an interrupted fit without refitting it. The full canonical bundle is
about 248 MiB; local checkpoint binaries add about 1.12 GiB.

See the [shared results guide](../README.md) for the optional staged-byte check
before pushing; existing tracked datasets remain tracked despite ignore rules.

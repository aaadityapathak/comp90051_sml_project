# TabM results

The completed run contains 80 outer fold/variant results, 891 inner fits, and
80 outer refits. All ten patient-separated folds, four main arms, and four
controls are included. Every fit used 100 epochs. Start with the
[summary table](output_data/summary.csv), [experiment report](reports/report.md),
[completion record](reports/completion.json), and
[current combined research report](../../report/COMP90051_Final_Report.pdf).

| Location | Contents |
| --- | --- |
| `run_config.json` | Settings, source/input hashes, and recorded versions |
| `output_data/summary.csv` | Mean and sample SD across ten outer folds |
| `output_data/fold_mean_metrics.csv` | Fold-level results underlying the summary |
| `output_data/paired_method_differences.csv` | Paired arm/control comparisons |
| `output_data/paired_stress_drops.csv` | Deterioration from observed conditions |
| `output_data/selected_hyperparameters.csv` | Blocks and frozen inner-OOF thresholds |
| `output_data/folds/` | Selected predictions, metrics, audits, and histories |
| `output_data/tuning/` | Individual inner-fit predictions, history, and audit |
| `reports/` | Figures, completion checks, and numerical validation |
| `source_provenance/` | Preserved source from the original execution layout |
| `pilots/` | Preliminary experiments excluded from final comparisons |

Stress repeats are averaged within folds before reporting mean and sample SD.
The secondary sensitivity threshold uses inner OOF predictions only. Fitted
checkpoints are not required to inspect results and are not submitted.

```powershell
python scripts/verify_result_artifacts.py --models tabm
python models/tabm/check_results.py
```

The first command needs no dataset or ML libraries. The second requires
externally prepared data and analysis dependencies and recalculates the 2,480
saved outer test conditions without training. Follow the
[model guide](../../models/tabm/README.md) for fresh training and use a new
output folder. See [notebook 05](../../notebooks/05_tabm_tuning.ipynb) for tuning
analysis. Original hashes and configuration records remain unchanged; a new
GPU training run may differ across hardware, drivers, and software versions.

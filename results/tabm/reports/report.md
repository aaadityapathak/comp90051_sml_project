# TabM experiment report — INCOMPLETE TABM PLAN

Generated 2026-10-12T14:18:02.062837+00:00. Requested scope: **incomplete**.
**3/80 outer-fold/variant checkpoints** and **0/10 complete outer folds** are available.
There are **33 completed inner fits**, **3 outer refits**, and **11 fully evaluated tuning candidates**.
The initial grid requires 720 inner fits; endpoint refinement can require more.
Completing a reduced run does not establish completion of the project plan.

Only completed outer checkpoints enter performance summaries. Partial tuning scores are validation scores, not final model results. A single completed outer fold has undefined sample SD. Empty figures mean the relevant results are not available; no values have been estimated or fabricated.

## Settings and runtime

Requested variants: A, B, C, D, availability_observed, availability_augmented, duplicate_original, duplicate_enhanced. Outer folds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; inner folds: 3.
Initial block grid: [2, 3, 4]; one-step endpoint refinement: True.
Every fit requests 100 epochs and batch size 256; stress repeats: 3 (31 conditions per outer checkpoint).
Device: cuda; seed: 20261008.
Sum of completed-fold wall times: 1176.0 seconds (not total elapsed experiment time).

```json
{
  "architecture": {
    "width": 128,
    "members": 16,
    "dropout": 0.1,
    "numerical_embeddings": false
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001
  },
  "versions": {
    "python": "3.12.10",
    "numpy": "2.5.1",
    "pandas": "3.0.6",
    "torch": "2.7.1+cu128",
    "tabm": "0.0.3"
  }
}
```

## Observed performance at the fixed 0.50 threshold

| variant | metric | mean | sd | n_folds | n_defined |
| --- | --- | --- | --- | --- | --- |
| A | macro_f1 | 0.9070 | undefined | 1 | 1 |
| A | sensitivity | 0.9103 | undefined | 1 | 1 |
| A | specificity | 0.9795 | undefined | 1 | 1 |
| A | brier | 0.0191 | undefined | 1 | 1 |
| B | macro_f1 | 0.9117 | undefined | 1 | 1 |
| B | sensitivity | 0.8966 | undefined | 1 | 1 |
| B | specificity | 0.9824 | undefined | 1 | 1 |
| B | brier | 0.0181 | undefined | 1 | 1 |
| C | macro_f1 | 0.9085 | undefined | 1 | 1 |
| C | sensitivity | 0.9103 | undefined | 1 | 1 |
| C | specificity | 0.9800 | undefined | 1 | 1 |
| C | brier | 0.0185 | undefined | 1 | 1 |

## Protocol and interpretation

All preprocessing, imputation, scaling and augmentation are fitted using training patients only. Block count is selected using mean inner-fold macro-F1; ties favor fewer blocks. The secondary threshold is selected from inner out-of-fold predictions to target sensitivity 0.90, then frozen for every outer condition. It is an exploratory threshold, not a clinical acceptance criterion.

Stress repeats are averaged within each outer fold before calculating the outer-fold mean and sample SD. Undefined metrics propagate through both aggregation stages. Error bars are descriptive sample SD, not confidence intervals; outer folds share training data. Positive paired stress deterioration means performance becomes worse (stress minus observed for Brier; observed minus stress otherwise). Method differences are left minus right on the intersection of completed folds, for both threshold policies.

The reports include B–A, D–C, C–A and D–B plus B/availability-observed, D/availability-augmented and both duplicate controls when paired results exist. Patient-weighted metrics give each patient total weight one. Quality groups use recorded source annotations; an absent flag does not prove a clean signal. The simulated tabular corruptions are not calibrated ECG waveform artefacts.

Majority baseline: computed from training labels only; current data manifest verified. It can be computed before any model finishes and is not evidence of completed model training.

## Figures

![Outer-fold progress](progress.png)

![Macro-F1 robustness](robustness_macro_f1.png)

![Sensitivity robustness](robustness_sensitivity.png)

![Brier-score robustness](robustness_brier.png)

![Calibration](calibration.png)

![Training](training_loss.png)

Calibration pools observed test records into ten probability bins across completed folds. Training curves average recorded histories at each epoch; they do not select the final model.

## Saved evidence

The sibling `output_data/` directory contains raw fold tables, predictions, tuning candidates, `summary.csv`, `fold_mean_metrics.csv`, paired contrasts, patient-weighted and quality-group summaries, majority baseline, and tuning progress. Checkpoint models are stored separately in `checkpoints/`. `completion.json` records missing checkpoints and precise coverage. Source hashes, processed-data manifest, and runtime versions are preserved in `../run_config.json`.

These report-generation coverage checks do not replace the independent `check_results.py` audit, which recalculates saved prediction metrics and checks patient separation, complete scenario coverage and settings. No results from other classifiers are required for this TabM report.

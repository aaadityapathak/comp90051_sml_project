> Supplementary analysis: see the [current four-page report](../report/COMP90051_Final_Report.pdf) for the latest three-model comparison, including retuned HGB. Earlier evidence-availability statements describe the original analysis.

# TabM observations

Best main arm: D, observed F1 0.9395 (0.0122) and six-lead-loss F1 0.9142. The duplicate-enhanced control has the highest observed mean across all variants, 0.9397 (0.0120), only 0.000231 above D; its six-lead-loss F1 is 0.7555. D is the stronger choice when missing-lead robustness matters.

## Experiment and observed results

The cohort contains 21,799 ECGs from 18,869 patients; 6.95% are AFIB-labelled. This class imbalance makes macro F1 and class-specific errors more informative than accuracy alone.

A: original measurements; B: original plus engineered fields; C: A with augmentation; D: B with augmentation. There are 504 original lead fields and 63 engineered fields. Augmentation adds one corrupted training view per ECG with total weight preserved.

Primary results use threshold 0.50, 10 outer patient folds and 3 inner folds. Filling, scaling and selection use training patients only. Three stress repeats are averaged within each fold, then summarized by mean and sample SD. SD is not a confidence interval.

| Arm | Macro F1 (SD) | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- |
| A | 0.9321 (0.0126) | 0.9108 | 0.9871 | 0.0142 |
| B | 0.9389 (0.0137) | 0.9199 | 0.9884 | 0.0133 |
| C | 0.9338 (0.0121) | 0.9134 | 0.9874 | 0.0140 |
| D | 0.9395 (0.0122) | 0.9166 | 0.9889 | 0.0129 |

Higher macro F1, sensitivity and specificity are better; lower Brier means lower probability error.

B minus A F1 is +0.0067; D minus B is +0.0006. The small observed augmentation gain alone does not justify D. Its stronger advantage is sustained performance under missing inputs.

## Tuning and training

Initial block counts were 2, 3 and 4, with one boundary refinement. Selection maximized mean inner observed F1; exact ties favored fewer blocks. Width 128, 16 members, dropout 0.1, AdamW, batch size 256 and 100 epochs were fixed. No early stopping was used.

| Variant | Setting and fold count | Selected inner F1 |
| --- | --- | --- |
| A | 2: 3; 3: 6; 4: 1 | 0.9311 |
| B | 1: 1; 2: 6; 3: 1; 4: 1; 5: 1 | 0.9353 |
| C | 2: 2; 3: 2; 4: 4; 5: 2 | 0.9322 |
| D | 1: 1; 2: 5; 3: 1; 4: 3 | 0.9360 |
| availability observed | 2: 4; 3: 4; 4: 1; 5: 1 | 0.9311 |
| availability augmented | 1: 2; 2: 1; 3: 2; 4: 4; 5: 1 | 0.9316 |
| duplicate original | 1: 1; 2: 2; 3: 4; 4: 2; 5: 1 | 0.9321 |
| duplicate enhanced | 2: 5; 3: 3; 4: 2 | 0.9362 |

57/80 selections triggered boundary refinement; 69/80 selected the middle of the final grid. Inner F1 averages each fold's selected candidate score; it is selection evidence, not an unbiased generalization estimate. Refined candidates have unequal fold coverage; their aggregate scores cannot establish one globally best setting.

The run contains 971 fits: 891 inner fits and 80 outer refits. Summed fold work is 13.07 hours, not end-to-end elapsed time. Training-loss histories alone cannot establish absence of overfitting or the best epoch count.

## Robustness and additional outputs

| Condition | A | B | C | D |
| --- | --- | --- | --- | --- |
| Observed | 0.9321 | 0.9389 | 0.9338 | 0.9395 |
| 1 missing lead | 0.9330 | 0.9373 | 0.9314 | 0.9383 |
| 3 missing leads | 0.8999 | 0.9145 | 0.9266 | 0.9299 |
| 6 missing leads | 0.6358 | 0.7513 | 0.9085 | 0.9142 |
| Noise 0.05 | 0.9323 | 0.9386 | 0.9342 | 0.9400 |
| Noise 0.10 | 0.9323 | 0.9390 | 0.9340 | 0.9398 |
| Noise 0.20 | 0.9322 | 0.9387 | 0.9340 | 0.9394 |
| Gain 0.05 | 0.9323 | 0.9386 | 0.9336 | 0.9405 |
| Gain 0.10 | 0.9327 | 0.9382 | 0.9334 | 0.9400 |
| Gain 0.20 | 0.9330 | 0.9377 | 0.9339 | 0.9393 |
| 3 leads + noise 0.10 | 0.8983 | 0.9115 | 0.9261 | 0.9304 |

Noise SD is the listed dose times the training-field IQR; gain is exp(dose times a standard-normal draw). Perturbations act on measurement tables, not raw ECG waveforms. Small F1 rises under mild corruption can reflect shifted threshold errors; they do not establish better measurement quality.

With six leads missing, A sensitivity is 0.1813; D retains 0.8492. Exposure to corrupted training views is consistent with this large robustness gain. Tested additive noise and gain changes have much smaller F1 effects than lead removal.

B exceeds its availability control by 0.0057 observed F1, supporting information in engineered values beyond their missingness indicators. Duplicate controls match augmented view/update counts but can select different block counts, leaving capacity and compute unmatched. Relative to observed-only arms, both double views and increase updates. Lower Brier establishes better probability error, not calibration by itself.

The inner-OOF sensitivity-target rule gives D observed F1 0.9402, sensitivity 0.9108 and specificity 0.9897. The 0.90 sensitivity target was met in 6/10 observed outer folds. This secondary operating point does not replace the primary ranking.

D patient-weighted F1 is 0.9407. F1 is 0.9376 with recorded quality flags and 0.9397 without flags. Flag absence does not establish clean signals; subgroup composition may differ.

## Next improvements and rationale

1. Repeat seeds and hold capacity and update budgets fixed in augmentation ablations to test whether tiny observed gains are stable. 2. Tune capacity, learning rate and epoch budget in inner folds: 11 final boundary selections suggest the limited block search may miss useful settings. 3. Ablate feature families and augmentation types to identify the sources of robustness. 4. Test lead-specific and correlated loss plus external data to assess transfer beyond the simulated stresses.

The one-fold, three-epoch pilot is excluded from ranking. No TabM ridge-filling experiment or external validation was completed.

Evidence: results/tabm/output_data/summary.csv, selected_hyperparameters.csv, paired_method_differences.csv, patient_weighted_metrics_summary.csv, quality_subgroups_summary.csv; run_config.json and reports/completion.json. Rebuild: python scripts/build_model_observations.py.

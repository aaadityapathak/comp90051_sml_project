# Logistic regression and TabM model observations

COMP90051 Statistical Machine Learning | Completed experiment review | 13 October 2026

TabM arm D has the highest observed macro-F1 among the eight main model-arm combinations: 0.9395 (0.0122), compared with 0.9032 (0.0147) for logistic regression D. The key practical finding is that strong observed performance does not ensure robustness to missing leads. Training augmentation substantially protects TabM under severe lead loss; logistic regression shows a stronger tradeoff between recovering AFIB sensitivity and avoiding false positives.

## Main findings

| Finding | Evidence and implication |
| --- | --- |
| Better observed ranking and precision for TabM | D average precision 0.9320 versus 0.8622 for logistic D. D precision is 0.8612 versus 0.7238; sensitivity is lower (0.9166 versus 0.9530). |
| Missing leads expose a major weakness | TabM A sensitivity falls to 0.1813 with six missing leads; augmented D retains 0.8492. Observed-only TabM should not be selected from clean performance alone. |
| Augmentation is not a universal gain | Under six-lead loss, logistic D sensitivity exceeds B, but D macro-F1 is 0.8501 versus B 0.8605. Choose the error tradeoff explicitly. |
| Small clean gains need uncertainty analysis | B to D gains are much smaller on observed data than TabM gains under lead loss. Fold SD is descriptive variability, not a confidence interval or significance test. |

## Scope and status

Both runs cover all ten outer patient folds. TabM completed 80 outer fold-variant checkpoints, 891 inner fits and 80 outer refits (971 total fits). Logistic regression completed nine variants including the separate ridge experiment, with 933 converged fits. Histogram gradient boosting remains pending, so the planned three-model comparison is incomplete.

Tables report record-level outer-fold means unless stated otherwise. Parentheses contain sample SD across ten outer folds. Stress repeats are averaged within each fold before averaging folds. All main tables use the prespecified 0.50 decision threshold. No p-values or claims of statistical significance are made.

## Study design and fair interpretation

The shared prepared package contains 21,799 ECGs from 18,869 patients and 1,514 AFIB-labelled records (6.95%). AFIB means presence in source SCP annotations, including likelihood zero. Models use 504 lead fields; identifiers, quality flags and 27 Global fields stay outside predictors. Enhanced construction adds 63 fields before encoding and missingness handling.

| Arm | Predictors | Training views |
| --- | --- | --- |
| A | Original lead measurements | Observed only |
| B | Original plus enhanced fields | Observed only |
| C | Original lead measurements | Observed plus one corrupted copy per ECG |
| D | Original plus enhanced fields | Observed plus one corrupted copy per ECG |

Training copies use three missing leads, additive noise 0.10, gain 0.10, or the combined condition. The augmentation stream is separate from evaluation. Evaluation includes observed data once and ten stress scenarios with three repeats each. Noise is scaled by training-field IQR; gain is exp(sigma times a standard-normal draw) per lead. These are measurement-space stress tests, not calibrated ECG waveform artefacts.

| Protocol | Logistic regression | TabM |
| --- | --- | --- |
| Validation | 10 outer and 3 inner patient folds | Same patient folds and prepared data hashes |
| Selection | Mean inner observed macro-F1 at 0.5 | Same criterion; fewer blocks on exact ties |
| Tuning | C 0.01, 0.1, 1; one endpoint refinement | Blocks 2, 3, 4; one endpoint refinement |
| Fit | Weighted L2 logistic loss; L-BFGS; max 2,000 iterations | Width 128; 16 members; dropout 0.1; AdamW; 100 epochs |
| Runtime | CPU numerical optimizer | CUDA float32; deterministic algorithms; batch 256 |
| Secondary threshold | Selected using inner OOF sensitivity target 0.90 | Same rule; fixed across outer stresses |

Learned filling, scaling, encoding and model selection use training patients only. The saved manifests match across models, as do all 2,480 saved test-condition perturbation audits across 80 shared fold-variants. The shared perturbation implementation keys draws by ECG ID, repeat and stream. Matched folds and stresses support paired descriptive comparisons, but overlap among outer training sets means the ten scores are not ten independent datasets. Differences in model capacity, tuning parameterization and training budget prevent attributing every difference to architecture alone.

## Observed performance across the four main arms

| Model and arm | Macro F1 (SD) | Sensitivity | Specificity | Precision | Brier |
| --- | --- | --- | --- | --- | --- |
| Logistic regression A | 0.8950 (0.0171) | 0.9570 | 0.9691 | 0.6988 | 0.0258 |
| Logistic regression B | 0.9003 (0.0151) | 0.9477 | 0.9721 | 0.7188 | 0.0237 |
| Logistic regression C | 0.8984 (0.0181) | 0.9541 | 0.9707 | 0.7095 | 0.0248 |
| Logistic regression D | 0.9032 (0.0147) | 0.9530 | 0.9727 | 0.7238 | 0.0234 |
| TabM A | 0.9321 (0.0126) | 0.9108 | 0.9871 | 0.8416 | 0.0142 |
| TabM B | 0.9389 (0.0137) | 0.9199 | 0.9884 | 0.8561 | 0.0133 |
| TabM C | 0.9338 (0.0121) | 0.9134 | 0.9874 | 0.8450 | 0.0140 |
| TabM D | 0.9395 (0.0122) | 0.9166 | 0.9889 | 0.8612 | 0.0129 |

Source: each model's output_data/summary.csv; observed, fixed_0.50. Higher is better except Brier.

## Ranking and probability quality

| Model and arm | AUROC | Average precision | Balanced accuracy |
| --- | --- | --- | --- |
| Logistic regression A | 0.9863 | 0.8420 | 0.9630 |
| Logistic regression D | 0.9870 | 0.8622 | 0.9628 |
| TabM A | 0.9920 | 0.9226 | 0.9489 |
| TabM D | 0.9923 | 0.9320 | 0.9528 |

TabM D improves the observed mean macro-F1 over logistic D by 0.0363, and lowers Brier by 0.0105. Its sensitivity is 0.0364 lower. This is a shift in the operating tradeoff as well as a gain in ranking metrics; macro-F1 alone does not describe which errors matter.

The training-majority baseline has mean accuracy 0.9305 and macro-F1 0.4820, with zero sensitivity. High accuracy alone is therefore inadequate for this imbalanced task. Average precision and class-specific errors remain central when assessing the benefit over a trivial baseline.

Observed means slightly favor TabM D over B, but the effect is small relative to fold variability. Do not select D solely because it has the largest clean mean. Its more persuasive advantage is robustness under lead loss, which must be weighed against runtime and the intended operating threshold.

## Logistic regression observations

Enhanced features improve observed macro-F1 by +0.0053 (paired-fold SD 0.0070) for B minus A. Augmentation adds +0.0028 (SD 0.0043) for D minus B. The improvements are modest; B slightly reduces sensitivity while improving specificity and precision. D offers the highest observed mean macro-F1 among the logistic main arms.

| Condition | Arm | Macro F1 | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| 3 missing leads | A | 0.8915 | 0.8201 | 0.9825 |
| 3 missing leads | B | 0.8919 | 0.8895 | 0.9751 |
| 3 missing leads | C | 0.8807 | 0.9376 | 0.9654 |
| 3 missing leads | D | 0.8870 | 0.9397 | 0.9678 |
| 6 missing leads | A | 0.7346 | 0.3613 | 0.9937 |
| 6 missing leads | B | 0.8605 | 0.7837 | 0.9754 |
| 6 missing leads | C | 0.8339 | 0.9031 | 0.9488 |
| 6 missing leads | D | 0.8501 | 0.9206 | 0.9541 |

Original observed-only A is fragile under six missing leads: sensitivity falls from 0.9570 to 0.3613. Enhanced B recovers sensitivity to 0.7837. This suggests that enhanced measurements help the linear classifier cope with incomplete inputs, although the aggregate comparison does not identify which constructed features are responsible.

Augmented D reaches sensitivity 0.9206 with six missing leads, but specificity drops to 0.9541 and precision to 0.6009. D therefore has lower macro-F1 than B under this stress. At three missing leads and under combined stress, augmentation likewise does not maximize logistic macro-F1 at 0.5. The positive class becomes easier to detect at the cost of more false positives.

## Optimization and selection

The saved completion report records all 933 fits as converged, with maximum gradient 1e-05. All 90 fold-variant selections finish at the middle of their final grid. This supports numerical completion within the specified search, not global optimality over all model or preprocessing choices.

| Arm | Selected C across ten outer folds |
| --- | --- |
| A | 0.01: 1 folds; 0.1: 9 folds |
| B | 0.1: 10 folds |
| C | 0.1: 7 folds; 1.0: 3 folds |
| D | 0.1: 9 folds; 1.0: 1 folds |

## TabM observations and training cost

Enhanced B improves observed macro-F1 over A by +0.0067 (paired-fold SD 0.0076). D improves over B by only +0.0006 (SD 0.0040) on observed data. The current results favor enhanced features for clean performance; evidence for augmentation comes mainly from stress behavior.

| Condition | Arm | Macro F1 | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| 3 missing leads | A | 0.8999 | 0.7402 | 0.9940 |
| 3 missing leads | B | 0.9145 | 0.7880 | 0.9935 |
| 3 missing leads | C | 0.9266 | 0.8979 | 0.9865 |
| 3 missing leads | D | 0.9299 | 0.9020 | 0.9872 |
| 6 missing leads | A | 0.6358 | 0.1813 | 0.9989 |
| 6 missing leads | B | 0.7513 | 0.3697 | 0.9979 |
| 6 missing leads | C | 0.9085 | 0.8281 | 0.9875 |
| 6 missing leads | D | 0.9142 | 0.8492 | 0.9872 |

Six missing leads reduce A macro-F1 to 0.6358 and B to 0.7513. Augmented C and D retain 0.9085 and 0.9142. High specificity in A/B coexists with very poor sensitivity, so a stable-looking overall accuracy would hide missed AFIB records. Augmentation addresses this behavior in the tested setting.

| Arm | Selected blocks across ten outer folds |
| --- | --- |
| A | 2: 3; 3: 6; 4: 1 |
| B | 1: 1; 2: 6; 3: 1; 4: 1; 5: 1 |
| C | 2: 2; 3: 2; 4: 4; 5: 2 |
| D | 1: 1; 2: 5; 3: 1; 4: 3 |

Across all 80 TabM fold-variants, 57 trigger endpoint refinement and 69 select a middle value. Endpoint selections motivate an inner-validation study of capacity; they do not authorize expanding the grid after inspecting outer test outcomes.

Completed-fold wall times sum to 13.07 hours. This is summed checkpoint work, not end-to-end elapsed time or GPU time, and parallel scheduling prevents direct interpretation as user waiting time. Training histories contain training loss; falling training loss alone does not establish generalization or an optimal epoch count.

## Complete stress profile at the primary threshold

### Logistic regression macro F1

| Scenario | A | B | C | D |
| --- | --- | --- | --- | --- |
| Observed | 0.8950 | 0.9003 | 0.8984 | 0.9032 |
| 1 missing lead | 0.8999 | 0.8991 | 0.8937 | 0.8976 |
| 3 missing leads | 0.8915 | 0.8919 | 0.8807 | 0.8870 |
| 6 missing leads | 0.7346 | 0.8605 | 0.8339 | 0.8501 |
| Noise 0.05 | 0.8953 | 0.9010 | 0.8991 | 0.9031 |
| Noise 0.10 | 0.8946 | 0.9010 | 0.8973 | 0.9032 |
| Noise 0.20 | 0.8905 | 0.9014 | 0.8915 | 0.9021 |
| Gain 0.05 | 0.8951 | 0.9002 | 0.8984 | 0.9029 |
| Gain 0.10 | 0.8947 | 0.9005 | 0.8975 | 0.9026 |
| Gain 0.20 | 0.8934 | 0.8987 | 0.8973 | 0.9009 |
| 3 leads + noise 0.10 | 0.8906 | 0.8908 | 0.8799 | 0.8878 |

### TabM macro F1

| Scenario | A | B | C | D |
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

Source: output_data/summary.csv, fixed_0.50. Noise doses are fractions of the training IQR; gain doses are lognormal sigma values. Three repeats per stress are averaged within folds. All rows use the same ten outer folds; the full comparison CSV retains SD and defined-fold counts.

Lead loss is the largest threat to observed-only TabM. Additive noise and gain perturbations up to the tested 0.20 doses have much smaller macro-F1 effects. Slight score increases under a stress are possible because a fixed threshold changes the balance of errors; they do not show that corruption improves measurement quality. Do not average all scenarios into one deployment score without a justified distribution of real operating conditions.

## Controls and what they establish

Availability controls preserve original predictors and 63 constructed missingness indicators while zeroing the constructed numerical values. Duplicate controls repeat observed rows with half weights. They address different questions: whether enhanced numerical values add information, and whether a change can be explained by repeating examples rather than corrupting them.

| Model | Comparison | Observed difference (SD) | 6 leads difference (SD) |
| --- | --- | --- | --- |
| Logistic regression | B minus availability_observed | +0.0053 (0.0070) | +0.1326 (0.0331) |
| Logistic regression | D minus availability_augmented | +0.0048 (0.0064) | +0.0158 (0.0105) |
| Logistic regression | duplicate_original minus A | +0.0000 (0.0000) | +0.0001 (0.0003) |
| Logistic regression | duplicate_enhanced minus B | +0.0000 (0.0000) | +0.0000 (0.0002) |
| TabM | B minus availability_observed | +0.0057 (0.0089) | +0.1162 (0.0335) |
| TabM | D minus availability_augmented | +0.0055 (0.0076) | +0.0060 (0.0062) |
| TabM | duplicate_original minus A | -0.0004 (0.0037) | -0.0008 (0.0296) |
| TabM | duplicate_enhanced minus B | +0.0008 (0.0036) | +0.0042 (0.0210) |

Differences are paired within outer folds for macro-F1 at 0.5. Positive values favor the first method. Source: output_data/paired_method_differences.csv. These are descriptive differences, with no multiple-comparison correction or independent significance test.

Enhanced B outperforms its availability control in both models, especially under six-lead loss. This supports useful information in the constructed numerical values beyond the control's missingness representation. It does not isolate any individual feature family or prove a causal physiological mechanism; feature-family ablations remain necessary.

Logistic duplicate controls reproduce observed macro-F1 exactly and differ only slightly under stress. Half-weight duplication preserves the logistic objective, consistent with their role as a check against sample-count effects. TabM duplicates have small nonzero differences. Repeated rows alter minibatch scheduling and the number of optimizer updates per epoch; neural training need not reproduce the same fitted function even with equal total sample weight.

TabM augmentation gains under severe lead loss are much larger than the corresponding duplicate control changes. A stronger future control would match optimizer update count and compute budget as well as total example weight, with several training seeds. Clean augmentation gains are small enough that seed and optimization variability deserve particular attention.

## Threshold choice and probability calibration

| Model and arm | F1 at 0.5 | F1 at inner rule | Sensitivity at inner rule | Specificity at inner rule |
| --- | --- | --- | --- | --- |
| Logistic regression A | 0.8950 | 0.9111 | 0.9112 | 0.9799 |
| Logistic regression B | 0.9003 | 0.9152 | 0.9125 | 0.9812 |
| Logistic regression C | 0.8984 | 0.9134 | 0.9175 | 0.9800 |
| Logistic regression D | 0.9032 | 0.9185 | 0.9153 | 0.9820 |
| TabM A | 0.9321 | 0.9325 | 0.9081 | 0.9874 |
| TabM B | 0.9389 | 0.9393 | 0.9106 | 0.9894 |
| TabM C | 0.9338 | 0.9357 | 0.9093 | 0.9884 |
| TabM D | 0.9395 | 0.9402 | 0.9108 | 0.9897 |

The secondary rule maximizes specificity among inner OOF thresholds reaching sensitivity at least 0.90. It is then frozen before scoring the outer fold and every stress. The target applies to inner OOF predictions; it is not a guaranteed outer sensitivity, per-fold minimum, or stress guarantee.

Logistic D moves from observed macro-F1 0.9032 at 0.5 to 0.9185 under the inner rule, while sensitivity moves from 0.9530 to 0.9153. The alternate threshold reduces false positives by accepting a lower recall level. TabM's observed threshold adjustment is smaller. This reinforces that model comparison should include matched operating objectives, rather than only the same numerical threshold.

## Calibration requires a separate assessment

Observed D Brier is 0.0234 for logistic regression and 0.0129 for TabM. Lower Brier indicates better mean squared probability error on this population, but does not by itself establish calibration across probability ranges or deployment prevalences. Class weighting changes the training objective; predicted scores should not automatically be interpreted as calibrated prevalence estimates.

Inspect saved ten-bin reliability tables with bin counts and AFIB counts, including sparse high-score bins. Add calibration intercept and slope, patient-resampled uncertainty and stress-specific reliability plots. Any recalibrator must be learned using training-only predictions inside the validation design. Test threshold policies at prespecified sensitivity or false-positive budgets and report the fraction of outer folds meeting the target.

The same held-out patients appear in several stress scenarios. Repeats, scenarios, ECGs from one patient and overlapping training folds are not independent replicates. Patient-cluster bootstrap intervals on aligned OOF predictions can describe test-patient uncertainty conditional on the fitted models; repeated complete experiments are needed to assess training and split variability.

## Patient weighting quality groups and filling

| Model and arm | Record macro F1 | Patient weighted macro F1 |
| --- | --- | --- |
| Logistic regression A | 0.8950 | 0.8956 |
| Logistic regression D | 0.9032 | 0.9028 |
| TabM A | 0.9321 | 0.9337 |
| TabM D | 0.9395 | 0.9407 |

Patient weighting gives each patient total weight one across their ECG records. The observed A/D results remain close to the record-level means, suggesting repeat ECGs are not driving the main observed ranking. This is a sensitivity check, not external validation.

| Model | Source quality group | Macro F1 | SD |
| --- | --- | --- | --- |
| Logistic regression D | no recorded quality flag | 0.9038 | 0.0174 |
| Logistic regression D | recorded quality flag | 0.8996 | 0.0285 |
| TabM D | no recorded quality flag | 0.9397 | 0.0147 |
| TabM D | recorded quality flag | 0.9376 | 0.0190 |

The flagged group contains 5,010 ECGs and 431 AFIB records; the unflagged group contains 16,789 ECGs and 1,083 AFIB records. Flags are source annotations, not measured severity or proof of clean signals. Group differences may reflect prevalence and patient composition; demographic fairness has not been established.

### Separate ridge filling experiment

| Mask | Median normalized MAE | Ridge normalized MAE |
| --- | --- | --- |
| cells_0.10 | 0.7349 | 0.4311 |
| missing_1 | 0.7363 | 0.6403 |
| missing_3 | 0.7355 | 0.6640 |
| missing_6 | 0.7351 | 0.6969 |

Recovery MAE uses eligible masked continuous cells and training-IQR scaling. Ridge improves this reconstruction measure, with less advantage for whole-lead loss. Its downstream logistic macro-F1 is 0.8947 observed (A 0.8950) and 0.8974 with three missing leads (A 0.8915). Reconstruction improvement is not equivalent to classification improvement. Inspect coverage, excluded targets and per-feature errors before generalizing beyond the scored cells. No TabM ridge experiment was run.

## Priority improvements and further evaluation

| Priority | Evaluation | Decision it should support |
| --- | --- | --- |
| 1 | Complete histogram gradient boosting with the same frozen folds, stresses, thresholds and controls. | Finish the planned three-model comparison without changing the two completed baselines. |
| 1 | Align OOF prediction keys; report paired per-patient differences and uncertainty. Predeclare primary model contrasts and stress endpoints. | Determine which gains are stable enough to support model and arm selection; avoid treating many comparisons as independent evidence. |
| 1 | Compare prespecified sensitivity targets and false-positive budgets; inspect reliability and calibrate only within training partitions. | Separate ranking improvements from threshold and probability-scale effects. |
| 2 | Repeat TabM with several training seeds; assess repeated patient splits when compute permits. | Quantify whether small clean gains exceed seed and split variability. |
| 2 | Ablate enhanced feature families and individual augmentation types, matching duplicate controls, update count and compute. | Identify the source of robustness and the augmentation that costs clean precision or runtime. |
| 2 | Test lead identities, correlated missingness, heavier doses and naturally poor-quality subgroups; validate on a genuinely external dataset. | Measure transfer beyond synthetic random lead loss and this source population. |
| 3 | Investigate TabM capacity, epoch count, learning rate, numerical embeddings or alternative preprocessing through inner validation only. | Explore better accuracy per unit compute as a new experiment with new provenance. |
| 3 | For ridge filling, analyze coverage and per-feature errors, then test downstream usefulness under additional stresses. | Decide whether the reconstruction improvement justifies complexity and cost. |

## Faster runs without changing the scientific experiment

Reuse complete, signature-matched inner candidates and outer checkpoints; regenerate reports from saved predictions instead of retraining. Profile preprocessing, tensor transfer, training and serialization separately before choosing a worker count. Small isolated pilots should measure throughput and GPU memory under the target hardware. Reusing cached prepared matrices can save CPU work only if the cache key includes fold, variant, input hashes and preprocessing configuration.

Changes to batch size, epoch count, mixed precision, early stopping, architecture or optimizer may change the result and require a new output directory and configuration. A reduced pilot cannot replace the completed fixed-protocol evidence. The current TabM fit uses 100 epochs without early stopping; the saved checkpoint restores completed inference state, not a partially completed optimizer trajectory.

## Fast refresh and reproducible environments

For the completed TabM experiment, refresh tables and figures and independently validate the full saved prediction bundle without fitting any models:

```powershell
.venv/Scripts/python algorithms/tabm/run_full_pipeline.py --report-only --skip-notebook
```

This command requires the complete plan to pass verification and writes reports/validation.json. Omit --skip-notebook to refresh notebook displays too. The quicker run_experiment.py --phase report command only rebuilds tables and figures; it does not independently recompute every prediction metric. Neither command publishes results unless the separate --publish option is explicitly used.

### Preparation reuse in future parallel runs

The parallel runner separates scheduling from the serial scientific protocol. Its inner mode groups all missing block candidates for one inner split into one worker job, so the prepared arrays are built once and reused across those candidates. A bounded pool controls total worker count; there are no nested pools. Inner mode releases the prepared split after its group finishes, while outer mode uses the serial runner's cache of up to three inner splits per worker.

```powershell
.venv/Scripts/python algorithms/tabm/run_parallel.py --device cuda --workers 2 --parallel-level inner --dry-run
```

The dry run inspects scheduling without data loading, CUDA access, model fitting or output writes. Actual parallel training defaults to results/tabm_parallel/, separate from the published serial run. Preparation reuse removes repeated CPU work by construction; no end-to-end training speedup is claimed until it is measured on the target GPU, dataset and worker configuration.

### Keep the recorded environments separate

| Recorded package | Logistic regression | TabM |
| --- | --- | --- |
| python | 3.12.15 | 3.12.10 |
| numpy | 2.5.3 | 2.5.1 |
| pandas | 3.0.6 | 3.0.6 |
| scipy | 1.18.1 | Not recorded |
| torch | Not recorded | 2.7.1+cu128 |
| tabm | Not recorded | 0.0.3 |

The Python and NumPy versions differ between the historical runs. Use algorithms/logistic_regression/requirements-recorded.txt for the recorded logistic packages and algorithms/tabm/requirements.txt with the documented CUDA PyTorch wheel for TabM. The broad root requirements are convenient development requirements, not a historical environment lock. Install each model in its own environment when reproducing fitted results. Document generation uses the separate requirements-docs.txt and does not require either training stack.

## Reproduction evidence and source map

The document builder reads the saved result CSVs and JSON configurations; it performs no fitting and never loads pickles. It writes the Word and Markdown versions from one content definition and records input SHA-256 hashes beside the comparison tables. Original model result files remain the numerical source of truth.

| Evidence | Repository location and role |
| --- | --- |
| Experiment identity | results/&lt;model&gt;/run_config.json: data hashes, software versions, selected scope and code hashes |
| Completion | results/&lt;model&gt;/reports/completion.json: counts and completed plan scope |
| Primary results | results/&lt;model&gt;/output_data/summary.csv and metrics.csv: fold-mean summaries and per-condition metrics |
| Controls and stresses | output_data/paired_method_differences.csv and paired_stress_drops.csv: within-model paired descriptive effects |
| Predictions and audit | output_data/folds/: inner OOF, outer predictions, selection and perturbation audits |
| Secondary analyses | output_data/patient_weighted_metrics_summary.csv, quality_subgroups*.csv, reliability.csv and logistic recovery*.csv |
| Generated comparison | results/comparison/output_data/observations_main_summary.csv, observations_method_differences.csv and observations_input_manifest.json |
| Stress matching | results/comparison/output_data/observations_pairing_validation.json: repeatable comparison of all common test perturbation audits |
| Prepared data reproduction | scripts/verify_preprocessing.py and results/comparison/reports/preprocessing_validation.json: rebuild and compare raw-to-prepared data hashes |
| Generator | scripts/build_model_observations.py and requirements-docs.txt |

### Commands from the repository root

```powershell
python -m pip install -r requirements-docs.txt
python scripts/build_model_observations.py
python scripts/verify_preprocessing.py
python algorithms/logistic_regression/check_results.py
python algorithms/tabm/check_results.py
```

The builder needs only python-docx plus Python's standard library and saved result tables. Numerical checkers additionally use the model environments and prepared data. Regenerating this document is distinct from independently validating all predictions and from retraining a model. See the algorithm READMEs for exact run and resume commands.

The saved preprocessing validation rebuilds all four prepared-data files from the source CSVs and matches their bytes and complete manifest to the existing package and both recorded model configurations. The existing processed package is not modified by that check.

For a full fresh reproduction, retrieve the cited source versions, verify the recorded SHA-256 hashes, rebuild the prepared package and use the recorded per-model environments. Preserve the historical configuration and source-provenance snapshots. Hardware, CUDA and library changes can alter neural numerical results despite fixed seeds; claim exact equality only after comparing outputs in a matched environment. Compare future experiments in separate folders and document every intentional protocol change.

Data attribution: PTB-XL+ 1.0.1 (https://physionet.org/content/ptb-xl-plus/1.0.1/) and PTB-XL 1.0.3 (https://physionet.org/content/ptb-xl/1.0.3/), CC BY 4.0. This is retrospective classification and controlled stress evaluation; it does not establish clinical diagnostic benefit, deployment reliability or external generalization.

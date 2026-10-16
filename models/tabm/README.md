# TabM

Train the project's TabM neural ensemble and reproduce its nested tuning, stress
evaluation and controls. Read the generated completion report for the actual run
status; the implementation alone does not mean the full experiment is complete.

## Inspect the completed run quickly

Regenerate figures using saved tables only, without data or retraining:

```powershell
python models/tabm/plot_results.py
```

The progress, robustness, calibration and training-curve PNGs are written to
`results/figures/tabm/`. Use `--destination` for another folder. This adapter
calls the same plotting functions as `reporting.py` and preserves the original
saved reports.

The saved experiment can be inspected without retraining. From the repository
root, regenerate its tables/figures and run the independent prediction checker:

```powershell
python models/tabm/run_full_pipeline.py --report-only --skip-notebook
```

This strict command requires the complete plan to pass verification, writes
`reports/validation.json`, and does not train models. Omit `--skip-notebook` to
also refresh notebook displays. For a quick table/figure refresh alone, use
`run_experiment.py --phase report`; it does not independently recheck every
prediction. Neither command publishes unless `--publish` is explicitly added
to `run_full_pipeline.py`.

| Module | Responsibility |
| --- | --- |
| `model.py` | Official TabM model, deterministic fit, probabilities, checkpoints |
| `data.py` | Fold isolation, preprocessing, controls, patient summaries |
| `run_experiment.py` | Fixed serial protocol, tuning, atomic fit completion, resume |
| `run_parallel.py` / `parallel_runtime.py` | Bounded scheduling and isolated worker resources |
| `reporting.py` / `check_results.py` | Saved-result aggregation and independent validation |
| `source_verification.py` | Verify the recorded path-only source migration |
| `run_full_pipeline.py` | Compose training/reporting, verification and optional notebook refresh |
| `build_notebook.py` | Build the notebook source; rebuilding clears notebook outputs |

## Install and run

From the repository root with Python 3.12:

```powershell
python -m venv ../afib_envs/tabm
../afib_envs/tabm/Scripts/Activate.ps1
# NVIDIA GPU build used for the recorded runs:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r models/tabm/requirements.txt
python -m ipykernel install --sys-prefix --name tabm --display-name 'Python 3 (TabM)'
python models/tabm/run_experiment.py --device cuda --output results/tabm_reproduction
```

For a CPU machine, install the requirements directly and use `--device cpu`.
The verified external prepared-data package must exist first. Its default
location is `../afib_data/processed`; set `AFIB_PROCESSED_DIR` to override it. The default serial runner
trains sequentially on one GPU; the optional parallel runner below schedules
independent fits concurrently. Completed candidate fits are cached; rerun the
identical command to resume. An interrupted fit restarts that fit from epoch one. Changes to data,
training code or configuration require a different output directory.

The notebook [05_tabm_tuning.ipynb](../../notebooks/05_tabm_tuning.ipynb) contains
the same runnable tuning workflow and result displays. `RUN_TRAINING=False`
is the default; explicitly set it to `True` to start or resume training. Inspection
leaves saved reports unchanged; set `TABM_REFRESH_REPORTS=1` to regenerate them. Open Jupyter with
`python -m jupyterlab notebooks/05_tabm_tuning.ipynb` and select
the registered `Python 3 (TabM)` kernel.

## Optional parallel runner

`run_parallel.py` schedules future runs using the same training, tuning and
evaluation functions as the serial runner. `parallel_runtime.py` provides its
spawned worker processes. The training commands below are for a future run;
adding the implementation does not launch training or establish any results.
Parallel runs default to **`results/tabm_parallel/`**, separate from the existing
serial output. The parallel runner rejects `--output results/tabm`.

Inspect a plan first without loading models or data, accessing CUDA, spawning
workers or writing output:

```powershell
python models/tabm/run_parallel.py --device cuda --workers 2 --dry-run
```

For a future run on one GPU:

```powershell
python models/tabm/run_parallel.py --device cuda --workers 2
# Alternatively, parallelize inner splits and reuse preparation across their candidates:
python models/tabm/run_parallel.py --device cuda --workers 2 --parallel-level inner
```

The defaults are two workers, `--parallel-level outer`, `--device auto`, and one
CPU math thread per worker. Outer mode runs independent outer-fold/variant jobs
concurrently; each job handles its own tuning, refit and evaluation. Inner mode
runs inner splits concurrently, keeping each split's missing candidates in one
worker so their prepared arrays are built once and reused. It waits for their scores,
runs any required endpoint refinements, then runs outer refit/evaluation jobs.
Both modes use one bounded process pool, with no nested pools multiplying the
worker count. Inner mode caches only one prepared split per worker and releases
it after its candidates finish; outer mode uses the serial runner's cache of
up to three inner splits per worker. Run one command at a time for a given output directory; the parent
holds its output lock until all workers have stopped.

For multiple GPUs or CPU execution:

```powershell
# Assign workers to GPUs round-robin:
python models/tabm/run_parallel.py --devices cuda:0 cuda:1 --workers 2 --output results/tabm_parallel_multi_gpu
python models/tabm/run_parallel.py --device cpu --workers 4 --threads-per-worker 1 --output results/tabm_parallel_cpu
```

Each CUDA worker defaults to a PyTorch allocator allowance of
`0.8 / workers sharing its GPU`; `--gpu-memory-fraction` overrides the allowance
per worker. This limit does not include every GPU allocation and cannot
guarantee sufficient memory or a speedup. On a 4 GB laptop GPU, start future
trials with two workers and reduce to one if memory is insufficient. CPU math
thread limits prevent every worker from independently claiming all CPU cores.

Resume using `run_parallel.py` and the same output directory. Completed fits
are verified and reused; an interrupted fit restarts from epoch one. Worker
count, allocator fraction and parallel level can change on resume, so reducing
workers after a memory failure preserves completed work. Scientific settings,
data, source hashes, assigned devices and threads per worker are immutable for
that output. Use a new output directory when changing those settings. Serial
and parallel run configurations are not interchangeable.

The parallel output has the same predictions, checkpoints and reports layout
described below, plus `execution.json` for the current launch settings,
`status.json` for aggregate progress, `worker_status/worker-N.json` for worker
progress, and `logs/worker-N.log` plus launch records in `logs/`. The frozen
`run_config.json` records the scientific configuration, source hashes and device
and thread settings. Successful training runs automatically generate reports
and run the independent checker, saving `reports/validation.json`. The parallel
runner does not execute the notebook or publish results.

## Fast development runs and reproducibility limits

Use a separate small pilot to check installation and execution before a new
full experiment. This example fits a single candidate on three inner splits
for one variant/fold at two epochs; it is **not a scientific result**:

```powershell
python models/tabm/run_parallel.py --phase tune --blocks 2 --variants A --outer-folds 0 --epochs 2 --repeats 1 --device cpu --workers 1 --output results/tabm_parallel/pilots/smoke
```

For a fresh full serial reproduction, add
`--output results/tabm_reproduction` to `run_experiment.py --device cuda`.
Use the exact saved `run_config.json` versions and verified processed data when
targeting the recorded run. Existing hashes and completion markers are immutable;
do not edit them to force a resume after changing training code or dependencies.

The recorded model, preprocessing and serial protocol remain unchanged by the
scheduling cleanup. Per-fit seeds, float32, deterministic algorithms and source
hashes support reproducibility, but do not promise identical floating-point
results on different hardware, devices or PyTorch versions. In this model,
available GPU memory determines whether training tensors and the shuffle
generator live on GPU or CPU. A different `data_device` can therefore change the
minibatch order even with the same seed. Each fit audit records `data_device`;
match it and the device/runtime when assessing an exact reproduction. Sharing a
GPU or changing worker counts can alter that choice. The serial and parallel
runs must remain separate experiments.

The split grouping removes repeated preprocessing within a candidate group;
GPU throughput and total speedup have not been benchmarked. More workers can
be slower or exceed available RAM/VRAM. Compare pilot wall time, memory use and
fit audit timings before committing to a full parallel run. Reducing epochs,
changing batches, using mixed precision or adding early stopping would change
the recorded protocol and needs a separately labelled experiment.

## Individual parameter scripts

Each script fits one setting on all three inner splits of all ten outer folds
and all eight variants, sharing the full experiment cache:

```powershell
python models/tabm/run_blocks_2.py
python models/tabm/run_blocks_3.py
python models/tabm/run_blocks_4.py
python models/tabm/run_experiment.py --phase run
```

The final command reuses the initial candidates, performs any permitted endpoint
refinement, selects parameters, refits and evaluates. Running just the final
command also runs missing candidate fits, giving outer-fold results sooner.
Do not launch concurrent trainers against the same output folder.

## Fixed protocol

| Setting | Value |
| --- | --- |
| Initial tuning grid | Blocks 2, 3, 4 |
| One optional endpoint refinement | 1, 2, 3 if 2 wins; 3, 4, 5 if 4 wins |
| Width / ensemble members | 128 / 16 |
| Dropout / numerical embeddings | 0.1 / none |
| Optimiser | AdamW, learning rate 0.0003, weight decay 0.0001 |
| Batch size / epochs | 256 / 100, no early stopping |
| Selection | Mean observed inner macro-F1, simpler exact ties |
| Validation | Shared manual 10 outer x 3 inner patient folds |
| Primary threshold | 0.5 |
| Exploratory threshold | Maximum specificity at inner OOF sensitivity >= 0.90 |
| Stress evaluation | Observed once; 10 corruption conditions x 3 matched repeats |

Arms A/B use original/enhanced features with observed training. C/D use the same
features with one corrupted training copy per ECG. Availability controls retain
raw predictors and missingness/categorical indicators while zeroing derived
numerical values; they are not pure missingness-only models. Duplicate
controls repeat observed rows with half weights. All learned preprocessing uses
current training patients only. Class/view weights are applied once, after mean
member BCE. Probabilities are the mean of member sigmoid probabilities.

Settings follow `docs/Project_Plan.docx`; the discrete refinement grids above make
its one-refinement rule explicit. Seeds, dependencies, source hashes, training
losses and elapsed times are recorded. Full float32 and deterministic PyTorch
algorithms are used. At least 800 fits are needed including controls, up to 1,040
with cached refinements. This is a substantial GPU run.

## Output locations

```text
results/tabm/
  run_config.json           # frozen experiment identity
  status.json               # current progress
  output_data/
    tuning/                 # each setting/split's predictions, history, audit
    folds/                  # selected inner OOF and outer stress predictions
  reports/                  # result tables, figures, report and completion status
  checkpoints/              # local fitted models and preprocessing
  pilots/                   # clearly separated engineering runs
  logs/                     # local live process logs
  source_provenance/        # preserved source evidence for the path migration
```

Output data and reports here are committed to GitHub. Binary resume checkpoints
and live logs remain local. No source data is copied into this folder. Full
experiments and short pilots cannot silently overwrite one another.

The former `models/tabm/` directory has moved intact to `results/tabm/`.
The submission preserves configurations, predictions and histories. Binary
resume checkpoints are local training artifacts and are excluded from it.
The runner and independent checker validate the recorded source hash against
`source_provenance/layout_v1/`, then compare the current runner's syntax with
that source, allowing only the exact output-path changes and compatibility
call. Training code, data, runtime and configuration changes still fail the
signature guard. A local run with its checkpoints can resume without rewriting
its configuration; use a fresh output directory to reproduce the submitted run.

```powershell
python models/tabm/run_experiment.py --phase report
python models/tabm/check_results.py --allow-partial
# Require every requested result to be complete:
python models/tabm/check_results.py
```

Reports average repeats within each outer fold, then calculate mean and sample
SD across folds. They include confusion counts, macro-F1, sensitivity,
specificity, precision, NPV, balanced accuracy, AUROC, average precision, Brier
scores, reliability diagrams, patient-weighted summaries, quality subgroups,
paired stress losses and paired method differences. Partial results are labelled.
These are research measurements, not evidence of clinical benefit.

## Sources

- Gorishniy, Kotelnikov and Babenko (2025), [TabM, ICLR](https://arxiv.org/abs/2410.24210).
- [Official implementation](https://github.com/yandex-research/tabm), pinned `tabm==0.0.3`.
- [PyTorch CUDA install instructions](https://pytorch.org/get-started/previous-versions/).
- [PTB-XL+ 1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/) and
  [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), CC BY 4.0.

The original data-loading source is preserved in `historical_sources/`. Its
recorded hash is checked before the source guard accepts the two reviewed
external-directory changes. Training, patient folds, labels and metrics must
still match the preserved calculations. These snapshots are provenance, not
alternative runners.

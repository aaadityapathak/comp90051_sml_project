# TabM

Train the project's TabM neural ensemble and reproduce its nested tuning, stress
evaluation and controls. Read the generated completion report for the actual run
status; the implementation alone does not mean the full experiment is complete.

## Install and run

From the repository root with Python 3.12:

```powershell
python -m venv .venv
# NVIDIA GPU build used for the recorded runs:
.venv/Scripts/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r algorithms/tabm/requirements.txt
.venv/Scripts/python -m ipykernel install --prefix .venv --name tabm --display-name 'Python 3 (TabM)'
.venv/Scripts/python algorithms/tabm/run_experiment.py --device cuda
```

For a CPU machine, install the requirements directly and use `--device cpu`.
The verified `processed_data/` package must exist first. Training is sequential
on one GPU. Completed candidate fits are cached; rerun the identical command to
resume. An interrupted fit restarts that fit from epoch one. Changes to data,
training code or configuration require a different output directory.

The notebook [05_tabm_tuning.ipynb](notebooks/05_tabm_tuning.ipynb) contains
the same runnable tuning workflow and result displays. Set `RUN_TRAINING=False`
to refresh reports while another trainer is active. Open Jupyter with
`.venv/Scripts/python -m jupyterlab algorithms/tabm/notebooks/05_tabm_tuning.ipynb` and select
the registered `Python 3 (TabM)` kernel.

## Individual parameter scripts

Each script fits one setting on all three inner splits of all ten outer folds
and all eight variants, sharing the full experiment cache:

```powershell
.venv/Scripts/python algorithms/tabm/run_blocks_2.py
.venv/Scripts/python algorithms/tabm/run_blocks_3.py
.venv/Scripts/python algorithms/tabm/run_blocks_4.py
.venv/Scripts/python algorithms/tabm/run_experiment.py --phase run
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
| Validation | Shared manual 10 outer × 3 inner patient folds |
| Primary threshold | 0.5 |
| Exploratory threshold | Maximum specificity at inner OOF sensitivity ≥ 0.90 |
| Stress evaluation | Observed once; 10 corruption conditions × 3 matched repeats |

Arms A/B use original/enhanced features with observed training. C/D use the same
features with one corrupted training copy per ECG. Availability-only controls
retain missingness indicators while zeroing derived numerical values. Duplicate
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
Existing configurations, predictions, checkpoints and histories are preserved.
The runner and independent checker validate the recorded source hash against
`source_provenance/layout_v1/`, then compare the current runner's syntax with
that source, allowing only the exact output-path changes and compatibility
call. Training code, data, runtime and configuration changes still fail the
signature guard. Existing runs can resume without rewriting their configuration.

```powershell
.venv/Scripts/python algorithms/tabm/run_experiment.py --phase report
.venv/Scripts/python algorithms/tabm/check_results.py --allow-partial
# Require every requested result to be complete:
.venv/Scripts/python algorithms/tabm/check_results.py
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

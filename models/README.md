# Selected algorithms

Each folder owns one classifier. Logistic regression and TabM training are
implemented and their saved results are present. Histogram gradient boosting
is implemented, but its reported result bundle is missing from this checkout.

Each algorithm keeps its implementation, runner, result checker, dependencies
and README directly in its folder. All notebooks are in the repository's
`notebooks/` folder.
Save outputs under the matching `results/<algorithm>/` folder using
`output_data/`, `reports/`, `predictions/` and `pilots/` as described in the
[results layout](../results/README.md). Create optional folders when needed;
histogram gradient boosting writes its fold artifacts under `results/histogram_gradient_boosting/formal/`.

| Folder | Purpose | Tuning candidates |
| --- | --- | --- |
| [logistic_regression](logistic_regression/) | Linear baseline | C: 0.01, 0.1, 1 |
| [histogram_gradient_boosting](histogram_gradient_boosting/) | Shallow nonlinear model | Maximum depth: 2, 3, 4 |
| [tabm](tabm/) | Neural ensemble | Blocks: 2, 3, 4 |

## Shared input

Run the [preprocessing notebook](../notebooks/01_preprocessing.ipynb) first. All classifiers consume the same
`processed_data/` folder; they do not load source datasets directly.

The folder contains selected lead measurements, separate labels/IDs, patient
folds and a manifest. Use the shared loader to obtain finite model matrices for
the current split. It learns filling, scaling and category rules from that
split's training patients only. This avoids leakage during nested validation.

From the repository root:

```python
import sys
sys.path.insert(0, "src")
from afib.data_preparation import load_prepared_split

# One inner split for tuning. Use inner_fold=None for the outer refit/test.
split = load_prepared_split(
    "processed_data", outer_fold=0, inner_fold=0, arm="A"
)
X_train, y_train = split["X_train"], split["y_train"]
X_valid, y_valid = split["X_valid"], split["y_valid"]
sample_weight = split["sample_weight"]
```

Training matrices, labels and weights share a `(view, ecg_id)` index. Arms C/D
include both observed and corrupted views. Validation matrices keep one row per
ECG. Use `train_metadata` and `valid_metadata` for IDs; never add them to X.

## Common experiment rules

- Use the supplied 10 outer and 3 inner patient folds. Select configurations
  using mean inner observed-data macro-F1; outer patients remain held out.
- Run A (original/observed), B (enhanced/observed), C (original/augmented) and
  D (enhanced/augmented). Pass the returned sample weights to the training loss;
  do not apply class weighting a second time.
- Keep labels and identifiers out of predictors. Use metadata for alignment,
  patient checks and saving held-out predictions.
- Follow the [project plan](../docs/Project_Plan.docx) for stress tests, controls and reporting.
  Save model artifacts/results separately from `processed_data/`.

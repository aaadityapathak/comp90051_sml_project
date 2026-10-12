# Shared model comparison

Place combined `inner_oof_predictions.csv` and `outer_predictions.csv` in
`predictions/` before running `notebooks/02_noise_evaluation.ipynb`. This preserves
the notebook's existing requirement for all three classifiers. Per-algorithm
exports live in `results/<algorithm>/predictions/`.

The combined files have not yet been produced. Shared evaluation reports remain
in `reports/evaluation/`.

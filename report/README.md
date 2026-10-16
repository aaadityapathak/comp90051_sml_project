# Research report

[`COMP90051_Final_Report.pdf`](COMP90051_Final_Report.pdf) is the four-page
submission report: A4, single column, 10-point text, and 16 mm page margins.
The editable narrative and layout are in [`build_report.py`](build_report.py).

Run these commands from the repository root:

```bash
python -m pip install -r report/requirements.txt
python report/analyze_results.py
python report/build_report.py
```

The analysis reads retained model results and requires no downloaded or
processed dataset. It does not train models. It checks 13,602,576 saved LR,
TabM and retuned HGB prediction rows against the recorded metrics, verifies
all 360 retuned HGB inner scores and 40 depth choices, and then rebuilds the
tables and figure. The prediction check can take several minutes.

| Report item | Generated file | Code |
| --- | --- | --- |
| Table 1: LR, earlier HGB, retuned HGB and TabM macro-F1 | `results/side_by_side_macro_f1.csv` | `analyze_results.py` computes fold means and SDs; `build_report.py` formats the table |
| Table 2: observed sensitivity, specificity, AP and Brier | `results/model_summary.csv` | `analyze_results.py` computes summaries; `build_report.py` selects the stated arms |
| Figure 1: augmentation effect, observed and six hidden leads | `results/augmentation_effect.png` | The final plotting block in `analyze_results.py` reads paired fold contrasts |
| Numerical contrasts and audit | `results/paired_differences.csv`, `results/validation.json` | `analyze_results.py` |

Stress repeats are averaged within each outer fold before the ten-fold mean
and sample SD are calculated. Figure error bars show the SD of within-fold
differences, not a confidence interval. The plot is created entirely from
the retained result files; no external image or local absolute path is needed.

The latest HGB bundle is `results/histogram_gradient_boosting/retuned7_8_9/`.
The earlier HGB results remain a distinct comparator. Retuned HGB prediction
labels and ECG/fold keys agree with the LR result bundle, but this does not
independently authenticate the original dataset or establish identical HGB
stress draws. Historical HGB metrics are included as published. Verifying
saved outputs is separate from reproducing all training runs.

Depths 7, 8 and 9 were selected 10, 15 and 15 times. The report therefore
retains the unresolved course requirement that the middle candidate should
be selected most often. It does not alter selections to satisfy that rule.

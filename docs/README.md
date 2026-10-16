# Supplementary project notes

The current submission document is the
[four-page research report](../report/COMP90051_Final_Report.pdf).
These longer notes preserve the project's preparation decisions, exploratory
analysis, controls, and earlier detailed observations.

- `Project_Plan.docx`: study design and research questions.
- `Data_Insights.docx`: descriptive dataset findings.
- `Preprocessing.docx`: preparation and feature construction.
- `Logistic_Regression_Observations.*` and `TabM_Observations.*`: detailed model analysis.
- `Histogram_Gradient_Boosting_Observations.*` and `Model_Comparison.*`: earlier analysis, supplemented by the latest retuned HGB evidence in the final report.

The Word versions and saved observation tables reflect their original
creation dates; statements about then-missing HGB outputs are historical.
The delivered retuned predictions and tuning records now live in
`results/histogram_gradient_boosting/retuned7_8_9/`. Markdown updates in these
notes may postdate their corresponding Word files.

The original `scripts/build_model_observations.py` workflow remains available
with `requirements-docs.txt` for reproducing that earlier analysis. Use
`report/analyze_results.py` and `report/build_report.py` for the current report.

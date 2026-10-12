# TabM experiment outputs

This folder owns TabM prediction data, tuning histories and reports. Check
[completion.json](reports/completion.json) and [the report](reports/report.md)
for actual progress. Incomplete experiments and short pilots are labelled.

- [Training and tuning notebook](../../notebooks/05_tabm_tuning.ipynb)
- [Implementation and individual parameter scripts](../../algorithms/tabm/)
- [Output data](output_data/)
- [Reports](reports/)
- [Pilot runs](pilots/)

The full experiment uses 100 epochs per fit, all ten outer folds, three inner
folds, four main arms and four controls. Pilots test execution only and do not
replace the full experiment. Binary checkpoints and live logs remain local.

Run `python algorithms/tabm/run_full_pipeline.py --publish` from the repository
root to resume the full experiment, verify results, refresh the notebook and
publish completed results to the current Git branch. Publishing is opt-in and
only stages generated TabM outputs and its notebook.

AFIB classification project

Use Python 3.12 from this folder.
1. Install: python -m pip install -r requirements.txt
2. Check: python -m unittest discover -s tests -v
3. Open: python -m jupyter lab notebooks/01_preprocessing.ipynb
4. Follow its source-file instructions and run every cell.

Preprocessing produces only processed_data/. All algorithms use that folder
through the shared load_prepared_split function, which fits final transforms
on each training partition and returns ready-to-fit matrices.

README.md is the project entry point. docs/ contains the project plan, dataset
insights and preprocessing guide. algorithms/ has one folder per selected
classifier. Logistic regression and TabM are implemented; histogram gradient
boosting and the full group comparison remain pending. Check generated reports
for actual experiment completion.

Data sources and attribution:
https://physionet.org/content/ptb-xl-plus/1.0.1/
https://physionet.org/content/ptb-xl/1.0.3/
Both datasets are CC BY 4.0. Keep data local and exclude it from submission.

The repository includes code, notebooks, docs and preserved experiment results.
Keep data/, processed_data/, reports/ and local environments local. TabM binary
checkpoints and live logs are also excluded. See results/README.md for layout.

Logistic regression:
python algorithms/logistic_regression/run_experiment.py --workers 2
Open algorithms/logistic_regression/notebooks/04_logistic_regression_results.ipynb.
Check results/logistic_regression/reports/completion.json for completion status.

TabM:
Install algorithms/tabm/requirements.txt (see its README for GPU installation),
then run: python algorithms/tabm/run_experiment.py
Open algorithms/tabm/notebooks/05_tabm_tuning.ipynb for parameter runs and reports.
Published TabM outputs are in results/tabm/output_data/ and results/tabm/reports/.
Binary checkpoints and live logs remain local. Short pilots are separate from
the full 100-epoch nested experiment.


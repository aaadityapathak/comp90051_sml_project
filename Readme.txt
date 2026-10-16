COMP90051: AFIB classification with noisy or missing ECG measurements
Run commands from this folder in an activated Python environment outside the repo.
Install: python -m pip install -r requirements.txt (then the selected model's requirements; see README.md).
Download the three linked PTB-XL+/PTB-XL CSVs into ../afib_data/raw/ as shown in README.md; do not submit data.
Prepare: python scripts/prepare_data.py (writes ../afib_data/processed/; AFIB_DATA_DIR and AFIB_PROCESSED_DIR override paths).
Check saved results without data: python scripts/verify_result_artifacts.py
Train using the model commands in README.md, with a new --output folder; saved results are already included.
Final report: report/COMP90051_Final_Report.pdf. All teammates share report and code maintenance.

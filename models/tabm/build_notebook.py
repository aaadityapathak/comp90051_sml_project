"""Build the TabM-only training and results notebook without extra dependencies."""
import json
from pathlib import Path
from textwrap import dedent

PROJECT = Path(__file__).resolve().parents[2]


def main():
    cells = []

    def cell(kind, source):
        item = {'cell_type': kind, 'metadata': {}, 'id': f'tabm-{len(cells):02d}',
                'source': dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''
    # TabM tuning and results

    Run the planned TabM experiment and inspect its actual saved results. This notebook owns
    TabM only. The command-line scripts and notebook use the same resumable experiment.

    **Protocol:** 10 patient outer folds, 3 inner folds, blocks 2/3/4, width 128,
    16 members, dropout 0.1, no numerical embeddings, AdamW learning rate 0.0003,
    weight decay 0.0001, batches of 256 and **100 epochs per fit**. Choose mean observed
    inner macro-F1, with simpler ties and at most one endpoint grid refinement.
    A-D and four availability/duplicate controls are included. A full run requires
    at least 800 fits and can take several days on a laptop GPU.

    Select the Python environment containing the TabM requirements. `RUN_TRAINING` defaults to false,
    so Run All inspects saved results. Set `TABM_RUN_TRAINING=1` before launching
    Jupyter, or explicitly set `RUN_TRAINING=True`, to start/resume training.
    The saved completion report records fold coverage; check it before interpreting results.
    ''')
    cell('code', '''
    from pathlib import Path
    import os, sys, json, subprocess
    import pandas as pd
    from IPython.display import display, Markdown, Image

    PROJECT = next(
        (candidate for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
         if (candidate / 'requirements.txt').is_file()
         and (candidate / 'src' / 'afib').is_dir()
         and (candidate / 'models').is_dir()),
        None,
    )
    if PROJECT is None:
        raise FileNotFoundError('Run this notebook from the repository root or a directory inside it.')
    if not (PROJECT / 'models/tabm/run_experiment.py').exists():
        raise RuntimeError('Open Jupyter from the repository root or a directory inside it.')
    sys.path.insert(0, str(PROJECT))
    OUTPUT = PROJECT / 'results/tabm'
    RUN_TRAINING = os.environ.get('TABM_RUN_TRAINING', '0') == '1'
    REFRESH_REPORTS = RUN_TRAINING or os.environ.get('TABM_REFRESH_REPORTS', '0') == '1'
    print('Training enabled:', RUN_TRAINING)
    print('Python:', sys.executable)
    print('Output:', OUTPUT)
    ''')
    cell('markdown', '''
    ## Run each initial parameter setting

    Each script fits one block setting across all 8 variants and all 30 inner splits.
    Existing matching results are reused. An incompatible code, data or runtime signature
    raises an error instead of mixing experiments. Do not run multiple trainers against
    the same output folder. Outer patients never select a parameter or threshold.
    ''')
    cell('code', '''
    def run_script(filename, *arguments):
        command = [sys.executable, str(PROJECT / 'models/tabm' / filename),
                   '--output', str(OUTPUT), *arguments]
        if not RUN_TRAINING:
            print('Inspection mode; skipped:', filename, *arguments)
            return
        log_dir = OUTPUT / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (Path(filename).stem + '_notebook.log')
        print('Running', filename, '; full log:', log_path, flush=True)
        with log_path.open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if any(word in line.lower() for word in ['complete', 'resume', 'error', 'traceback']):
                        print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError(f'{filename} failed; inspect {log_path}')
            except BaseException:
                process.terminate()
                process.wait()
                raise
    ''')
    for blocks in (2, 3, 4):
        cell('code', f"run_script('run_blocks_{blocks}.py')")
    cell('markdown', '''
    ## Select, refit and evaluate

    This step reuses the three initial candidates, evaluates block 1 or 5 only if
    an initial endpoint wins, and freezes the chosen setting. It refits on outer
    training patients and tests all 31 scenario/repeat conditions. Threshold 0.5
    is primary; the exploratory 0.90-sensitivity threshold comes from selected
    inner out-of-fold predictions and remains frozen for every outer stress.
    ''')
    cell('code', "run_script('run_experiment.py', '--phase', 'run')")
    cell('markdown', '''
    ## Inspect completion and optionally refresh reports

    Predictions, histories and audits live in `results/tabm/output_data/`;
    aggregate tables also live there; figures and the written report live in `results/tabm/reports/`.
    Binary resume checkpoints stay locally in `results/tabm/checkpoints/`.
    Inspection leaves saved reports unchanged. Set `TABM_REFRESH_REPORTS=1` to rebuild
    reports explicitly. Reports label partial results explicitly. Repeats are averaged within each
    outer fold before calculating fold means and sample SD.
    ''')
    cell('code', '''
    from models.tabm.reporting import generate_reports
    if REFRESH_REPORTS and (OUTPUT / 'run_config.json').exists():
        generate_reports(OUTPUT)
    elif not (OUTPUT / 'run_config.json').exists():
        print('No full experiment has started yet.')
    for name in ['status.json', 'reports/completion.json', 'completion.json']:
        path = OUTPUT / name
        if path.exists():
            display(Markdown(f'**{name}**'))
            display(json.loads(path.read_text(encoding='utf-8')))
    ''')
    cell('code', '''
    # Show published result tables without assuming all outer folds exist.
    for name in ['summary.csv', 'selected_hyperparameters.csv', 'tuning_progress.csv',
                 'paired_method_differences.csv', 'paired_stress_drops.csv',
                 'patient_weighted_metrics_summary.csv', 'quality_subgroups_summary.csv']:
        paths = [OUTPUT / 'reports' / name, OUTPUT / 'output_data' / name]
        path = next((p for p in paths if p.exists()), None)
        if path:
            display(Markdown(f'**{name}**'))
            try:
                display(pd.read_csv(path).head(40))
            except pd.errors.EmptyDataError:
                print('No completed results for this table yet.')
    ''')
    cell('code', '''
    for path in sorted((OUTPUT / 'reports').glob('*.png')):
        display(Markdown(f'**{path.stem.replace("_", " ")}**'))
        display(Image(filename=str(path)))
    for name in ['report.md', 'README.md']:
        path = OUTPUT / 'reports' / name
        if path.exists():
            display(Markdown(path.read_text(encoding='utf-8')))
            break
    ''')
    cell('code', '''
    # Verify all requested folds before interpreting a complete experiment.
    command = [sys.executable, str(PROJECT / 'models/tabm/check_results.py'),
               '--output', str(OUTPUT)]
    if (OUTPUT / 'run_config.json').exists():
        checked = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        print(checked.stdout)
        if checked.returncode:
            print(checked.stderr)
            raise RuntimeError('Result validation failed.')
    ''')
    cell('markdown', '''
    ## Interpretation and sources

    Compare B-A and D-C for added feature values, C-A and D-B for augmentation,
    and the availability/duplicate controls before attributing an improvement.
    Error bars describe variation across dependent outer folds; they are not
    clinical confidence intervals. Missing extracted measurements do not simulate
    physical electrode removal. No clinical benefit follows from this experiment alone.

    Protocol: `docs/Project_Plan.docx` (8 October 2026).
    [Official TabM implementation](https://github.com/yandex-research/tabm),
    [TabM paper](https://arxiv.org/abs/2410.24210),
    [PTB-XL+ 1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/) and
    [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (CC BY 4.0).
    ''')
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3 (TabM)',
                                            'language': 'python', 'name': 'tabm'},
                             'language_info': {'name': 'python', 'version': '3.12.10'}}}
    path = PROJECT / 'notebooks/05_tabm_tuning.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()

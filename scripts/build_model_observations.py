"""Rebuild four concise Word/Markdown reports without training any model.

LR and TabM use saved result tables. HGB uses the retained historical document;
its figures are explicitly reported-only because the raw bundle is unavailable.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
from docx import Document
from observation_support import ARMS, LABELS, MODELS, NAMES, POLICY, SCENARIOS, Report, Results

HGB_SOURCE = 'results/histogram_gradient_boosting/source_provenance/reported_summary.docx'
SECONDARY = 'inner_sensitivity_0.90'


def hgb_reference(r):
    path = r.root / HGB_SOURCE
    payload = path.read_bytes()
    r.inputs[HGB_SOURCE] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    observed, stress = {}, {}
    for table in Document(path).tables:
        rows = [[c.text.strip() for c in row.cells] for row in table.rows]
        if len(rows[0]) == 5 and rows[0][1].startswith('Macro-F1'):
            for row in rows[1:]:
                observed[row[0]] = {name: [float(x) for x in re.findall(r'\d+\.\d+', cell)]
                                    for name, cell in zip(('macro_f1', 'sensitivity', 'specificity', 'auroc'), row[1:])}
        if rows[0] == ['Arm', 'Observed', '1 missing', '3 missing', '6 missing']:
            stress = {row[0]: dict(zip(('observed', 'missing_1', 'missing_3', 'missing_6'), map(float, row[1:]))) for row in rows[1:]}
    if set(observed) != set(ARMS) or set(stress) != set(ARMS):
        raise ValueError('HGB reference tables do not match the expected four-arm report.')
    return observed, stress


def protocol(d):
    d.p('The cohort contains 21,799 ECGs from 18,869 patients; 6.95% are AFIB-labelled. '
        'This class imbalance makes macro F1 and class-specific errors more informative than accuracy alone.')
    d.p('A: original measurements; B: original plus engineered fields; C: A with augmentation; D: B with augmentation. '
        'There are 504 original lead fields and 63 engineered fields. Augmentation adds one corrupted training view per ECG with total weight preserved.')
    d.p('Primary results use threshold 0.50, 10 outer patient folds and 3 inner folds. Filling, scaling and selection use training patients only. '
        'Three stress repeats are averaged within each fold, then summarized by mean and sample SD. SD is not a confidence interval.')


def observed_table(d, r, m):
    d.table(['Arm', 'Macro F1 (SD)', 'Sensitivity', 'Specificity', 'Brier'],
            [[a, r.fmt(m, a, sd=True)] + [r.fmt(m, a, k) for k in ('sensitivity', 'specificity', 'brier')] for a in ARMS],
            [1.2, 4.5, 3.9, 3.9, 3.9])
    d.p('Higher macro F1, sensitivity and specificity are better; lower Brier means lower probability error.')


def tuning_table(d, r, m):
    parameter = 'C' if m == 'logistic_regression' else 'n_blocks'
    rows = r.csv(m, 'selected_hyperparameters')
    d.table(['Variant', 'Setting and fold count', 'Selected inner F1'], [
        [a.replace('_', ' '), '; '.join(f'{float(k):g}: {v}' for k, v in sorted(Counter(x[parameter] for x in rows if x['variant'] == a).items(), key=lambda x: float(x[0]))),
         f'{statistics.mean(float(x["inner_mean_macro_f1"]) for x in rows if x["variant"] == a):.4f}']
        for a in r.config[m]['variants']], [5.1, 8.1, 4.2])
    refined, middle = (sum(x[k] == 'True' for x in rows) for k in ('refined', 'selected_middle'))
    d.p(f'{refined}/{len(rows)} selections triggered boundary refinement; {middle}/{len(rows)} selected the middle of the final grid. '
        'Inner F1 averages each fold\'s selected candidate score; it is selection evidence, not an unbiased generalization estimate. '
        'Refined candidates have unequal fold coverage; their aggregate scores cannot establish one globally best setting.')


def stress_table(d, r, m):
    d.table(['Condition', 'A', 'B', 'C', 'D'], [[LABELS[s]] + [r.fmt(m, a, scenario=s) for a in ARMS] for s in SCENARIOS],
            [6.2, 2.8, 2.8, 2.8, 2.8])
    d.p('Noise SD is the listed dose times the training-field IQR; gain is exp(dose times a standard-normal draw). '
        'Perturbations act on measurement tables, not raw ECG waveforms. Small F1 rises under mild corruption can reflect shifted threshold errors; they do not establish better measurement quality.')


def additional_checks(d, r, m):
    target_rows = [x for x in r.csv(m, 'fold_mean_metrics') if x['variant'] == 'D' and x['scenario'] == 'observed'
                   and x['threshold_policy'] == SECONDARY]
    target_hits = sum(float(x['sensitivity']) >= .9 for x in target_rows)
    d.p(f'The inner-OOF sensitivity-target rule gives D observed F1 {r.fmt(m, "D", policy=SECONDARY)}, '
        f'sensitivity {r.fmt(m, "D", "sensitivity", policy=SECONDARY)} and specificity {r.fmt(m, "D", "specificity", policy=SECONDARY)}. '
        f'The 0.90 sensitivity target was met in {target_hits}/{len(target_rows)} observed outer folds. '
        'This secondary operating point does not replace the primary ranking.')
    patients = [x for x in r.csv(m, 'patient_weighted_metrics_summary') if x['variant'] == 'D' and x['scenario'] == 'observed'
                and x['threshold_policy'] == POLICY and x.get('metric', 'macro_f1') == 'macro_f1']
    patient_f1 = float(patients[0].get('mean', patients[0].get('macro_f1_mean')))
    q = {}
    for group in ('recorded_quality_flag', 'no_recorded_quality_flag'):
        row = next(x for x in r.csv(m, 'quality_subgroups_summary') if x['variant'] == 'D' and x['scenario'] == 'observed'
                   and x['threshold_policy'] == POLICY and x['subgroup'] == group and x.get('metric', 'macro_f1') == 'macro_f1')
        q[group] = float(row.get('mean', row.get('macro_f1_mean')))
    d.p(f'D patient-weighted F1 is {patient_f1:.4f}. F1 is {q["recorded_quality_flag"]:.4f} with recorded quality flags '
        f'and {q["no_recorded_quality_flag"]:.4f} without flags. Flag absence does not establish clean signals; subgroup composition may differ.')


def sources(d, m):
    d.p(f'Evidence: results/{m}/output_data/summary.csv, selected_hyperparameters.csv, paired_method_differences.csv, '
        'patient_weighted_metrics_summary.csv, quality_subgroups_summary.csv; run_config.json and reports/completion.json. '
        'Rebuild: python scripts/build_model_observations.py.')
    if m == 'logistic_regression':
        d.p('Ridge reconstruction evidence: output_data/recovery_summary.csv in the same result folder.')


def logistic_report(r):
    m, d = 'logistic_regression', Report()
    d.title('Logistic regression observations')
    d.p(f'Best primary observed result: D, macro F1 {r.fmt(m, "D", sd=True)}. It has the highest observed mean across all saved variants. '
        'B performs better under severe lead loss; there is no single best arm for every condition.')
    d.heading('Experiment and observed results')
    protocol(d)
    observed_table(d, r, m)
    d.p(f'B minus A F1 is {r.difference(m, "B minus A")[0]:+.4f}; D minus B is {r.difference(m, "D minus B")[0]:+.4f}. '
        'Engineered fields supply relationships a linear decision function cannot construct itself. The results support their usefulness, without identifying which feature family produced the gain.')
    d.heading('Tuning and optimization')
    d.p('Weighted L2 logistic regression used L-BFGS, up to 2,000 iterations, and initial C values 0.01, 0.1 and 1, with one boundary refinement. '
        'Larger C weakens regularization. Selection used mean inner observed macro F1 at 0.50.')
    tuning_table(d, r, m)
    d.p(f'All {r.complete[m]["n_fits"]} saved fits converged; maximum absolute gradient was {r.complete[m]["max_gradient"]:.2g}. '
        'This confirms numerical completion of the specified search, not optimality across other model or feature choices.')
    d.page()
    d.heading('Robustness and additional outputs')
    stress_table(d, r, m)
    d.p(f'With six leads missing, B F1 is {r.fmt(m, "B", scenario="missing_6")} versus D {r.fmt(m, "D", scenario="missing_6")}. '
        f'D retains sensitivity {r.fmt(m, "D", "sensitivity", "missing_6")}, but specificity falls to {r.fmt(m, "D", "specificity", "missing_6")}. '
        'Augmentation trades fewer missed positives for more false positives, reducing F1 in this condition.')
    d.p(f'B exceeds its availability control by {r.difference(m, "B minus availability_observed")[0]:.4f} observed F1. '
        'That control retains original predictors and engineered missingness indicators but zeroes engineered values. Half-weight duplicates reproduce observed F1, consistent with an unchanged logistic objective.')
    d.p(f'Ridge filling reaches observed F1 {r.fmt(m, "ridge")} and three-lead-loss F1 {r.fmt(m, "ridge", scenario="missing_3")} '
        f'(A: {r.fmt(m, "A")} and {r.fmt(m, "A", scenario="missing_3")}). Only these two downstream conditions were tested. '
        'Reconstruction error improves over median filling for cell masking and 1, 3 and 6 missing leads; better reconstruction alone does not establish a better classifier.')
    recovery = r.csv(m, 'recovery_summary')
    recovery_value = lambda method, scenario: float(next(x['mean'] for x in recovery if x['method'] == method and x['scenario'] == scenario))
    d.p(f'IQR-normalized reconstruction MAE, median to ridge: 10% cell masking '
        f'{recovery_value("median", "cells_0.10"):.4f} to {recovery_value("ridge", "cells_0.10"):.4f}; six missing leads '
        f'{recovery_value("median", "missing_6"):.4f} to {recovery_value("ridge", "missing_6"):.4f}. Only eligible masked continuous cells are scored.')
    additional_checks(d, r, m)
    d.heading('Next improvements and rationale')
    d.p('1. Tune augmentation dose and mixture inside training folds to reduce specificity loss under lead removal. '
        '2. Test feature-family ablations and elastic-net regularization to identify useful predictors and control redundancy. '
        '3. Fit calibration and thresholds inside training folds to compare sensitivity at a fixed false-positive budget. '
        '4. Repeat patient splits and test external data to quantify uncertainty and transfer beyond synthetic corruption.')
    d.p('The one-fold pilot is a workflow check, not additional ten-fold evidence. External validation and statistical superiority are not established.')
    sources(d, m)
    return d


def tabm_report(r):
    m, d = 'tabm', Report()
    d.title('TabM observations')
    d.p(f'Best main arm: D, observed F1 {r.fmt(m, "D", sd=True)} and six-lead-loss F1 {r.fmt(m, "D", scenario="missing_6")}. '
        f'The duplicate-enhanced control has the highest observed mean across all variants, {r.fmt(m, "duplicate_enhanced", sd=True)}, '
        f'only {r.value(m, "duplicate_enhanced") - r.value(m, "D"):.6f} above D; its six-lead-loss F1 is {r.fmt(m, "duplicate_enhanced", scenario="missing_6")}. '
        'D is the stronger choice when missing-lead robustness matters.')
    d.heading('Experiment and observed results')
    protocol(d)
    observed_table(d, r, m)
    d.p(f'B minus A F1 is {r.difference(m, "B minus A")[0]:+.4f}; D minus B is {r.difference(m, "D minus B")[0]:+.4f}. '
        'The small observed augmentation gain alone does not justify D. Its stronger advantage is sustained performance under missing inputs.')
    d.heading('Tuning and training')
    d.p('Initial block counts were 2, 3 and 4, with one boundary refinement. Selection maximized mean inner observed F1; exact ties favored fewer blocks. '
        'Width 128, 16 members, dropout 0.1, AdamW, batch size 256 and 100 epochs were fixed. No early stopping was used.')
    tuning_table(d, r, m)
    d.p(f'The run contains {r.complete[m]["n_fits"]} fits: {r.complete[m]["completed_inner_fits"]} inner fits and '
        f'{r.complete[m]["completed_outer_refits"]} outer refits. Summed fold work is {r.complete[m]["completed_fold_runtime_seconds"]/3600:.2f} hours, '
        'not end-to-end elapsed time. Training-loss histories alone cannot establish absence of overfitting or the best epoch count.')
    d.page()
    d.heading('Robustness and additional outputs')
    stress_table(d, r, m)
    d.p(f'With six leads missing, A sensitivity is {r.fmt(m, "A", "sensitivity", "missing_6")}; D retains {r.fmt(m, "D", "sensitivity", "missing_6")}. '
        'Exposure to corrupted training views is consistent with this large robustness gain. Tested additive noise and gain changes have much smaller F1 effects than lead removal.')
    d.p(f'B exceeds its availability control by {r.difference(m, "B minus availability_observed")[0]:.4f} observed F1, supporting information in engineered values beyond their missingness indicators. '
        'Duplicate controls match augmented view/update counts but can select different block counts, leaving capacity and compute unmatched. '
        'Relative to observed-only arms, both double views and increase updates. Lower Brier establishes better probability error, not calibration by itself.')
    additional_checks(d, r, m)
    d.heading('Next improvements and rationale')
    d.p('1. Repeat seeds and hold capacity and update budgets fixed in augmentation ablations to test whether tiny observed gains are stable. '
        '2. Tune capacity, learning rate and epoch budget in inner folds: 11 final boundary selections suggest the limited block search may miss useful settings. '
        '3. Ablate feature families and augmentation types to identify the sources of robustness. '
        '4. Test lead-specific and correlated loss plus external data to assess transfer beyond the simulated stresses.')
    d.p('The one-fold, three-epoch pilot is excluded from ranking. No TabM ridge-filling experiment or external validation was completed.')
    sources(d, m)
    return d


def hgb_report(r, observed, stress):
    d = Report()
    d.title('Histogram gradient boosting observations')
    d.p('Best reported observed arm: B, F1 0.9230 (SD 0.0143). Best reported six-lead-loss arm: C, F1 0.8892. '
        'The retained report contains transcribed results; underlying predictions, fold metrics and tuning files are absent, so these figures cannot currently be independently recalculated.')
    d.heading('Experiment and reported results')
    protocol(d)
    d.table(['Arm', 'Macro F1 (SD)', 'Sensitivity', 'Specificity', 'AUROC'],
            [[a, f'{observed[a]["macro_f1"][0]:.4f} ({observed[a]["macro_f1"][1]:.4f})'] +
             [f'{observed[a][k][0]:.4f}' for k in ('sensitivity', 'specificity', 'auroc')] for a in ARMS], [1.2, 4.5, 3.9, 3.9, 3.9])
    d.p('B improves observed F1 over A by a reported paired mean 0.0018 (SD 0.0047), with higher specificity and slightly lower sensitivity. '
        'Its best F1 reflects this error balance. Trees can capture nonlinear feature interactions; the report does not isolate which interactions or engineered fields produce the gain.')
    d.heading('Tuning and evidence limits')
    d.p('Initial maximum depths were 2, 3 and 4 with one boundary refinement. The current runner permits depth 1 or 5 depending on the boundary winner; '
        'the historical report describes upper refinement to 5. Mean inner observed F1 selects depth; ties prefer shallower trees.')
    d.p('Fixed settings: 150 iterations, learning rate 0.05, maximum 16 leaves, minimum 30 samples per leaf, L2 penalty 1, no early stopping and seed 42. '
        'These constrain complexity; they do not establish an optimal tuning budget.')
    d.p('The report states 40 completed arm-fold runs and 31 condition/repeat evaluations per run. Candidate scores, selected-depth counts, '
        'run versions and source hashes are unavailable. No specific depth can be called the winner from the retained evidence.')
    d.heading('Missing leads and augmentation')
    d.table(['Arm', 'Observed', '1 missing', '3 missing', '6 missing'],
            [[a] + [f'{stress[a][s]:.4f}' for s in ('observed', 'missing_1', 'missing_3', 'missing_6')] for a in ARMS], [1.2, 4.05, 4.05, 4.05, 4.05])
    d.p('Augmentation lowers reported observed F1: C minus A is -0.0057; D minus B is -0.0051. With six leads removed, C and D strongly outperform A and B. '
        'C exceeds D under three and six missing leads, so enhancement adds no robustness gain in these augmented comparisons.')
    d.p('Under combined three-lead loss and noise 0.10, reported F1 is A 0.8493, B 0.8876, C 0.9062 and D 0.9019. '
        'The source describes smaller effects for isolated noise and gain but omits their complete numerical tables. '
        'Brier, average precision, availability/duplication controls and subgroup results are not available in the report.')
    d.heading('Next improvements and rationale')
    d.p('1. Retain predictions, selections, data/source hashes and package versions on a future authorized run so scores can be independently checked. '
        '2. Tune augmentation mixtures in inner folds to retain robustness with less observed F1 loss. '
        '3. Study depth, leaves, learning rate and iterations within a declared budget to balance capacity and overfitting. '
        '4. Add the same controls, probability metrics and patient-level comparisons as the other algorithms to complete the comparison.')
    d.p(f'Source: {HGB_SOURCE}. Future output: results/histogram_gradient_boosting/formal/. '
        'Runner and checker: models/histogram_gradient_boosting/. This review does not regenerate HGB results.')
    return d


def comparison_report(r, hgb):
    d = Report()
    d.title('Algorithm result comparison')
    d.p('TabM D has the best primary observed macro F1 among the main arms and the strongest reported six-lead-loss F1. '
        'LR and TabM have saved evidence that can be checked; HGB is represented by a historical summary, so the three-model ranking is descriptive and provisional.')
    d.heading('Best observed main arm per algorithm')
    d.table(['Algorithm and arm', 'Macro F1 (SD)', 'Sensitivity', 'Specificity', 'Brier'], [
        ['Logistic D', r.fmt('logistic_regression', 'D', sd=True)] + [r.fmt('logistic_regression', 'D', k) for k in ('sensitivity', 'specificity', 'brier')],
        ['HGB B reported', '0.9230 (0.0143)', '0.9496', '0.9803', 'Unavailable'],
        ['TabM D', r.fmt('tabm', 'D', sd=True)] + [r.fmt('tabm', 'D', k) for k in ('sensitivity', 'specificity', 'brier')],
    ], [4.2, 4.2, 3.0, 3.0, 3.0])
    d.p('Observed data, threshold 0.50, ten outer patient folds; parentheses contain sample SD. Best arm means the highest outer-fold mean among A-D. '
        'Selecting an arm after seeing these outcomes does not create a new independent evaluation of that choice.')
    d.p(f'TabM D exceeds logistic D by {r.value("tabm", "D") - r.value("logistic_regression", "D"):.4f} F1 and reported HGB B by '
        f'{r.value("tabm", "D") - hgb["B"]["macro_f1"][0]:.4f}. Its precision is {r.fmt("tabm", "D", "precision")} versus logistic D '
        f'{r.fmt("logistic_regression", "D", "precision")}. Higher specificity reduces false positives and improves F1, but lower sensitivity means more missed positives.')
    d.p(f'TabM D average precision is {r.fmt("tabm", "D", "average_precision")} versus logistic D {r.fmt("logistic_regression", "D", "average_precision")}. '
        f'HGB B reported AUROC 0.9935 exceeds TabM D {r.fmt("tabm", "D", "auroc")}; TabM does not win every metric. HGB Brier and average precision are unavailable.')
    d.heading('Robustness changes the best arm')
    d.table(['Algorithm', 'Best observed arm', 'Best six-lead-loss arm', 'F1 under six-lead loss'], [
        ['Logistic', 'D', 'B', r.fmt('logistic_regression', 'B', scenario='missing_6')],
        ['HGB reported', 'B', 'C', '0.8892'], ['TabM', 'D', 'D', r.fmt('tabm', 'D', scenario='missing_6')]], [4.1, 4.1, 4.6, 4.6])
    d.p('TabM D combines the strongest main-arm observed and missing-lead F1. Nonlinear modeling and corrupted training views are plausible contributors; '
        'different tuning budgets and optimization prevent attributing the full advantage to architecture.')
    d.p(f'Across all TabM variants, duplicate-enhanced has the largest observed mean ({r.fmt("tabm", "duplicate_enhanced")}), only '
        f'{r.value("tabm", "duplicate_enhanced") - r.value("tabm", "D"):.6f} above D, with much weaker missing-lead F1. This does not overturn D\'s robustness advantage.')
    d.heading('Comparison limits and next steps')
    d.p(f'LR and TabM have identical prepared-data manifests and {r.pairing["conditions_compared"]:,} matching perturbation audits across 80 common fold-variants. '
        'Patient folds keep repeat ECGs together. HGB matching cannot be independently verified. Outer training sets overlap; fold SD establishes neither significance nor external validity.')
    d.p('1. Recover the HGB bundle or generate it on a future authorized run before claiming a fully verified three-model comparison. '
        '2. Predeclare contrasts, quantify paired patient-level uncertainty and repeat seeds for small observed gains. '
        '3. Compare at a common sensitivity or false-positive budget, fitting thresholds/calibration inside training folds. '
        '4. Test external data and realistic lead loss before deployment. Do not average stresses without a justified operating distribution.')
    d.p('Evidence: LR/TabM output_data/summary.csv, run_config.json and saved perturbation audits; '
        f'{HGB_SOURCE}. Pilots are excluded from ranking. Rebuild: python scripts/build_model_observations.py.')
    return d


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--comparison-output', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    r = Results(root)
    hgb, stress = hgb_reference(r)
    output = args.output_dir or root / 'docs'
    comparison = args.comparison_output or root / 'results/comparison/output_data'
    comparison.mkdir(parents=True, exist_ok=True)
    reports = {'Logistic_Regression_Observations': logistic_report(r), 'TabM_Observations': tabm_report(r),
               'Histogram_Gradient_Boosting_Observations': hgb_report(r, hgb, stress), 'Model_Comparison': comparison_report(r, hgb)}
    for stem, report in reports.items():
        report.doc.core_properties.title = report.md[0].removeprefix('# ').strip()
        report.save(output / f'{stem}.docx')
    rows = [{'model': m, **row} for m in MODELS for row in r.csv(m, 'summary') if row['variant'] in ARMS]
    rows.sort(key=lambda x: (x['model'], x['variant'], x['scenario'], x['threshold_policy'], x['metric']))
    write_csv(comparison / 'observations_main_summary.csv', rows,
              ['model', 'variant', 'scenario', 'threshold_policy', 'metric', 'mean', 'sd', 'n_folds', 'n_defined'])
    paired = [{'model': m, 'threshold_policy': POLICY, 'n_defined': '', **row} for m in MODELS for row in r.csv(m, 'paired_method_differences')]
    paired.sort(key=lambda x: (x['model'], x['comparison'], x['scenario'], x['threshold_policy'], x['metric']))
    write_csv(comparison / 'observations_method_differences.csv', paired,
              ['model', 'comparison', 'scenario', 'threshold_policy', 'metric', 'direction', 'n_folds', 'n_defined', 'mean_difference', 'sd_difference'])
    (comparison / 'observations_pairing_validation.json').write_text(json.dumps(r.pairing, indent=2) + '\n', encoding='utf-8')
    (comparison / 'hgb_reported_reference.json').write_text(json.dumps({'source': HGB_SOURCE,
        'status': 'reported_only_raw_bundle_absent', 'observed': hgb, 'missing_lead_f1': stress}, indent=2) + '\n', encoding='utf-8')
    provenance = {'schema_version': 2, 'report_date': '2026-10-15', 'models_with_raw_artifacts': list(MODELS),
        'hgb_status': 'Historical report only; raw model outputs were not regenerated.', 'documents': list(reports),
        'aggregation': 'Stress repeats averaged within fold; mean and sample SD across ten folds.',
        'inference': 'Descriptive comparisons; SD is not a confidence interval.', 'sources': dict(sorted(r.inputs.items())),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'support_sha256': hashlib.sha256(Path(__file__).with_name('observation_support.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    (comparison / 'observations_input_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote four Word reports and matching Markdown copies to {output}')
    print(f'Wrote comparison evidence and input hashes to {comparison}')


if __name__ == '__main__':
    main()

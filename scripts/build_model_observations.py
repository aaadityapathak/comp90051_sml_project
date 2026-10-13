"""Rebuild the model observation documents and comparison tables from saved results.

No training, source data, GPU, or pickle loading is required. Install
``requirements-docs.txt`` and run this file from any working directory. Source
tables are read only. Both document formats are generated from the same content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


MODELS = ("logistic_regression", "tabm")
NAMES = {"logistic_regression": "Logistic regression", "tabm": "TabM"}
ARMS = ("A", "B", "C", "D")
POLICY = "fixed_0.50"
SCENARIOS = ("observed", "missing_1", "missing_3", "missing_6", "noise_0.05",
             "noise_0.10", "noise_0.20", "gain_0.05", "gain_0.10", "gain_0.20",
             "combined_3lead_noise0.10")
LABELS = {"observed": "Observed", "missing_1": "1 missing lead",
          "missing_3": "3 missing leads", "missing_6": "6 missing leads",
          "combined_3lead_noise0.10": "3 leads + noise 0.10"}
LABELS.update({f"{kind}_{dose}": f"{kind.title()} {dose}"
               for kind in ("noise", "gain") for dose in ("0.05", "0.10", "0.20")})


class Results:
    def __init__(self, root: Path):
        self.root, self.inputs, self.cache = root, {}, {}
        self.config = {m: self.json(f"results/{m}/run_config.json") for m in MODELS}
        self.complete = {m: self.json(f"results/{m}/reports/completion.json") for m in MODELS}
        if not self.complete["tabm"].get("complete_tabm_plan"):
            raise ValueError("This observation report requires a complete TabM run.")
        if not self.complete["logistic_regression"].get("complete_logistic_plan"):
            raise ValueError("This observation report requires a complete logistic run.")
        if self.config[MODELS[0]]["manifest"] != self.config[MODELS[1]]["manifest"]:
            raise ValueError("The two runs must use the same prepared data manifest.")
        for m in MODELS:
            rows = self.csv(m, "summary")
            self.cache[m, "summary_index"] = {
                (r["variant"], r["scenario"], r["threshold_policy"], r["metric"]): r
                for r in rows}
            if len(self.cache[m, "summary_index"]) != len(rows):
                raise ValueError(f"Duplicate summary keys for {m}.")
            for arm in ARMS:
                for scenario in SCENARIOS:
                    for metric in ("macro_f1", "sensitivity", "specificity", "brier"):
                        r = self.row(m, arm, metric, scenario)
                        if int(r["n_folds"]) != 10 or int(r["n_defined"]) != 10:
                            raise ValueError(f"Incomplete metric: {m}/{arm}/{scenario}/{metric}")
        self.pairing = self.verify_stress_pairing()

    def verify_stress_pairing(self):
        """Check common fold-variant stress audits without loading model checkpoints."""
        variants = sorted(set(self.config[MODELS[0]]["variants"]) &
                          set(self.config[MODELS[1]]["variants"]))
        conditions = 0
        for outer in range(10):
            for variant in variants:
                audits = []
                for model in MODELS:
                    relative = (f"results/{model}/output_data/folds/{outer:02d}_{variant}/"
                                "perturbation_audit.json")
                    rows = self.json(relative)["test"]
                    if len(rows) != 31:
                        raise ValueError(f"Expected 31 stress conditions in {relative}.")
                    audits.append([{k: v for k, v in row.items() if k != "model"} for row in rows])
                if audits[0] != audits[1]:
                    raise ValueError(f"Cross-model perturbation audit mismatch: {outer}/{variant}.")
                conditions += len(audits[0])
        return {"status": "passed", "shared_prepared_manifest": True,
                "fold_variants_compared": 10 * len(variants), "conditions_compared": conditions,
                "excluded_metadata_fields": ["model"],
                "scope": "Exact equality of saved test perturbation diagnostics; no models refitted."}

    def read(self, relative):
        payload = (self.root / relative).read_bytes()
        self.inputs[relative] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        return payload.decode("utf-8-sig")

    def json(self, relative):
        return json.loads(self.read(relative))

    def csv(self, model, stem):
        key = (model, stem)
        if key not in self.cache:
            self.cache[key] = list(csv.DictReader(io.StringIO(
                self.read(f"results/{model}/output_data/{stem}.csv"))))
        return self.cache[key]

    def row(self, model, arm, metric, scenario="observed", policy=POLICY):
        return self.cache[model, "summary_index"][arm, scenario, policy, metric]

    def value(self, model, arm, metric="macro_f1", scenario="observed", policy=POLICY):
        return float(self.row(model, arm, metric, scenario, policy)["mean"])

    def fmt(self, model, arm, metric="macro_f1", scenario="observed", policy=POLICY, sd=False):
        r = self.row(model, arm, metric, scenario, policy)
        value = f"{float(r['mean']):.4f}"
        return value + (f" ({float(r['sd']):.4f})" if sd else "")

    def difference(self, model, comparison, scenario="observed", metric="macro_f1"):
        rows = [r for r in self.csv(model, "paired_method_differences")
                if r["comparison"] == comparison and r["scenario"] == scenario
                and r["metric"] == metric and r.get("threshold_policy", POLICY) == POLICY]
        if len(rows) != 1:
            raise ValueError(f"Expected one paired difference: {model}, {comparison}, {scenario}")
        return float(rows[0]["mean_difference"]), float(rows[0]["sd_difference"])


class Report:
    """Build matching Word and Markdown with modest, readable report styling."""

    def __init__(self):
        self.doc, self.md = Document(), []
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(1.9)
        section.left_margin = section.right_margin = Cm(1.8)
        for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2"):
            style = self.doc.styles[name]
            style.font.name = "Calibri"
            style.font.color.rgb = RGBColor(0, 0, 0)
        normal = self.doc.styles["Normal"]
        normal.font.size = Pt(10)
        normal.paragraph_format.space_after = Pt(7)
        normal.paragraph_format.line_spacing = 1.08
        self.doc.styles["Title"].font.size = Pt(24)
        self.doc.styles["Heading 1"].font.size = Pt(16)
        self.doc.styles["Heading 2"].font.size = Pt(12)
        for name in ("Heading 1", "Heading 2"):
            self.doc.styles[name].paragraph_format.space_before = Pt(12)
            self.doc.styles[name].paragraph_format.space_after = Pt(6)
        self.doc.core_properties.title = "Logistic regression and TabM model observations"
        self.doc.core_properties.author = "COMP90051 project team"
        self.doc.core_properties.subject = "Results, limitations and further evaluation"
        stamp = datetime(2026, 10, 13, tzinfo=timezone.utc)
        self.doc.core_properties.created = self.doc.core_properties.modified = stamp

    def title(self, text):
        self.doc.add_paragraph(text, "Title")
        self.md.append(f"# {text}\n")

    def heading(self, text, level=1):
        self.doc.add_heading(text, level)
        self.md.append(f"{'#' * (level + 1)} {text}\n")

    def p(self, text):
        self.doc.add_paragraph(text)
        self.md.append(text + "\n")

    def code(self, lines):
        paragraph = self.doc.add_paragraph()
        for index, line in enumerate(lines):
            run = paragraph.add_run(("\n" if index else "") + line)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        self.md.extend(["```powershell", *lines, "```", ""])

    def page(self):
        self.doc.add_page_break()

    def table(self, headers, rows, widths=None):
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.autofit = False
        if widths is None:
            widths = [17.4 / len(headers)] * len(headers)
        for c, width in zip(table.columns, widths):
            c.width = Cm(width)
        for cell, width, text in zip(table.rows[0].cells, widths, headers):
            cell.width = Cm(width)
            cell.text = str(text)
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "E8EEF3")
            cell._tc.get_or_add_tcPr().append(shade)
            for run in cell.paragraphs[0].runs:
                run.bold = True
        header = OxmlElement("w:tblHeader")
        table.rows[0]._tr.get_or_add_trPr().append(header)
        for values in rows:
            for cell, width, text in zip(table.add_row().cells, widths, values):
                cell.width = Cm(width)
                cell.text = str(text)
        for row in table.rows:
            prevent_split = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(prevent_split)
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(4)
                    paragraph.paragraph_format.space_before = Pt(4)
                    paragraph.paragraph_format.line_spacing = 1
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
        self.md.append("| " + " | ".join(headers) + " |")
        self.md.append("| " + " | ".join("---" for _ in headers) + " |")
        self.md.extend("| " + " | ".join(html.escape(str(x)).replace("\n", " ").replace("|", "\\|")
                                            for x in row) + " |" for row in rows)
        self.md.append("")

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = io.BytesIO()
        self.doc.save(stream)
        # Normalize ZIP member timestamps so identical inputs produce identical files.
        with ZipFile(stream) as source, ZipFile(path, "w", ZIP_DEFLATED) as dest:
            for member in source.infolist():
                info = ZipInfo(member.filename, (2026, 10, 13, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                dest.writestr(info, source.read(member.filename))
        path.with_suffix(".md").write_text("\n".join(self.md), encoding="utf-8", newline="\n")


def build_report(r):
    doc = Report()
    lr, tm = MODELS
    f = r.fmt
    v = r.value
    doc.title("Logistic regression and TabM model observations")
    doc.p("COMP90051 Statistical Machine Learning | Completed experiment review | 13 October 2026")
    doc.p(f"TabM arm D has the highest observed macro-F1 among the eight main model-arm combinations: "
          f"{f(tm, 'D', sd=True)}, compared with {f(lr, 'D', sd=True)} for logistic regression D. "
          "The key practical finding is that strong observed performance does not ensure robustness to missing leads. "
          "Training augmentation substantially protects TabM under severe lead loss; logistic regression "
          "shows a stronger tradeoff between recovering AFIB sensitivity and avoiding false positives.")
    doc.heading("Main findings")
    doc.table(["Finding", "Evidence and implication"], [
        ["Better observed ranking and precision for TabM", f"D average precision {f(tm, 'D', 'average_precision')} "
         f"versus {f(lr, 'D', 'average_precision')} for logistic D. D precision is {f(tm, 'D', 'precision')} "
         f"versus {f(lr, 'D', 'precision')}; sensitivity is lower ({f(tm, 'D', 'sensitivity')} "
         f"versus {f(lr, 'D', 'sensitivity')})."],
        ["Missing leads expose a major weakness", f"TabM A sensitivity falls to {f(tm, 'A', 'sensitivity', 'missing_6')} "
         f"with six missing leads; augmented D retains {f(tm, 'D', 'sensitivity', 'missing_6')}. "
         "Observed-only TabM should not be selected from clean performance alone."],
        ["Augmentation is not a universal gain", f"Under six-lead loss, logistic D sensitivity exceeds B, "
         f"but D macro-F1 is {f(lr, 'D', scenario='missing_6')} versus B {f(lr, 'B', scenario='missing_6')}. "
         "Choose the error tradeoff explicitly."],
        ["Small clean gains need uncertainty analysis", "B to D gains are much smaller on observed data than "
         "TabM gains under lead loss. Fold SD is descriptive variability, not a confidence interval or significance test."],
    ], [5.3, 12.1])
    doc.heading("Scope and status")
    doc.p(f"Both runs cover all ten outer patient folds. TabM completed "
          f"{r.complete[tm]['completed_outer_fold_variants']} outer fold-variant checkpoints, "
          f"{r.complete[tm]['completed_inner_fits']} inner fits and {r.complete[tm]['completed_outer_refits']} "
          f"outer refits ({r.complete[tm]['n_fits']} total fits). Logistic regression completed nine variants "
          f"including the separate ridge experiment, with {r.complete[lr]['n_fits']} converged fits. "
          "Histogram gradient boosting remains pending, so the planned three-model comparison is incomplete.")
    doc.p("Tables report record-level outer-fold means unless stated otherwise. Parentheses contain sample SD "
          "across ten outer folds. Stress repeats are averaged within each fold before averaging folds. "
          "All main tables use the prespecified 0.50 decision threshold. No p-values or claims of statistical "
          "significance are made.")

    doc.page()
    doc.heading("Study design and fair interpretation")
    manifest = r.config[tm]["manifest"]
    positives = sum(int(row["n_positive"]) for row in r.csv(tm, "majority_baseline"))
    doc.p(f"The shared prepared package contains {manifest['rows']:,} ECGs from {manifest['patients']:,} patients "
          f"and {positives:,} AFIB-labelled records ({positives / manifest['rows'] * 100:.2f}%). "
          "AFIB means presence in source SCP annotations, including "
          "likelihood zero. Models use 504 lead fields; identifiers, quality flags and 27 Global fields stay "
          "outside predictors. Enhanced construction adds 63 fields before encoding and missingness handling.")
    doc.table(["Arm", "Predictors", "Training views"], [
        ["A", "Original lead measurements", "Observed only"],
        ["B", "Original plus enhanced fields", "Observed only"],
        ["C", "Original lead measurements", "Observed plus one corrupted copy per ECG"],
        ["D", "Original plus enhanced fields", "Observed plus one corrupted copy per ECG"],
    ], [1.2, 7.1, 9.1])
    doc.p("Training copies use three missing leads, additive noise 0.10, gain 0.10, or the combined condition. "
          "The augmentation stream is separate from evaluation. Evaluation includes observed data once and "
          "ten stress scenarios with three repeats each. Noise is scaled by training-field IQR; gain is "
          "exp(sigma times a standard-normal draw) per lead. These are measurement-space stress tests, "
          "not calibrated ECG waveform artefacts.")
    doc.table(["Protocol", "Logistic regression", "TabM"], [
        ["Validation", "10 outer and 3 inner patient folds", "Same patient folds and prepared data hashes"],
        ["Selection", "Mean inner observed macro-F1 at 0.5", "Same criterion; fewer blocks on exact ties"],
        ["Tuning", "C 0.01, 0.1, 1; one endpoint refinement", "Blocks 2, 3, 4; one endpoint refinement"],
        ["Fit", "Weighted L2 logistic loss; L-BFGS; max 2,000 iterations", "Width 128; 16 members; dropout 0.1; AdamW; 100 epochs"],
        ["Runtime", "CPU numerical optimizer", "CUDA float32; deterministic algorithms; batch 256"],
        ["Secondary threshold", "Selected using inner OOF sensitivity target 0.90", "Same rule; fixed across outer stresses"],
    ], [3.3, 7.05, 7.05])
    doc.p("Learned filling, scaling, encoding and model selection use training patients only. The saved manifests "
          f"match across models, as do all {r.pairing['conditions_compared']:,} saved test-condition perturbation "
          f"audits across {r.pairing['fold_variants_compared']} shared fold-variants. "
          "The shared perturbation implementation keys draws by ECG ID, repeat and stream. "
          "Matched folds and stresses support paired descriptive comparisons, but overlap among outer training "
          "sets means the ten scores are not ten independent datasets. Differences in model capacity, tuning "
          "parameterization and training budget prevent attributing every difference to architecture alone.")

    doc.page()
    doc.heading("Observed performance across the four main arms")
    rows = [[NAMES[m] + " " + arm, f(m, arm, sd=True), f(m, arm, "sensitivity"),
             f(m, arm, "specificity"), f(m, arm, "precision"), f(m, arm, "brier")]
            for m in MODELS for arm in ARMS]
    doc.table(["Model and arm", "Macro F1 (SD)", "Sensitivity", "Specificity", "Precision", "Brier"],
              rows, [3.9, 3.5, 2.5, 2.5, 2.5, 2.5])
    doc.p("Source: each model's output_data/summary.csv; observed, fixed_0.50. Higher is better except Brier.")
    doc.heading("Ranking and probability quality")
    doc.table(["Model and arm", "AUROC", "Average precision", "Balanced accuracy"],
              [[NAMES[m] + " " + arm, f(m, arm, "auroc"), f(m, arm, "average_precision"),
                f(m, arm, "balanced_accuracy")] for m in MODELS for arm in ("A", "D")], [6.5, 3.4, 3.6, 3.9])
    doc.p(f"TabM D improves the observed mean macro-F1 over logistic D by "
          f"{v(tm, 'D') - v(lr, 'D'):.4f}, and lowers Brier by "
          f"{v(lr, 'D', 'brier') - v(tm, 'D', 'brier'):.4f}. Its sensitivity is "
          f"{v(lr, 'D', 'sensitivity') - v(tm, 'D', 'sensitivity'):.4f} lower. "
          "This is a shift in the operating tradeoff as well as a gain in ranking metrics; "
          "macro-F1 alone does not describe which errors matter.")
    baseline = r.csv(tm, "majority_baseline")
    doc.p(f"The training-majority baseline has mean accuracy "
          f"{statistics.mean(float(x['accuracy']) for x in baseline):.4f} and macro-F1 "
          f"{statistics.mean(float(x['macro_f1']) for x in baseline):.4f}, with zero sensitivity. "
          "High accuracy alone is therefore inadequate for this imbalanced task. Average precision and "
          "class-specific errors remain central when assessing the benefit over a trivial baseline.")
    doc.p("Observed means slightly favor TabM D over B, but the effect is small relative to fold variability. "
          "Do not select D solely because it has the largest clean mean. Its more persuasive advantage is "
          "robustness under lead loss, which must be weighed against runtime and the intended operating threshold.")

    doc.page()
    doc.heading("Logistic regression observations")
    ba, ba_sd = r.difference(lr, "B minus A")
    db, db_sd = r.difference(lr, "D minus B")
    doc.p(f"Enhanced features improve observed macro-F1 by {ba:+.4f} (paired-fold SD {ba_sd:.4f}) for B minus A. "
          f"Augmentation adds {db:+.4f} (SD {db_sd:.4f}) for D minus B. The improvements are modest; "
          "B slightly reduces sensitivity while improving specificity and precision. D offers the highest "
          "observed mean macro-F1 among the logistic main arms.")
    doc.table(["Condition", "Arm", "Macro F1", "Sensitivity", "Specificity"],
              [[LABELS[s], arm, f(lr, arm, scenario=s), f(lr, arm, "sensitivity", s),
                f(lr, arm, "specificity", s)] for s in ("missing_3", "missing_6") for arm in ARMS],
              [4.3, 1.3, 3.6, 4.1, 4.1])
    doc.p(f"Original observed-only A is fragile under six missing leads: sensitivity falls from "
          f"{f(lr, 'A', 'sensitivity')} to {f(lr, 'A', 'sensitivity', 'missing_6')}. "
          f"Enhanced B recovers sensitivity to {f(lr, 'B', 'sensitivity', 'missing_6')}. "
          "This suggests that enhanced measurements help the linear classifier cope with incomplete inputs, "
          "although the aggregate comparison does not identify which constructed features are responsible.")
    doc.p(f"Augmented D reaches sensitivity {f(lr, 'D', 'sensitivity', 'missing_6')} with six missing leads, "
          f"but specificity drops to {f(lr, 'D', 'specificity', 'missing_6')} and precision to "
          f"{f(lr, 'D', 'precision', 'missing_6')}. D therefore has lower macro-F1 than B under this stress. "
          "At three missing leads and under combined stress, augmentation likewise does not maximize logistic "
          "macro-F1 at 0.5. The positive class becomes easier to detect at the cost of more false positives.")
    selections = r.csv(lr, "selected_hyperparameters")
    doc.heading("Optimization and selection")
    doc.p(f"The saved completion report records all {r.complete[lr]['n_fits']} fits as converged, with maximum "
          f"gradient {r.complete[lr]['max_gradient']:.3g}. All {len(selections)} fold-variant selections finish "
          "at the middle of their final grid. This supports numerical completion within the specified search, "
          "not global optimality over all model or preprocessing choices.")
    doc.table(["Arm", "Selected C across ten outer folds"],
              [[arm, "; ".join(f"{key}: {count} folds" for key, count in sorted(Counter(
                  row["C"] for row in selections if row["variant"] == arm).items()))] for arm in ARMS], [2.0, 15.4])

    doc.page()
    doc.heading("TabM observations and training cost")
    ba, ba_sd = r.difference(tm, "B minus A")
    db, db_sd = r.difference(tm, "D minus B")
    doc.p(f"Enhanced B improves observed macro-F1 over A by {ba:+.4f} (paired-fold SD {ba_sd:.4f}). "
          f"D improves over B by only {db:+.4f} (SD {db_sd:.4f}) on observed data. "
          "The current results favor enhanced features for clean performance; evidence for augmentation "
          "comes mainly from stress behavior.")
    doc.table(["Condition", "Arm", "Macro F1", "Sensitivity", "Specificity"],
              [[LABELS[s], arm, f(tm, arm, scenario=s), f(tm, arm, "sensitivity", s),
                f(tm, arm, "specificity", s)] for s in ("missing_3", "missing_6") for arm in ARMS],
              [4.3, 1.3, 3.6, 4.1, 4.1])
    doc.p(f"Six missing leads reduce A macro-F1 to {f(tm, 'A', scenario='missing_6')} and B to "
          f"{f(tm, 'B', scenario='missing_6')}. Augmented C and D retain "
          f"{f(tm, 'C', scenario='missing_6')} and {f(tm, 'D', scenario='missing_6')}. "
          "High specificity in A/B coexists with very poor sensitivity, so a stable-looking overall "
          "accuracy would hide missed AFIB records. Augmentation addresses this behavior in the tested setting.")
    selections = r.csv(tm, "selected_hyperparameters")
    doc.table(["Arm", "Selected blocks across ten outer folds"],
              [[arm, "; ".join(f"{key}: {count}" for key, count in sorted(Counter(
                  row["n_blocks"] for row in selections if row["variant"] == arm).items()))] for arm in ARMS], [2, 15.4])
    doc.p(f"Across all {len(selections)} TabM fold-variants, "
          f"{sum(x['refined'] == 'True' for x in selections)} trigger endpoint refinement and "
          f"{sum(x['selected_middle'] == 'True' for x in selections)} select a middle value. "
          "Endpoint selections motivate an inner-validation study of capacity; they do not authorize "
          "expanding the grid after inspecting outer test outcomes.")
    doc.p(f"Completed-fold wall times sum to {r.complete[tm]['completed_fold_runtime_seconds'] / 3600:.2f} hours. "
          "This is summed checkpoint work, not end-to-end elapsed time or GPU time, and parallel scheduling "
          "prevents direct interpretation as user waiting time. Training histories contain training loss; "
          "falling training loss alone does not establish generalization or an optimal epoch count.")

    doc.page()
    doc.heading("Complete stress profile at the primary threshold")
    for model in MODELS:
        doc.heading(NAMES[model] + " macro F1", level=2)
        doc.table(["Scenario", "A", "B", "C", "D"],
                  [[LABELS[s]] + [f(model, a, scenario=s) for a in ARMS] for s in SCENARIOS],
                  [6.2, 2.8, 2.8, 2.8, 2.8])
    doc.p("Source: output_data/summary.csv, fixed_0.50. Noise doses are fractions of the training IQR; gain "
          "doses are lognormal sigma values. Three repeats per stress are averaged within folds. All rows "
          "use the same ten outer folds; the full comparison CSV retains SD and defined-fold counts.")
    doc.p("Lead loss is the largest threat to observed-only TabM. Additive noise and gain perturbations up to "
          "the tested 0.20 doses have much smaller macro-F1 effects. Slight score increases under a stress "
          "are possible because a fixed threshold changes the balance of errors; they do not show that "
          "corruption improves measurement quality. Do not average all scenarios into one deployment score "
          "without a justified distribution of real operating conditions.")

    doc.page()
    doc.heading("Controls and what they establish")
    doc.p("Availability controls preserve original predictors and 63 constructed missingness indicators while "
          "zeroing the constructed numerical values. Duplicate controls repeat observed rows with half "
          "weights. They address different questions: whether enhanced numerical values add information, "
          "and whether a change can be explained by repeating examples rather than corrupting them.")
    comparisons = ("B minus availability_observed", "D minus availability_augmented",
                   "duplicate_original minus A", "duplicate_enhanced minus B")
    rows = []
    for model in MODELS:
        for comparison in comparisons:
            clean, clean_sd = r.difference(model, comparison)
            stress, stress_sd = r.difference(model, comparison, "missing_6")
            rows.append([NAMES[model], comparison, f"{clean:+.4f} ({clean_sd:.4f})",
                         f"{stress:+.4f} ({stress_sd:.4f})"])
    doc.table(["Model", "Comparison", "Observed difference (SD)", "6 leads difference (SD)"],
              rows, [3.4, 6.0, 4.0, 4.0])
    doc.p("Differences are paired within outer folds for macro-F1 at 0.5. Positive values favor the first "
          "method. Source: output_data/paired_method_differences.csv. These are descriptive differences, "
          "with no multiple-comparison correction or independent significance test.")
    doc.p("Enhanced B outperforms its availability control in both models, especially under six-lead loss. "
          "This supports useful information in the constructed numerical values beyond the control's "
          "missingness representation. It does not isolate any individual feature family or prove a causal "
          "physiological mechanism; feature-family ablations remain necessary.")
    doc.p("Logistic duplicate controls reproduce observed macro-F1 exactly and differ only slightly under "
          "stress. Half-weight duplication preserves the logistic objective, consistent with their role as "
          "a check against sample-count effects. TabM duplicates have small nonzero differences. Repeated "
          "rows alter minibatch scheduling and the number of optimizer updates per epoch; neural training "
          "need not reproduce the same fitted function even with equal total sample weight.")
    doc.p("TabM augmentation gains under severe lead loss are much larger than the corresponding duplicate "
          "control changes. A stronger future control would match optimizer update count and compute budget "
          "as well as total example weight, with several training seeds. Clean augmentation gains are small "
          "enough that seed and optimization variability deserve particular attention.")

    doc.page()
    doc.heading("Threshold choice and probability calibration")
    secondary = "inner_sensitivity_0.90"
    doc.table(["Model and arm", "F1 at 0.5", "F1 at inner rule", "Sensitivity at inner rule", "Specificity at inner rule"],
              [[NAMES[m] + " " + a, f(m, a), f(m, a, policy=secondary),
                f(m, a, "sensitivity", policy=secondary), f(m, a, "specificity", policy=secondary)]
               for m in MODELS for a in ARMS], [4.2, 2.6, 3.1, 3.75, 3.75])
    doc.p("The secondary rule maximizes specificity among inner OOF thresholds reaching sensitivity at least "
          "0.90. It is then frozen before scoring the outer fold and every stress. The target applies to "
          "inner OOF predictions; it is not a guaranteed outer sensitivity, per-fold minimum, or stress guarantee.")
    doc.p(f"Logistic D moves from observed macro-F1 {f(lr, 'D')} at 0.5 to "
          f"{f(lr, 'D', policy=secondary)} under the inner rule, while sensitivity moves from "
          f"{f(lr, 'D', 'sensitivity')} to {f(lr, 'D', 'sensitivity', policy=secondary)}. "
          "The alternate threshold reduces false positives by accepting a lower recall level. TabM's "
          "observed threshold adjustment is smaller. This reinforces that model comparison should include "
          "matched operating objectives, rather than only the same numerical threshold.")
    doc.heading("Calibration requires a separate assessment")
    doc.p(f"Observed D Brier is {f(lr, 'D', 'brier')} for logistic regression and {f(tm, 'D', 'brier')} "
          "for TabM. Lower Brier indicates better mean squared probability error on this population, but "
          "does not by itself establish calibration across probability ranges or deployment prevalences. "
          "Class weighting changes the training objective; predicted scores should not automatically be "
          "interpreted as calibrated prevalence estimates.")
    doc.p("Inspect saved ten-bin reliability tables with bin counts and AFIB counts, including sparse "
          "high-score bins. Add calibration intercept and slope, patient-resampled uncertainty and "
          "stress-specific reliability plots. Any recalibrator must be learned using training-only "
          "predictions inside the validation design. Test threshold policies at prespecified sensitivity "
          "or false-positive budgets and report the fraction of outer folds meeting the target.")
    doc.p("The same held-out patients appear in several stress scenarios. Repeats, scenarios, ECGs from one "
          "patient and overlapping training folds are not independent replicates. Patient-cluster bootstrap "
          "intervals on aligned OOF predictions can describe test-patient uncertainty conditional on the "
          "fitted models; repeated complete experiments are needed to assess training and split variability.")

    doc.page()
    doc.heading("Patient weighting quality groups and filling")
    rows = []
    for model in MODELS:
        for arm in ("A", "D"):
            subset = [x for x in r.csv(model, "patient_weighted_metrics_summary") if
                      x["variant"] == arm and x["scenario"] == "observed" and
                      x["threshold_policy"] == POLICY and x.get("metric", "macro_f1") == "macro_f1"]
            value = float(subset[0].get("mean", subset[0].get("macro_f1_mean")))
            rows.append([NAMES[model] + " " + arm, f(model, arm), f"{value:.4f}"])
    doc.table(["Model and arm", "Record macro F1", "Patient weighted macro F1"], rows, [6.5, 5.45, 5.45])
    doc.p("Patient weighting gives each patient total weight one across their ECG records. The observed "
          "A/D results remain close to the record-level means, suggesting repeat ECGs are not driving "
          "the main observed ranking. This is a sensitivity check, not external validation.")
    rows = []
    for model in MODELS:
        source = r.csv(model, "quality_subgroups_summary")
        for group in ("no_recorded_quality_flag", "recorded_quality_flag"):
            entries = [x for x in source if x["variant"] == "D" and x["scenario"] == "observed"
                       and x["threshold_policy"] == POLICY and x["subgroup"] == group
                       and x.get("metric", "macro_f1") == "macro_f1"]
            item = entries[0]
            rows.append([NAMES[model] + " D", group.replace("_", " "),
                         f"{float(item.get('mean', item.get('macro_f1_mean'))):.4f}",
                         f"{float(item.get('sd', item.get('macro_f1_std'))):.4f}"])
    doc.table(["Model", "Source quality group", "Macro F1", "SD"], rows, [4.2, 7.0, 3.1, 3.1])
    q = [x for x in r.csv(lr, "quality_subgroups") if x["variant"] == "A"
         and x["scenario"] == "observed" and x["threshold_policy"] == POLICY]
    counts = {g: (sum(int(x["n"]) for x in q if x["subgroup"] == g),
                  sum(int(x["n_positive"]) for x in q if x["subgroup"] == g))
              for g in ("recorded_quality_flag", "no_recorded_quality_flag")}
    doc.p(f"The flagged group contains {counts['recorded_quality_flag'][0]:,} ECGs and "
          f"{counts['recorded_quality_flag'][1]:,} AFIB records; the unflagged group contains "
          f"{counts['no_recorded_quality_flag'][0]:,} ECGs and {counts['no_recorded_quality_flag'][1]:,} AFIB records. "
          "Flags are source annotations, not measured severity or proof of clean signals. Group differences "
          "may reflect prevalence and patient composition; demographic fairness has not been established.")
    doc.heading("Separate ridge filling experiment", level=2)
    recovery = r.csv(lr, "recovery_summary")
    doc.table(["Mask", "Median normalized MAE", "Ridge normalized MAE"],
              [[s, *[f"{float(next(x['mean'] for x in recovery if x['method'] == method and x['scenario'] == s)):.4f}"
                     for method in ("median", "ridge")]]
               for s in ("cells_0.10", "missing_1", "missing_3", "missing_6")], [5.2, 6.1, 6.1])
    doc.p(f"Recovery MAE uses eligible masked continuous cells and training-IQR scaling. Ridge improves "
          "this reconstruction measure, with less advantage for whole-lead loss. Its downstream logistic "
          f"macro-F1 is {f(lr, 'ridge')} observed (A {f(lr, 'A')}) and "
          f"{f(lr, 'ridge', scenario='missing_3')} with three missing leads (A "
          f"{f(lr, 'A', scenario='missing_3')}). Reconstruction improvement is not equivalent to "
          "classification improvement. Inspect coverage, excluded targets and per-feature errors before "
          "generalizing beyond the scored cells. No TabM ridge experiment was run.")

    doc.page()
    doc.heading("Priority improvements and further evaluation")
    doc.table(["Priority", "Evaluation", "Decision it should support"], [
        ["1", "Complete histogram gradient boosting with the same frozen folds, stresses, thresholds and controls.",
         "Finish the planned three-model comparison without changing the two completed baselines."],
        ["1", "Align OOF prediction keys; report paired per-patient differences and uncertainty. Predeclare primary model contrasts and stress endpoints.",
         "Determine which gains are stable enough to support model and arm selection; avoid treating many comparisons as independent evidence."],
        ["1", "Compare prespecified sensitivity targets and false-positive budgets; inspect reliability and calibrate only within training partitions.",
         "Separate ranking improvements from threshold and probability-scale effects."],
        ["2", "Repeat TabM with several training seeds; assess repeated patient splits when compute permits.",
         "Quantify whether small clean gains exceed seed and split variability."],
        ["2", "Ablate enhanced feature families and individual augmentation types, matching duplicate controls, update count and compute.",
         "Identify the source of robustness and the augmentation that costs clean precision or runtime."],
        ["2", "Test lead identities, correlated missingness, heavier doses and naturally poor-quality subgroups; validate on a genuinely external dataset.",
         "Measure transfer beyond synthetic random lead loss and this source population."],
        ["3", "Investigate TabM capacity, epoch count, learning rate, numerical embeddings or alternative preprocessing through inner validation only.",
         "Explore better accuracy per unit compute as a new experiment with new provenance."],
        ["3", "For ridge filling, analyze coverage and per-feature errors, then test downstream usefulness under additional stresses.",
         "Decide whether the reconstruction improvement justifies complexity and cost."],
    ], [1.6, 8.2, 7.6])
    doc.heading("Faster runs without changing the scientific experiment")
    doc.p("Reuse complete, signature-matched inner candidates and outer checkpoints; regenerate reports from "
          "saved predictions instead of retraining. Profile preprocessing, tensor transfer, training and "
          "serialization separately before choosing a worker count. Small isolated pilots should measure "
          "throughput and GPU memory under the target hardware. Reusing cached prepared matrices can save "
          "CPU work only if the cache key includes fold, variant, input hashes and preprocessing configuration.")
    doc.p("Changes to batch size, epoch count, mixed precision, early stopping, architecture or optimizer "
          "may change the result and require a new output directory and configuration. A reduced pilot "
          "cannot replace the completed fixed-protocol evidence. The current TabM fit uses 100 epochs "
          "without early stopping; the saved checkpoint restores completed inference state, not a "
          "partially completed optimizer trajectory.")

    doc.page()
    doc.heading("Fast refresh and reproducible environments")
    doc.p("For the completed TabM experiment, refresh tables and figures and independently validate "
          "the full saved prediction bundle without fitting any models:")
    doc.code((".venv/Scripts/python algorithms/tabm/run_full_pipeline.py --report-only --skip-notebook",))
    doc.p("This command requires the complete plan to pass verification and writes reports/validation.json. "
          "Omit --skip-notebook to refresh notebook displays too. The quicker run_experiment.py --phase report "
          "command only rebuilds tables and figures; it does not independently recompute every prediction "
          "metric. Neither command publishes results unless the separate --publish option is explicitly used.")
    doc.heading("Preparation reuse in future parallel runs", level=2)
    doc.p("The parallel runner separates scheduling from the serial scientific protocol. Its inner mode "
          "groups all missing block candidates for one inner split into one worker job, so the prepared "
          "arrays are built once and reused across those candidates. A bounded pool controls total worker "
          "count; there are no nested pools. Inner mode releases the prepared split after its group finishes, "
          "while outer mode uses the serial runner's cache of up to three inner splits per worker.")
    doc.code((".venv/Scripts/python algorithms/tabm/run_parallel.py --device cuda --workers 2 --parallel-level inner --dry-run",))
    doc.p("The dry run inspects scheduling without data loading, CUDA access, model fitting or output writes. "
          "Actual parallel training defaults to results/tabm_parallel/, separate from the published serial "
          "run. Preparation reuse removes repeated CPU work by construction; no end-to-end training speedup "
          "is claimed until it is measured on the target GPU, dataset and worker configuration.")
    doc.heading("Keep the recorded environments separate", level=2)
    versions = {model: r.config[model]["versions"] for model in MODELS}
    doc.table(["Recorded package", "Logistic regression", "TabM"],
              [[key, versions[lr].get(key, "Not recorded"), versions[tm].get(key, "Not recorded")]
               for key in ("python", "numpy", "pandas", "scipy", "torch", "tabm")], [4.8, 6.3, 6.3])
    doc.p("The Python and NumPy versions differ between the historical runs. Use "
          "algorithms/logistic_regression/requirements-recorded.txt for the recorded logistic packages "
          "and algorithms/tabm/requirements.txt with the documented CUDA PyTorch wheel for TabM. "
          "The broad root requirements are convenient development requirements, not a historical environment "
          "lock. Install each model in its own environment when reproducing fitted results. Document "
          "generation uses the separate requirements-docs.txt and does not require either training stack.")
    for relative in ("algorithms/tabm/run_full_pipeline.py", "algorithms/tabm/run_parallel.py",
                     "algorithms/tabm/parallel_runtime.py", "algorithms/tabm/requirements.txt",
                     "algorithms/logistic_regression/requirements-recorded.txt"):
        r.read(relative)

    doc.page()
    doc.heading("Reproduction evidence and source map")
    doc.p("The document builder reads the saved result CSVs and JSON configurations; it performs no fitting "
          "and never loads pickles. It writes the Word and Markdown versions from one content definition "
          "and records input SHA-256 hashes beside the comparison tables. Original model result files "
          "remain the numerical source of truth.")
    doc.table(["Evidence", "Repository location and role"], [
        ["Experiment identity", "results/<model>/run_config.json: data hashes, software versions, selected scope and code hashes"],
        ["Completion", "results/<model>/reports/completion.json: counts and completed plan scope"],
        ["Primary results", "results/<model>/output_data/summary.csv and metrics.csv: fold-mean summaries and per-condition metrics"],
        ["Controls and stresses", "output_data/paired_method_differences.csv and paired_stress_drops.csv: within-model paired descriptive effects"],
        ["Predictions and audit", "output_data/folds/: inner OOF, outer predictions, selection and perturbation audits"],
        ["Secondary analyses", "output_data/patient_weighted_metrics_summary.csv, quality_subgroups*.csv, reliability.csv and logistic recovery*.csv"],
        ["Generated comparison", "results/comparison/output_data/observations_main_summary.csv, observations_method_differences.csv and observations_input_manifest.json"],
        ["Stress matching", "results/comparison/output_data/observations_pairing_validation.json: repeatable comparison of all common test perturbation audits"],
        ["Prepared data reproduction", "scripts/verify_preprocessing.py and results/comparison/reports/preprocessing_validation.json: rebuild and compare raw-to-prepared data hashes"],
        ["Generator", "scripts/build_model_observations.py and requirements-docs.txt"],
    ], [4.0, 13.4])
    doc.heading("Commands from the repository root", level=2)
    doc.code(("python -m pip install -r requirements-docs.txt",
                    "python scripts/build_model_observations.py",
                    "python scripts/verify_preprocessing.py",
                    "python algorithms/logistic_regression/check_results.py",
                    "python algorithms/tabm/check_results.py"))
    doc.p("The builder needs only python-docx plus Python's standard library and saved result tables. "
          "Numerical checkers additionally use the model environments and prepared data. Regenerating "
          "this document is distinct from independently validating all predictions and from retraining "
          "a model. See the algorithm READMEs for exact run and resume commands.")
    preprocessing_path = "results/comparison/reports/preprocessing_validation.json"
    if (r.root / preprocessing_path).is_file():
        preprocessing = r.json(preprocessing_path)
        if preprocessing.get("status") != "passed" or not all(
                row.get("matches") for row in preprocessing.get("comparisons", {}).values()):
            raise ValueError("Prepared-data reproduction evidence does not pass.")
        doc.p("The saved preprocessing validation rebuilds all four prepared-data files from the source "
              "CSVs and matches their bytes and complete manifest to the existing package and both "
              "recorded model configurations. The existing processed package is not modified by that check.")
    doc.p("For a full fresh reproduction, retrieve the cited source versions, verify the recorded SHA-256 "
          "hashes, rebuild the prepared package and use the recorded per-model environments. Preserve "
          "the historical configuration and source-provenance snapshots. Hardware, CUDA and library "
          "changes can alter neural numerical results despite fixed seeds; claim exact equality only "
          "after comparing outputs in a matched environment. Compare future experiments in separate "
          "folders and document every intentional protocol change.")
    doc.p("Data attribution: PTB-XL+ 1.0.1 (https://physionet.org/content/ptb-xl-plus/1.0.1/) and "
          "PTB-XL 1.0.3 (https://physionet.org/content/ptb-xl/1.0.3/), CC BY 4.0. "
          "This is retrospective classification and controlled stress evaluation; it does not establish "
          "clinical diagnostic benefit, deployment reliability or external generalization.")
    return doc


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Word output path; Markdown is saved alongside it")
    parser.add_argument("--comparison-output", type=Path, help="Directory for generated comparison tables")
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = Results(root)
    output = args.output or root / "docs" / "Model_Observations.docx"
    comparison = args.comparison_output or root / "results" / "comparison" / "output_data"
    comparison.mkdir(parents=True, exist_ok=True)
    report = build_report(result)
    report.save(output)
    main_rows, paired = [], []
    for model in MODELS:
        main_rows.extend({"model": model, **row} for row in result.csv(model, "summary")
                         if row["variant"] in ARMS)
        for row in result.csv(model, "paired_method_differences"):
            paired.append({"model": model, "threshold_policy": POLICY, "n_defined": "", **row})
    main_rows.sort(key=lambda x: (x["model"], x["variant"], x["scenario"], x["threshold_policy"], x["metric"]))
    paired.sort(key=lambda x: (x["model"], x["comparison"], x["scenario"], x["threshold_policy"], x["metric"]))
    write_csv(comparison / "observations_main_summary.csv", main_rows,
              ["model", "variant", "scenario", "threshold_policy", "metric", "mean", "sd", "n_folds", "n_defined"])
    write_csv(comparison / "observations_method_differences.csv", paired,
              ["model", "comparison", "scenario", "threshold_policy", "metric", "direction",
               "n_folds", "n_defined", "mean_difference", "sd_difference"])
    (comparison / "observations_pairing_validation.json").write_text(
        json.dumps(result.pairing, indent=2) + "\n", encoding="utf-8", newline="\n")
    provenance = {"schema_version": 1, "report_date": "2026-10-13", "models": list(MODELS),
                  "aggregation": "Three stress repeats averaged within each fold; mean and sample SD across ten folds.",
                  "inference": "Descriptive comparisons only; SD is not a confidence interval.",
                  "schema_normalization": "Logistic paired tables record fixed_0.50 only; n_defined is blank because the source omits it.",
                  "sources": dict(sorted(result.inputs.items())),
                  "generator_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    (comparison / "observations_input_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {output}\nWrote {output.with_suffix('.md')}\nWrote comparison tables to {comparison}")


if __name__ == "__main__":
    main()

"""Build the four-page report after running ``report/analyze_results.py``.

The narrative is kept here with the layout so every submitted sentence remains
editable. Numerical tables and the figure come from saved experiment outputs.
"""
from pathlib import Path
import csv
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Spacer

REPORT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_PDF = REPORT_DIRECTORY / 'COMP90051_Final_Report.pdf'
# Platypus reserves six points of padding on each side of its text frame.
CONTENT_WIDTH = A4[0] - 32*mm - 12
paragraph_styles = {
    'body': ParagraphStyle('body', fontName='Times-Roman', fontSize=10, leading=11.6, spaceAfter=5, textColor=colors.black),
    'heading': ParagraphStyle('heading', fontName='Times-Bold', fontSize=10, leading=11.7, spaceBefore=6, spaceAfter=4, keepWithNext=True, textColor=colors.black),
    'title': ParagraphStyle('title', fontName='Times-Bold', fontSize=10, leading=12, spaceAfter=5, textColor=colors.black),
    'caption': ParagraphStyle('caption', fontName='Times-Roman', fontSize=10, leading=11.2, spaceBefore=3, spaceAfter=5, textColor=colors.black),
    'cell': ParagraphStyle('cell', fontName='Times-Roman', fontSize=10, leading=11.2, spaceAfter=0, textColor=colors.black),
    'header': ParagraphStyle('header', fontName='Times-Bold', fontSize=10, leading=11.2, textColor=colors.black),
    'reference': ParagraphStyle('reference', fontName='Times-Roman', fontSize=10, leading=11.1, leftIndent=14, firstLineIndent=-14, spaceAfter=4, textColor=colors.black),
}
metric_rows = list(csv.DictReader((REPORT_DIRECTORY/'results/model_summary.csv').open()))
summary_metrics = {(r['model'],r['variant'],r['scenario'],r['metric']):(float(r['mean']),float(r['sd'])) for r in metric_rows}
MODEL_NAMES = {'LR':'logistic_regression',
               'HGB':'histogram_gradient_boosting_retuned','TabM':'tabm'}
TABLE_MODELS = {'LR':'logistic_regression',
                'HGB earlier':'histogram_gradient_boosting_earlier',
                'HGB 7-9':'histogram_gradient_boosting_retuned',
                'TabM':'tabm'}
report_elements = []
text_blocks = []
content_blocks = []

def add_paragraph(text, style='body'):
    """Record the narrative alongside the flowable used for PDF layout."""
    report_elements.append(Paragraph(text, paragraph_styles[style]))
    text_blocks.append(text)
    content_blocks.append({'type': 'paragraph', 'text': text, 'style': style})


def add_heading(text):
    add_paragraph(text, 'heading')


def format_metric(model, arm, scenario='observed', metric='macro_f1', digits=3):
    """Format equal-fold means and sample SDs calculated from result files."""
    model_key = MODEL_NAMES.get(model, TABLE_MODELS.get(model))
    mean,sd=summary_metrics[(model_key,arm,scenario,metric)]
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'


def add_table(headers, data, widths):
    """Wrap table cells at the same 10-point font used in the report body."""
    cells=[[Paragraph(escape(str(v)),paragraph_styles['header']) for v in headers]]
    cells += [[Paragraph(escape(str(v)),paragraph_styles['cell']) for v in row] for row in data]
    table = Table(cells, colWidths=[width*CONTENT_WIDTH for width in widths], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF2')),
        ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D9D9D9')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
    ]))
    report_elements.append(table)
    report_elements.append(Spacer(1, 4))
    text_blocks.append('\n'.join(' | '.join(map(str,row)) for row in [headers]+data))
    content_blocks.append({'type': 'table', 'headers': headers, 'data': data,
                           'widths': widths})


def add_page_break():
    report_elements.append(PageBreak())
    text_blocks.append('\n[PAGE BREAK]\n')
    content_blocks.append({'type': 'page_break'})

add_paragraph('AFIB classification under missing and perturbed ECG measurements','title')
add_paragraph('Jiayi Huang 1668763  |  Siwen Wang 1721961  |  Aaditya Pathak 1703474')
add_heading('1 Introduction')
add_paragraph('Can cross-lead feature construction and training with corrupted measurements preserve atrial fibrillation (AFIB) classification when inputs are missing or perturbed, and does the benefit depend on model flexibility? We compare these interventions within linear, tree and neural models, rather than simply rank classifiers. A separate experiment asks whether learned filling improves AFIB prediction as well as measurement reconstruction. We evaluate absolute performance and deterioration together: a weak model can lose little under stress while remaining unhelpful. Our main finding is that construction helps under severe lead loss, whereas augmentation helps the flexible models most and can worsen the balance of errors for logistic regression.')
add_paragraph('We join public PTB-XL+ 1.0.1 ECGDeli features [1] to PTB-XL 1.0.3 annotations [2]: 21,799 ECGs from 18,869 patients, including 1,514 AFIB-labelled records (6.95%). A record is positive when its annotation contains AFIB, including a likelihood value of zero; non-AFIB does not mean healthy. The 504 lead fields comprise 480 numerical and 24 categorical fields, exceeding the required dataset size without padding. We exclude 27 Global fields, identifiers, diagnoses and quality flags from predictors. In particular, excluding Global summaries prevents hidden leads from remaining visible through precomputed aggregates. Data URLs are <link href="https://physionet.org/content/ptb-xl-plus/1.0.1/">https://physionet.org/content/ptb-xl-plus/1.0.1/</link> and <link href="https://physionet.org/content/ptb-xl/1.0.3/">https://physionet.org/content/ptb-xl/1.0.3/</link> (CC BY 4.0).')
add_paragraph('Always predicting non-AFIB gives 93.05% accuracy while missing every positive record, motivating macro-F1 and class-specific errors. Natural gaps occur in 49.08% of ECGs but only 2.61% of lead-feature cells. Complete-case deletion would therefore discard almost half the cohort and could change its composition. We retain incomplete records and keep all recordings from each patient in the same validation partition.')
add_heading('2 Related work')
add_paragraph('PTB-XL supplies annotated waveforms [2], while PTB-XL+ supplies extracted ECG measurements [1], enabling a reproducible study of feature representations. Sisk et al. [3] show that imputation and missing indicators must be assessed in the intended prediction setting; reconstruction quality alone is insufficient. This motivates our separate recovery and downstream tests. TabM [4], published at the ICLR 2025 main conference and not covered in our classes, combines multilayer perceptrons (MLPs) through parameter-efficient ensembling. Its tabular benchmark results motivate a flexible comparator, but do not establish robustness for AFIB. Our contribution is to test representation and augmentation jointly across model classes under a common patient-separated protocol.')
add_heading('3 Methods')
add_paragraph('<b>Representation.</b> Before filling gaps, we construct 63 fields: 52 summaries across 13 measurement families (median, interquartile range, range and chest-minus-limb median); seven median ratios of beat-level IQR to duration; and four squares/products involving QRS duration, QT interval and P-wave amplitude/duration. Aggregates use available leads, and undefined values remain missing. We rebuild them after every corruption so hidden measurements cannot leak through derived features. Aggregation can retain redundant information from surviving leads; products and squares expand the decision functions available to a linear classifier.')
add_table(['Arm','Feature view','Training views per ECG'],[
    ['A','504 original lead fields','Observed'],['B','Original plus 63 constructed fields','Observed'],
    ['C','504 original lead fields','Observed plus one corrupted copy'],['D','Original plus 63 constructed fields','Observed plus one corrupted copy']], [.09,.46,.45])
add_paragraph('<b>Preprocessing.</b> Numerical medians, means and standard deviations are learned only from the current observed training partition. We retain missingness indicators, standardise numerical fields and one-hot encode morphology with explicit missing/unknown categories. All-missing training columns use zero; constant columns use scale one. Frozen transformations are applied to held-out records. Common finite inputs help control the comparison, although HGB consequently does not use its native missing-value routing.')
add_paragraph('<b>Implementation checks.</b> Tests cover patient separation, training-only transformations, preserved gaps, corruption pairing and hand-calculated metrics. Recalculation of 10.9 million LR/TabM prediction rows and 2.7 million newly supplied HGB rows reproduces saved classification and probability metrics. HGB labels and held-out ECG/fold keys match LR; all 360 inner scores support the 40 saved depth selections. HGB historical environment and perturbation manifests remain incomplete: these checks validate saved outputs, not independent retraining or identical stress draws across models.')

add_page_break()
add_heading('3 Methods continued')
add_table(['Classifier and role','Tuned parameter and fixed settings'],[
    ['Logistic regression (LR)\nLinear reference','L2 inverse penalty C = {0.01, 0.1, 1}; unpenalised intercept; own weighted loss and analytic gradient; SciPy L-BFGS, at most 2,000 iterations.'],
    ['Histogram gradient boosting (HGB)\nConstrained nonlinear trees','Maximum depth = {7, 8, 9}; 150 iterations, learning rate 0.05, at most 16 leaves, at least 30 rows per leaf, L2 = 1; no early stopping.'],
    ['TabM\nNeural ensemble [4]','Blocks = {2, 3, 4}; width 128, 16 members, dropout 0.1; AdamW learning rate 0.0003, weight decay 0.0001, batch size 256, 100 epochs; no numerical embeddings or early stopping.']], [.28,.72])
add_paragraph('LR is linear in its supplied features; regularisation limits reliance on redundant predictors, while constructed interactions give B/D nonlinear boundaries in the original measurements. Capacity-limited boosted trees learn threshold interactions. TabM shares MLP weights while member-specific factors allow different predictions. Training averages binary cross-entropy over members before applying class/view weights; inference averages probabilities, not logits. Shared parameters make an ensemble cheaper than independent networks, but do not guarantee robustness. HGB and TabM use library estimators; we implement the splits, nested search and metrics ourselves.')
add_paragraph('<b>Nested validation and tuning.</b> Using seed 20261008, our code stratifies patients by whether they have any AFIB-positive recording and allocates them to ten outer folds. Each outer training set has three inner patient folds. Every learned preprocessing step is refitted inside each inner training split. We select the parameter with highest mean inner observed-input macro-F1 at threshold 0.5, favouring the simpler setting on exact ties. The chosen model is refitted on all outer-training patients, then frozen for observed and stressed evaluation. All algorithms use the same patient folds; outer scores never choose hyperparameters.')
add_paragraph('LR and TabM permit one grid shift toward a winning boundary using inner scores. LR selects the middle of its final grid in all 40 main runs (90/90 including controls); all 933 saved fits converge. It chooses C = 0.1 in 35/40 main runs, consistent with balancing shrinkage against flexibility. TabM selects the middle in 35/40 main runs (69/80 including controls), with 971 fits overall. Its initial candidate means differ by at most 0.0016 within an arm with no consistent depth ordering. Refined candidates have unequal fold coverage, so their overall means cannot establish a globally best setting.')
add_paragraph('Earlier HGB tuning selected depth 5 in all 40 runs, motivating the new fixed grid {7, 8, 9}. The 360 recovered inner scores confirm depths 7/8/9 in 10/15/15 cases. Depth 8 ties the upper boundary; the course requirement for the middle value to win most often remains unresolved. The 16-leaf cap limits complexity: increasing depth need not add leaves, although it permits more unbalanced paths. This post hoc grid extension is exploratory and does not establish a globally optimal model.')
add_paragraph('<b>Training corruption.</b> C/D retain each observed ECG and add one copy, assigned without labels approximately equally to three hidden leads, noise 0.10, gain 0.10, or three hidden leads plus noise 0.10. Each ECG has inverse-frequency weight equal to the training size divided by twice its class count. Each view receives half, preserving total weight and the LR penalty scale. Corruption acts on raw measurements, then features are rebuilt and transformed using the observed-training fit.')
add_paragraph('<b>Held-out stresses.</b> We hide 1, 3 or 6 leads (all 42 fields per lead), add Gaussian noise with SD 0.05, 0.10 or 0.20 times each training-field IQR, apply amplitude gain exp(sigma Z), where Z is standard normal, at the same log-scale doses, or combine three hidden leads with noise 0.10. Gain is positive with median one and is shared within an ECG/lead. Noise excludes counts/morphology; gain excludes durations. Natural gaps remain, and synthetic negative durations/IQRs are clipped at zero. Each stress has three repeats. ECG-keyed draws and separate training/evaluation streams support paired comparisons. These are measurement-space assumptions, not clinical tolerances or simulated physical electrode removal.')
add_paragraph('<b>Outcomes and uncertainty.</b> We calculate confusion counts, both class F1 scores, sensitivity, specificity, AUROC, non-interpolated average precision (AP) and unweighted Brier score from scratch. Macro-F1 averages the two class F1 scores. Primary classification uses 0.5. Stress repeats are averaged within each outer fold; we then report the ten-fold mean and sample SD. Paired contrasts are formed within folds. SD bars describe variation, not confidence intervals: training sets overlap and stress repeats are not independent patient samples. We keep corruption types separate rather than construct an arbitrary overall robustness score.')

add_page_break()
add_heading('4 Results and discussion')
add_paragraph('<b>Table 1.</b> Side-by-side macro-F1 at 0.5, mean ± sample SD over ten folds after averaging stress repeats. Earlier HGB selects depth 5; HGB 7-9 is the latest run. Observed inputs retain natural gaps. C/D add corrupted views; B/D add constructed features.','caption')
data=[]
for scenario, label in [('observed','Observed'),('missing_3','3 hidden leads'),('missing_6','6 hidden leads')]:
    for arm in 'ABCD':
        data.append([label if arm == 'A' else '', arm] + [format_metric(model,arm,scenario) for model in TABLE_MODELS])
add_table(['Input','Arm','LR','HGB earlier','HGB 7-9','TabM'],data,[.145,.07,.19625,.19625,.19625,.19625])
add_paragraph('<b>Construction helps most under severe lead loss.</b> B minus A observed F1 gains are 0.0053 ± 0.0070 for LR, 0.0002 ± 0.0031 for HGB and 0.0067 ± 0.0076 for TabM. With six hidden leads, gains rise to 0.1259 ± 0.0338, 0.1438 ± 0.0213 and 0.1155 ± 0.0332. Pooling surviving measurements plausibly reduces dependence on one lead; explicit interactions also help LR. These are pipeline contrasts because tuning may change capacity, not proof that one feature family caused the gain. HGB construction helps mainly under severe missingness.')
img=Image(str(REPORT_DIRECTORY/'results/augmentation_effect.png'),width=174*mm,height=36.8*mm)
img.hAlign='CENTER'; report_elements.append(img)
content_blocks.append({'type': 'image', 'path': 'results/augmentation_effect.png',
                       'width_mm': 174, 'height_mm': 36.8})
add_paragraph('<b>Figure 1.</b> Enhanced-feature augmentation effect (D minus B), using latest HGB. Points are paired fold means; bars are sample SD. Panels use different scales.','caption')
add_paragraph('<b>Augmentation is model dependent.</b> Observed D minus B F1 is +0.0028 for LR, -0.0069 for HGB and +0.0006 for TabM. Under six-lead loss it is -0.0104, +0.1518 and +0.1629 (Figure 1). Exposure to corrupted views may reduce reliance on fragile inputs. Flexibility alone is insufficient: unaugmented TabM A falls from 0.932 to 0.636, below LR A at 0.735. A smaller performance drop also does not make a low-scoring model preferable.')
add_paragraph('<b>Retuning preserves the main HGB trade-off.</b> Earlier/new observed F1 is 0.9230/0.9244 for B and six-lead F1 is 0.8892/0.8955 for C. Gains are modest: new A/B six-lead F1 falls by 0.0041/0.0047, and observed D falls by 0.0005. Thus deeper search is not uniformly better. In the latest run, C exceeds D by 0.0129 under six-lead loss; construction and augmentation are not uniformly additive. Derived fields may repeat tree interactions. Cross-run changes are descriptive because exact stress draws and historical software environments are not fully established.')
add_paragraph('<b>Controls qualify the explanation.</b> Under six-lead loss, LR B beats a control retaining raw values and derived missingness flags by 0.1326 ± 0.0331 F1. TabM B exceeds its equivalent control by 0.0057 observed F1. TabM\'s unchanged-copy control reaches 0.9397 observed F1 but only 0.7555 under six-lead loss, supporting a corruption benefit beyond duplication. Copied rows increase optimiser updates and tuning can change capacity; this is not an equal-compute comparison. HGB lacks these controls.')

add_page_break()
add_heading('4 Results and discussion continued')
add_paragraph('<b>Table 2.</b> Observed operating points of the highest mean-F1 main arm per model. Entries are mean ± sample SD. Higher AP and lower Brier are preferable.','caption')
add_table(['Model arm','Sensitivity','Specificity','AP','Brier'],[[label]+[format_metric(m,a,metric=k) for k in ['sensitivity','specificity','average_precision','brier']] for m,a,label in [('LR','D','LR D'),('HGB','B','HGB B'),('TabM','D','TabM D')]],[.20,.20,.20,.20,.20])
add_paragraph('<b>Model choice depends on the error budget.</b> TabM D has the highest main-arm F1 observed (0.9395) and with six hidden leads (0.9142). Across 21,799 observed held-out ECGs, it produces 225 false positives and 126 false negatives, versus LR D\'s 555 and 71: its F1 advantage conceals 55 additional missed AFIB records. Verified HGB B AP slightly exceeds TabM D (0.9353 versus 0.9320); F1 does not capture every aspect of ranking. Lower Brier indicates lower probability error, not proof of calibration. Best-arm selection is descriptive after viewing outer scores, not an independently evaluated procedure.')
add_paragraph('Additive noise and gain at the tested doses change F1 much less than lead masking, so our conclusion concerns missing measurements rather than general noise immunity. Equal patient weighting gives observed D F1 of 0.9028 (LR) and 0.9407 (TabM), close to record-weighted results. An optional LR/TabM threshold selected from inner out-of-fold predictions targets sensitivity of at least 0.90; it does not guarantee that sensitivity on unseen patients. Class balancing and threshold 0.5 do not define a clinical operating point. Training-only calibration and threshold selection at a common error budget would make a more relevant next comparison.')
add_paragraph('<b>Learned filling improves reconstruction more than classification.</b> In a separate LR A branch, ridge regressions (alpha 1) estimate 216 non-P-wave continuous fields from up to eight training-selected predictors, requiring at least 20 supporting observations. Targets are predicted independently: labels and previously generated values are never inputs. Unsupported predictions fall back to the training median; measured entries and original missingness flags remain. P-related targets stay on baseline filling rather than assuming a missing value represents normal physiology. The filler is refitted inside each training split.')
add_paragraph('Under 10% random-cell masking, IQR-normalised recovery MAE falls from 0.7349 with median filling to 0.4311 with ridge. Yet observed downstream F1 changes by -0.0003 ± 0.0015 and three-lead F1 by only +0.0059 ± 0.0048. Reconstruction and classification optimise different goals, so a large recovery gain need not change the decision boundary usefully. Recovery is scored only on deliberately hidden, originally observed eligible cells; natural gaps have no known truth. We retain median filling for A-D because it is simple and keeps the main representation/augmentation comparison consistent. Complete-case deletion would alter the cohort; waveform models could capture temporal structure but would change both representation and computational budget. Neither alternative was empirically compared.')
add_heading('5 Limitations and conclusion')
add_paragraph('EDA and the earlier HGB run informed the design, so these are exploratory internal results. Patient separation prevents repeat-record leakage but does not establish external validity, correlated-failure robustness or clinical benefit. One split scheme, multiple contrasts, fixed budgets and incomplete historical run manifests limit inference. Next steps are to resolve HGB boundary selection using inner data only, ablate feature families at fixed capacity and update counts, compare calibration/thresholds, and validate externally. Missing quality flags do not certify clean signals.')
add_paragraph('Without augmentation, cross-lead construction improves severe-missingness performance across the tested model classes. Augmentation provides the largest protection for flexible models, while its benefit depends on the operating threshold and competing errors. TabM D is the strongest tested F1 candidate, but LR misses fewer observed AFIB records. Robustness is conditional on representation, corruption and the cost of mistakes; no model is universally superior.')
add_heading('References')
refs=[
'[1] Strodthoff, N., et al. (2023). PTB-XL+, a comprehensive electrocardiographic feature dataset. <i>Scientific Data, 10</i>, 279. <link href="https://doi.org/10.1038/s41597-023-02153-8">https://doi.org/10.1038/s41597-023-02153-8</link>',
'[2] Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. <i>Scientific Data, 7</i>, 154. <link href="https://doi.org/10.1038/s41597-020-0495-6">https://doi.org/10.1038/s41597-020-0495-6</link>',
'[3] Sisk, R., et al. (2023). Imputation and missing indicators for handling missing data in the development and deployment of clinical prediction models: A simulation study. <i>Statistical Methods in Medical Research, 32</i>(8), 1461-1477. <link href="https://doi.org/10.1177/09622802231165001">https://doi.org/10.1177/09622802231165001</link>',
'[4] Gorishniy, Y., Kotelnikov, A., and Babenko, A. (2025). TabM: Advancing tabular deep learning with parameter-efficient ensembling. <i>ICLR 2025</i>. <link href="https://openreview.net/forum?id=Sd4wYYOhmY">https://openreview.net/forum?id=Sd4wYYOhmY</link>',
]
for reference in refs:
    add_paragraph(reference, 'reference')

def footer(canvas, doc):
    """Keep the page number inside the required page margins."""
    canvas.saveState()
    canvas.setFont('Times-Roman', 10)
    canvas.drawRightString(A4[0]-16*mm, 18*mm, str(doc.page))
    canvas.restoreState()

if __name__ == '__main__':
    doc=SimpleDocTemplate(str(OUTPUT_PDF),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=16*mm,bottomMargin=16*mm,title='AFIB classification under missing and perturbed ECG measurements',author='Jiayi Huang; Siwen Wang; Aaditya Pathak',pageCompression=1)
    doc.build(report_elements,onFirstPage=footer,onLaterPages=footer)
    if doc.page != 4:
        raise RuntimeError(f'The report must be four pages; generated {doc.page}. Review the layout.')
    print(OUTPUT_PDF)
    print('Approximate words:',len(re.sub('<[^>]+>',' ',' '.join(text_blocks)).split()))

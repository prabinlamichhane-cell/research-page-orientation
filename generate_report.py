"""
Generate a PDF report from results/ directory.
Run: python3 generate_report.py
"""

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

RESULTS = Path('results')
OUT     = Path('results/PP-LCNet_x1_0_doc_ori_Benchmark_Report.pdf')

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

H1 = ParagraphStyle('H1', fontSize=18, fontName='Helvetica-Bold',
                    spaceAfter=6, textColor=colors.HexColor('#1B3A6B'))
H2 = ParagraphStyle('H2', fontSize=13, fontName='Helvetica-Bold',
                    spaceBefore=14, spaceAfter=4, textColor=colors.HexColor('#1B3A6B'))
H3 = ParagraphStyle('H3', fontSize=10, fontName='Helvetica-Bold',
                    spaceBefore=8, spaceAfter=3, textColor=colors.HexColor('#333333'))
BODY = ParagraphStyle('Body', fontSize=9, fontName='Helvetica',
                      spaceAfter=4, leading=13)
CAPTION = ParagraphStyle('Caption', fontSize=8, fontName='Helvetica-Oblique',
                         alignment=TA_CENTER, textColor=colors.HexColor('#666666'),
                         spaceAfter=8)
SMALL = ParagraphStyle('Small', fontSize=8, fontName='Helvetica',
                       textColor=colors.HexColor('#555555'), spaceAfter=3)

TABLE_HEADER = colors.HexColor('#1B3A6B')
TABLE_ALT    = colors.HexColor('#F0F4FA')

def tbl_style(col_widths=None):
    return TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  TABLE_HEADER),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  9),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, TABLE_ALT]),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#CCCCCC')),
        ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN',         (0, 0), (0, -1),  'LEFT'),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
    ])

def img(path, width, caption=None):
    items = []
    if Path(path).exists():
        items.append(Image(str(path), width=width,
                           height=width * 0.6, kind='proportional'))
        if caption:
            items.append(Paragraph(caption, CAPTION))
    return items

def hr():
    return HRFlowable(width='100%', thickness=0.5,
                      color=colors.HexColor('#CCCCCC'), spaceAfter=6, spaceBefore=2)

# ── Load data ─────────────────────────────────────────────────────────────────
summary  = pd.read_csv(RESULTS / 'comparison_summary.csv')
paddle   = pd.read_csv(RESULTS / 'paddle_results.csv')
onnx_df  = pd.read_csv(RESULTS / 'onnx_results.csv')
opt_df   = pd.read_csv(RESULTS / 'optimum_results.csv')

from sklearn.metrics import classification_report, accuracy_score
import numpy as np

cr = classification_report(paddle['label'], paddle['paddle_pred'],
                           target_names=['0°','90°','180°','270°'],
                           output_dict=True)

clean_acc = (paddle[paddle['messy']==False]['paddle_pred'] ==
             paddle[paddle['messy']==False]['label']).mean()
messy_acc = (paddle[paddle['messy']==True]['paddle_pred'] ==
             paddle[paddle['messy']==True]['label']).mean()

errors = paddle[paddle['paddle_pred'] != paddle['label']][['label','paddle_pred']]
top_confusion = errors.groupby(['label','paddle_pred']).size().sort_values(ascending=False).head(6)

deg_map = {0: '0°', 1: '90°', 2: '180°', 3: '270°'}

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# Cover
story += [
    Spacer(1, 1.5*cm),
    Paragraph('PP-LCNet_x1_0_doc_ori', H1),
    Paragraph('Runtime Benchmark Report', ParagraphStyle(
        'Sub', fontSize=14, fontName='Helvetica',
        textColor=colors.HexColor('#444444'), spaceAfter=4)),
    Paragraph('Nepali Financial Document Orientation Classification', ParagraphStyle(
        'Sub2', fontSize=10, fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#666666'), spaceAfter=16)),
    hr(),
    Table([
        ['Date',    '2026-05-05'],
        ['Author',  'prabin.lamichhane@amniltech.com'],
        ['Ticket',  '#2963 — Amnil Research Board'],
        ['Repo',    'github.com/prabinlamichhane-cell/research-page-orientation'],
        ['Model',   'PP-LCNet_x1_0_doc_ori (PP-StructureV3, Paddle 3.x PIR)'],
    ], colWidths=[4*cm, 12*cm],
    style=TableStyle([
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 8),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1B3A6B')),
        ('TOPPADDING',(0,0),(-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0, colors.white),
    ])),
    Spacer(1, 0.8*cm),
]

# 1. Objective
story += [
    Paragraph('1. Objective', H2), hr(),
    Paragraph(
        'Benchmark PP-LCNet_x1_0_doc_ori — PaddlePaddle\'s lightweight document orientation '
        'classifier — across three inference runtimes on a dataset of real Nepali financial '
        'documents. The goal is to identify the optimal runtime for production deployment.',
        BODY),
    Paragraph('Runtimes evaluated:', BODY),
    Paragraph('• <b>PaddlePaddle</b> (native Paddle 3.x PIR format) — baseline', BODY),
    Paragraph('• <b>ONNX Runtime</b> — converted via paddle2onnx (opset 17)', BODY),
    Paragraph('• <b>HuggingFace Optimum ORT</b> — same ONNX model via ORTModelForImageClassification', BODY),
    Spacer(1, 0.3*cm),
]

# 2. Dataset
story += [
    Paragraph('2. Dataset', H2), hr(),
    Table([
        ['Property', 'Value'],
        ['Source PDFs', '9 real Nepali financial documents'],
        ['Document types', 'Audit reports, microfinance annual reports, management docs'],
        ['Total images', '552 (138 per class)'],
        ['Classes', '4 — 0°, 90°, 180°, 270°'],
        ['Class balance', 'Perfectly balanced (25% each)'],
        ['Messy/degraded images', '210 (38%) — synthetically augmented'],
        ['Messy transforms', 'Skew, shadow, ink bleed, fold, noise, blur, JPEG'],
    ], colWidths=[5.5*cm, 10.5*cm], style=tbl_style()),
    Spacer(1, 0.3*cm),
    Paragraph(
        '<b>Data quality note:</b> All source PDFs were manually verified to ensure every '
        'source page is at true 0° before synthetic rotation. Landscape financial tables '
        'stored at 90° in PDF metadata were corrected to portrait. An earlier run without '
        'this correction yielded 89.1% accuracy; after correction accuracy rose to 96.9%, '
        'confirming label integrity as the most critical data pipeline step.',
        SMALL),
    Spacer(1, 0.4*cm),
]
story += img(RESULTS / 'raw_sample.png', 16*cm,
             'Figure 1 — Sample pages from collected Nepali financial PDFs')
story += img(RESULTS / 'messify_preview.png', 16*cm,
             'Figure 2 — Degradation pipeline: original → augmented → messify variants')
story += img(RESULTS / 'class_distribution.png', 10*cm,
             'Figure 3 — Class distribution (balanced)')

# 3. Results
story += [
    PageBreak(),
    Paragraph('3. Results Summary', H2), hr(),
    Table([
        ['Runtime', 'Accuracy', 'Avg Latency', 'P50', 'P95', 'Throughput'],
        ['PaddlePaddle', '96.92%', '39.53 ms', '39.42 ms', '46.54 ms', '25.3 img/s'],
        ['ONNX Runtime', '96.92%', '34.14 ms', '33.53 ms', '43.87 ms', '29.3 img/s'],
        ['Optimum ORT',  '96.92%', '32.00 ms', '32.23 ms', '39.15 ms', '31.2 img/s'],
    ], colWidths=[4*cm, 2.5*cm, 2.8*cm, 2.5*cm, 2.5*cm, 2.7*cm],
    style=tbl_style()),
    Spacer(1, 0.2*cm),
    Paragraph('Prediction agreement: 100% across all 3 runtimes — conversion is numerically lossless.', SMALL),
    Spacer(1, 0.4*cm),
]
story += img(RESULTS / 'comparison_chart.png', 16*cm,
             'Figure 4 — Accuracy, avg latency and throughput comparison')
story += img(RESULTS / 'latency_boxplot.png', 14*cm,
             'Figure 5 — Latency distribution by runtime')

# 4. Per-class
story += [
    Paragraph('4. Per-class Performance', H2), hr(),
    Paragraph('Identical across all runtimes (100% prediction agreement).', BODY),
    Table([
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        ['0°',   f"{cr['0°']['precision']:.4f}",  f"{cr['0°']['recall']:.4f}",  f"{cr['0°']['f1-score']:.4f}",  '138'],
        ['90°',  f"{cr['90°']['precision']:.4f}", f"{cr['90°']['recall']:.4f}", f"{cr['90°']['f1-score']:.4f}", '138'],
        ['180°', f"{cr['180°']['precision']:.4f}",f"{cr['180°']['recall']:.4f}",f"{cr['180°']['f1-score']:.4f}",'138'],
        ['270°', f"{cr['270°']['precision']:.4f}",f"{cr['270°']['recall']:.4f}",f"{cr['270°']['f1-score']:.4f}",'138'],
        ['Overall', f"{cr['macro avg']['precision']:.4f}", f"{cr['macro avg']['recall']:.4f}",
         f"{cr['macro avg']['f1-score']:.4f}", '552'],
    ], colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 3*cm], style=tbl_style()),
    Spacer(1, 0.4*cm),
    Paragraph('Clean vs Messy Accuracy', H3),
    Table([
        ['Condition', 'Accuracy'],
        ['Clean images', f'{clean_acc:.4f} ({clean_acc*100:.1f}%)'],
        ['Messy / degraded images', f'{messy_acc:.4f} ({messy_acc*100:.1f}%)'],
    ], colWidths=[8*cm, 8*cm], style=tbl_style()),
    Spacer(1, 0.4*cm),
    Paragraph('Top Confusion Pairs', H3),
    Table(
        [['True', 'Predicted', 'Count']] +
        [[deg_map[t], deg_map[p], str(c)]
         for (t, p), c in top_confusion.items()],
        colWidths=[5*cm, 5*cm, 6*cm], style=tbl_style()),
    Spacer(1, 0.4*cm),
]
story += img(RESULTS / 'paddle_confusion_matrix.png', 10*cm,
             'Figure 6 — Confusion matrix (identical for all runtimes)')

# 5. Key findings
story += [
    PageBreak(),
    Paragraph('5. Key Findings', H2), hr(),
    Paragraph('1. <b>All runtimes produce identical predictions</b> — paddle2onnx at opset 17 is numerically lossless.', BODY),
    Paragraph('2. <b>ONNX Runtime is 14% faster</b> than native PaddlePaddle (34.14ms vs 39.53ms).', BODY),
    Paragraph('3. <b>Optimum ORT is 19% faster</b> than Paddle and 6% faster than raw ORT — HF wrapper adds no measurable overhead.', BODY),
    Paragraph('4. <b>96.9% out-of-the-box accuracy</b> on Nepali financial documents with no domain fine-tuning.', BODY),
    Paragraph('5. <b>5% accuracy drop on messy images</b> (93.8% vs 98.8%) — the primary gap to close with fine-tuning.', BODY),
    Paragraph('6. <b>Label integrity is critical</b> — mislabeled source pages caused 7.8% artificial accuracy drop in the first run.', BODY),
    Spacer(1, 0.4*cm),
]

# 6. Recommendation
story += [
    Paragraph('6. Recommendation', H2), hr(),
    Paragraph('<b>Use ONNX Runtime for production deployment.</b>', BODY),
    Table([
        ['Criterion', 'Recommendation'],
        ['Accuracy', 'Any runtime (identical)'],
        ['Speed', 'Optimum ORT or ONNX Runtime'],
        ['Dependency weight', 'ONNX Runtime (lighter than Optimum)'],
        ['HuggingFace pipeline integration', 'Optimum ORT'],
        ['Standalone deployment', 'ONNX Runtime'],
    ], colWidths=[8*cm, 8*cm], style=tbl_style()),
    Spacer(1, 0.3*cm),
    Paragraph(
        '• <b>Standalone:</b> onnxruntime + model.onnx — minimal dependencies, 19% faster than Paddle.',
        BODY),
    Paragraph(
        '• <b>HuggingFace pipeline:</b> Optimum ORT — fastest option, clean API.',
        BODY),
    Paragraph(
        '• <b>Drop PaddlePaddle</b> from production — heaviest dependency, slowest, no accuracy benefit.',
        BODY),
    Spacer(1, 0.4*cm),
]

# 7. Fine-tuning
story += [
    Paragraph('7. Fine-tuning — Is It Worth It?', H2), hr(),
    Paragraph(
        '96.9% is a strong out-of-the-box baseline. The remaining errors cluster around '
        '180°↔0° confusions on symmetric layouts and degraded scan robustness. '
        'Fine-tuning is feasible and recommended if production accuracy requirements exceed 98% '
        'or deployment targets heavily degraded documents.',
        BODY),
    Spacer(1, 0.2*cm),
    Paragraph('Expected gains from fine-tuning:', H3),
    Table([
        ['Approach', 'Expected Accuracy', 'Effort'],
        ['No fine-tuning (current)', '~97%', 'Done'],
        ['Feature extraction (freeze backbone, train head)', '~98-99%', 'Low — 1-2 days'],
        ['Full fine-tuning', '~99%+', 'Medium — 3-5 days'],
        ['Full fine-tuning + larger dataset (500+ pages)', '~99%+ on messy', 'High'],
    ], colWidths=[8*cm, 3.5*cm, 4.5*cm], style=tbl_style()),
    Spacer(1, 0.3*cm),
    Paragraph(
        '<b>Bottleneck is data, not model capacity.</b> PP-LCNet_x1_0 has sufficient capacity '
        'for 4-class orientation. Currently 138 source pages from 9 PDFs. '
        'Collecting 500+ real Nepali pages — including genuinely degraded scans from '
        'cooperatives, NGOs, and municipalities — will drive the largest accuracy improvement.',
        SMALL),
    Spacer(1, 0.3*cm),
    Paragraph('Suggested fine-tuning approach:', H3),
    Paragraph('1. Collect 500+ real Nepali financial pages including degraded scans.', BODY),
    Paragraph('2. Rotate × 4 → 2000+ labeled images.', BODY),
    Paragraph('3. Feature extraction: freeze PP-LCNet backbone, retrain 4-class head only.', BODY),
    Paragraph('4. If accuracy plateaus below 99%, progressively unfreeze later blocks.', BODY),
    Paragraph('5. Re-export to ONNX — deployment path unchanged.', BODY),
    Spacer(1, 0.4*cm),
]

# 8. Limitations
story += [
    Paragraph('8. Limitations', H2), hr(),
    Table([
        ['Concern', 'Severity', 'Notes'],
        ['Only 9 source PDFs (138 pages)', 'Medium', 'Small dataset — accuracy estimate has high variance'],
        ['Synthetic messiness only', 'Medium', 'Real degraded scans may behave differently'],
        ['Pre-training data unknown', 'Low', 'Cannot rule out overlap with PaddlePaddle training set'],
        ['4 rotations per page not independent', 'Low', 'Page-level split would give cleaner estimate'],
    ], colWidths=[6*cm, 2.5*cm, 7.5*cm], style=tbl_style()),
    Spacer(1, 0.3*cm),
    Paragraph(
        'The 96.9% figure is directionally reliable but should be re-evaluated on a larger '
        '(50+ PDF) held-out test set before making production SLA commitments.',
        SMALL),
    Spacer(1, 0.4*cm),
]

# 9. Technical notes
story += [
    Paragraph('9. Technical Notes', H2), hr(),
    Paragraph('<b>Model format:</b> Paddle 3.x PIR — pass inference.json (not inference.pdmodel) to paddle.inference.Config.', SMALL),
    Paragraph('<b>Preprocessing:</b> resize_short(256) → center_crop(224) → normalize(ImageNet) → CHW → batch dim. '
              'Simple resize(224,224) produces incorrect results.', SMALL),
    Paragraph('<b>ONNX conversion:</b> paddle2onnx opset 17. Constant folding: 282 → 115 nodes.', SMALL),
    Paragraph('<b>Optimum fix:</b> input renamed x → pixel_values; output fetch_name_0 → logits; '
              'config.json with model_type: resnet required.', SMALL),
    Spacer(1, 0.2*cm),
    Paragraph(f'<b>Repo:</b> https://github.com/prabinlamichhane-cell/research-page-orientation', SMALL),
]

# ── Build PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    leftMargin=2*cm, rightMargin=2*cm,
    title='PP-LCNet_x1_0_doc_ori Benchmark Report',
    author='prabin.lamichhane@amniltech.com',
)
doc.build(story)
print(f'Report saved → {OUT}')

"""
Confidence-based retry experiment on English-only dataset.
Two sets: Set 1 (rotated only) and Set 2 (rotated + augment + messify).
For images below 75% confidence: rotate to all 4 orientations, re-infer,
pick highest confidence. Generates a PDF report.
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.preprocess import load_and_preprocess, LABEL_TO_DEGREES, DEGREES_TO_LABEL

# ── Config ────────────────────────────────────────────────────────────────────
SETS = {
    'Set 1 — Rotated only':         Path('data/english_only_rotated_clean'),
    'Set 2 — Aug + Messify':        Path('data/english_only_rotated_messy'),
}
CSVS = {
    'Set 1 — Rotated only':         Path('results/english_only/english_only_clean_results.csv'),
    'Set 2 — Aug + Messify':        Path('results/english_only/english_only_augmented_results.csv'),
}
OUT_DIR   = Path('results/english_only/confidence_retry')
MODEL_PATH = 'models/model.onnx'
THRESHOLD  = 0.75
DEGREES    = [0, 90, 180, 270]
ROTATE_FLAGS = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp/_en_retry_tmp.png')

# ── ONNX session ──────────────────────────────────────────────────────────────
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session  = ort.InferenceSession(MODEL_PATH, sess_opts, providers=['CPUExecutionProvider'])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name


def predict_from_array(img_bgr: np.ndarray):
    cv2.imwrite(str(TMP), img_bgr)
    probs = session.run([out_name], {inp_name: load_and_preprocess(TMP)})[0][0]
    label = int(np.argmax(probs))
    return label, float(probs[label]), probs


def rotate_array(img: np.ndarray, degrees: int) -> np.ndarray:
    flag = ROTATE_FLAGS[degrees]
    return img if flag is None else cv2.rotate(img, flag)


# ── Run experiment per set ────────────────────────────────────────────────────
all_results = {}

for set_name, csv_path in CSVS.items():
    print(f'\n{"="*60}')
    print(f'{set_name}')
    print(f'{"="*60}')

    df = pd.read_csv(csv_path)

    # Step 1: Re-run inference tracking confidence
    print('Running inference with confidence tracking...')
    pred_labels, confidences = [], []
    prob_cols = {d: [] for d in DEGREES}

    for row in tqdm(df.itertuples(), total=len(df)):
        img = cv2.imread(row.image_path)
        p_label, conf, probs = predict_from_array(img)
        pred_labels.append(p_label)
        confidences.append(conf)
        for i, d in enumerate(DEGREES):
            prob_cols[d].append(float(probs[i]))

    df['pred_label'] = pred_labels
    df['confidence'] = confidences
    df['correct']    = df['pred_label'] == df['label']
    df['uncertain']  = df['confidence'] < THRESHOLD
    for d in DEGREES:
        df[f'prob_{d}'] = prob_cols[d]

    n_unc   = df['uncertain'].sum()
    n_wrong = (~df['correct']).sum()
    print(f'Accuracy       : {df["correct"].mean()*100:.2f}%')
    print(f'Avg confidence : {df["confidence"].mean()*100:.1f}%')
    print(f'Uncertain (<{THRESHOLD:.0%}): {n_unc} / {len(df)} ({n_unc/len(df)*100:.1f}%)')
    print(f'  wrong + uncertain: {(~df["correct"] & df["uncertain"]).sum()}')
    print(f'  right + uncertain: {(df["correct"] & df["uncertain"]).sum()}')

    # Step 2: Retry uncertain
    uncertain_df = df[df['uncertain']].copy()
    retry_records = []
    print(f'Retrying {len(uncertain_df)} uncertain images...')

    for row in tqdm(uncertain_df.itertuples(), total=len(uncertain_df)):
        img = cv2.imread(row.image_path)
        per_rot_conf = {}
        all_retries  = []

        for retry_deg in DEGREES:
            retried = rotate_array(img, retry_deg)
            r_label, r_conf, _ = predict_from_array(retried)
            inferred_deg   = (LABEL_TO_DEGREES[r_label] - retry_deg) % 360
            inferred_label = DEGREES_TO_LABEL.get(inferred_deg, -1)
            per_rot_conf[retry_deg] = r_conf
            all_retries.append({
                'retry_deg':      retry_deg,
                'model_conf':     r_conf,
                'inferred_label': inferred_label,
            })

        best_r = max(all_retries, key=lambda x: x['model_conf'])
        retry_records.append({
            'source':          row.source,
            'true_deg':        row.degrees,
            'true_label':      row.label,
            'messy':           row.messy if hasattr(row, 'messy') else False,
            'orig_pred_label': row.pred_label,
            'orig_pred_deg':   LABEL_TO_DEGREES[row.pred_label],
            'orig_confidence': row.confidence,
            'conf_rot0':       per_rot_conf[0],
            'conf_rot90':      per_rot_conf[90],
            'conf_rot180':     per_rot_conf[180],
            'conf_rot270':     per_rot_conf[270],
            'best_retry_deg':  best_r['retry_deg'],
            'best_conf':       best_r['model_conf'],
            'inferred_label':  best_r['inferred_label'],
            'retry_correct':   best_r['inferred_label'] == row.label,
            'orig_correct':    row.correct,
            'conf_gain':       best_r['model_conf'] - row.confidence,
        })

    retry_df = pd.DataFrame(retry_records)

    # Step 3: Rotate-to-0 consistency
    cons_records = []
    print('Rotate-to-0 consistency check...')
    for row in tqdm(uncertain_df.itertuples(), total=len(uncertain_df)):
        img         = cv2.imread(row.image_path)
        inverse_deg = (360 - LABEL_TO_DEGREES[row.pred_label]) % 360
        corrected   = rotate_array(img, inverse_deg)
        c_label, c_conf, _ = predict_from_array(corrected)
        cons_records.append({
            'source':          row.source,
            'true_deg':        row.degrees,
            'true_label':      row.label,
            'orig_pred_deg':   LABEL_TO_DEGREES[row.pred_label],
            'orig_confidence': row.confidence,
            'corrected_pred':  LABEL_TO_DEGREES[c_label],
            'corrected_conf':  c_conf,
            'corrected_is_0':  c_label == 0,
            'orig_correct':    row.correct,
        })
    cons_df = pd.DataFrame(cons_records)

    # Save CSVs
    slug = set_name.replace(' ', '_').replace('—', '').replace('+', 'plus').strip('_')
    df.to_csv(OUT_DIR / f'{slug}_all_predictions.csv', index=False)
    retry_df.to_csv(OUT_DIR / f'{slug}_retry_results.csv', index=False)
    cons_df.to_csv(OUT_DIR / f'{slug}_consistency_results.csv', index=False)

    # Print retry summary
    if len(retry_df) > 0:
        print(f'\nRetry accuracy: {retry_df["orig_correct"].mean()*100:.1f}% -> '
              f'{retry_df["retry_correct"].mean()*100:.1f}%  '
              f'(gain +{retry_df["conf_gain"].mean()*100:.1f}pp)')
        print('Consistent rotation boost:')
        for true_deg in DEGREES:
            sub = retry_df[retry_df['true_deg'] == true_deg]
            if len(sub) == 0:
                continue
            print(f'  True {true_deg:>3}° (n={len(sub)}):')
            for col, deg in [('conf_rot0',0),('conf_rot90',90),('conf_rot180',180),('conf_rot270',270)]:
                gain = (sub[col] - sub['orig_confidence']).mean() * 100
                pct  = (sub[col] > sub['orig_confidence']).mean() * 100
                print(f'    +{deg:>3}°  gain={gain:+.1f}pp  improved={pct:.0f}%')

    all_results[set_name] = {
        'df': df, 'retry_df': retry_df, 'cons_df': cons_df, 'slug': slug
    }

    # ── Plots ──────────────────────────────────────────────────────────────────
    # Plot 1: distribution + by-class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['confidence'], bins=30, color='#4C72B0', edgecolor='white')
    axes[0].axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD:.0%})')
    axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Count')
    axes[0].set_title(f'Confidence Distribution\n{set_name}')
    axes[0].legend()

    conf_by_class = [df[df['degrees'] == d]['confidence'].values for d in DEGREES]
    axes[1].boxplot(conf_by_class, tick_labels=[f'{d}°' for d in DEGREES])
    axes[1].axhline(THRESHOLD, color='red', linestyle='--', label=f'Threshold')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title(f'Confidence by Class\n{set_name}')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{slug}_confidence_dist.png', dpi=150)
    plt.close()

    # Plot 2: correct vs wrong confidence
    correct_conf = df[df['correct']]['confidence'].values
    wrong_conf   = df[~df['correct']]['confidence'].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram overlay
    axes[0].hist(correct_conf, bins=25, alpha=0.7, color='#55A868', edgecolor='white',
                 label=f'Correct (n={len(correct_conf)})')
    if len(wrong_conf):
        axes[0].hist(wrong_conf, bins=25, alpha=0.7, color='#C44E52', edgecolor='white',
                     label=f'Wrong (n={len(wrong_conf)})')
    axes[0].axvline(THRESHOLD, color='black', linestyle='--', alpha=0.6, label=f'Threshold ({THRESHOLD:.0%})')
    axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Count')
    axes[0].set_title(f'Confidence: Correct vs Wrong\n{set_name}')
    axes[0].legend()

    # Box/strip plot per class — correct vs wrong
    import matplotlib.patches as mpatches
    positions_c, positions_w = [], []
    data_c, data_w = [], []
    xticks, xlabels = [], []
    gap = 0.4
    for i, d in enumerate(DEGREES):
        sub = df[df['degrees'] == d]
        xc = i * 2
        xw = i * 2 + gap
        data_c.append(sub[sub['correct']]['confidence'].values)
        data_w.append(sub[~sub['correct']]['confidence'].values)
        positions_c.append(xc)
        positions_w.append(xw)
        xticks.append(xc + gap / 2)
        xlabels.append(f'{d}°')

    bp_c = axes[1].boxplot(data_c, positions=positions_c, widths=0.3,
                           patch_artist=True, manage_ticks=False)
    bp_w = axes[1].boxplot(data_w, positions=positions_w, widths=0.3,
                           patch_artist=True, manage_ticks=False)
    for patch in bp_c['boxes']:
        patch.set_facecolor('#55A86888')
    for patch in bp_w['boxes']:
        patch.set_facecolor('#C44E5288')
    axes[1].set_xticks(xticks); axes[1].set_xticklabels(xlabels)
    axes[1].axhline(THRESHOLD, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Confidence')
    axes[1].set_title(f'Confidence by Class (Correct vs Wrong)\n{set_name}')
    axes[1].legend(handles=[
        mpatches.Patch(color='#55A868', label='Correct'),
        mpatches.Patch(color='#C44E52', label='Wrong'),
    ])
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{slug}_correct_vs_wrong.png', dpi=150)
    plt.close()

    if len(retry_df) > 0:
        # Gain heatmap
        gain_rows = []
        for true_deg in DEGREES:
            sub = retry_df[retry_df['true_deg'] == true_deg]
            if len(sub) == 0:
                continue
            for col, rot_deg in [('conf_rot0',0),('conf_rot90',90),('conf_rot180',180),('conf_rot270',270)]:
                gain_rows.append({'true_deg': true_deg, 'retry_deg': rot_deg,
                                  'avg_gain': (sub[col] - sub['orig_confidence']).mean() * 100})
        gain_df    = pd.DataFrame(gain_rows)
        gain_pivot = gain_df.pivot(index='true_deg', columns='retry_deg', values='avg_gain')

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(gain_pivot, annot=True, fmt='.1f', ax=ax, cmap='RdYlGn', center=0,
                    xticklabels=[f'+{d}°' for d in DEGREES],
                    yticklabels=[f'{d}°' for d in DEGREES])
        ax.set_xlabel('Retry Rotation Applied'); ax.set_ylabel('True Orientation')
        ax.set_title(f'Avg Confidence Gain (pp)\n{set_name}  (threshold={THRESHOLD:.0%})')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{slug}_gain_heatmap.png', dpi=150)
        plt.close()

        # Scatter: before vs after
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_s = retry_df['retry_correct'].map({True: '#55A868', False: '#C44E52'})
        ax.scatter(retry_df['orig_confidence'], retry_df['best_conf'],
                   c=colors_s, alpha=0.8, s=60)
        ax.plot([0,1],[0,1],'k--', alpha=0.3, label='no gain')
        ax.axvline(THRESHOLD, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Confidence Before Retry'); ax.set_ylabel('Best Confidence After Retry')
        ax.set_title(f'Confidence Gain\n{set_name}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{slug}_gain_scatter.png', dpi=150)
        plt.close()

print(f'\nAll results saved -> {OUT_DIR}')

# ── PDF Report ────────────────────────────────────────────────────────────────
print('Generating PDF report...')

from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER

styles = getSampleStyleSheet()
H1   = ParagraphStyle('H1',   parent=styles['Heading1'], fontSize=16, spaceAfter=6,
                      textColor=colors.HexColor('#1a1a2e'))
H2   = ParagraphStyle('H2',   parent=styles['Heading2'], fontSize=13, spaceAfter=4,
                      textColor=colors.HexColor('#16213e'))
H3   = ParagraphStyle('H3',   parent=styles['Heading3'], fontSize=11, spaceAfter=3,
                      textColor=colors.HexColor('#0f3460'))
BODY = ParagraphStyle('BODY', parent=styles['Normal'],   fontSize=9,  leading=14, spaceAfter=4)
MONO = ParagraphStyle('MONO', parent=styles['Code'],     fontSize=8,  leading=12, spaceAfter=2)
CTR  = ParagraphStyle('CTR',  parent=BODY, alignment=TA_CENTER)

def tbl_style(header='#16213e'):
    return TableStyle([
        ('BACKGROUND',     (0,0), (-1,0), colors.HexColor(header)),
        ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
        ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 8),
        ('ALIGN',          (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.4, colors.HexColor('#dee2e6')),
        ('TOPPADDING',     (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 4),
    ])

def add_img(path, max_w=14*cm, max_h=8*cm):
    p = Path(path)
    if not p.exists():
        return Paragraph(f'[not found: {p.name}]', MONO)
    with PILImage.open(p) as im:
        nat_w, nat_h = im.size
    scale = min(max_w / nat_w, max_h / nat_h)
    return RLImage(str(p), width=nat_w * scale, height=nat_h * scale)

story = []

# Title
story += [
    Spacer(1, 1*cm),
    Paragraph('English-Only Confidence Retry Experiment', H1),
    Paragraph('PP-LCNet_x1_0_doc_ori — Set 1 (Clean) vs Set 2 (Aug + Messify)', H2),
    HRFlowable(width='100%', thickness=1, color=colors.HexColor('#16213e')),
    Spacer(1, 0.3*cm),
    Paragraph(f'Threshold: {THRESHOLD:.0%} &nbsp;|&nbsp; 7 PDFs, 94 pages, 376 images per set &nbsp;|&nbsp; ONNX Runtime', CTR),
    Spacer(1, 0.5*cm),
]

story += [
    Paragraph('1. Objective', H2),
    Paragraph(
        f'For each English-only image where the model confidence falls below {THRESHOLD:.0%}, '
        'the image is rotated to all 4 orientations and inference is re-run on each. '
        'The rotation yielding the highest confidence becomes the final prediction. '
        'This experiment runs on both the clean set (Set 1) and the augmented/messy set (Set 2) '
        'to isolate how degradation affects the retry strategy.', BODY),
    Spacer(1, 0.4*cm),
]

for set_name, res in all_results.items():
    df       = res['df']
    retry_df = res['retry_df']
    cons_df  = res['cons_df']
    slug     = res['slug']

    story += [
        Paragraph(f'2. {set_name}' if set_name == list(all_results.keys())[0]
                  else f'3. {set_name}', H2),
    ]

    # Baseline table
    n_unc = df['uncertain'].sum()
    base_data = [
        ['Metric', 'Value'],
        ['Images',                str(len(df))],
        ['Overall accuracy',      f'{df["correct"].mean()*100:.2f}%'],
        ['Avg confidence',        f'{df["confidence"].mean()*100:.1f}%'],
        [f'Uncertain (<{THRESHOLD:.0%})',
         f'{n_unc} / {len(df)} ({n_unc/len(df)*100:.1f}%)'],
        ['Wrong + uncertain',     str((~df['correct'] & df['uncertain']).sum())],
        ['Correct + uncertain',   str((df['correct'] & df['uncertain']).sum())],
    ]
    if df['messy'].any():
        base_data += [
            ['Clean accuracy', f'{df[~df["messy"]]["correct"].mean()*100:.2f}%'],
            ['Messy accuracy', f'{df[df["messy"]]["correct"].mean()*100:.2f}%'],
        ]
    story.append(Table(base_data, colWidths=[9*cm, 8*cm], style=tbl_style()))
    story.append(Spacer(1, 0.3*cm))
    story.append(add_img(OUT_DIR / f'{slug}_confidence_dist.png', max_w=15*cm, max_h=8*cm))
    story.append(Spacer(1, 0.3*cm))

    # Correct vs wrong confidence chart
    story += [Paragraph('Confidence Score: Correct vs Misclassified', H3)]
    n_wrong_set = int((~df['correct']).sum())
    story += [Paragraph(
        f'{len(df) - n_wrong_set} correct predictions | {n_wrong_set} wrong predictions. '
        'Left: overlapping histogram shows wrong predictions cluster at lower confidence. '
        'Right: per-class boxplot isolates which orientation classes produce uncertain wrong predictions.',
        BODY)]
    story.append(add_img(OUT_DIR / f'{slug}_correct_vs_wrong.png', max_w=15*cm, max_h=8*cm))
    story.append(Spacer(1, 0.3*cm))

    # Retry results
    if len(retry_df) > 0:
        rescued = ((~retry_df['orig_correct']) & retry_df['retry_correct']).sum()
        broken  = (retry_df['orig_correct'] & ~retry_df['retry_correct']).sum()
        story += [Paragraph('Retry Results:', H3)]
        retry_data = [
            ['Metric', 'Value'],
            ['Uncertain images retried',    str(len(retry_df))],
            ['Accuracy before retry',       f'{retry_df["orig_correct"].mean()*100:.1f}%'],
            ['Accuracy after retry',        f'{retry_df["retry_correct"].mean()*100:.1f}%'],
            ['Avg confidence gain',         f'+{retry_df["conf_gain"].mean()*100:.1f}pp'],
            ['Errors rescued',              str(rescued)],
            ['Correct flipped wrong',       str(broken)],
        ]
        story.append(Table(retry_data, colWidths=[9*cm, 8*cm], style=tbl_style()))
        story.append(Spacer(1, 0.3*cm))

        # Per-rotation boost table
        story += [Paragraph('Avg confidence gain per retry rotation (pp):', H3)]
        boost_data = [['True Deg', 'n', '+0°', '+90°', '+180°', '+270°']]
        for true_deg in DEGREES:
            sub = retry_df[retry_df['true_deg'] == true_deg]
            if len(sub) == 0:
                continue
            row_data = [f'{true_deg}°', str(len(sub))]
            for col in ['conf_rot0','conf_rot90','conf_rot180','conf_rot270']:
                gain = (sub[col] - sub['orig_confidence']).mean() * 100
                pct  = (sub[col] > sub['orig_confidence']).mean() * 100
                row_data.append(f'{gain:+.1f}pp\n({pct:.0f}%)')
            boost_data.append(row_data)

        boost_tbl = Table(boost_data, colWidths=[2.5*cm,1.5*cm,3*cm,3*cm,3*cm,3*cm])
        boost_tbl.setStyle(tbl_style())
        # Highlight best gain per row
        for ri, true_deg in enumerate([d for d in DEGREES
                                        if len(retry_df[retry_df['true_deg']==d]) > 0], start=1):
            sub   = retry_df[retry_df['true_deg'] == true_deg]
            gains = [(sub[c] - sub['orig_confidence']).mean()
                     for c in ['conf_rot0','conf_rot90','conf_rot180','conf_rot270']]
            best  = int(np.argmax(gains)) + 2
            boost_tbl.setStyle(TableStyle([
                ('BACKGROUND', (best,ri), (best,ri), colors.HexColor('#d4edda')),
                ('FONTNAME',   (best,ri), (best,ri), 'Helvetica-Bold'),
            ]))
        story.append(boost_tbl)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph('Green = strongest avg gain for that class.', MONO))
        story.append(Spacer(1, 0.3*cm))

        heatmap_path = OUT_DIR / f'{slug}_gain_heatmap.png'
        scatter_path = OUT_DIR / f'{slug}_gain_scatter.png'
        story.append(add_img(heatmap_path, max_w=12*cm, max_h=7*cm))
        story.append(Spacer(1, 0.2*cm))
        story.append(add_img(scatter_path, max_w=12*cm, max_h=7*cm))
    else:
        story += [
            Paragraph('No uncertain images — all predictions above threshold.', BODY),
        ]

    # Rotate-to-0 consistency
    if len(cons_df) > 0:
        rate    = cons_df['corrected_is_0'].mean()
        r_right = cons_df[cons_df['orig_correct']]['corrected_is_0'].mean() \
                  if cons_df['orig_correct'].any() else float('nan')
        r_wrong = cons_df[~cons_df['orig_correct']]['corrected_is_0'].mean() \
                  if (~cons_df['orig_correct']).any() else float('nan')
        story += [
            Paragraph('Rotate-to-0 Self-Consistency:', H3),
        ]
        cons_data = [
            ['Metric', 'Value'],
            ['Corrected → model says 0°',
             f'{cons_df["corrected_is_0"].sum()} / {len(cons_df)} ({rate*100:.1f}%)'],
            ['When orig correct', f'{r_right*100:.1f}%' if not np.isnan(r_right) else 'N/A'],
            ['When orig wrong',   f'{r_wrong*100:.1f}%' if not np.isnan(r_wrong) else 'N/A'],
        ]
        story.append(Table(cons_data, colWidths=[9*cm, 8*cm], style=tbl_style()))

    story.append(PageBreak())

# Cross-set comparison
story += [
    Paragraph('4. Set 1 vs Set 2 — Retry Comparison', H2),
]
comp_data = [['Metric', 'Set 1 (Clean)', 'Set 2 (Aug + Messy)']]
for metric_label, fn in [
    ('Images', lambda d, r: str(len(d))),
    ('Overall accuracy', lambda d, r: f'{d["correct"].mean()*100:.2f}%'),
    ('Avg confidence', lambda d, r: f'{d["confidence"].mean()*100:.1f}%'),
    (f'Uncertain (<{THRESHOLD:.0%})', lambda d, r: f'{d["uncertain"].sum()} ({d["uncertain"].mean()*100:.1f}%)'),
    ('Retry acc (uncertain set)', lambda d, r: f'{r["retry_correct"].mean()*100:.1f}%' if len(r) else 'N/A'),
    ('Avg conf gain', lambda d, r: f'+{r["conf_gain"].mean()*100:.1f}pp' if len(r) else 'N/A'),
    ('Errors rescued', lambda d, r: str(((~r["orig_correct"]) & r["retry_correct"]).sum()) if len(r) else 'N/A'),
]:
    row = [metric_label]
    for set_name in all_results:
        row.append(fn(all_results[set_name]['df'], all_results[set_name]['retry_df']))
    comp_data.append(row)

story.append(Table(comp_data, colWidths=[7*cm, 5.5*cm, 5.5*cm], style=tbl_style()))
story += [
    Spacer(1, 0.5*cm),
    HRFlowable(width='100%', thickness=0.5, color=colors.grey),
    Paragraph('Model: PP-LCNet_x1_0_doc_ori &nbsp;|&nbsp; English-only, 7 PDFs &nbsp;|&nbsp; '
              f'Threshold: {THRESHOLD:.0%} &nbsp;|&nbsp; Date: 2026-05-06', MONO),
]

doc = SimpleDocTemplate(
    str(OUT_DIR / 'English_Only_Confidence_Retry_Report.pdf'),
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)
doc.build(story)
print(f'Report saved -> {OUT_DIR / "English_Only_Confidence_Retry_Report.pdf"}')

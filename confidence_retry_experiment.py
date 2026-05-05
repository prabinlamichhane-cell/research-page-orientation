"""
Confidence-based retry experiment — runs on the main dataset (data/rotated/).

For each image in data/dataset.csv:
  1. Run inference, record (prediction, confidence).
  2. Flag uncertain predictions where confidence < THRESHOLD (default 0.50).
  3. For uncertain images: rotate the image to all 3 other orientations,
     run inference on each, pick highest confidence.
     Inferred true orientation = (model_pred_deg - retry_rotation) % 360
  4. Rotate-to-0 self-consistency: apply inverse of predicted rotation,
     check if model says 0° with high confidence.
  5. Report + charts.

Model already outputs softmax probabilities — no additional softmax applied.
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

from src.preprocess import load_and_preprocess, LABEL_TO_DEGREES, DEGREES_TO_LABEL

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_CSV  = Path('data/dataset.csv')
OUT_DIR      = Path('results/confidence_retry')
MODEL_PATH   = 'models/model.onnx'
THRESHOLD    = 0.75
DEGREES      = [0, 90, 180, 270]
ROTATE_FLAGS = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ONNX session ──────────────────────────────────────────────────────────────
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(MODEL_PATH, sess_opts,
                               providers=['CPUExecutionProvider'])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

TMP = Path('/tmp/_retry_tmp.png')


def predict_from_array(img_bgr: np.ndarray):
    """Model outputs softmax probs directly — no additional softmax."""
    cv2.imwrite(str(TMP), img_bgr)
    tensor = load_and_preprocess(TMP)
    probs  = session.run([out_name], {inp_name: tensor})[0][0]
    label  = int(np.argmax(probs))
    return label, float(probs[label]), probs


def rotate_array(img: np.ndarray, degrees: int) -> np.ndarray:
    flag = ROTATE_FLAGS[degrees]
    return img if flag is None else cv2.rotate(img, flag)


# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_CSV)
print(f'Loaded {len(df)} images from {DATASET_CSV}')
print(f'  Messy: {df["messy"].sum()} ({df["messy"].mean()*100:.0f}%)  '
      f'Clean: {(~df["messy"]).sum()} ({(~df["messy"]).mean()*100:.0f}%)')

# ── Step 1: Run inference on all images ───────────────────────────────────────
print('\nRunning inference on main dataset...')
pred_labels, confidences = [], []
prob_cols = {d: [] for d in DEGREES}

for row in tqdm(df.itertuples(), total=len(df)):
    img = cv2.imread(row.image_path)
    p_label, conf, probs = predict_from_array(img)
    pred_labels.append(p_label)
    confidences.append(conf)
    for i, d in enumerate(DEGREES):
        prob_cols[d].append(float(probs[i]))

df['pred_label']  = pred_labels
df['confidence']  = confidences
df['correct']     = df['pred_label'] == df['label']
df['uncertain']   = df['confidence'] < THRESHOLD
for d in DEGREES:
    df[f'prob_{d}'] = prob_cols[d]

print(f'\nBaseline:')
print(f'  Accuracy      : {df["correct"].mean()*100:.2f}%')
print(f'  Avg confidence: {df["confidence"].mean()*100:.1f}%')
print(f'  Uncertain (<{THRESHOLD:.0%}): {df["uncertain"].sum()} / {len(df)} ({df["uncertain"].mean()*100:.1f}%)')
print(f'    wrong + uncertain: {(~df["correct"] & df["uncertain"]).sum()}')
print(f'    right + uncertain: {(df["correct"] & df["uncertain"]).sum()}')

# ── Step 2: Retry uncertain predictions ───────────────────────────────────────
uncertain_df = df[df['uncertain']].copy()
print(f'\nRetrying {len(uncertain_df)} uncertain predictions...')

retry_records = []
for row in tqdm(uncertain_df.itertuples(), total=len(uncertain_df)):
    img = cv2.imread(row.image_path)

    best = {'conf': row.confidence, 'inferred_label': row.pred_label, 'retry_deg': 0}
    all_retries = []
    per_rot_conf = {}  # retry_deg → confidence after retry

    for retry_deg in DEGREES:
        retried = rotate_array(img, retry_deg)
        r_label, r_conf, _ = predict_from_array(retried)
        inferred_deg   = (LABEL_TO_DEGREES[r_label] - retry_deg) % 360
        inferred_label = DEGREES_TO_LABEL.get(inferred_deg, -1)
        per_rot_conf[retry_deg] = r_conf
        all_retries.append({
            'retry_deg':       retry_deg,
            'model_pred_deg':  LABEL_TO_DEGREES[r_label],
            'model_conf':      r_conf,
            'inferred_deg':    inferred_deg,
            'inferred_label':  inferred_label,
        })
        if r_conf > best['conf']:
            best = {'conf': r_conf, 'inferred_label': inferred_label, 'retry_deg': retry_deg}

    best_r = max(all_retries, key=lambda x: x['model_conf'])
    retry_records.append({
        'source':              row.source,
        'true_deg':            row.degrees,
        'true_label':          row.label,
        'messy':               row.messy,
        'orig_pred_label':     row.pred_label,
        'orig_pred_deg':       LABEL_TO_DEGREES[row.pred_label],
        'orig_confidence':     row.confidence,
        'conf_rot0':           per_rot_conf[0],
        'conf_rot90':          per_rot_conf[90],
        'conf_rot180':         per_rot_conf[180],
        'conf_rot270':         per_rot_conf[270],
        'best_retry_deg':      best_r['retry_deg'],
        'best_model_pred_deg': best_r['model_pred_deg'],
        'best_conf':           best_r['model_conf'],
        'inferred_label':      best_r['inferred_label'],
        'retry_correct':       best_r['inferred_label'] == row.label,
        'orig_correct':        row.correct,
        'conf_gain':           best_r['model_conf'] - row.confidence,
    })

retry_df = pd.DataFrame(retry_records)

# ── Step 3: Rotate-to-0 self-consistency check ────────────────────────────────
print('\nRunning rotate-to-0 self-consistency check...')
cons_records = []
for row in tqdm(uncertain_df.itertuples(), total=len(uncertain_df)):
    img = cv2.imread(row.image_path)
    inverse_deg = (360 - LABEL_TO_DEGREES[row.pred_label]) % 360
    corrected   = rotate_array(img, inverse_deg)
    c_label, c_conf, _ = predict_from_array(corrected)
    cons_records.append({
        'source':          row.source,
        'true_deg':        row.degrees,
        'true_label':      row.label,
        'messy':           row.messy,
        'orig_pred_deg':   LABEL_TO_DEGREES[row.pred_label],
        'orig_confidence': row.confidence,
        'inverse_applied': inverse_deg,
        'corrected_pred':  LABEL_TO_DEGREES[c_label],
        'corrected_conf':  c_conf,
        'corrected_is_0':  c_label == 0,
        'orig_correct':    row.correct,
    })

cons_df = pd.DataFrame(cons_records)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
df.to_csv(OUT_DIR / 'all_predictions.csv', index=False)
retry_df.to_csv(OUT_DIR / 'retry_results.csv', index=False)
cons_df.to_csv(OUT_DIR / 'consistency_results.csv', index=False)

# ── Report ────────────────────────────────────────────────────────────────────
sep = '='*65
print(f'\n{sep}')
print('CONFIDENCE-BASED RETRY EXPERIMENT — REPORT')
print(f'{sep}')
print(f'Dataset   : {len(df)} images (9 Nepali PDFs, main dataset)')
print(f'Threshold : {THRESHOLD:.0%}\n')

print('── Baseline ──────────────────────────────────────────────────')
print(f'Accuracy               : {df["correct"].mean()*100:.2f}%')
print(f'Avg confidence         : {df["confidence"].mean()*100:.1f}%')
print(f'Uncertain predictions  : {df["uncertain"].sum()} ({df["uncertain"].mean()*100:.1f}%)')
print(f'  wrong + uncertain    : {(~df["correct"] & df["uncertain"]).sum()}')
print(f'  right + uncertain    : {(df["correct"] & df["uncertain"]).sum()}  '
      f'(correct but model unsure — retry may flip these)')

print('\n── Confidence by class ───────────────────────────────────────')
for deg in DEGREES:
    sub = df[df['degrees'] == deg]
    print(f'  {deg:>3}°  avg={sub["confidence"].mean()*100:.1f}%  '
          f'uncertain={sub["uncertain"].sum()}/{len(sub)}  '
          f'acc={sub["correct"].mean()*100:.1f}%')

print('\n── Clean vs Messy confidence ─────────────────────────────────')
for label, mask in [('Clean', ~df['messy']), ('Messy', df['messy'])]:
    sub = df[mask]
    print(f'  {label:<6}  avg_conf={sub["confidence"].mean()*100:.1f}%  '
          f'uncertain={sub["uncertain"].sum()}/{len(sub)} ({sub["uncertain"].mean()*100:.1f}%)  '
          f'acc={sub["correct"].mean()*100:.1f}%')

if len(retry_df) > 0:
    print('\n── Retry results ─────────────────────────────────────────────')
    acc_before = retry_df['orig_correct'].mean()
    acc_after  = retry_df['retry_correct'].mean()
    avg_gain   = retry_df['conf_gain'].mean()
    rescued    = ((~retry_df['orig_correct']) & retry_df['retry_correct']).sum()
    broken     = (retry_df['orig_correct'] & (~retry_df['retry_correct'])).sum()
    print(f'  Accuracy before retry : {acc_before*100:.1f}%')
    print(f'  Accuracy after retry  : {acc_after*100:.1f}%')
    print(f'  Avg confidence gain   : +{avg_gain*100:.1f}pp')
    print(f'  Errors rescued        : {rescued}')
    print(f'  Correct flipped wrong : {broken}  (retry hurt these)')

    print('\n  Which retry rotation gave highest confidence?')
    for deg in DEGREES:
        sub = retry_df[retry_df['best_retry_deg'] == deg]
        if len(sub) == 0:
            continue
        print(f'  +{deg:>3}°  → {len(sub):>3} cases ({len(sub)/len(retry_df)*100:.0f}%)  '
              f'retry_acc={sub["retry_correct"].mean()*100:.1f}%')

    print('\n  Pattern — true orientation vs best retry rotation:')
    pivot = retry_df.groupby(['true_deg', 'best_retry_deg']).size().unstack(fill_value=0)
    pivot.index.name   = 'True deg'
    pivot.columns.name = 'Best retry'
    print(pivot.to_string())

    if retry_df['messy'].any():
        print('\n  Retry accuracy by image quality:')
        for label, mask in [('Clean', ~retry_df['messy']), ('Messy', retry_df['messy'])]:
            sub = retry_df[mask]
            if len(sub):
                print(f'    {label:<6}: before={sub["orig_correct"].mean()*100:.1f}%  '
                      f'after={sub["retry_correct"].mean()*100:.1f}%  n={len(sub)}')

if len(cons_df) > 0:
    print('\n── Rotate-to-0 self-consistency ──────────────────────────────')
    rate     = cons_df['corrected_is_0'].mean()
    avg_conf = cons_df['corrected_conf'].mean()
    print(f'  After rotating by -predicted_deg, model says 0°: '
          f'{cons_df["corrected_is_0"].sum()}/{len(cons_df)} ({rate*100:.1f}%)')
    print(f'  Avg confidence on corrected image: {avg_conf*100:.1f}%')
    if cons_df['orig_correct'].any() and (~cons_df['orig_correct']).any():
        r_right = cons_df[cons_df['orig_correct']]['corrected_is_0'].mean()
        r_wrong = cons_df[~cons_df['orig_correct']]['corrected_is_0'].mean()
        print(f'  Corrected-to-0 when orig RIGHT : {r_right*100:.1f}%')
        print(f'  Corrected-to-0 when orig WRONG : {r_wrong*100:.1f}%')
        useful = abs(r_right - r_wrong) > 0.2
        print(f'  Error detector signal          : {"STRONG" if useful else "WEAK"}')

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['confidence'], bins=40, color='#4C72B0', edgecolor='white')
axes[0].axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD:.0%})')
axes[0].set_xlabel('Confidence')
axes[0].set_ylabel('Count')
axes[0].set_title('Confidence Distribution — All Images')
axes[0].legend()

conf_by_class = [df[df['degrees'] == d]['confidence'].values for d in DEGREES]
axes[1].boxplot(conf_by_class, tick_labels=[f'{d}°' for d in DEGREES])
axes[1].axhline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD:.0%})')
axes[1].set_ylabel('Confidence')
axes[1].set_title('Confidence by True Orientation')
axes[1].legend()
plt.suptitle('Confidence Analysis — Main Dataset (9 Nepali PDFs)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'confidence_distribution.png', dpi=150)
plt.close()

# 2. Confidence: clean vs messy
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df[~df['messy']]['confidence'], bins=30, alpha=0.6, label='Clean', color='#55A868')
ax.hist(df[df['messy']]['confidence'],  bins=30, alpha=0.6, label='Messy', color='#C44E52')
ax.axvline(THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD:.0%})')
ax.set_xlabel('Confidence')
ax.set_ylabel('Count')
ax.set_title('Confidence Distribution: Clean vs Messy')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'confidence_clean_vs_messy.png', dpi=150)
plt.close()

# 3. Retry rotation heatmap
if len(retry_df) > 0:
    try:
        pivot = retry_df.groupby(['true_deg', 'best_retry_deg']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Best Retry Rotation Applied')
        ax.set_ylabel('True Orientation')
        ax.set_title(f'Which Retry Rotation Helped Most?\n(uncertain predictions, threshold={THRESHOLD:.0%})')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'retry_rotation_heatmap.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f'  (heatmap skipped: {e})')

# 4. Confidence before vs after retry scatter
if len(retry_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = retry_df['retry_correct'].map({True: '#55A868', False: '#C44E52'})
    ax.scatter(retry_df['orig_confidence'], retry_df['best_conf'],
               c=colors, alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='no gain')
    ax.axvline(THRESHOLD, color='red', linestyle=':', alpha=0.5, label=f'threshold')
    ax.set_xlabel('Confidence Before Retry')
    ax.set_ylabel('Best Confidence After Retry')
    ax.set_title('Confidence Gain from Retry\n(green=correct after retry, red=wrong)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'confidence_gain_scatter.png', dpi=150)
    plt.close()

# 5. Per-rotation confidence — which rotation consistently boosts confidence?
if len(retry_df) > 0:
    print('\n── Consistent rotation boost (avg conf gain per retry angle) ──')
    print('  (positive = retry at this angle reliably increases confidence)')
    for true_deg in DEGREES:
        sub = retry_df[retry_df['true_deg'] == true_deg]
        if len(sub) == 0:
            continue
        print(f'\n  True {true_deg:>3}°  (n={len(sub)}):')
        for rot_col, rot_deg in [('conf_rot0',0),('conf_rot90',90),('conf_rot180',180),('conf_rot270',270)]:
            if rot_col not in sub.columns:
                continue
            gain     = (sub[rot_col] - sub['orig_confidence']).mean()
            pct_up   = (sub[rot_col] > sub['orig_confidence']).mean() * 100
            print(f'    +{rot_deg:>3}°  avg_gain={gain*100:+.1f}pp  '
                  f'improved={pct_up:.0f}% of cases')

    # Heatmap: avg confidence gain per (true_deg, retry_deg)
    gain_rows = []
    for true_deg in DEGREES:
        sub = retry_df[retry_df['true_deg'] == true_deg]
        if len(sub) == 0:
            continue
        for rot_col, rot_deg in [('conf_rot0',0),('conf_rot90',90),('conf_rot180',180),('conf_rot270',270)]:
            if rot_col not in sub.columns:
                continue
            gain_rows.append({
                'true_deg':  true_deg,
                'retry_deg': rot_deg,
                'avg_gain':  (sub[rot_col] - sub['orig_confidence']).mean() * 100,
            })
    gain_df = pd.DataFrame(gain_rows)
    gain_pivot = gain_df.pivot(index='true_deg', columns='retry_deg', values='avg_gain')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(gain_pivot, annot=True, fmt='.1f', ax=ax,
                cmap='RdYlGn', center=0,
                xticklabels=[f'+{d}°' for d in DEGREES],
                yticklabels=[f'{d}°' for d in DEGREES])
    ax.set_xlabel('Retry Rotation Applied')
    ax.set_ylabel('True Orientation')
    ax.set_title(f'Avg Confidence Gain (pp) per Retry Rotation\n'
                 f'(threshold={THRESHOLD:.0%}, n={len(retry_df)} uncertain images)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'retry_confidence_gain_heatmap.png', dpi=150)
    plt.close()

print(f'\nResults → {OUT_DIR}')

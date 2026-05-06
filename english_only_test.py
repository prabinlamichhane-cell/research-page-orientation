"""
Isolated test on English-only Nepali financial document.
Two experiment sets:
  Set 1 — Rotated only       (clean, no augmentation)
  Set 2 — Rotated + augment + messify (same pipeline as main dataset)

Runs inference across all 3 runtimes for each set.
"""

import sys
sys.path.insert(0, '.')

import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.preprocess import load_and_preprocess, rotate_image, DEGREES_TO_LABEL

PDF_DIR  = Path('data/pdfs_english_only')
IMG_DIR  = Path('data/english_only_raw')
OUT_DIR  = Path('results/english_only')

IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI        = 150
MAX_PAGES  = 40
MESSY_PROB = 0.4

rng = np.random.default_rng(seed=42)

# ── Augmentation pipeline (identical to 00_dataset_creation.ipynb) ────────────

def augment(img: np.ndarray) -> np.ndarray:
    img   = img.copy().astype(np.float32)
    alpha = rng.uniform(0.85, 1.15)
    beta  = float(rng.integers(-15, 15))
    img   = np.clip(img * alpha + beta, 0, 255)
    img   = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255)
    img   = img.astype(np.uint8)
    h, w  = img.shape[:2]
    t = rng.integers(0, max(1, int(h * 0.02)))
    b = rng.integers(0, max(1, int(h * 0.02)))
    l = rng.integers(0, max(1, int(w * 0.02)))
    r = rng.integers(0, max(1, int(w * 0.02)))
    img = cv2.resize(img[t:h-b, l:w-r], (w, h), interpolation=cv2.INTER_LINEAR)
    q   = int(rng.integers(70, 95))
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def messify(img: np.ndarray) -> np.ndarray:
    img  = img.copy()
    h, w = img.shape[:2]

    if rng.random() < 0.5:
        angle = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        direction       = rng.choice(['left', 'right', 'top', 'bottom'])
        shadow_strength = rng.uniform(0.3, 0.6)
        shadow_width    = rng.uniform(0.2, 0.5)
        mask = np.ones((h, w), dtype=np.float32)
        if direction == 'left':
            end = int(w * shadow_width)
            mask[:, :end] = np.linspace(1 - shadow_strength, 1, end)
        elif direction == 'right':
            start = int(w * (1 - shadow_width))
            mask[:, start:] = np.linspace(1, 1 - shadow_strength, w - start)
        elif direction == 'top':
            end = int(h * shadow_width)
            mask[:end, :] = np.linspace(1 - shadow_strength, 1, end).reshape(-1, 1)
        else:
            start = int(h * (1 - shadow_width))
            mask[start:, :] = np.linspace(1, 1 - shadow_strength, h - start).reshape(-1, 1)
        img = np.clip(img.astype(np.float32) * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k      = int(rng.choice([2, 3]))
        kernel = np.ones((k, k), np.uint8)
        img    = cv2.dilate(img, kernel, iterations=1)
    if rng.random() < 0.3:
        fold_axis = rng.choice(['h', 'v'])
        if fold_axis == 'h':
            y = int(rng.integers(h // 4, 3 * h // 4))
            img[y-1:y+2, :] = np.clip(
                img[y-1:y+2, :].astype(np.float32) * rng.uniform(0.5, 0.75), 0, 255
            ).astype(np.uint8)
        else:
            x = int(rng.integers(w // 4, 3 * w // 4))
            img[:, x-1:x+2] = np.clip(
                img[:, x-1:x+2].astype(np.float32) * rng.uniform(0.5, 0.75), 0, 255
            ).astype(np.uint8)
    if rng.random() < 0.4:
        amount   = rng.uniform(0.002, 0.01)
        n_pixels = int(amount * h * w)
        ys = rng.integers(0, h, n_pixels); xs = rng.integers(0, w, n_pixels)
        img[ys, xs] = 255
        ys = rng.integers(0, h, n_pixels); xs = rng.integers(0, w, n_pixels)
        img[ys, xs] = 0
    if rng.random() < 0.35:
        k   = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        q      = int(rng.integers(20, 50))
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img    = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


# ── Step 1: PDF → source images ───────────────────────────────────────────────
pdfs = sorted(PDF_DIR.glob('*.pdf'))
print(f'Found {len(pdfs)} PDFs in {PDF_DIR}:')
for p in pdfs:
    print(f'  {p.name}')

mat          = fitz.Matrix(DPI / 72, DPI / 72)
source_paths = []

for pdf_path in pdfs:
    doc       = fitz.open(str(pdf_path))
    extracted = 0
    for i in range(min(len(doc), MAX_PAGES)):
        pix     = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img     = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if np.sum(gray >= 250) / gray.size >= 0.97:
            continue
        out = IMG_DIR / f'{pdf_path.stem}_page_{i+1:04d}.png'
        cv2.imwrite(str(out), img_bgr)
        source_paths.append(out)
        extracted += 1
    doc.close()
    print(f'  {pdf_path.name}: {extracted} pages extracted')

print(f'Total: {len(source_paths)} source pages')


# ── Step 2: Build both datasets ───────────────────────────────────────────────
print('\nBuilding Set 1 (rotated only) and Set 2 (rotated + augment + messify)...')

# Reset rng to same seed so both sets are deterministic and comparable
rng = np.random.default_rng(seed=42)

rot_dir_clean = Path('data/english_only_rotated_clean')
rot_dir_messy = Path('data/english_only_rotated_messy')
rot_dir_clean.mkdir(parents=True, exist_ok=True)
rot_dir_messy.mkdir(parents=True, exist_ok=True)

records_clean, records_messy = [], []

for src in source_paths:
    img = cv2.imread(str(src))
    for deg in [0, 90, 180, 270]:
        rotated = rotate_image(src, deg)

        # Set 1: rotated only
        out_clean = rot_dir_clean / f'{src.stem}_rot{deg}.png'
        cv2.imwrite(str(out_clean), rotated)
        records_clean.append({
            'image_path': str(out_clean.resolve()),
            'label':      DEGREES_TO_LABEL[deg],
            'degrees':    deg,
            'source':     src.name,
            'messy':      False,
        })

        # Set 2: rotated + augment + messify(40%)
        processed = augment(rotated)
        is_messy  = rng.random() < MESSY_PROB
        if is_messy:
            processed = messify(processed)
        out_messy = rot_dir_messy / f'{src.stem}_rot{deg}.png'
        cv2.imwrite(str(out_messy), processed)
        records_messy.append({
            'image_path': str(out_messy.resolve()),
            'label':      DEGREES_TO_LABEL[deg],
            'degrees':    deg,
            'source':     src.name,
            'messy':      is_messy,
        })

df_clean = pd.DataFrame(records_clean)
df_messy = pd.DataFrame(records_messy)
n_messy  = df_messy['messy'].sum()

print(f'  Set 1 (clean)  : {len(df_clean)} images — rotated only')
print(f'  Set 2 (aug+messy): {len(df_messy)} images — '
      f'{n_messy} messy ({n_messy/len(df_messy)*100:.0f}%) / '
      f'{len(df_messy)-n_messy} clean ({(1-n_messy/len(df_messy))*100:.0f}%)')


# ── Step 3: Inference helpers ─────────────────────────────────────────────────
import paddle
from paddle.inference import Config, create_predictor
import onnxruntime as ort
import torch
from optimum.onnxruntime import ORTModelForImageClassification

# PaddlePaddle
paddle_config = Config(
    'models/PP-LCNet_x1_0_doc_ori_infer/inference.json',
    'models/PP-LCNet_x1_0_doc_ori_infer/inference.pdiparams',
)
paddle_config.disable_gpu()
paddle_config.enable_mkldnn()
paddle_config.set_cpu_math_library_num_threads(4)
paddle_config.switch_ir_optim(True)
predictor   = create_predictor(paddle_config)
input_names  = predictor.get_input_names()
output_names = predictor.get_output_names()

def predict_paddle(image_path):
    tensor = load_and_preprocess(image_path)
    h = predictor.get_input_handle(input_names[0])
    h.reshape(tensor.shape)
    h.copy_from_cpu(tensor)
    predictor.run()
    out = predictor.get_output_handle(output_names[0]).copy_to_cpu()
    return int(np.argmax(out, axis=1)[0]), float(np.max(out))

# ONNX Runtime
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session  = ort.InferenceSession('models/model.onnx', sess_opts,
                                providers=['CPUExecutionProvider'])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

def predict_onnx(image_path):
    tensor = load_and_preprocess(image_path)
    logits = session.run([out_name], {inp_name: tensor})[0]
    t0 = time.perf_counter()
    return int(np.argmax(logits, axis=1)[0])

# Optimum ORT
opt_model = ORTModelForImageClassification.from_pretrained(
    'models/optimum', local_files_only=True)

def predict_optimum(image_path):
    pv     = torch.from_numpy(load_and_preprocess(image_path))
    logits = opt_model(pixel_values=pv).logits.detach().numpy()
    return int(np.argmax(logits, axis=1)[0])


# ── Step 4: Run inference on both sets ────────────────────────────────────────
def run_inference(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    for runtime, pred_fn, pred_col, lat_col in [
        ('PaddlePaddle', predict_paddle,  'paddle_pred',  'paddle_lat'),
        ('ONNX Runtime', predict_onnx,    'onnx_pred',    'onnx_lat'),
        ('Optimum ORT',  predict_optimum, 'optimum_pred', 'optimum_lat'),
    ]:
        preds, lats = [], []
        for row in tqdm(df.itertuples(), total=len(df), desc=f'  {label} | {runtime}'):
            t0 = time.perf_counter()
            if runtime == 'PaddlePaddle':
                p, _ = predict_paddle(row.image_path)
            elif runtime == 'ONNX Runtime':
                tensor = load_and_preprocess(row.image_path)
                logits = session.run([out_name], {inp_name: tensor})[0]
                p = int(np.argmax(logits, axis=1)[0])
            else:
                pv = torch.from_numpy(load_and_preprocess(row.image_path))
                logits = opt_model(pixel_values=pv).logits.detach().numpy()
                p = int(np.argmax(logits, axis=1)[0])
            lats.append((time.perf_counter() - t0) * 1000)
            preds.append(p)
        df[pred_col] = preds
        df[lat_col]  = lats
    return df

print('\n--- Set 1: Rotated only ---')
df_clean = run_inference(df_clean, 'Set1-Clean')

print('\n--- Set 2: Rotated + augment + messify ---')
df_messy = run_inference(df_messy, 'Set2-Augmented')

df_clean.to_csv(OUT_DIR / 'english_only_clean_results.csv', index=False)
df_messy.to_csv(OUT_DIR / 'english_only_augmented_results.csv', index=False)


# ── Step 5: Report ────────────────────────────────────────────────────────────
sep = '='*65
print(f'\n{sep}')
print('ENGLISH-ONLY ISOLATED TEST — TWO-SET RESULTS')
print(f'{sep}')
print(f'PDFs     : {len(pdfs)} documents from {PDF_DIR}')
print(f'Pages    : {len(source_paths)} source pages\n')

for set_label, df in [('Set 1 — Rotated only (clean)', df_clean),
                       ('Set 2 — Rotated + augment + messify', df_messy)]:
    print(f'\n{set_label}')
    print('-'*50)
    for name, pred_col, lat_col in [
        ('PaddlePaddle', 'paddle_pred',  'paddle_lat'),
        ('ONNX Runtime', 'onnx_pred',    'onnx_lat'),
        ('Optimum ORT',  'optimum_pred', 'optimum_lat'),
    ]:
        acc = (df[pred_col] == df['label']).mean()
        avg = df[lat_col].mean()
        print(f'  {name:<15}  acc={acc:.4f} ({acc*100:.1f}%)  avg_lat={avg:.2f}ms')

    # Per-class
    print(f'\n  Per-class (ONNX Runtime):')
    print(classification_report(df['label'], df['onnx_pred'],
          target_names=['0deg','90deg','180deg','270deg'], digits=4))

    # Messy breakdown if applicable
    if df['messy'].any():
        clean_acc = (df.loc[~df['messy'], 'onnx_pred'] == df.loc[~df['messy'], 'label']).mean()
        messy_acc = (df.loc[df['messy'],  'onnx_pred'] == df.loc[df['messy'],  'label']).mean()
        print(f'  Clean subset: {clean_acc*100:.1f}%  |  Messy subset: {messy_acc*100:.1f}%')

agree = (df_clean['onnx_pred'] == df_messy['onnx_pred']).mean()
print(f'\nAgreement between Set1 and Set2 predictions: {agree:.4f}')


# ── Step 6: Charts ────────────────────────────────────────────────────────────
runtimes = ['PaddlePaddle', 'ONNX Runtime', 'Optimum ORT']
pred_cols = ['paddle_pred', 'onnx_pred', 'optimum_pred']
lat_cols  = ['paddle_lat',  'onnx_lat',  'optimum_lat']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bar_w = 0.35
x = np.arange(len(runtimes))

accs_clean = [(df_clean[c] == df_clean['label']).mean() * 100 for c in pred_cols]
accs_messy = [(df_messy[c] == df_messy['label']).mean() * 100 for c in pred_cols]

bars1 = axes[0].bar(x - bar_w/2, accs_clean, bar_w, label='Set 1: Rotated only',  color='#4C72B0')
bars2 = axes[0].bar(x + bar_w/2, accs_messy, bar_w, label='Set 2: Aug + Messify', color='#DD8452')
axes[0].set_xticks(x); axes[0].set_xticklabels(runtimes, rotation=10)
axes[0].set_ylim(0, 105); axes[0].set_ylabel('%')
axes[0].set_title('Accuracy — English Only')
axes[0].legend()
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8)

lats_clean = [df_clean[c].mean() for c in lat_cols]
lats_messy = [df_messy[c].mean() for c in lat_cols]
axes[1].bar(x - bar_w/2, lats_clean, bar_w, label='Set 1: Rotated only',  color='#4C72B0')
axes[1].bar(x + bar_w/2, lats_messy, bar_w, label='Set 2: Aug + Messify', color='#DD8452')
axes[1].set_xticks(x); axes[1].set_xticklabels(runtimes, rotation=10)
axes[1].set_ylabel('ms'); axes[1].set_title('Avg Latency — English Only')
axes[1].legend()

plt.suptitle(f'English-Only ({len(pdfs)} PDFs, {len(source_paths)} pages x4): Clean vs Augmented', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'english_only_comparison.png', dpi=150)
plt.close()

# Confusion matrices side by side (ONNX)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, df, title in [
    (axes[0], df_clean, 'Set 1: Rotated only'),
    (axes[1], df_messy, 'Set 2: Aug + Messify'),
]:
    cm = confusion_matrix(df['label'], df['onnx_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=['0°','90°','180°','270°'],
                yticklabels=['0°','90°','180°','270°'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
plt.suptitle('English-Only — Confusion Matrices (ONNX Runtime)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'english_only_confusion_matrix.png', dpi=150)
plt.close()

print(f'\nResults saved -> {OUT_DIR}')

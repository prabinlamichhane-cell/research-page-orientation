"""
Isolated test on English-only Nepali financial document.
Extracts pages, rotates x4, runs inference across all 3 runtimes.
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

from src.preprocess import load_and_preprocess, DEGREES_TO_LABEL

PDF_PATH   = Path('data/pdfs_english_only/Annual-Report-2023-24-English.pdf')
IMG_DIR    = Path('data/english_only_raw')
ROT_DIR    = Path('data/english_only_rotated')
OUT_DIR    = Path('results/english_only')

IMG_DIR.mkdir(parents=True, exist_ok=True)
ROT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 150
MAX_PAGES = 40   # cap at 40 pages for a representative sample

# ── Step 1: PDF → images ──────────────────────────────────────────────────────
print(f'Extracting pages from {PDF_PATH.name}...')
doc = fitz.open(str(PDF_PATH))
mat = fitz.Matrix(DPI / 72, DPI / 72)
source_paths = []

for i in range(min(len(doc), MAX_PAGES)):
    pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # skip blank pages
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.sum(gray >= 250) / gray.size >= 0.97:
        continue
    out = IMG_DIR / f'page_{i+1:04d}.png'
    cv2.imwrite(str(out), img_bgr)
    source_paths.append(out)

doc.close()
print(f'  {len(source_paths)} pages extracted')

# ── Step 2: Rotate x4 → balanced dataset ─────────────────────────────────────
print('Creating rotated dataset...')
rotate_flags = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

records = []
for src in source_paths:
    img = cv2.imread(str(src))
    for deg, flag in rotate_flags.items():
        out_img = img if flag is None else cv2.rotate(img, flag)
        out_path = ROT_DIR / f'{src.stem}_rot{deg}.png'
        cv2.imwrite(str(out_path), out_img)
        records.append({
            'image_path': str(out_path.resolve()),
            'label':      DEGREES_TO_LABEL[deg],
            'degrees':    deg,
            'source':     src.name,
        })

df = pd.DataFrame(records)
print(f'  {len(df)} images ({len(df)//4} source pages x 4 rotations)')

# ── Step 3: PaddlePaddle inference ────────────────────────────────────────────
print('\nRunning PaddlePaddle inference...')
import paddle
from paddle.inference import Config, create_predictor

config = Config(
    'models/PP-LCNet_x1_0_doc_ori_infer/inference.json',
    'models/PP-LCNet_x1_0_doc_ori_infer/inference.pdiparams',
)
config.disable_gpu()
config.enable_mkldnn()
config.set_cpu_math_library_num_threads(4)
config.switch_ir_optim(True)
predictor = create_predictor(config)
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

preds, lats = [], []
for row in tqdm(df.itertuples(), total=len(df), desc='  Paddle'):
    t0 = time.perf_counter()
    p, _ = predict_paddle(row.image_path)
    lats.append((time.perf_counter() - t0) * 1000)
    preds.append(p)
df['paddle_pred'] = preds
df['paddle_lat']  = lats

# ── Step 4: ONNX Runtime inference ───────────────────────────────────────────
print('Running ONNX Runtime inference...')
import onnxruntime as ort

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession('models/model.onnx', sess_opts,
                               providers=['CPUExecutionProvider'])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

preds, lats = [], []
for row in tqdm(df.itertuples(), total=len(df), desc='  ONNX'):
    t0 = time.perf_counter()
    tensor = load_and_preprocess(row.image_path)
    logits = session.run([out_name], {inp_name: tensor})[0]
    lats.append((time.perf_counter() - t0) * 1000)
    preds.append(int(np.argmax(logits, axis=1)[0]))
df['onnx_pred'] = preds
df['onnx_lat']  = lats

# ── Step 5: Optimum ORT inference ────────────────────────────────────────────
print('Running Optimum ORT inference...')
import torch
from optimum.onnxruntime import ORTModelForImageClassification

model = ORTModelForImageClassification.from_pretrained(
    'models/optimum', local_files_only=True)

preds, lats = [], []
for row in tqdm(df.itertuples(), total=len(df), desc='  Optimum'):
    t0 = time.perf_counter()
    pv = torch.from_numpy(load_and_preprocess(row.image_path))
    logits = model(pixel_values=pv).logits.detach().numpy()
    lats.append((time.perf_counter() - t0) * 1000)
    preds.append(int(np.argmax(logits, axis=1)[0]))
df['optimum_pred'] = preds
df['optimum_lat']  = lats

df.to_csv(OUT_DIR / 'english_only_results.csv', index=False)

# ── Step 6: Report ────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('ENGLISH-ONLY ISOLATED TEST RESULTS')
print('='*60)
print(f'Document : {PDF_PATH.name}')
print(f'Pages    : {len(source_paths)} source  →  {len(df)} test images\n')

for name, pred_col, lat_col in [
    ('PaddlePaddle', 'paddle_pred',  'paddle_lat'),
    ('ONNX Runtime', 'onnx_pred',    'onnx_lat'),
    ('Optimum ORT',  'optimum_pred', 'optimum_lat'),
]:
    acc = (df[pred_col] == df['label']).mean()
    avg = df[lat_col].mean()
    print(f'{name:<15}  acc={acc:.4f} ({acc*100:.1f}%)  '
          f'avg_lat={avg:.2f}ms  tput={1000/avg:.1f} img/s')

print()
print('Per-class (PaddlePaddle):')
print(classification_report(df['label'], df['paddle_pred'],
      target_names=['0°','90°','180°','270°'], digits=4))

agree_onnx = (df['paddle_pred'] == df['onnx_pred']).mean()
agree_opt  = (df['paddle_pred'] == df['optimum_pred']).mean()
print(f'Agreement Paddle vs ONNX   : {agree_onnx:.4f}')
print(f'Agreement Paddle vs Optimum: {agree_opt:.4f}')

# ── Plots ─────────────────────────────────────────────────────────────────────
cm_data = confusion_matrix(df['label'], df['paddle_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_data, annot=True, fmt='d', ax=ax,
            xticklabels=['0°','90°','180°','270°'],
            yticklabels=['0°','90°','180°','270°'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'English-Only — Confusion Matrix\n{PDF_PATH.name}')
plt.tight_layout()
plt.savefig(OUT_DIR / 'english_only_confusion_matrix.png', dpi=150)
plt.close()

# Summary bar chart
runtimes = ['PaddlePaddle', 'ONNX Runtime', 'Optimum ORT']
accs  = [(df[c] == df['label']).mean() * 100
         for c in ['paddle_pred','onnx_pred','optimum_pred']]
lats  = [df[c].mean()
         for c in ['paddle_lat','onnx_lat','optimum_lat']]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(runtimes, accs, color=['#4C72B0','#DD8452','#55A868'])
axes[0].set_ylim(0, 105)
axes[0].set_title('Accuracy — English Only')
axes[0].set_ylabel('%')
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

axes[1].bar(runtimes, lats, color=['#4C72B0','#DD8452','#55A868'])
axes[1].set_title('Avg Latency — English Only')
axes[1].set_ylabel('ms')
for i, v in enumerate(lats):
    axes[1].text(i, v + 0.3, f'{v:.1f}ms', ha='center', fontsize=9)

plt.suptitle(f'English-Only Isolated Test  ({len(df)} images)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'english_only_comparison.png', dpi=150)
plt.close()

print(f'\nResults saved → {OUT_DIR}')

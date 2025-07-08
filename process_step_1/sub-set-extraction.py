#!/usr/bin/env python3
# extract_features.py
# ---------------------------------------------
# Generates poster_features.jsonl with 2×2-grid
# color histograms (8 bins/channel) per image.
# ---------------------------------------------

import cv2
import os
import json
import numpy as np              # ← added import
from glob import glob

# === CONFIG ===
POSTER_DIR      = 'subset/'          # <-- update
OUTPUT_JSONL    = 'poster_features.jsonl'
GRID_SIZE       = (2, 2)      # rows, cols
BINS_PER_CH     = 8           # histogram bins per channel

# === FUNCTION ===
def grid_color_histogram(image,
                         grid_size=GRID_SIZE,
                         bins_per_channel=BINS_PER_CH):
    h, w = image.shape[:2]
    rows, cols = grid_size
    cell_h, cell_w = h // rows, w // cols
    hist_vecs = []

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell_h, c * cell_w
            cell   = image[y0:y0+cell_h, x0:x0+cell_w]
            hist   = cv2.calcHist([cell], [0,1,2], None,
                                  [bins_per_channel]*3,
                                  [0,256,0,256,0,256]).flatten()
            hist   = hist / hist.sum() if hist.sum() else hist
            hist_vecs.append(hist.astype('float32'))

    return np.concatenate(hist_vecs)

# === EXECUTION ===
poster_files = glob(os.path.join(POSTER_DIR, '*.jpg'))
if not poster_files:
    print(f'❌ No JPEGs found in {POSTER_DIR}')
    exit(1)

print(f'▶ Extracting features from {len(poster_files)} images…')

written, skipped = 0, 0
with open(OUTPUT_JSONL, 'w') as out_f:
    for fp in poster_files:
        img = cv2.imread(fp)
        if img is None:
            skipped += 1
            continue

        hist_vec = grid_color_histogram(img)
        record = {
            'filename'        : os.path.basename(fp),
            'color_histogram' : hist_vec.tolist()
        }
        out_f.write(json.dumps(record) + '\n')
        written += 1
        if written % 500 == 0:
            print(f'  …{written} done')

print(f'✅ Finished: {written} written, {skipped} skipped.')
print(f'   Output → {os.path.abspath(OUTPUT_JSONL)}')

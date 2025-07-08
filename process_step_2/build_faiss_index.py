#!/usr/bin/env python3
# build_index.py
# ---------------------------------------------
# Builds FAISS L2 index from poster_features.jsonl
# ---------------------------------------------

import faiss
import json
import numpy as np
import os

# === CONFIG ===
FEATURES_JSONL   = '../process_step_1/poster_features.jsonl'          # output of extract_features.py
FAISS_INDEX_OUT  = 'poster_index.faiss'
METADATA_OUT     = 'poster_files.json'

# === LOAD FEATURES ===
vectors, filenames = [], []

with open(FEATURES_JSONL, 'r') as f:
    for line in f:
        rec = json.loads(line)
        vec = np.array(rec['color_histogram'], dtype='float32')
        vectors.append(vec)
        filenames.append(rec['filename'])

if not vectors:
    print('❌ No vectors found. Check FEATURES_JSONL path.')
    exit(1)

mat = np.vstack(vectors)
dim = mat.shape[1]

# === BUILD INDEX ===
index = faiss.IndexFlatL2(dim)
index.add(mat)

# === SAVE ===
faiss.write_index(index, FAISS_INDEX_OUT)
with open(METADATA_OUT, 'w') as f:
    json.dump(filenames, f)

print(f'✅ Index built ({len(filenames)} vectors, dim={dim}).')
print(f'   • {FAISS_INDEX_OUT}')
print(f'   • {METADATA_OUT}')

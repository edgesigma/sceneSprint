#!/usr/bin/env python3
# enhanced_build_faiss_index.py
# ---------------------------------------------
# Builds FAISS L2 index from enhanced_poster_features.jsonl
# Uses combined pose + color features
# ---------------------------------------------

import faiss
import json
import numpy as np
import os

# === CONFIG ===
FEATURES_JSONL   = '../process_step_1/enhanced_poster_features.jsonl'
FAISS_INDEX_OUT  = 'enhanced_poster_index.faiss'
METADATA_OUT     = 'enhanced_poster_files.json'

# === LOAD ENHANCED FEATURES ===
vectors, filenames = [], []

print('▶ Loading enhanced features...')

with open(FEATURES_JSONL, 'r') as f:
    for line in f:
        rec = json.loads(line)
        # Use combined features for indexing
        vec = np.array(rec['combined_features'], dtype='float32')
        vectors.append(vec)
        filenames.append(rec['filename'])

if not vectors:
    print('❌ No vectors found. Check FEATURES_JSONL path.')
    exit(1)

mat = np.vstack(vectors)
dim = mat.shape[1]

print(f'  - Loaded {len(filenames)} feature vectors')
print(f'  - Feature dimension: {dim}')

# === BUILD INDEX ===
print('▶ Building FAISS index...')
index = faiss.IndexFlatL2(dim)
index.add(mat)

# === SAVE ===
print('▶ Saving index and metadata...')
faiss.write_index(index, FAISS_INDEX_OUT)
with open(METADATA_OUT, 'w') as f:
    json.dump(filenames, f, indent=2)

print(f'✅ Enhanced index built successfully!')
print(f'   • Index: {FAISS_INDEX_OUT} ({len(filenames)} vectors, dim={dim})')
print(f'   • Metadata: {METADATA_OUT}')
print(f'   • Features: pose + color combined')

#!/usr/bin/env python3
# app.py – drop-in ready Flask server
# ---------------------------------------------
# • Accepts selfie  → /match  (POST)
# • Computes 2×2-grid (8-bin) color histogram
# • Searches FAISS index built with same vectors
# • Returns composite (selfie | poster) URL
# ---------------------------------------------

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import faiss, cv2, json, uuid, os, numpy as np

# === CONFIG – update absolute paths & public host ===
FAISS_INDEX_PATH     = '../process_step_2/poster_index.faiss'
POSTER_METADATA_PATH = '../process_step_2/poster_files.json'
POSTER_DIR           = '../process_step_1/subset'
COMPOSITE_DIR        = 'composites/'
PUBLIC_HOST          = 'http://localhost:5000'  # <— change to IP / domain

GRID_SIZE        = (2, 2)   # rows, cols
BINS_PER_CHANNEL = 8        # per color channel

# === ENSURE DIRECTORIES ===
os.makedirs(COMPOSITE_DIR, exist_ok=True)

# === LOAD INDEX & FILENAMES ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(POSTER_METADATA_PATH, 'r') as f:
    poster_files = json.load(f)

# === GRID HISTOGRAM ===
def grid_color_histogram(image, grid_size=GRID_SIZE, bins=BINS_PER_CHANNEL):
    h, w = image.shape[:2]
    rows, cols = grid_size
    cell_h, cell_w = h // rows, w // cols
    parts = []
    for r in range(rows):
        for c in range(cols):
            cell = image[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            hist = cv2.calcHist([cell], [0,1,2], None,
                                [bins]*3, [0,256,0,256,0,256]).flatten()
            hist = hist / hist.sum() if hist.sum() else hist
            parts.append(hist.astype('float32'))
    return np.concatenate(parts)

# === FLASK ===
app = Flask(__name__)
CORS(app)

def resize_height(img, h=600):
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*ratio), h))

@app.route('/match', methods=['POST'])
def match():
    file = request.files['image']
    temp = os.path.join(COMPOSITE_DIR, 'selfie_tmp.jpg')
    file.save(temp)

    selfie = cv2.imread(temp)
    if selfie is None:
        return jsonify({'error': 'Invalid image'}), 400

    # --- FEATURE VECTOR (color only) ---
    q_vec = grid_color_histogram(selfie)

    # --- SEARCH ---
    D, I = index.search(q_vec.reshape(1,-1), 1)
    match_name = poster_files[I[0][0]]
    poster_path = os.path.join(POSTER_DIR, match_name)
    poster_img  = cv2.imread(poster_path)
    if poster_img is None:
        return jsonify({'error': 'Poster not found'}), 500

    # --- COMPOSITE ---
    selfie_r = resize_height(selfie)
    poster_r = resize_height(poster_img, selfie_r.shape[0])
    composite = np.hstack((selfie_r, poster_r))

    fname = f'comp_{uuid.uuid4().hex}.jpg'
    comp_path = os.path.join(COMPOSITE_DIR, fname)
    cv2.imwrite(comp_path, composite)

    return jsonify({'match_composite_url': f'{PUBLIC_HOST}/composites/{fname}'})

@app.route('/composites/<path:fn>')
def serve_comp(fn):
    return send_from_directory(COMPOSITE_DIR, fn)

@app.route('/posters/<path:fn>')  # optional direct poster serving
def serve_poster(fn):
    return send_from_directory(POSTER_DIR, fn)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

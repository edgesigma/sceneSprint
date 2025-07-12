#!/usr/bin/env python3
# context_aware_app.py – Context-Aware Movie Poster Matching Server

from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import faiss, cv2, json, uuid, os, numpy as np
import sys
from pathlib import Path

# Get the project root directory (parent of the 'server' directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Add process_step_1 to path to import feature extraction logic
sys.path.append(str(PROJECT_ROOT / 'process_step_1'))
from context_aware_feature_extraction import extract_features, load_weights

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH     = PROJECT_ROOT / 'process_step_2' / 'context_aware_poster_index.faiss'
POSTER_METADATA_PATH = PROJECT_ROOT / 'process_step_2' / 'context_aware_poster_metadata.json'
POSTER_DIR           = PROJECT_ROOT / 'covers'
COMPOSITE_DIR        = Path(__file__).parent / 'composites'
CONFIG_FILE          = PROJECT_ROOT / 'config.json'

# ─── INIT ────────────────────────────────────────────────────────────────────
os.makedirs(COMPOSITE_DIR, exist_ok=True)

# Load FAISS index and metadata
print("🚀 Loading FAISS index and metadata...")
faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(POSTER_METADATA_PATH) as f:
    metadata = json.load(f)

poster_files = metadata['filenames']
print(f"✅ Loaded index with {faiss_index.ntotal:,} vectors ({faiss_index.d} dimensions)")

# Load feature weights
weights = load_weights(str(CONFIG_FILE))
print(f"⚖️ Loaded feature weights: {weights}")


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(PROJECT_ROOT, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from parent directory"""
    return send_from_directory(PROJECT_ROOT, filename)

@app.route('/covers/<filename>')
def serve_cover(filename):
    """Serve poster images"""
    return send_from_directory(POSTER_DIR, filename)

@app.route('/composites/<filename>')
def serve_composite(filename):
    """Serve generated composite images"""
    return send_from_directory(COMPOSITE_DIR, filename)

@app.route('/api/search', methods=['POST'])
@app.route('/match', methods=['POST'])
def search_similar():
    """Movie poster matching using weighted features"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    temp_img_path = None
    try:
        # Read and process uploaded image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Save temp file to pass to extract_features
        temp_img_path = f"temp_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_img_path, img)

        # Extract weighted features for the query image
        query_vector = extract_features(temp_img_path, weights).reshape(1, -1)
        
        os.remove(temp_img_path) # Clean up temp file

        # Search FAISS index
        k_results = min(int(request.form.get('num_results', 10)), 50)
        distances, indices = faiss_index.search(query_vector, k_results)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(poster_files):
                results.append({
                    'rank': i + 1,
                    'filename': poster_files[idx],
                    'distance': float(dist),
                    'similarity_score': max(0, 100 - dist * 10),
                    'url': url_for('serve_cover', filename=poster_files[idx]),
                })
        
        # Generate composite image
        composite_filename = f"composite_{uuid.uuid4().hex[:8]}.jpg"
        composite_path = os.path.join(COMPOSITE_DIR, composite_filename)
        
        composite_results = results[:12]
        if len(composite_results) > 0:
            create_composite_image(img, composite_results, composite_path)
        
        return jsonify({
            'query_analysis': {
                'applied_weights': weights,
                'feature_dimensions': len(query_vector[0])
            },
            'results': results,
            'composite_url': url_for('serve_composite', filename=composite_filename) if composite_results else None,
            'match_composite_url': url_for('serve_composite', filename=composite_filename) if composite_results else None,
            'total_database_size': faiss_index.ntotal,
            'context_ranking_applied': False # Context ranking removed
        })
        
    except Exception as e:
        # Clean up temp file in case of error
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def create_composite_image(query_img, results, output_path):
    """Create a composite image showing query + top matches"""
    
    try:
        target_size = (200, 300)
        query_resized = cv2.resize(query_img, target_size)
        
        result_images = [query_resized]
        
        for result in results[:11]:
            poster_path = os.path.join(POSTER_DIR, result['filename'])
            if os.path.exists(poster_path):
                poster_img = cv2.imread(poster_path)
                if poster_img is not None:
                    poster_resized = cv2.resize(poster_img, target_size)
                    result_images.append(poster_resized)
        
        # Create 3x4 grid
        rows, cols = 3, 4
        grid_width = cols * target_size[0]
        grid_height = rows * target_size[1]
        composite = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, img in enumerate(result_images[:12]):
            row = i // cols
            col = i % cols
            y_start = row * target_size[1]
            y_end = y_start + target_size[1]
            x_start = col * target_size[0]
            x_end = x_start + target_size[0]
            
            if img.shape == (target_size[1], target_size[0], 3):
                 composite[y_start:y_end, x_start:x_end] = img

        cv2.imwrite(output_path, composite)
        print(f"🖼️  Saved composite image to {output_path}")

    except Exception as e:
        print(f"❌ Error creating composite image: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

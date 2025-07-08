#!/usr/bin/env python3
# enhanced_app.py – Flask SPA + API with pose detection

from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import faiss, cv2, json, uuid, os, numpy as np
import mediapipe as mp

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH     = '../process_step_2/enhanced_poster_index.faiss'
POSTER_METADATA_PATH = '../process_step_2/enhanced_poster_files.json'
POSTER_DIR           = '../process_step_1/subset'
COMPOSITE_DIR        = 'composites'
GRID_SIZE            = (2, 2)
BINS_PER_CHANNEL     = 8
NUM_KEYPOINTS        = 33
POSE_WEIGHT          = 0.6
COLOR_WEIGHT         = 0.4

# ─── INIT ────────────────────────────────────────────────────────────────────
os.makedirs(COMPOSITE_DIR, exist_ok=True)

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(POSTER_METADATA_PATH) as f:
    poster_files = json.load(f)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def grid_color_histogram(img, grid=GRID_SIZE, bins=BINS_PER_CHANNEL):
    """Extract color histogram from grid cells"""
    h, w = img.shape[:2]; rows, cols = grid
    cell_h, cell_w = h // rows, w // cols
    feats = []
    for r in range(rows):
        for c in range(cols):
            cell = img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            hist = cv2.calcHist([cell], [0,1,2], None, [bins]*3, [0,256]*3).flatten()
            feats.append((hist / hist.sum() if hist.sum() else hist).astype('float32'))
    return np.concatenate(feats)

def extract_pose_features(image):
    """Extract pose keypoints using MediaPipe"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    
    pose_vector = []
    for i in range(NUM_KEYPOINTS):
        if results.pose_landmarks and i < len(results.pose_landmarks.landmark):
            lm = results.pose_landmarks.landmark[i]
            pose_vector.extend([lm.x, lm.y])
        else:
            pose_vector.extend([0.0, 0.0])
    
    return np.array(pose_vector, dtype='float32')

def extract_combined_features(image):
    """Extract and combine pose + color features for query"""
    # Extract pose features
    pose_features = extract_pose_features(image) * POSE_WEIGHT
    
    # Extract color features  
    color_features = grid_color_histogram(image) * COLOR_WEIGHT
    
    # Combine features
    combined_features = np.concatenate([pose_features, color_features])
    
    return combined_features

app = Flask(__name__, static_folder='..', static_url_path='')
CORS(app)

def resize_height(img, h=600):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*scale), h))

# ─── SPA ENTRY ───────────────────────────────────────────────────────────────
@app.route('/')
def spa():
    return app.send_static_file('index.html')

# ─── API ─────────────────────────────────────────────────────────────────────
@app.route('/match', methods=['POST'])
def match():
    file = request.files.get('image')
    if not file:
        return jsonify(error='No image'), 400

    tmp = os.path.join(COMPOSITE_DIR, 'selfie_tmp.jpg')
    file.save(tmp)
    selfie = cv2.imread(tmp)
    if selfie is None:
        return jsonify(error='Invalid image'), 400

    try:
        # Extract combined pose + color features
        q_vec = extract_combined_features(selfie)
        
        # Search FAISS index
        _, I = index.search(q_vec.reshape(1, -1), 1)
        poster_path = os.path.join(POSTER_DIR, poster_files[I[0][0]])
        poster = cv2.imread(poster_path)
        
        if poster is None:
            return jsonify(error='Poster not found'), 500

        # Create composite image
        comp = np.hstack((resize_height(selfie), resize_height(poster, 600)))
        fname = f'comp_{uuid.uuid4().hex}.jpg'
        cv2.imwrite(os.path.join(COMPOSITE_DIR, fname), comp)

        return jsonify(
            match_composite_url=url_for('serve_comp', fn=fname),
            matched_poster=poster_files[I[0][0]]
        )
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify(error='Processing error'), 500

@app.route('/composites/<path:fn>')
def serve_comp(fn):
    return send_from_directory(COMPOSITE_DIR, fn)

@app.route('/posters/<path:fn>')
def serve_poster(fn):
    return send_from_directory(POSTER_DIR, fn)

# ─── HEALTH CHECK ────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'features': 'pose + color',
        'index_size': index.ntotal,
        'pose_detection': True
    })

if __name__ == '__main__':
    print(f'🚀 Enhanced Movie Poster Match server starting...')
    print(f'   • Pose detection: ✅ MediaPipe')
    print(f'   • Color analysis: ✅ Grid histogram')
    print(f'   • Index size: {index.ntotal} posters')
    app.run(host='0.0.0.0', port=5000, debug=False)

#!/usr/bin/env python3
# context_aware_app.py – Context-Aware Movie Poster Matching Server

from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import faiss, cv2, json, uuid, os, numpy as np
import mediapipe as mp

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH     = '../process_step_2/context_aware_poster_index.faiss'
POSTER_METADATA_PATH = '../process_step_2/context_aware_poster_metadata.json'
POSTER_DIR           = '../covers'
COMPOSITE_DIR        = 'composites'
GRID_SIZE            = (3, 3)
BINS_PER_CHANNEL     = 8
NUM_KEYPOINTS        = 33
MIN_POSE_CONFIDENCE  = 0.1
CONTEXT_FEATURE_SIZE = 20

# Pose type encodings (must match feature extraction)
POSE_TYPES = {
    'none': 0,
    'face_only': 1, 
    'portrait': 2,
    'half_body': 3,
    'full_body': 4,
    'partial': 5
}

# ─── INIT ────────────────────────────────────────────────────────────────────
os.makedirs(COMPOSITE_DIR, exist_ok=True)

# Load FAISS index and metadata
print("🚀 Loading context-aware FAISS index and metadata...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(POSTER_METADATA_PATH) as f:
    metadata = json.load(f)

poster_files = metadata['filenames']
pose_confidences = metadata['pose_confidences']
pose_types = metadata['pose_types']
face_prominences = metadata['face_prominences']
body_coverages = metadata['body_coverages']
context_metadata = metadata['context_metadata']

print(f"✅ Loaded context-aware index with {faiss_index.ntotal:,} vectors ({faiss_index.d} dimensions)")
print(f"📊 Average pose confidence: {metadata['index_stats']['average_pose_confidence']:.3f}")
print(f"🎭 Pose type distribution: {metadata['index_stats']['pose_type_distribution']}")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, 
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

def analyze_pose_context(image):
    """Advanced pose context analysis (matching feature extraction)"""
    
    strategies = [
        ("Original", image),
        ("Resized_640", cv2.resize(image, (640, 480))),
        ("Enhanced_Contrast", cv2.convertScaleAbs(image, alpha=1.3, beta=15)),
        ("Histogram_Equalized", cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
    ]
    
    best_result = None
    best_confidence = 0
    best_strategy = "none"
    
    # Try multiple detection strategies
    for strategy_name, processed_img in strategies:
        if len(processed_img.shape) == 2:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)
            
            if avg_visibility > best_confidence:
                best_confidence = avg_visibility
                best_result = results
                best_strategy = strategy_name
    
    # Analyze pose context
    context = {
        'pose_type': 'none',
        'body_parts_visible': [],
        'face_prominence': 0.0,
        'body_coverage': 0.0,
        'pose_confidence': best_confidence,
        'detection_strategy': best_strategy,
        'keypoints_detected': 0,
        'face_angle_estimate': 0.0,
        'body_symmetry': 0.0
    }
    
    if not best_result or not best_result.pose_landmarks:
        return context
    
    landmarks = best_result.pose_landmarks.landmark
    
    # Define body part keypoint groups
    body_parts = {
        'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'shoulders': [11, 12],
        'arms': [13, 14, 15, 16],
        'torso': [11, 12, 23, 24],
        'hips': [23, 24],
        'legs': [25, 26, 27, 28, 29, 30, 31, 32]
    }
    
    # Calculate visibility for each body part
    visible_parts = {}
    total_keypoints = 0
    
    for part_name, keypoint_indices in body_parts.items():
        visible_count = 0
        for i in keypoint_indices:
            if i < len(landmarks) and landmarks[i].visibility > 0.3:
                visible_count += 1
                total_keypoints += 1
        
        visible_parts[part_name] = visible_count / len(keypoint_indices)
    
    context['keypoints_detected'] = total_keypoints
    
    # Determine pose type based on visible body parts
    face_vis = visible_parts['face']
    shoulder_vis = visible_parts['shoulders'] 
    torso_vis = visible_parts['torso']
    leg_vis = visible_parts['legs']
    
    if face_vis > 0.6:
        if leg_vis > 0.5:
            context['pose_type'] = 'full_body'
        elif torso_vis > 0.5:
            context['pose_type'] = 'half_body'
        elif shoulder_vis > 0.5:
            context['pose_type'] = 'portrait'
        else:
            context['pose_type'] = 'face_only'
    elif leg_vis > 0.4 or torso_vis > 0.4:
        context['pose_type'] = 'partial'
    else:
        context['pose_type'] = 'none'
    
    # Calculate face prominence
    if face_vis > 0.4:
        face_points = []
        for i in range(11):
            if i < len(landmarks) and landmarks[i].visibility > 0.3:
                face_points.append((landmarks[i].x, landmarks[i].y))
        
        if len(face_points) >= 4:
            face_coords = np.array(face_points)
            face_width = np.max(face_coords[:, 0]) - np.min(face_coords[:, 0])
            face_height = np.max(face_coords[:, 1]) - np.min(face_coords[:, 1])
            context['face_prominence'] = min(100.0, (face_width * face_height) * 150)
    
    # Additional context calculations
    if face_vis > 0.5 and len(landmarks) > 10:
        nose = landmarks[0]
        left_eye = landmarks[2] if landmarks[2].visibility > 0.3 else None
        right_eye = landmarks[5] if landmarks[5].visibility > 0.3 else None
        
        if left_eye and right_eye:
            eye_diff = abs(left_eye.x - right_eye.x)
            context['face_angle_estimate'] = min(90.0, eye_diff * 180)
    
    if shoulder_vis > 0.5 and len(landmarks) > 12:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3:
            shoulder_balance = 1.0 - abs(left_shoulder.y - right_shoulder.y)
            context['body_symmetry'] = max(0.0, min(1.0, shoulder_balance))
    
    context['body_parts_visible'] = [part for part, ratio in visible_parts.items() if ratio > 0.3]
    context['body_coverage'] = sum(visible_parts.values()) / len(visible_parts)
    
    return context

def extract_context_features(context):
    """Convert pose context to numerical feature vector (matching extraction)"""
    
    features = np.zeros(CONTEXT_FEATURE_SIZE, dtype=np.float32)
    
    # Pose type one-hot encoding (6 dimensions)
    pose_type_idx = POSE_TYPES.get(context['pose_type'], 0)
    if pose_type_idx < 6:
        features[pose_type_idx] = 1.0
    
    # Body part visibility (6 dimensions)
    body_part_names = ['face', 'shoulders', 'arms', 'torso', 'hips', 'legs']
    for i, part in enumerate(body_part_names):
        if part in [bp.split('_')[0] for bp in context['body_parts_visible']]:
            features[6 + i] = 1.0
    
    # Continuous features (8 dimensions)
    features[12] = min(1.0, context['face_prominence'] / 100.0)
    features[13] = min(1.0, context['body_coverage'])
    features[14] = min(1.0, context['pose_confidence'])
    features[15] = min(1.0, context['keypoints_detected'] / 33)
    features[16] = min(1.0, context['face_angle_estimate'] / 90)
    features[17] = min(1.0, context['body_symmetry'])
    features[18] = 1.0 if len(context['body_parts_visible']) > 3 else 0.0
    features[19] = 1.0 if context['detection_strategy'] != 'none' else 0.0
    
    return features

def extract_pose_features_robust(image, context=None):
    """Extract pose features with context awareness"""
    
    if context is None:
        context = analyze_pose_context(image)
    
    pose_vector = np.zeros(NUM_KEYPOINTS * 2, dtype=np.float32)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    
    if results and results.pose_landmarks and context['pose_confidence'] > MIN_POSE_CONFIDENCE:
        landmarks = results.pose_landmarks.landmark
        
        for i in range(min(NUM_KEYPOINTS, len(landmarks))):
            lm = landmarks[i]
            if lm.visibility > 0.3:
                pose_vector[i*2] = lm.x
                pose_vector[i*2 + 1] = lm.y
    
    return pose_vector

def grid_color_histogram_enhanced(image, grid_size=GRID_SIZE, bins_per_channel=BINS_PER_CHANNEL):
    """Enhanced color histogram with better spatial resolution"""
    h, w = image.shape[:2]
    rows, cols = grid_size
    cell_h, cell_w = h // rows, w // cols
    hist_vecs = []

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell_h, c * cell_w
            y1, x1 = min(y0 + cell_h, h), min(x0 + cell_w, w)
            cell = image[y0:y1, x0:x1]
            
            if cell.size == 0:
                hist_vecs.append(np.zeros(bins_per_channel**3, dtype=np.float32))
                continue
            
            hist = cv2.calcHist([cell], [0,1,2], None,
                              [bins_per_channel]*3,
                              [0,256,0,256,0,256]).flatten()
            
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            else:
                hist = np.ones(len(hist)) / len(hist)
            
            hist_vecs.append(hist.astype(np.float32))

    return np.concatenate(hist_vecs)

def extract_context_aware_features(image):
    """Extract complete context-aware feature set (matching extraction)"""
    
    # Step 1: Analyze pose context
    context = analyze_pose_context(image)
    
    # Step 2: Extract pose features
    pose_features = extract_pose_features_robust(image, context)
    
    # Step 3: Extract color features  
    color_features = grid_color_histogram_enhanced(image)
    
    # Step 4: Extract context features
    context_features = extract_context_features(context)
    
    # Step 5: Context-aware weighting (matching extraction logic)
    pose_type = context['pose_type']
    
    if pose_type == 'face_only':
        pose_weight = 0.2
        color_weight = 0.7
        context_weight = 0.1
    elif pose_type == 'portrait':
        pose_weight = 0.4
        color_weight = 0.5
        context_weight = 0.1
    elif pose_type in ['half_body', 'full_body']:
        pose_weight = 0.6
        color_weight = 0.3
        context_weight = 0.1
    else:  # none or partial
        pose_weight = 0.1
        color_weight = 0.8
        context_weight = 0.1
    
    # Apply weights
    weighted_pose = pose_features * pose_weight
    weighted_color = color_features * color_weight
    weighted_context = context_features * context_weight
    
    # Combine all features
    combined_features = np.concatenate([weighted_pose, weighted_color, weighted_context])
    
    return {
        'combined_features': combined_features,
        'context': context,
        'weights': {
            'pose_weight': pose_weight,
            'color_weight': color_weight,
            'context_weight': context_weight
        }
    }

def compute_context_compatibility(query_context, poster_context):
    """Compute context compatibility for smart ranking"""
    
    # Pose type compatibility matrix
    compatibility_matrix = {
        ('face_only', 'face_only'): 1.0,
        ('face_only', 'portrait'): 0.8,
        ('face_only', 'half_body'): 0.3,
        ('face_only', 'full_body'): 0.1,
        ('portrait', 'portrait'): 1.0,
        ('portrait', 'face_only'): 0.8,
        ('portrait', 'half_body'): 0.7,
        ('portrait', 'full_body'): 0.4,
        ('half_body', 'half_body'): 1.0,
        ('half_body', 'full_body'): 0.8,
        ('full_body', 'full_body'): 1.0
    }
    
    query_pose = query_context['pose_type']
    poster_pose = poster_context['pose_type']
    
    compatibility_key = (query_pose, poster_pose)
    reverse_key = (poster_pose, query_pose)
    
    base_compatibility = 0.2  # Default low compatibility
    if compatibility_key in compatibility_matrix:
        base_compatibility = compatibility_matrix[compatibility_key]
    elif reverse_key in compatibility_matrix:
        base_compatibility = compatibility_matrix[reverse_key]
    
    # Adjust based on face prominence similarity (for face-heavy queries)
    if query_pose in ['face_only', 'portrait']:
        face_diff = abs(query_context['face_prominence'] - poster_context.get('face_prominence', 0))
        face_similarity = max(0, 1.0 - face_diff / 100.0)
        base_compatibility = (base_compatibility + face_similarity) / 2
    
    return base_compatibility

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('../', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from parent directory"""
    return send_from_directory('../', filename)

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
    """Context-aware movie poster matching"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and process uploaded image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Extract context-aware features
        feature_data = extract_context_aware_features(img)
        query_vector = feature_data['combined_features'].reshape(1, -1)
        query_context = feature_data['context']
        
        # Search FAISS index (get more candidates for re-ranking)
        k_candidates = min(100, faiss_index.ntotal)
        distances, indices = faiss_index.search(query_vector, k_candidates)
        
        # Context-aware re-ranking
        ranked_results = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(poster_files):
                poster_context = context_metadata[idx]
                
                # Compute context compatibility
                compatibility = compute_context_compatibility(query_context, poster_context)
                
                # Adjust distance based on context compatibility
                adjusted_distance = dist * (2.0 - compatibility)  # Lower distance = better match
                
                ranked_results.append({
                    'rank': i + 1,
                    'filename': poster_files[idx],
                    'original_distance': float(dist),
                    'adjusted_distance': float(adjusted_distance),
                    'context_compatibility': float(compatibility),
                    'similarity_score': max(0, 100 - adjusted_distance * 10),
                    'url': url_for('serve_cover', filename=poster_files[idx]),
                    'database_pose_type': poster_context['pose_type'],
                    'database_face_prominence': poster_context.get('face_prominence', 0),
                    'database_pose_confidence': float(pose_confidences[idx]) if idx < len(pose_confidences) else 0
                })
        
        # Sort by adjusted distance (context-aware ranking)
        ranked_results.sort(key=lambda x: x['adjusted_distance'])
        
        # Take top results
        num_results = min(int(request.form.get('num_results', 10)), 20)
        final_results = ranked_results[:num_results]
        
        # Generate composite image
        composite_filename = f"composite_{uuid.uuid4().hex[:8]}.jpg"
        composite_path = os.path.join(COMPOSITE_DIR, composite_filename)
        
        composite_results = final_results[:12]
        if len(composite_results) > 0:
            create_composite_image(img, composite_results, composite_path)
        
        return jsonify({
            'query_analysis': {
                'pose_type': query_context['pose_type'],
                'pose_confidence': float(query_context['pose_confidence']),
                'face_prominence': float(query_context['face_prominence']),
                'body_coverage': float(query_context['body_coverage']),
                'detected_keypoints': int(query_context['keypoints_detected']),
                'detection_strategy': query_context['detection_strategy'],
                'adaptive_weights': feature_data['weights'],
                'feature_dimensions': len(feature_data['combined_features'])
            },
            'results': final_results,
            'composite_url': url_for('serve_composite', filename=composite_filename) if composite_results else None,
            'match_composite_url': url_for('serve_composite', filename=composite_filename) if composite_results else None,
            'total_database_size': faiss_index.ntotal,
            'context_ranking_applied': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def create_composite_image(query_img, results, output_path):
    """Create a composite image showing query + top matches with context info"""
    
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
            composite[y_start:y_end, x_start:x_end] = img
        
        # Add labels with context information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Label query image
        cv2.putText(composite, "QUERY", (10, 25), font, font_scale, color, thickness)
        
        # Label results with similarity and context
        for i, result in enumerate(results[:11]):
            if i + 1 < len(result_images):
                row = (i + 1) // cols
                col = (i + 1) % cols
                x_start = col * target_size[0]
                y_start = row * target_size[1]
                
                # Similarity score
                score_text = f"{result['similarity_score']:.0f}%"
                cv2.putText(composite, score_text, (x_start + 5, y_start + 25), 
                           font, font_scale, color, thickness)
                
                # Context compatibility
                compat_text = f"C:{result['context_compatibility']:.2f}"
                cv2.putText(composite, compat_text, (x_start + 5, y_start + 45), 
                           font, 0.4, (0, 255, 255), 1)
        
        cv2.imwrite(output_path, composite)
        
    except Exception as e:
        print(f"Error creating composite: {e}")

@app.route('/api/stats')
def get_stats():
    """Get database statistics"""
    return jsonify({
        'total_posters': faiss_index.ntotal,
        'feature_dimensions': faiss_index.d,
        'average_pose_confidence': metadata['index_stats']['average_pose_confidence'],
        'pose_type_distribution': metadata['index_stats']['pose_type_distribution'],
        'average_face_prominence': metadata['index_stats']['average_face_prominence'],
        'server_type': 'context_aware_intelligent_matching'
    })

if __name__ == '__main__':
    print(f"🎬 Context-Aware Movie Poster Matching Server")
    print(f"=" * 60)
    print(f"📊 Database: {faiss_index.ntotal:,} posters")
    print(f"🧠 Features: Context-aware pose + color + metadata")
    print(f"🎯 Matching: Intelligent pose-type compatibility")
    print(f"🌐 Server: http://localhost:5000")
    print(f"=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

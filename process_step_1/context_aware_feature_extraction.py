#!/usr/bin/env python3
"""
Context-Aware Feature Extraction for Movie Poster Matching
==========================================================

This script extracts enhanced features that consider pose context:
- Analyzes pose type (face_only, portrait, half_body, full_body)
- Extracts context-specific features
- Applies adaptive weighting based on content
- Stores rich metadata for smart matching

Features extracted:
- Pose features (66D) - MediaPipe landmarks
- Enhanced color features (4608D) - 3x3 grid histograms  
- Context features (20D) - pose type, body parts, face prominence
- Metadata: pose_type, compatibility_scores, detected_keypoints, etc.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# ─── CONFIG ──────────────────────────────────────────────────────────────────
COVERS_DIR = '../covers'
OUTPUT_FILE = 'context_aware_poster_features.jsonl'
GRID_SIZE = (3, 3)
BINS_PER_CHANNEL = 8
NUM_KEYPOINTS = 33
MIN_POSE_CONFIDENCE = 0.1

# Context feature dimensions
CONTEXT_FEATURE_SIZE = 20

# Pose type encodings
POSE_TYPES = {
    'none': 0,
    'face_only': 1, 
    'portrait': 2,
    'half_body': 3,
    'full_body': 4,
    'partial': 5
}

# ─── POSE CONTEXT ANALYSIS ──────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

def analyze_pose_context(image):
    """Advanced pose context analysis for movie posters"""
    
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
        'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Face landmarks
        'shoulders': [11, 12],                         # Shoulder landmarks  
        'arms': [13, 14, 15, 16],                     # Arm landmarks
        'torso': [11, 12, 23, 24],                    # Torso landmarks
        'hips': [23, 24],                             # Hip landmarks
        'legs': [25, 26, 27, 28, 29, 30, 31, 32]     # Leg landmarks
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
    
    # Calculate face prominence (percentage of image occupied by face)
    if face_vis > 0.4:
        face_points = []
        for i in range(11):  # Face keypoints
            if i < len(landmarks) and landmarks[i].visibility > 0.3:
                face_points.append((landmarks[i].x, landmarks[i].y))
        
        if len(face_points) >= 4:
            face_coords = np.array(face_points)
            face_width = np.max(face_coords[:, 0]) - np.min(face_coords[:, 0])
            face_height = np.max(face_coords[:, 1]) - np.min(face_coords[:, 1])
            context['face_prominence'] = min(100.0, (face_width * face_height) * 150)
    
    # Estimate face angle (basic frontal vs profile detection)
    if face_vis > 0.5 and len(landmarks) > 10:
        nose = landmarks[0]
        left_eye = landmarks[2] if landmarks[2].visibility > 0.3 else None
        right_eye = landmarks[5] if landmarks[5].visibility > 0.3 else None
        
        if left_eye and right_eye:
            # Calculate face angle based on eye positions
            eye_diff = abs(left_eye.x - right_eye.x)
            context['face_angle_estimate'] = min(90.0, eye_diff * 180)  # Rough estimate
    
    # Calculate body symmetry
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
    """Convert pose context to numerical feature vector"""
    
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
    features[12] = min(1.0, context['face_prominence'] / 100.0)  # Normalized face prominence
    features[13] = min(1.0, context['body_coverage'])           # Body coverage ratio
    features[14] = min(1.0, context['pose_confidence'])         # Pose detection confidence
    features[15] = min(1.0, context['keypoints_detected'] / 33) # Keypoint ratio
    features[16] = min(1.0, context['face_angle_estimate'] / 90) # Face angle
    features[17] = min(1.0, context['body_symmetry'])           # Body symmetry
    features[18] = 1.0 if len(context['body_parts_visible']) > 3 else 0.0  # Multi-part visibility
    features[19] = 1.0 if context['detection_strategy'] != 'none' else 0.0  # Successful detection
    
    return features

def extract_pose_features_robust(image, context=None):
    """Extract pose features with context awareness"""
    
    if context is None:
        context = analyze_pose_context(image)
    
    pose_vector = np.zeros(NUM_KEYPOINTS * 2, dtype=np.float32)
    
    # Re-run pose detection for feature extraction
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
    """Extract complete context-aware feature set"""
    
    # Step 1: Analyze pose context
    context = analyze_pose_context(image)
    
    # Step 2: Extract pose features
    pose_features = extract_pose_features_robust(image, context)
    
    # Step 3: Extract color features  
    color_features = grid_color_histogram_enhanced(image)
    
    # Step 4: Extract context features
    context_features = extract_context_features(context)
    
    # Step 5: Context-aware weighting
    pose_type = context['pose_type']
    
    # Adaptive weights based on pose type
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
        'pose_features': pose_features,
        'color_features': color_features,
        'context_features': context_features,
        'weights': {
            'pose_weight': pose_weight,
            'color_weight': color_weight,
            'context_weight': context_weight
        },
        'context': context,
        'feature_dimensions': {
            'pose': len(pose_features),
            'color': len(color_features), 
            'context': len(context_features),
            'total': len(combined_features)
        }
    }

def process_images():
    """Process all poster images and extract context-aware features"""
    
    covers_path = Path(COVERS_DIR)
    if not covers_path.exists():
        print(f"❌ Covers directory not found: {COVERS_DIR}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in covers_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"🎬 Found {len(image_files)} poster images")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"🧠 Features: Context-aware pose + enhanced color + metadata")
    print("=" * 60)
    
    # Check for existing progress
    processed_files = set()
    if os.path.exists(OUTPUT_FILE):
        print("📂 Found existing output file, checking progress...")
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                processed_files.add(data['filename'])
        print(f"✅ {len(processed_files)} files already processed")
    
    # Process images with progress bar
    stats = {
        'total_processed': len(processed_files),
        'pose_types': {pt: 0 for pt in POSE_TYPES.keys()},
        'average_confidence': 0.0,
        'processing_errors': 0
    }
    
    with open(OUTPUT_FILE, 'a') as f:
        for img_file in tqdm(image_files, desc="Extracting features"):
            filename = img_file.name
            
            # Skip if already processed
            if filename in processed_files:
                continue
            
            try:
                # Load and process image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"⚠️  Could not load: {filename}")
                    stats['processing_errors'] += 1
                    continue
                
                # Extract context-aware features
                feature_data = extract_context_aware_features(img)
                
                # Update statistics
                stats['total_processed'] += 1
                pose_type = feature_data['context']['pose_type']
                stats['pose_types'][pose_type] += 1
                stats['average_confidence'] += feature_data['context']['pose_confidence']
                
                # Prepare output record
                record = {
                    'filename': filename,
                    'features': feature_data['combined_features'].tolist(),
                    'pose_confidence': float(feature_data['context']['pose_confidence']),
                    'pose_type': pose_type,
                    'detected_keypoints': int(feature_data['context']['keypoints_detected']),
                    'face_prominence': float(feature_data['context']['face_prominence']),
                    'body_coverage': float(feature_data['context']['body_coverage']),
                    'detection_strategy': feature_data['context']['detection_strategy'],
                    'body_parts_visible': feature_data['context']['body_parts_visible'],
                    'adaptive_weights': feature_data['weights'],
                    'feature_dimensions': feature_data['feature_dimensions'],
                    'context_metadata': feature_data['context']
                }
                
                # Write to file
                f.write(json.dumps(record) + '\n')
                f.flush()
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                stats['processing_errors'] += 1
                continue
    
    # Final statistics
    if stats['total_processed'] > 0:
        stats['average_confidence'] /= stats['total_processed']
    
    print("\n" + "="*60)
    print("🎯 EXTRACTION COMPLETE")
    print("="*60)
    print(f"📊 Total images processed: {stats['total_processed']:,}")
    print(f"📈 Average pose confidence: {stats['average_confidence']:.3f}")
    print(f"❌ Processing errors: {stats['processing_errors']}")
    print("\n📋 Pose Type Distribution:")
    for pose_type, count in stats['pose_types'].items():
        if count > 0:
            percentage = (count / stats['total_processed']) * 100
            print(f"   {pose_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n💾 Features saved to: {OUTPUT_FILE}")
    file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"📦 File size: {file_size:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract context-aware features from movie posters")
    parser.add_argument('--covers-dir', default=COVERS_DIR, help='Directory containing poster images')
    parser.add_argument('--output', default=OUTPUT_FILE, help='Output JSONL file')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    
    args = parser.parse_args()
    
    COVERS_DIR = args.covers_dir
    OUTPUT_FILE = args.output
    
    process_images()

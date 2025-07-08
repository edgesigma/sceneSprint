#!/usr/bin/env python3
# enhanced_feature_extraction.py
# ---------------------------------------------
# Generates poster_features.jsonl with both:
# - 2×2-grid color histograms (8 bins/channel)
# - MediaPipe pose features (33 keypoints)
# ---------------------------------------------

import cv2
import os
import json
import numpy as np
import mediapipe as mp
from glob import glob

# === CONFIG ===
POSTER_DIR      = 'subset/'
OUTPUT_JSONL    = 'enhanced_poster_features.jsonl'
GRID_SIZE       = (2, 2)      # rows, cols
BINS_PER_CH     = 8           # histogram bins per channel
NUM_KEYPOINTS   = 33          # MediaPipe pose keypoints
POSE_WEIGHT     = 0.6         # Weight for pose features
COLOR_WEIGHT    = 0.4         # Weight for color features

# === INITIALIZE MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# === FUNCTIONS ===
def grid_color_histogram(image,
                         grid_size=GRID_SIZE,
                         bins_per_channel=BINS_PER_CH):
    """Extract color histogram from 2x2 grid cells"""
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
            # Use zeros for missing keypoints
            pose_vector.extend([0.0, 0.0])
    
    return np.array(pose_vector, dtype='float32')

def extract_combined_features(image):
    """Extract and combine both pose and color features"""
    # Extract pose features
    pose_features = extract_pose_features(image) * POSE_WEIGHT
    
    # Extract color features
    color_features = grid_color_histogram(image) * COLOR_WEIGHT
    
    # Combine features
    combined_features = np.concatenate([pose_features, color_features])
    
    return {
        'pose_features': pose_features.tolist(),
        'color_features': color_features.tolist(),
        'combined_features': combined_features.tolist()
    }

# === EXECUTION ===
poster_files = glob(os.path.join(POSTER_DIR, '*.jpg'))
if not poster_files:
    print(f'❌ No JPEGs found in {POSTER_DIR}')
    exit(1)

print(f'▶ Extracting enhanced features from {len(poster_files)} images…')
print(f'  - Pose features: {NUM_KEYPOINTS * 2} dimensions (weight: {POSE_WEIGHT})')
print(f'  - Color features: {BINS_PER_CH**3 * GRID_SIZE[0] * GRID_SIZE[1]} dimensions (weight: {COLOR_WEIGHT})')

written, skipped = 0, 0
with open(OUTPUT_JSONL, 'w') as out_f:
    for fp in poster_files:
        img = cv2.imread(fp)
        if img is None:
            skipped += 1
            continue

        try:
            features = extract_combined_features(img)
            record = {
                'filename': os.path.basename(fp),
                'pose_features': features['pose_features'],
                'color_features': features['color_features'],
                'combined_features': features['combined_features']
            }
            out_f.write(json.dumps(record) + '\n')
            written += 1
            
            if written % 100 == 0:
                print(f'  …{written} done')
                
        except Exception as e:
            print(f'  ⚠ Error processing {fp}: {e}')
            skipped += 1

print(f'✅ Finished: {written} written, {skipped} skipped.')
print(f'   Output → {os.path.abspath(OUTPUT_JSONL)}')

# Clean up MediaPipe
pose.close()

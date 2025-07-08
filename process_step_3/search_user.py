import cv2
import mediapipe as mp
import numpy as np
import faiss
import json
import sys

# ================================
# CONFIGURATION
# ================================

FAISS_INDEX_PATH = 'poster_index.faiss'
POSTER_FILES_PATH = 'poster_files.json'
NUM_RESULTS = 5

POSE_WEIGHT = 0.6
COLOR_WEIGHT = 0.4
NUM_KEYPOINTS = 33

# ================================
# LOAD INDEX AND METADATA
# ================================

index = faiss.read_index(FAISS_INDEX_PATH)

with open(POSTER_FILES_PATH, 'r') as f:
    poster_files = json.load(f)

# ================================
# INITIALIZE MEDIAPIPE POSE
# ================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# ================================
# USER INPUT: PROCESS SELFIE IMAGE
# ================================

# Pass selfie image path as first argument
if len(sys.argv) < 2:
    print("Usage: python search_user.py path_to_selfie.jpg")
    sys.exit(1)

selfie_path = sys.argv[1]
image = cv2.imread(selfie_path)

if image is None:
    print("Failed to load image:", selfie_path)
    sys.exit(1)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --------- Extract Pose Keypoints ---------
results = pose.process(rgb_image)

pose_vector = []
for i in range(NUM_KEYPOINTS):
    if results.pose_landmarks and i < len(results.pose_landmarks.landmark):
        lm = results.pose_landmarks.landmark[i]
        pose_vector.extend([lm.x, lm.y])
    else:
        pose_vector.extend([0,0])

pose_vector = np.array(pose_vector, dtype='float32') * POSE_WEIGHT

# --------- Extract Color Histogram ---------
hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()
hist_sum = hist.sum()
if hist_sum > 0:
    hist_normalized = hist / hist_sum
else:
    hist_normalized = hist

hist_vector = hist_normalized.astype('float32') * COLOR_WEIGHT

# --------- Combine Pose + Color Vectors ---------
query_vector = np.concatenate([pose_vector, hist_vector])

# ================================
# SEARCH FAISS INDEX
# ================================

D, I = index.search(np.expand_dims(query_vector, axis=0), NUM_RESULTS)

# ================================
# OUTPUT RESULTS
# ================================

print("Top matches:")
for rank, idx in enumerate(I[0]):
    filename = poster_files[idx]
    print(f"{rank+1}. {filename} (distance: {D[0][rank]:.4f})")

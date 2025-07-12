#!/usr/bin/env python3
"""
Context-Aware Feature Extraction for Movie Poster Matching
==========================================================

This script extracts features for movie posters, focusing on:
- 4x4 grid-based HSV color histograms for spatial color information.
- Binned person count (0, 1, 2, 3+) to capture audience size context.

The output is a TSV file mapping each image filename to its feature vector.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from PIL import Image

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
COVERS_DIR = SCRIPT_DIR / '../covers'
OUTPUT_FILE = SCRIPT_DIR / 'features.tsv'
GRID_SIZE = (4, 4)
# HSV bins: 8 for Hue, 4 for Saturation, 4 for Value
H_BINS = 8
S_BINS = 4
V_BINS = 4
CONFIG_FILE = SCRIPT_DIR / '../config.json'

def load_weights(config_file=CONFIG_FILE):
    """Loads feature weights from the config file."""
    if not os.path.exists(config_file):
        print(f"Warning: Config file not found at {config_file}. Using default weights.")
        return {"color": 1.0, "person": 1.0}
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config.get("feature_weights", {"color": 1.0, "person": 1.0})

# ─── POSE DETECTION FOR PERSON COUNT ─────────────────────────────────────────
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose # type: ignore
# Use a try-except block for environments where mediapipe might not be fully installed
try:
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
except Exception as e:
    print(f"Warning: Could not initialize MediaPipe Pose. Person detection will be disabled. Error: {e}")
    pose_detector = None

def detect_person_count(image):
    """
    Detects the number of people in an image.
    NOTE: MediaPipe's Pose model is optimized for a single person.
    This function provides a basic estimate and will count 1 if a pose is detected.
    """
    if pose_detector is None:
        return 0
        
    # Ensure image is in BGR format for OpenCV
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_image)
    return 1 if results.pose_landmarks else 0

# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def calculate_color_histogram(image, grid_size=GRID_SIZE):
    """
    Calculates a grid-based color histogram in the HSV color space.
    The image is divided into a grid, and a histogram is computed for each cell.
    """
    # Ensure image is in BGR format for OpenCV
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv_image.shape
    histograms = []

    cell_h, cell_w = h // grid_size[0], w // grid_size[1]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w
            
            cell = hsv_image[y1:y2, x1:x2]
            
            if cell.size == 0:
                histograms.append(np.zeros(H_BINS * S_BINS * V_BINS, dtype=np.float32))
                continue

            hist = cv2.calcHist(
                [cell], [0, 1, 2], None, 
                [H_BINS, S_BINS, V_BINS], 
                [0, 180, 0, 256, 0, 256]
            )
            
            # Normalize the histogram
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            histograms.append(hist.flatten())
            
    return np.concatenate(histograms).astype(np.float32)

def bin_person_count(count):
    """
    Bins the person count into a one-hot encoded vector of size 4.
    Categories are: 0, 1, 2, and 3+.
    """
    binned_vector = np.zeros(4, dtype=np.float32)
    if count == 0:
        binned_vector[0] = 1.0
    elif count == 1:
        binned_vector[1] = 1.0
    elif count == 2:
        binned_vector[2] = 1.0
    else:
        binned_vector[3] = 1.0
    return binned_vector

def extract_features(image_path, weights, person_count_override=None):
    """
    Extracts a combined feature vector for a single image.
    """
    image = None
    try:
        # Try with OpenCV first
        image = cv2.imread(str(image_path))
        if image is None:
            # If OpenCV fails, try with PIL as a fallback
            pil_img = Image.open(image_path).convert('RGB')
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if image is None:
            print(f"Warning: Could not read image {image_path.name}. Skipping.")
            return None
            
        # ─── PRE-PROCESSING: RESIZE SMALL IMAGES ─────────────────────────────
        # If the image is smaller than the grid, resize it to prevent zero vectors.
        min_height = GRID_SIZE[0] * 2
        min_width = GRID_SIZE[1] * 2
        h, w, _ = image.shape
        
        if h < min_height or w < min_width:
            # Enlarge small images, maintaining aspect ratio
            if h < w:
                new_h = min_height
                new_w = int(w * (new_h / h))
            else:
                new_w = min_width
                new_h = int(h * (new_w / w))
            
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    except Exception as e:
        print(f"Warning: Failed to load or resize {image_path.name}. Error: {e}. Skipping.")
        return None

    try:
        # 1. Color Histogram
        color_features = calculate_color_histogram(image)

        # 2. Person Count
        if person_count_override is not None:
            person_count = person_count_override
        else:
            person_count = detect_person_count(image)
        
        person_features = bin_person_count(person_count)

        # 3. Apply weights and combine
        weighted_color = color_features * weights.get('color', 1.0)
        weighted_person = person_features * weights.get('person', 1.0)

        return np.concatenate([weighted_color, weighted_person])
    except Exception as e:
        print(f"Warning: Failed to extract features for {image_path.name}. Error: {e}. Skipping.")
        return None

# ─── MAIN PROCESSING ─────────────────────────────────────────────────────────

def process_images_in_directory(input_dir, output_file, weights, person_count_override=None, limit=None):
    """
    Processes all images in a directory, extracts features, and saves to a TSV file.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory not found at {input_dir}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if limit:
        image_files = image_files[:limit]
        print(f"Processing a limited set of {len(image_files)} images.")

    print(f"Found {len(image_files)} images to process in '{input_dir}'.")
    print(f"Output will be saved to '{output_file}'.")

    processed_files = set()
    if os.path.exists(output_file):
        print("Output file exists. Resuming...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_files.add(line.split('\t')[0])
                except IndexError:
                    continue # Skip malformed lines
        print(f"Skipping {len(processed_files)} already processed files.")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for image_file in tqdm(image_files, desc="Extracting Features"):
            if image_file.name in processed_files:
                continue

            features = extract_features(image_file, weights, person_count_override=person_count_override)
            if features is not None:
                feature_str = ",".join([f"{x:.8f}" for x in features])
                f_out.write(f"{image_file.name}\t{feature_str}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features from movie posters.")
    parser.add_argument('--input_dir', type=str, default=COVERS_DIR, help="Directory of poster images.")
    parser.add_argument('--output_file', type=str, default=OUTPUT_FILE, help="Path to the output TSV file.")
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help="Path to the config file for feature weights.")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of images to process for testing.")
    args = parser.parse_args()

    weights = load_weights(args.config)
    print(f"Using feature weights: {weights}")

    process_images_in_directory(args.input_dir, args.output_file, weights, limit=args.limit)

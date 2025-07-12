#!/usr/bin/env python3
"""
Analyze and Visualize Feature Vectors
======================================

This script reads the feature vectors from a TSV file,
applies dimensionality reduction (t-SNE), and visualizes the results
to help understand the feature space.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
FEATURES_FILE = SCRIPT_DIR / 'features.tsv'
OUTPUT_PLOT_FILE = SCRIPT_DIR / 'feature_visualization.png'

def load_features(file_path):
    """Loads features from the TSV file."""
    if not file_path.exists():
        print(f"Error: Features file not found at {file_path}")
        return None, None
    
    df = pd.read_csv(file_path, sep='\t', header=None, names=['filename', 'features'])
    
    # Convert feature strings to numpy arrays
    features = df['features'].apply(lambda x: np.fromstring(x, sep=','))
    
    # Stack features into a 2D numpy array
    feature_matrix = np.vstack(features.values)
    
    return df['filename'], feature_matrix

def get_person_count_from_feature(feature_vector):
    """
    Extracts the binned person count from the end of a feature vector.
    Assumes the last 4 elements are the one-hot encoded person count.
    """
    person_vector = feature_vector[-4:]
    # Find the index of the '1' in the one-hot vector.
    count = np.argmax(person_vector)
    # Map index to a descriptive label.
    if count == 0:
        return "0 People"
    if count == 1:
        return "1 Person"
    if count == 2:
        return "2 People"
    return "3+ People"

def visualize_features(filenames, feature_matrix, output_file):
    """
    Applies t-SNE and visualizes the feature vectors.
    """
    print("Applying t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix) - 1))
    reduced_features = tsne.fit_transform(feature_matrix)
    
    # --- New: Save coordinates and filenames for analysis ---
    print("Saving t-SNE coordinates to process_step_1/tsne_coordinates.csv...")
    analysis_df = pd.DataFrame({
        'filename': filenames,
        'x': reduced_features[:, 0],
        'y': reduced_features[:, 1]
    })
    analysis_df.to_csv('process_step_1/tsne_coordinates.csv', index=False)
    print("Coordinates saved.")
    # --- End New ---

    # --- New: Color by Person Count ---
    print("Coloring visualization by person count...")
    person_counts = np.apply_along_axis(get_person_count_from_feature, 1, feature_matrix)
    
    # Define a specific order for the legend to ensure consistency
    hue_order = ["0 People", "1 Person", "2 People", "3+ People"]
    # Filter for categories that are actually present in the data
    present_categories = [cat for cat in hue_order if cat in np.unique(person_counts)]

    plt.figure(figsize=(16, 12))
    ax = sns.scatterplot(
        x=reduced_features[:, 0], 
        y=reduced_features[:, 1], 
        hue=person_counts,
        hue_order=present_categories, # Use the filtered, ordered list
        palette="bright", # Use a high-contrast palette
        alpha=0.8,
    )
    ax.legend(title="Pose: Person Count") # More descriptive legend title
    plt.title('t-SNE Visualization by Person Count')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    
    # --- New: Handle incrementing filenames ---
    output_path = Path(output_file)
    base_name = output_path.stem
    suffix = output_path.suffix
    
    # Find the next available filename
    counter = 1
    new_base_name = f"{base_name}_{counter}"
    new_output_file = output_path.with_stem(new_base_name)
    while new_output_file.exists():
        counter += 1
        new_base_name = f"{base_name}_{counter}"
        new_output_file = output_path.with_stem(new_base_name)
    # --- End New ---

    person_count_plot_file = str(new_output_file).replace(suffix, f'_by_person_count{suffix}')
    print(f"Saving person count visualization to {person_count_plot_file}...")
    plt.savefig(person_count_plot_file)
    plt.close()
    print("Person count visualization saved.")
    # --- End New ---
    
    plt.figure(figsize=(12, 12))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], alpha=0.7)
    plt.title('t-SNE Visualization of Movie Poster Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    
    print(f"Saving visualization to {new_output_file}...")
    plt.savefig(new_output_file)
    plt.close()
    print("Visualization saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and visualize movie poster features.")
    parser.add_argument('--features_file', type=str, default=FEATURES_FILE, help="Path to the features TSV file.")
    parser.add_argument('--output_plot', type=str, default=OUTPUT_PLOT_FILE, help="Path to save the output plot.")
    args = parser.parse_args()

    filenames, feature_matrix = load_features(Path(args.features_file))
    
    if feature_matrix is not None:
        visualize_features(filenames, feature_matrix, args.output_plot)

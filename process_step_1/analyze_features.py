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

def visualize_features(filenames, feature_matrix, output_file):
    """
    Applies t-SNE and visualizes the feature vectors.
    """
    print("Applying t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix) - 1))
    reduced_features = tsne.fit_transform(feature_matrix)
    
    plt.figure(figsize=(12, 12))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], alpha=0.7)
    plt.title('t-SNE Visualization of Movie Poster Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    
    print(f"Saving visualization to {output_file}...")
    plt.savefig(output_file)
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

#!/usr/bin/env python3
"""
Context-Aware FAISS Index Builder
=================================

Builds a FAISS index from context-aware features with rich metadata
for intelligent movie poster matching.
"""

import faiss
import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FEATURES_FILE = '../process_step_1/context_aware_poster_features.jsonl'
INDEX_OUTPUT = 'context_aware_poster_index.faiss'
METADATA_OUTPUT = 'context_aware_poster_metadata.json'
CHECKPOINT_FILE = 'index_build_checkpoint.json'

def check_existing_progress():
    """Check if there's an existing checkpoint to resume from"""
    
    if os.path.exists(CHECKPOINT_FILE):
        print("📂 Found existing checkpoint file...")
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        
        # Check if the checkpoint is valid and complete
        if (os.path.exists(INDEX_OUTPUT) and 
            os.path.exists(METADATA_OUTPUT) and
            checkpoint.get('status') == 'complete'):
            print("✅ Index already built and complete!")
            return 'complete', checkpoint
        elif checkpoint.get('status') == 'partial':
            print("🔄 Partial build found, ready to resume...")
            return 'partial', checkpoint
    
    return 'new', None

def save_checkpoint(status, data=None):
    """Save checkpoint information"""
    
    checkpoint = {
        'status': status,
        'timestamp': str(Path(CHECKPOINT_FILE).stat().st_mtime if os.path.exists(CHECKPOINT_FILE) else 0),
        'data': data or {}
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_features_and_metadata():
    """Load features and extract comprehensive metadata"""
    
    if not os.path.exists(FEATURES_FILE):
        print(f"❌ Features file not found: {FEATURES_FILE}")
        print("   Run context_aware_feature_extraction.py first!")
        return None, None
    
    print(f"📂 Loading features from: {FEATURES_FILE}")
    
    # Count total lines for progress bar
    total_lines = 0
    with open(FEATURES_FILE, 'r') as f:
        for _ in f:
            total_lines += 1
    
    print(f"📊 Processing {total_lines:,} feature records...")
    
    features_list = []
    filenames = []
    pose_confidences = []
    pose_types = []
    face_prominences = []
    body_coverages = []
    detected_keypoints = []
    detection_strategies = []
    body_parts_visible = []
    adaptive_weights = []
    context_metadata = []
    
    line_count = 0
    error_count = 0
    
    with open(FEATURES_FILE, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Loading features", unit="records"):
            line_count += 1
            try:
                data = json.loads(line.strip())
                
                features_list.append(data['features'])
                filenames.append(data['filename'])
                pose_confidences.append(data['pose_confidence'])
                pose_types.append(data['pose_type'])
                face_prominences.append(data['face_prominence'])
                body_coverages.append(data['body_coverage'])
                detected_keypoints.append(data['detected_keypoints'])
                detection_strategies.append(data['detection_strategy'])
                body_parts_visible.append(data['body_parts_visible'])
                adaptive_weights.append(data['adaptive_weights'])
                context_metadata.append(data['context_metadata'])
                
            except (json.JSONDecodeError, KeyError) as e:
                error_count += 1
                if error_count <= 10:  # Only show first 10 errors
                    print(f"⚠️  Error parsing line {line_count}: {e}")
                elif error_count == 11:
                    print(f"⚠️  ... (suppressing further parsing errors)")
                continue
    
    if error_count > 0:
        print(f"⚠️  Total parsing errors: {error_count:,}")
    
    if not features_list:
        print("❌ No valid features found!")
        return None, None
    
    # Convert to numpy array
    print("🔢 Converting features to numpy array...")
    features_matrix = np.array(features_list, dtype='float32')
    
    print(f"✅ Loaded {len(features_list):,} feature vectors")
    print(f"📐 Feature dimensions: {features_matrix.shape[1]}")
    
    # Save checkpoint for features loaded
    save_checkpoint('features_loaded', {
        'feature_count': len(features_list),
        'feature_dimensions': features_matrix.shape[1]
    })
    
    # Compile metadata
    metadata = {
        'filenames': filenames,
        'pose_confidences': pose_confidences,
        'pose_types': pose_types,
        'face_prominences': face_prominences,
        'body_coverages': body_coverages,
        'detected_keypoints': detected_keypoints,
        'detection_strategies': detection_strategies,
        'body_parts_visible': body_parts_visible,
        'adaptive_weights': adaptive_weights,
        'context_metadata': context_metadata,
        'index_stats': {
            'total_posters': len(features_list),
            'feature_dimensions': features_matrix.shape[1],
            'average_pose_confidence': float(np.mean(pose_confidences)),
            'average_face_prominence': float(np.mean(face_prominences)),
            'average_body_coverage': float(np.mean(body_coverages)),
            'pose_type_distribution': {},
            'detection_strategy_distribution': {}
        }
    }
    
    # Calculate pose type distribution
    from collections import Counter
    pose_counter = Counter(pose_types)
    strategy_counter = Counter(detection_strategies)
    
    total_posters = len(pose_types)
    for pose_type, count in pose_counter.items():
        metadata['index_stats']['pose_type_distribution'][pose_type] = {
            'count': count,
            'percentage': (count / total_posters) * 100
        }
    
    for strategy, count in strategy_counter.items():
        metadata['index_stats']['detection_strategy_distribution'][strategy] = {
            'count': count,
            'percentage': (count / total_posters) * 100
        }
    
    return features_matrix, metadata

def build_faiss_index(features_matrix, resume_checkpoint=None):
    """Build optimized FAISS index for context-aware features"""
    
    n_vectors, n_dimensions = features_matrix.shape
    
    print(f"🏗️  Building FAISS index...")
    print(f"   Vectors: {n_vectors:,}")
    print(f"   Dimensions: {n_dimensions}")
    
    # Check if we can resume from checkpoint
    if resume_checkpoint and os.path.exists(INDEX_OUTPUT):
        print("🔄 Resuming from existing index...")
        try:
            index = faiss.read_index(INDEX_OUTPUT)
            if index.ntotal == n_vectors:
                print("✅ Index already complete!")
                return index
            else:
                print("⚠️  Incomplete index found, rebuilding...")
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            print("🔨 Building new index...")
    
    # Create FAISS index
    if n_vectors > 10000:
        # Use IVF index for large datasets
        n_clusters = min(int(np.sqrt(n_vectors)), 4096)
        quantizer = faiss.IndexFlatL2(n_dimensions)
        index = faiss.IndexIVFFlat(quantizer, n_dimensions, n_clusters)
        
        print(f"   Index type: IVF with {n_clusters} clusters")
        
        # Training phase with progress
        print("🧠 Training index...")
        with tqdm(total=1, desc="Training", unit="phase") as pbar:
            index.train(features_matrix)
            pbar.update(1)
        
        # Adding vectors with progress
        print("📥 Adding vectors to index...")
        batch_size = min(10000, n_vectors // 10)  # Dynamic batch size
        
        for i in tqdm(range(0, n_vectors, batch_size), desc="Adding vectors", unit="batch"):
            end_idx = min(i + batch_size, n_vectors)
            batch = features_matrix[i:end_idx]
            index.add(batch)
            
            # Save progress checkpoint
            if i % (batch_size * 5) == 0:  # Every 5 batches
                temp_index_path = f"{INDEX_OUTPUT}.tmp"
                faiss.write_index(index, temp_index_path)
                save_checkpoint('building', {
                    'vectors_added': index.ntotal,
                    'total_vectors': n_vectors,
                    'progress': (index.ntotal / n_vectors) * 100
                })
        
        # Set search parameters
        index.nprobe = min(32, max(1, n_clusters // 32))
        print(f"   Search parameter nprobe: {index.nprobe}")
        
    else:
        # Use flat index for smaller datasets
        index = faiss.IndexFlatL2(n_dimensions)
        print("   Index type: Flat L2")
        
        print("📥 Adding vectors to index...")
        # Add with progress bar even for small datasets
        batch_size = min(1000, n_vectors // 5)
        
        for i in tqdm(range(0, n_vectors, batch_size), desc="Adding vectors", unit="batch"):
            end_idx = min(i + batch_size, n_vectors)
            batch = features_matrix[i:end_idx]
            index.add(batch)
    
    print(f"✅ Index built successfully!")
    print(f"   Total vectors: {index.ntotal:,}")
    print(f"   Is trained: {index.is_trained}")
    
    return index

def save_index_and_metadata(index, metadata):
    """Save FAISS index and metadata to disk"""
    
    print(f"💾 Saving index to: {INDEX_OUTPUT}")
    faiss.write_index(index, INDEX_OUTPUT)
    
    print(f"💾 Saving metadata to: {METADATA_OUTPUT}")
    with open(METADATA_OUTPUT, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # File size information
    index_size = os.path.getsize(INDEX_OUTPUT) / (1024**2)  # MB
    metadata_size = os.path.getsize(METADATA_OUTPUT) / (1024**2)  # MB
    
    print(f"📦 Index file size: {index_size:.1f} MB")
    print(f"📦 Metadata file size: {metadata_size:.1f} MB")

def print_statistics(metadata):
    """Print comprehensive index statistics"""
    
    stats = metadata['index_stats']
    
    print("\n" + "="*60)
    print("📊 CONTEXT-AWARE INDEX STATISTICS")
    print("="*60)
    
    print(f"🎬 Total posters: {stats['total_posters']:,}")
    print(f"📐 Feature dimensions: {stats['feature_dimensions']}")
    print(f"🎯 Average pose confidence: {stats['average_pose_confidence']:.3f}")
    print(f"👤 Average face prominence: {stats['average_face_prominence']:.1f}%")
    print(f"🫂 Average body coverage: {stats['average_body_coverage']:.3f}")
    
    print("\n📋 Pose Type Distribution:")
    for pose_type, data in stats['pose_type_distribution'].items():
        print(f"   {pose_type:12}: {data['count']:6,} ({data['percentage']:5.1f}%)")
    
    print("\n🔍 Detection Strategy Distribution:")
    for strategy, data in stats['detection_strategy_distribution'].items():
        print(f"   {strategy:20}: {data['count']:6,} ({data['percentage']:5.1f}%)")
    
    print("="*60)

def main():
    """Main index building process with resume capability"""
    
    parser = argparse.ArgumentParser(description="Build context-aware FAISS index with resume capability")
    parser.add_argument('--features-file', default=FEATURES_FILE, help='Input features JSONL file')
    parser.add_argument('--index-output', default=INDEX_OUTPUT, help='Output FAISS index file')
    parser.add_argument('--metadata-output', default=METADATA_OUTPUT, help='Output metadata JSON file')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild even if index exists')
    
    args = parser.parse_args()
    
    global FEATURES_FILE, INDEX_OUTPUT, METADATA_OUTPUT
    FEATURES_FILE = args.features_file
    INDEX_OUTPUT = args.index_output
    METADATA_OUTPUT = args.metadata_output
    
    print("🚀 Context-Aware FAISS Index Builder")
    print("="*60)
    
    # Check for existing progress
    checkpoint = None
    if not args.force_rebuild:
        status, checkpoint = check_existing_progress()
        
        if status == 'complete':
            print("✅ Index already built and complete!")
            print("   Use --force-rebuild to rebuild anyway.")
            return 0
        elif status == 'partial':
            print("🔄 Resuming from checkpoint...")
    
    # Step 1: Load features and metadata
    features_matrix, metadata = load_features_and_metadata()
    if features_matrix is None:
        return 1
    
    # Step 2: Build FAISS index (with resume capability)
    index = build_faiss_index(features_matrix, checkpoint)
    
    # Step 3: Save index and metadata
    save_index_and_metadata(index, metadata)
    
    # Step 4: Mark as complete
    save_checkpoint('complete', {
        'total_vectors': index.ntotal,
        'feature_dimensions': features_matrix.shape[1],
        'index_file': INDEX_OUTPUT,
        'metadata_file': METADATA_OUTPUT
    })
    
    # Step 5: Print statistics
    print_statistics(metadata)
    
    print(f"\n✅ Context-aware index building complete!")
    print(f"🎯 Ready for intelligent movie poster matching!")
    
    # Clean up temporary files
    temp_files = [f"{INDEX_OUTPUT}.tmp", f"{METADATA_OUTPUT}.tmp"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🧹 Cleaned up: {temp_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())

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
from datetime import datetime
import argparse

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FEATURES_FILE = SCRIPT_DIR / '../process_step_1/features.tsv'
INDEX_OUTPUT = SCRIPT_DIR / 'context_aware_poster_index.faiss'
METADATA_OUTPUT = SCRIPT_DIR / 'context_aware_poster_metadata.json'
CHECKPOINT_FILE = SCRIPT_DIR / 'index_build_checkpoint.json'
FEATURES_CACHE = SCRIPT_DIR / 'features_cache.npz'
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'

def check_existing_progress():
    """Check if there's an existing checkpoint to resume from"""
    
    if os.path.exists(CHECKPOINT_FILE):
        print("📂 Found existing checkpoint file...")
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        
        print(f"   Checkpoint status: {checkpoint.get('status', 'unknown')}")
        
        # Check if the checkpoint is valid and complete
        if (os.path.exists(INDEX_OUTPUT) and 
            os.path.exists(METADATA_OUTPUT) and
            checkpoint.get('status') == 'complete'):
            print("✅ Index already built and complete!")
            return 'complete', checkpoint
        elif checkpoint.get('status') in ['building', 'partial', 'features_loaded', 'training', 'starting', 'counting_complete', 'loading_features', 'converting_to_numpy']:
            print("🔄 Partial build found, ready to resume...")
            return 'partial', checkpoint
    
    # Check for temporary index file even without checkpoint
    if os.path.exists(f"{INDEX_OUTPUT}.tmp"):
        print("📂 Found temporary index file without checkpoint...")
        return 'partial', {'status': 'building', 'data': {}}
    
    return 'new', None


def save_checkpoint(status, data=None):
    """Save checkpoint information"""
    
    checkpoint = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'data': data or {}
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_features_and_metadata(use_cache=True, resume_checkpoint=None):
    """Load features and extract comprehensive metadata with caching and resume capability"""
    
    # Save initial checkpoint
    save_checkpoint('starting', {'phase': 'feature_loading'})
    
    # Check if we can resume from a previous feature loading session
    resume_lines = 0
    if resume_checkpoint and resume_checkpoint.get('status') in ['loading_features', 'counting_complete']:
        print("🔄 Attempting to resume feature loading...")
        checkpoint_data = resume_checkpoint.get('data', {})
        if 'lines_processed' in checkpoint_data:
            resume_lines = checkpoint_data['lines_processed']
            print(f"   Previous session processed {resume_lines:,} lines")
            print(f"   Resuming feature loading from line {resume_lines+1:,}")
    
    # Check if we have cached features
    if use_cache and os.path.exists(FEATURES_CACHE) and os.path.exists(f"{METADATA_OUTPUT}.cache"):
        print("🚀 Loading features from cache...")
        try:
            cache_data = np.load(FEATURES_CACHE, allow_pickle=True)
            features_matrix = cache_data['features_matrix']
            
            with open(f"{METADATA_OUTPUT}.cache", 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Loaded {features_matrix.shape[0]:,} cached feature vectors")
            print(f"📐 Feature dimensions: {features_matrix.shape[1]}")
            
            # Update checkpoint with cached data
            save_checkpoint('features_loaded', {
                'feature_count': features_matrix.shape[0],
                'feature_dimensions': features_matrix.shape[1],
                'from_cache': True
            })
            
            return features_matrix, metadata
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            print("📂 Falling back to loading from source...")
    
    if not os.path.exists(FEATURES_FILE):
        print(f"❌ Features file not found: {FEATURES_FILE}")
        print("   Run context_aware_feature_extraction.py first!")
        return None, None
    
    print(f"📂 Loading features from: {FEATURES_FILE}")
    
    # Count total lines for progress bar
    print("📊 Counting total records...")
    total_lines = 0
    try:
        with open(FEATURES_FILE, 'r') as f:
            for _ in f:
                total_lines += 1
    except Exception as e:
        print(f"❌ Error reading features file: {e}")
        return None, None
    
    print(f"📊 Processing {total_lines:,} feature records...")
    
    # Save checkpoint with total count
    save_checkpoint('counting_complete', {
        'total_lines': total_lines,
        'phase': 'loading_features'
    })
    
    features_list = []
    filenames = []
    
    error_count = 0
    
    try:
        with open(FEATURES_FILE, 'r') as f:
            # Skip lines if resuming
            for _ in range(resume_lines):
                next(f)
            for line_num, line in enumerate(f, resume_lines + 1):
                try:
                    filename, features_str = line.strip().split('\t')
                    features = [float(x) for x in features_str.split(',')]
                    features_list.append(features)
                    filenames.append(filename)

                    # Progress update and checkpoint every 1000 records
                    if line_num % 1000 == 0:
                        progress = (line_num / total_lines) * 100
                        print(f"   Loading progress: {progress:.1f}% ({line_num:,}/{total_lines:,})")
                        # Save frequent checkpoint during loading
                        save_checkpoint('loading_features', {
                            'lines_processed': line_num,
                            'total_lines': total_lines,
                            'progress': progress,
                            'features_loaded': len(features_list) + resume_lines
                        })
                except (ValueError, IndexError) as e:
                    error_count += 1
                    if error_count <= 10:  # Only show first 10 errors
                        print(f"⚠️  Error parsing line {line_num}: {e}")
                    elif error_count == 11:
                        print(f"⚠️  ... (suppressing further parsing errors)")
                    continue
    except Exception as e:
        print(f"❌ Critical error during feature loading: {e}")
        return None, None
    
    if error_count > 0:
        print(f"⚠️  Total parsing errors: {error_count:,}")
    
    if not features_list:
        print("❌ No valid features found!")
        return None, None
    
    # Convert to numpy array
    print("🔢 Converting features to numpy array...")
    save_checkpoint('converting_to_numpy', {
        'features_loaded': len(features_list),
        'phase': 'converting'
    })
    
    try:
        features_matrix = np.array(features_list, dtype='float32')
    except Exception as e:
        print(f"❌ Error converting to numpy array: {e}")
        return None, None
    
    print(f"✅ Loaded {len(features_list):,} feature vectors")
    print(f"📐 Feature dimensions: {features_matrix.shape[1]}")
    
    # Save checkpoint for features loaded
    save_checkpoint('features_loaded', {
        'feature_count': len(features_list),
        'feature_dimensions': features_matrix.shape[1],
        'from_cache': False
    })
    
    # Compile metadata
    metadata = {
        'filenames': filenames,
        'index_stats': {
            'total_posters': len(features_list),
            'feature_dimensions': features_matrix.shape[1],
        }
    }
    
    # Cache the features and metadata for faster resume
    print("💾 Caching features for faster resume...")
    try:
        np.savez_compressed(FEATURES_CACHE, features_matrix=features_matrix)
        with open(f"{METADATA_OUTPUT}.cache", 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Features cached successfully")
    except Exception as e:
        print(f"⚠️  Could not cache features: {e}")
    
    return features_matrix, metadata

def build_faiss_index(features_matrix, resume_checkpoint=None, override_batch_size=None):
    """Build optimized FAISS index for context-aware features"""
    
    n_vectors, n_dimensions = features_matrix.shape
    
    print(f"🏗️  Building FAISS index...")
    print(f"   Vectors: {n_vectors:,}")
    print(f"   Dimensions: {n_dimensions}")
    
    # Check if we can resume from checkpoint
    if resume_checkpoint and (os.path.exists(str(INDEX_OUTPUT)) or os.path.exists(f"{str(INDEX_OUTPUT)}.tmp")):
        print("🔄 Checking existing index for resume...")
        
        # Validate checkpoint and index consistency
        checkpoint_data = resume_checkpoint.get('data', {})
        expected_vectors = checkpoint_data.get('vectors_added', 0)
        expected_dimensions = checkpoint_data.get('feature_dimensions', n_dimensions)
        
        try:
            # Try loading the main index first, then temporary
            index_path = str(INDEX_OUTPUT) if os.path.exists(str(INDEX_OUTPUT)) else f"{str(INDEX_OUTPUT)}.tmp"
            index = faiss.read_index(index_path)
            
            # Comprehensive validation
            index_valid = True
            validation_errors = []
            
            # Check dimensions match
            if index.d != n_dimensions:
                validation_errors.append(f"Dimension mismatch: index={index.d}, expected={n_dimensions}")
                index_valid = False
            
            # Check vector count consistency with checkpoint
            if expected_vectors > 0 and abs(index.ntotal - expected_vectors) > 1000:  # Allow small variance
                validation_errors.append(f"Vector count mismatch: index={index.ntotal}, checkpoint={expected_vectors}")
                index_valid = False
                
            # Check if index is trained (for IVF indices)
            if hasattr(index, 'is_trained') and not index.is_trained and n_vectors > 10000:
                validation_errors.append("Index is not properly trained")
                index_valid = False
            
            # Try a simple test operation to verify index integrity
            try:
                if index.ntotal > 0:
                    # Test search with a small sample
                    test_vector = features_matrix[0:1].copy()
                    _, _ = index.search(test_vector, min(5, index.ntotal))
            except Exception as test_error:
                validation_errors.append(f"Index test search failed: {test_error}")
                index_valid = False
            
            if not index_valid:
                print("⚠️  Index validation failed:")
                for error in validation_errors:
                    print(f"   - {error}")
                print("🧹 Cleaning up corrupted files and restarting...")
                
                # Clean up corrupted files
                cleanup_files = [str(INDEX_OUTPUT), f"{str(INDEX_OUTPUT)}.tmp", str(CHECKPOINT_FILE), f"{str(METADATA_OUTPUT)}.cache"]
                for cleanup_file in cleanup_files:
                    if os.path.exists(cleanup_file):
                        try:
                            os.remove(cleanup_file)
                            print(f"   Removed: {cleanup_file}")
                        except Exception as e:
                            print(f"   Warning: Could not remove {cleanup_file}: {e}")
                
                print("🔨 Starting fresh build...")
                # Reset resume_checkpoint to None to force fresh start
                resume_checkpoint = None
                
            elif index.ntotal == n_vectors:
                print("✅ Index already complete and valid!")
                # If we loaded from temp, save as final
                if index_path.endswith('.tmp'):
                    faiss.write_index(index, str(INDEX_OUTPUT))
                    print("💾 Promoted temporary index to final")
                return index
                
            elif index.ntotal > 0:
                print(f"✅ Valid partial index found: {index.ntotal:,}/{n_vectors:,} ({(index.ntotal/n_vectors)*100:.1f}%)")
                
                # Resume from validated index
                resume_batch_size = override_batch_size or checkpoint_data.get('batch_size', min(1000, max(250, n_vectors // 100)))
                next_start = index.ntotal  # Use actual index count, not checkpoint
                
                print(f"   Continuing from vector {next_start:,}")
                print(f"   Remaining vectors: {n_vectors - next_start:,}")
                print(f"   Using batch size: {resume_batch_size:,}")
                
                try:
                    for i in range(next_start, n_vectors, resume_batch_size):
                        end_idx = min(i + resume_batch_size, n_vectors)
                        batch = features_matrix[i:end_idx]
                        index.add(x=batch)
                        
                        progress = (index.ntotal / n_vectors) * 100
                        batch_num = (i - next_start) // resume_batch_size + 1
                        remaining_batches = ((n_vectors - next_start) + resume_batch_size - 1) // resume_batch_size
                        
                        print(f"   Resume Batch {batch_num}/{remaining_batches}: {progress:.1f}% ({index.ntotal:,}/{n_vectors:,})")
                        
                        # Save progress more frequently for VS Code stability
                        temp_index_path = f"{str(INDEX_OUTPUT)}.tmp"
                        try:
                            faiss.write_index(index, temp_index_path)
                            save_checkpoint('building', {
                                'vectors_added': index.ntotal,
                                'total_vectors': n_vectors,
                                'progress': progress,
                                'last_completed_batch': batch_num,
                                'batch_size': resume_batch_size,
                                'next_start_index': index.ntotal,
                                'resumed_from': next_start
                            })
                        except Exception as save_error:
                            print(f"⚠️  Warning: Could not save checkpoint: {save_error}")
                        
                        import gc
                        gc.collect()
                
                    print("✅ Resume completed!")
                    return index
                except Exception as resume_error:
                    print(f"⚠️  Resume failed: {resume_error}")
                    print("🔨 Starting fresh build...")
            else:
                print("⚠️  Empty index found, rebuilding...")
                
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            print("🔨 Building new index...")
            
            # Clean up potentially corrupted files
            for cleanup_file in [str(INDEX_OUTPUT), f"{str(INDEX_OUTPUT)}.tmp"]:
                if os.path.exists(cleanup_file):
                    try:
                        os.remove(cleanup_file)
                        print(f"🧹 Cleaned up: {cleanup_file}")
                    except:
                        pass
    
    # Determine batch size
    if override_batch_size:
        batch_size = override_batch_size
    else:
        batch_size = min(1000, max(250, n_vectors // 100))  # Much smaller default batches
    
    print(f"   Using batch size: {batch_size:,}")
    
    # Create FAISS index
    if n_vectors > 10000:
        # Use IVF index for large datasets
        n_clusters = min(int(np.sqrt(n_vectors)), 4096)
        quantizer = faiss.IndexFlatL2(n_dimensions)
        index = faiss.IndexIVFFlat(quantizer, n_dimensions, n_clusters)
        
        print(f"   Index type: IVF with {n_clusters} clusters")
        
        # Training phase - save checkpoint before training
        print("🧠 Training index...")
        save_checkpoint('training', {
            'total_vectors': n_vectors,
            'feature_dimensions': n_dimensions,
            'batch_size': batch_size,
            'clusters': n_clusters
        })
        
        try:
            index.train(features_matrix)
            print("✅ Training completed")
        except Exception as train_error:
            print(f"❌ Training failed: {train_error}")
            return None
        
        # Adding vectors with progress
        print("📥 Adding vectors to index...")
        print(f"   Total batches: {(n_vectors + batch_size - 1) // batch_size}")
        
        # Save initial checkpoint
        save_checkpoint('building', {
            'vectors_added': 0,
            'total_vectors': n_vectors,
            'progress': 0.0,
            'batch_size': batch_size,
            'next_start_index': 0
        })
        
        try:
            for i in range(0, n_vectors, batch_size):
                end_idx = min(i + batch_size, n_vectors)
                batch = features_matrix[i:end_idx]
                
                # Add batch with error handling
                try:
                    # Corrected call to index.add, which expects only the data
                    index.add(batch)
                except Exception as add_error:
                    print(f"❌ Failed to add batch at {i}: {add_error}")
                    # Save what we have so far
                    temp_index_path = f"{INDEX_OUTPUT}.tmp"
                    try:
                        faiss.write_index(index, temp_index_path)
                    except:
                        pass
                    raise add_error
                
                # Progress update every batch for better tracking
                progress = (index.ntotal / n_vectors) * 100
                batch_num = i // batch_size + 1
                total_batches = (n_vectors + batch_size - 1) // batch_size
                
                print(f"   Batch {batch_num:,}/{total_batches:,}: {progress:.1f}% ({index.ntotal:,}/{n_vectors:,})")
                
                # Save progress checkpoint EVERY batch for reliable resume
                temp_index_path = f"{str(INDEX_OUTPUT)}.tmp"
                try:
                    faiss.write_index(index, temp_index_path)
                    save_checkpoint('building', {
                        'vectors_added': index.ntotal,
                        'total_vectors': n_vectors,
                        'progress': progress,
                        'last_completed_batch': batch_num,
                        'batch_size': batch_size,
                        'next_start_index': index.ntotal
                    })
                except Exception as save_error:
                    print(f"⚠️  Warning: Could not save checkpoint: {save_error}")
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
        
        except Exception as build_error:
            print(f"❌ Build failed: {build_error}")
            return None
        
        # Set search parameters
        index.nprobe = min(32, max(1, n_clusters // 32))
        print(f"   Search parameter nprobe: {index.nprobe}")
        
    else:
        # Use flat index for smaller datasets
        index = faiss.IndexFlatL2(n_dimensions)
        print("   Index type: Flat L2")
        
        print("📥 Adding vectors to index...")
        print(f"   Using batch size: {batch_size:,}")
        
        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch = features_matrix[i:end_idx]
            # Corrected call to index.add, which expects only the data
            index.add(batch)
            
            # Simple progress update
            progress = (index.ntotal / n_vectors) * 100
            if i % batch_size == 0:
                print(f"   Progress: {progress:.1f}% ({index.ntotal:,}/{n_vectors:,})")
    
    print(f"✅ Index built successfully!")
    print(f"   Total vectors: {index.ntotal:,}")
    print(f"   Is trained: {index.is_trained}")
    
    return index

def save_index_and_metadata(index, metadata):
    """Save FAISS index and metadata to disk"""
    
    print(f"💾 Saving index to: {INDEX_OUTPUT}")
    faiss.write_index(index, str(INDEX_OUTPUT))
    
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
    
    print("="*60)

def main():
    """Main index building process with resume capability"""
    
    global FEATURES_FILE, INDEX_OUTPUT, METADATA_OUTPUT, FEATURES_CACHE, CHECKPOINT_FILE, CHECKPOINT_DIR
    parser = argparse.ArgumentParser(description="Build context-aware FAISS index with resume capability")
    parser.add_argument('--features-file', default=str(FEATURES_FILE), help='Input features JSONL file')
    parser.add_argument('--index-output', default=str(INDEX_OUTPUT), help='Output FAISS index file')
    parser.add_argument('--metadata-output', default=str(METADATA_OUTPUT), help='Output metadata JSON file')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild even if index exists')
    parser.add_argument('--vscode-friendly', action='store_true', help='Use simpler output for VS Code terminal')
    parser.add_argument('--batch-size', type=int, help='Override default batch size')
    parser.add_argument('--clear-cache', action='store_true', help='Clear feature cache and start fresh')
    parser.add_argument('--use-cache', action='store_true', default=True, help='Use cached features if available')
    
    args = parser.parse_args()
    
    # Use command line arguments for file paths
    FEATURES_FILE = Path(args.features_file)
    INDEX_OUTPUT = Path(args.index_output)
    METADATA_OUTPUT = Path(args.metadata_output)
    
    # Re-derive script-relative paths from the potentially new INDEX_OUTPUT path
    SCRIPT_DIR = INDEX_OUTPUT.parent
    CHECKPOINT_FILE = SCRIPT_DIR / 'index_build_checkpoint.json'
    FEATURES_CACHE = SCRIPT_DIR / 'features_cache.npz'
    CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'

    print("🚀 Context-Aware FAISS Index Builder")
    print("="*60)
    
    # Clear cache if requested
    if args.clear_cache:
        cache_files = [str(FEATURES_CACHE), f"{str(METADATA_OUTPUT)}.cache", str(CHECKPOINT_FILE), f"{str(INDEX_OUTPUT)}.tmp"]
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🧹 Cleared cache: {cache_file}")
        
        # Clear checkpoint directory
        if os.path.exists(CHECKPOINT_DIR):
            import shutil
            shutil.rmtree(CHECKPOINT_DIR)
            print(f"🧹 Cleared checkpoint directory: {CHECKPOINT_DIR}")
        
        print("✅ Cache cleared, starting fresh")
    
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
    features_matrix, metadata = load_features_and_metadata(use_cache=args.use_cache, resume_checkpoint=checkpoint)
    if features_matrix is None:
        return 1
    
    # Step 2: Build FAISS index (with resume capability)
    index = build_faiss_index(features_matrix, checkpoint, override_batch_size=args.batch_size)
    
    # Step 3: Save index and metadata
    save_index_and_metadata(index, metadata)
    
    # Step 4: Mark as complete
    save_checkpoint('complete', {
        'total_vectors': index.ntotal,
        'feature_dimensions': features_matrix.shape[1],
        'index_file': str(INDEX_OUTPUT),
        'metadata_file': str(METADATA_OUTPUT)
    })
    
    # Step 5: Print statistics
    print_statistics(metadata)
    
    print(f"\n✅ Context-aware index building complete!")
    print(f"🎯 Ready for intelligent movie poster matching!")
    
    # Clean up temporary files
    temp_files = [f"{str(INDEX_OUTPUT)}.tmp", f"{str(METADATA_OUTPUT)}.tmp"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🧹 Cleaned up: {temp_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())

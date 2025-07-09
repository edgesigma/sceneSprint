#!/usr/bin/env python3
"""
Context-Aware Pipeline Builder
=============================

Automates the complete context-aware movie poster matching pipeline:
1. Feature extraction with pose context analysis
2. FAISS index building with intelligent metadata
3. Validation and testing
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
STEPS = {
    'feature_extraction': {
        'script': 'process_step_1/context_aware_feature_extraction.py',
        'output': 'process_step_1/context_aware_poster_features.jsonl',
        'description': 'Context-aware feature extraction with pose analysis'
    },
    'index_building': {
        'script': 'process_step_2/context_aware_build_faiss_index.py', 
        'output_index': 'process_step_2/context_aware_poster_index.faiss',
        'output_metadata': 'process_step_2/context_aware_poster_metadata.json',
        'description': 'Build FAISS index with context metadata'
    }
}

def check_dependencies():
    """Check if required dependencies are available"""
    
    print("🔍 Checking dependencies...")
    
    required_packages = ['cv2', 'numpy', 'mediapipe', 'faiss', 'flask', 'tqdm']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install opencv-python numpy mediapipe faiss-cpu flask tqdm")
        return False
    
    # Check covers directory
    if not os.path.exists('covers'):
        print("❌ Covers directory not found: covers")
        return False
    
    covers_count = len([f for f in Path('covers').iterdir() 
                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    print(f"   📁 Found {covers_count:,} poster images")
    
    if covers_count == 0:
        print("❌ No poster images found in covers directory")
        return False
    
    print("✅ All dependencies available")
    return True

def get_file_info(filepath):
    """Get file information (size, modification time, record count)"""
    
    if not os.path.exists(filepath):
        return None
    
    stat = os.stat(filepath)
    size_mb = stat.st_size / (1024**2)
    mtime = time.ctime(stat.st_mtime)
    
    info = {
        'size_mb': size_mb,
        'modified': mtime,
        'records': 0
    }
    
    # Count records for JSONL files
    if filepath.endswith('.jsonl'):
        try:
            with open(filepath, 'r') as f:
                info['records'] = sum(1 for _ in f)
        except:
            info['records'] = 0
    
    return info

def run_step(step_name, step_config, force=False):
    """Run a pipeline step"""
    
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_config['description']}")
    print(f"{'='*60}")
    
    script_path = step_config['script']
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Check if outputs already exist
    outputs_exist = False
    if 'output' in step_config:
        output_path = step_config['output']
        if os.path.exists(output_path):
            outputs_exist = True
            info = get_file_info(output_path)
            print(f"📁 Existing output: {output_path}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Records: {info['records']:,}")
            print(f"   Modified: {info['modified']}")
    
    if 'output_index' in step_config:
        index_path = step_config['output_index']
        metadata_path = step_config['output_metadata']
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            outputs_exist = True
            index_info = get_file_info(index_path)
            metadata_info = get_file_info(metadata_path)
            print(f"📁 Existing index: {index_path}")
            print(f"   Size: {index_info['size_mb']:.1f} MB")
            print(f"📁 Existing metadata: {metadata_path}")
            print(f"   Size: {metadata_info['size_mb']:.1f} MB")
    
    if outputs_exist and not force:
        response = input(f"\n⚠️  Output files exist. Rebuild? [y/N]: ").strip().lower()
        if response != 'y':
            print("⏭️  Skipping step")
            return True
    
    # Backup existing outputs
    if outputs_exist:
        print("🔄 Creating backups...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if 'output' in step_config:
            output_path = step_config['output']
            backup_path = f"{output_path}.backup_{timestamp}"
            os.rename(output_path, backup_path)
            print(f"   📦 {output_path} → {backup_path}")
        
        if 'output_index' in step_config:
            for output_key in ['output_index', 'output_metadata']:
                if output_key in step_config:
                    output_path = step_config[output_key]
                    backup_path = f"{output_path}.backup_{timestamp}"
                    os.rename(output_path, backup_path)
                    print(f"   📦 {output_path} → {backup_path}")
    
    # Run the script
    print(f"\n🔄 Running: python3 {script_path}")
    start_time = time.time()
    
    try:
        result = subprocess.run(['python3', script_path], 
                              capture_output=True, text=True, cwd='.')
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Step completed successfully in {duration:.1f}s")
            
            # Show output file info
            if 'output' in step_config:
                output_path = step_config['output']
                if os.path.exists(output_path):
                    info = get_file_info(output_path)
                    print(f"📊 Output: {info['records']:,} records, {info['size_mb']:.1f} MB")
            
            if 'output_index' in step_config:
                index_path = step_config['output_index']
                if os.path.exists(index_path):
                    info = get_file_info(index_path)
                    print(f"📊 Index: {info['size_mb']:.1f} MB")
            
            return True
        else:
            print(f"❌ Step failed (exit code {result.returncode})")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running step: {e}")
        return False

def validate_outputs():
    """Validate that all pipeline outputs are correct"""
    
    print(f"\n{'='*60}")
    print("🔍 VALIDATING PIPELINE OUTPUTS")
    print(f"{'='*60}")
    
    # Check feature extraction output
    features_file = STEPS['feature_extraction']['output']
    if not os.path.exists(features_file):
        print(f"❌ Features file missing: {features_file}")
        return False
    
    print(f"📊 Validating features file...")
    feature_info = get_file_info(features_file)
    print(f"   Records: {feature_info['records']:,}")
    print(f"   Size: {feature_info['size_mb']:.1f} MB")
    
    # Validate feature file format
    sample_features = []
    pose_types = set()
    
    try:
        with open(features_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 records
                    break
                
                data = json.loads(line.strip())
                required_fields = ['filename', 'features', 'pose_type', 'pose_confidence']
                
                for field in required_fields:
                    if field not in data:
                        print(f"❌ Missing field '{field}' in record {i+1}")
                        return False
                
                sample_features.append(len(data['features']))
                pose_types.add(data['pose_type'])
        
        print(f"   ✅ Feature dimensions: {sample_features[0]} (consistent: {len(set(sample_features)) == 1})")
        print(f"   ✅ Pose types found: {sorted(pose_types)}")
        
    except Exception as e:
        print(f"❌ Error validating features: {e}")
        return False
    
    # Check index outputs
    index_file = STEPS['index_building']['output_index']
    metadata_file = STEPS['index_building']['output_metadata']
    
    if not os.path.exists(index_file):
        print(f"❌ Index file missing: {index_file}")
        return False
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file missing: {metadata_file}")
        return False
    
    print(f"📊 Validating index and metadata...")
    index_info = get_file_info(index_file)
    metadata_info = get_file_info(metadata_file)
    
    print(f"   Index size: {index_info['size_mb']:.1f} MB")
    print(f"   Metadata size: {metadata_info['size_mb']:.1f} MB")
    
    # Validate metadata format
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        required_metadata_fields = ['filenames', 'pose_types', 'index_stats']
        for field in required_metadata_fields:
            if field not in metadata:
                print(f"❌ Missing metadata field: {field}")
                return False
        
        total_posters = len(metadata['filenames'])
        print(f"   ✅ Total posters in index: {total_posters:,}")
        print(f"   ✅ Average pose confidence: {metadata['index_stats']['average_pose_confidence']:.3f}")
        
        pose_dist = metadata['index_stats']['pose_type_distribution']
        print(f"   ✅ Pose distribution: {len(pose_dist)} types")
        
    except Exception as e:
        print(f"❌ Error validating metadata: {e}")
        return False
    
    print("✅ All outputs validated successfully!")
    return True

def print_summary():
    """Print pipeline summary"""
    
    print(f"\n{'='*60}")
    print("🎉 CONTEXT-AWARE PIPELINE COMPLETE!")
    print(f"{'='*60}")
    
    # File sizes and counts
    features_info = get_file_info(STEPS['feature_extraction']['output'])
    index_info = get_file_info(STEPS['index_building']['output_index'])
    metadata_info = get_file_info(STEPS['index_building']['output_metadata'])
    
    print(f"📊 OUTPUTS:")
    print(f"   Features: {features_info['records']:,} records ({features_info['size_mb']:.1f} MB)")
    print(f"   Index: {index_info['size_mb']:.1f} MB")
    print(f"   Metadata: {metadata_info['size_mb']:.1f} MB")
    
    total_size = features_info['size_mb'] + index_info['size_mb'] + metadata_info['size_mb']
    print(f"   Total: {total_size:.1f} MB")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Start context-aware server: python3 server/context_aware_app.py")
    print(f"   2. Or use server manager: python3 server_manager.py context-aware")
    print(f"   3. Test face-only selfies vs movie posters!")
    print(f"\n🎯 Your face-only selfies will now match appropriate movie posters!")

def main():
    """Main pipeline execution"""
    
    print("🎬 Context-Aware Movie Poster Pipeline Builder")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Parse command line arguments
    force_rebuild = '--force' in sys.argv
    
    if force_rebuild:
        print("🔄 Force rebuild mode enabled")
    
    # Run pipeline steps
    for step_name, step_config in STEPS.items():
        success = run_step(step_name, step_config, force=force_rebuild)
        
        if not success:
            print(f"\n❌ Pipeline failed at step: {step_name}")
            return 1
    
    # Validate outputs
    if not validate_outputs():
        print(f"\n❌ Pipeline validation failed")
        return 1
    
    # Print summary
    print_summary()
    
    return 0

if __name__ == "__main__":
    exit(main())

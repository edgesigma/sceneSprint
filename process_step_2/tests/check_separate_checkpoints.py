#!/usr/bin/env python3
"""
Check Separate Checkpoint Status
===============================

Utility to check the status of separate checkpoint files.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_aware_build_faiss_index import (
    load_separate_checkpoint,
    CHECKPOINT_DIR
)

def check_checkpoint_status():
    """Check the status of all separate checkpoint files"""
    
    print("📋 Separate Checkpoint Status")
    print("=" * 40)
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory does not exist: {CHECKPOINT_DIR}")
        return
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('_checkpoint.json')]
    
    if not checkpoint_files:
        print(f"📁 No checkpoint files found in {CHECKPOINT_DIR}")
        return
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint files:\n")
    
    for checkpoint_file in sorted(checkpoint_files):
        phase = checkpoint_file.replace('_checkpoint.json', '')
        checkpoint_data = load_separate_checkpoint(phase)
        
        if checkpoint_data:
            print(f"🔸 {phase.upper()}")
            print(f"   Status: {checkpoint_data.get('status', 'unknown')}")
            
            if 'timestamp' in checkpoint_data:
                try:
                    timestamp = datetime.fromisoformat(checkpoint_data['timestamp'])
                    print(f"   Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    print(f"   Last updated: {checkpoint_data['timestamp']}")
            
            # Show phase-specific information
            if phase == 'feature_loading':
                if 'processed' in checkpoint_data:
                    print(f"   Processed: {checkpoint_data['processed']:,}")
                if 'n_vectors' in checkpoint_data:
                    print(f"   Vectors: {checkpoint_data['n_vectors']:,}")
                if 'feature_dimensions' in checkpoint_data:
                    print(f"   Dimensions: {checkpoint_data['feature_dimensions']}")
                if 'source' in checkpoint_data:
                    print(f"   Source: {checkpoint_data['source']}")
            
            elif phase == 'index_building':
                if 'vectors_added' in checkpoint_data:
                    print(f"   Vectors added: {checkpoint_data['vectors_added']:,}")
                if 'total_vectors' in checkpoint_data:
                    print(f"   Total vectors: {checkpoint_data['total_vectors']:,}")
                if 'feature_dimensions' in checkpoint_data:
                    print(f"   Dimensions: {checkpoint_data['feature_dimensions']}")
                if 'batch_size' in checkpoint_data:
                    print(f"   Batch size: {checkpoint_data['batch_size']}")
                if 'index_type' in checkpoint_data:
                    print(f"   Index type: {checkpoint_data['index_type']}")
                if 'current_batch' in checkpoint_data:
                    print(f"   Current batch: {checkpoint_data['current_batch']}")
                if 'total_batches' in checkpoint_data:
                    print(f"   Total batches: {checkpoint_data['total_batches']}")
            
            # Show error information if present
            if 'error' in checkpoint_data:
                print(f"   ❌ Error: {checkpoint_data['error']}")
            
            # Show completion progress
            if phase == 'feature_loading' and checkpoint_data.get('status') == 'partial':
                processed = checkpoint_data.get('processed', 0)
                if 'total' in checkpoint_data:
                    total = checkpoint_data['total']
                    progress = (processed / total) * 100
                    print(f"   Progress: {progress:.1f}%")
            
            elif phase == 'index_building' and checkpoint_data.get('status') == 'adding_vectors':
                vectors_added = checkpoint_data.get('vectors_added', 0)
                total_vectors = checkpoint_data.get('total_vectors', 0)
                if total_vectors > 0:
                    progress = (vectors_added / total_vectors) * 100
                    print(f"   Progress: {progress:.1f}%")
            
            print()  # Empty line for readability
    
    print(f"📂 Checkpoint directory: {CHECKPOINT_DIR}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python check_separate_checkpoints.py")
        print("Check the status of all separate checkpoint files.")
        return
    
    check_checkpoint_status()

if __name__ == "__main__":
    main()

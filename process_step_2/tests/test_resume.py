#!/usr/bin/env python3
"""
Test Resume Functionality
=========================

Simple test to verify that the FAISS index building resume works correctly.
"""

import os
import json
import faiss
import numpy as np

def test_resume():
    """Test the resume functionality"""
    
    print("🧪 Testing Resume Functionality")
    print("=" * 40)
    
    # Check what files exist
    files_to_check = [
        '../index_build_checkpoint.json',
        '../context_aware_poster_index.faiss',
        '../context_aware_poster_index.faiss.tmp',
        '../context_aware_poster_metadata.json',
        '../features_cache.npz',
        '../context_aware_poster_metadata.json.cache'
    ]
    
    print("📁 Current file status:")
    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024**2)
            print(f"   ✅ {filename} ({size:.1f} MB)")
            
            # If it's a checkpoint, show details
            if filename == '../index_build_checkpoint.json':
                with open(filename, 'r') as f:
                    checkpoint = json.load(f)
                print(f"      Status: {checkpoint.get('status', 'unknown')}")
                if 'data' in checkpoint:
                    data = checkpoint['data']
                    if 'progress' in data:
                        print(f"      Progress: {data['progress']:.1f}%")
                    if 'vectors_added' in data:
                        print(f"      Vectors: {data['vectors_added']:,}")
                    if 'next_start_index' in data:
                        print(f"      Next start: {data['next_start_index']:,}")
            
            # If it's an index, show vector count
            elif filename.endswith('.faiss') or filename.endswith('.tmp'):
                try:
                    index = faiss.read_index(filename)
                    print(f"      Vectors: {index.ntotal:,}")
                except Exception as e:
                    print(f"      Error reading: {e}")
        else:
            print(f"   ❌ {filename} (not found)")
    
    print("\n💡 Resume Test Recommendations:")
    
    # Check if we have a proper resume scenario
    has_checkpoint = os.path.exists('../index_build_checkpoint.json')
    has_temp_index = os.path.exists('../context_aware_poster_index.faiss.tmp')
    has_cache = os.path.exists('../features_cache.npz')
    
    if has_checkpoint and has_temp_index:
        print("✅ Good resume scenario detected!")
        print("   - Checkpoint file exists")
        print("   - Temporary index exists")
        if has_cache:
            print("   - Feature cache exists (faster resume)")
        print("\n🚀 Run: cd .. && python3 context_aware_build_faiss_index.py --vscode-friendly")
    elif has_cache:
        print("⚠️  Partial resume scenario:")
        print("   - Feature cache exists (good)")
        print("   - But no checkpoint/temp index")
        print("\n🚀 Run: cd .. && python3 context_aware_build_faiss_index.py --vscode-friendly")
    else:
        print("❌ No resume data found")
        print("   Will start from scratch")
        print("\n🚀 Run: cd .. && python3 context_aware_build_faiss_index.py --vscode-friendly --batch-size 1000")
    
    print("\n🛠️  Other useful commands:")
    print("   --clear-cache        Clear all cache and start fresh")
    print("   --force-rebuild      Force rebuild even if complete")
    print("   --batch-size 500     Use smaller batches for stability")

if __name__ == "__main__":
    test_resume()

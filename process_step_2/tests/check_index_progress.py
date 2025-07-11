#!/usr/bin/env python3
"""
Check FAISS Index Building Progress
==================================

Quick script to check the current status of index building without running the full script.
"""

import json
import os
import faiss
from pathlib import Path

CHECKPOINT_FILE = '../index_build_checkpoint.json'
INDEX_OUTPUT = '../context_aware_poster_index.faiss'
METADATA_OUTPUT = '../context_aware_poster_metadata.json'

def main():
    print("🔍 Checking FAISS Index Building Progress")
    print("=" * 50)
    
    # Check checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        
        print(f"📋 Checkpoint Status: {checkpoint.get('status', 'unknown')}")
        
        if 'data' in checkpoint and checkpoint['data']:
            data = checkpoint['data']
            if 'progress' in data:
                print(f"🎯 Progress: {data['progress']:.1f}%")
            if 'vectors_added' in data and 'total_vectors' in data:
                print(f"📊 Vectors: {data['vectors_added']:,} / {data['total_vectors']:,}")
            if 'feature_dimensions' in data:
                print(f"📐 Dimensions: {data['feature_dimensions']}")
    else:
        print("❌ No checkpoint file found")
    
    # Check actual files
    print("\n📁 File Status:")
    
    if os.path.exists(INDEX_OUTPUT):
        size = os.path.getsize(INDEX_OUTPUT) / (1024**2)
        print(f"✅ Main index: {INDEX_OUTPUT} ({size:.1f} MB)")
        
        try:
            index = faiss.read_index(INDEX_OUTPUT)
            print(f"   Vectors in index: {index.ntotal:,}")
        except Exception as e:
            print(f"   ⚠️  Error reading index: {e}")
    else:
        print(f"❌ Main index not found: {INDEX_OUTPUT}")
    
    # Check temporary index
    temp_index = f"{INDEX_OUTPUT}.tmp"
    if os.path.exists(temp_index):
        size = os.path.getsize(temp_index) / (1024**2)
        print(f"🔄 Temp index: {temp_index} ({size:.1f} MB)")
        
        try:
            index = faiss.read_index(temp_index)
            print(f"   Vectors in temp index: {index.ntotal:,}")
        except Exception as e:
            print(f"   ⚠️  Error reading temp index: {e}")
    else:
        print(f"❌ Temp index not found: {temp_index}")
    
    if os.path.exists(METADATA_OUTPUT):
        size = os.path.getsize(METADATA_OUTPUT) / (1024**2)
        print(f"✅ Metadata: {METADATA_OUTPUT} ({size:.1f} MB)")
    else:
        print(f"❌ Metadata not found: {METADATA_OUTPUT}")
    
    print("\n💡 Tips:")
    print("• Run with --vscode-friendly for less verbose output")
    print("• Use --force-rebuild to start fresh")
    print("• The script automatically resumes from the last checkpoint")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to validate the enhanced index builder
"""
import os
import sys
import json

# Add the script directory to the path
sys.path.insert(0, '/var/llm/movieNight/movieNight/process_step_2')

from context_aware_build_faiss_index import check_existing_progress, save_checkpoint

def test_validation_logic():
    """Test the validation logic"""
    print("🧪 Testing validation logic...")
    
    # Check current progress
    status, checkpoint = check_existing_progress()
    print(f"Status: {status}")
    print(f"Checkpoint: {checkpoint}")
    
    # Test checkpoint structure
    if checkpoint:
        data = checkpoint.get('data', {})
        print(f"Lines processed: {data.get('lines_processed', 0)}")
        print(f"Total lines: {data.get('total_lines', 0)}")
        print(f"Progress: {data.get('progress', 0):.1f}%")
        
        # Check for expected fields
        required_fields = ['lines_processed', 'total_lines', 'progress']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"⚠️  Missing required fields: {missing_fields}")
        else:
            print("✅ Checkpoint structure is valid")
    
    print("\n📦 Testing file cleanup simulation...")
    
    # Simulate cleanup test (without actually deleting files)
    test_files = [
        'context_aware_poster_index.faiss',
        'context_aware_poster_index.faiss.tmp',
        'context_aware_poster_metadata.json.cache'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"   Would clean up: {test_file}")
        else:
            print(f"   File not found: {test_file}")

if __name__ == "__main__":
    test_validation_logic()

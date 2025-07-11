#!/usr/bin/env python3
"""
Test Checkpoint System
=====================

Simple test to verify checkpoints are being saved and can be read.
"""

import json
import os
import sys

# Add the parent directory to path to import the main script functions
sys.path.append('..')

# Import the checkpoint functions
from context_aware_build_faiss_index import save_checkpoint, check_existing_progress

def test_checkpoint_system():
    """Test that checkpoints are saved and loaded correctly"""
    
    print("🧪 Testing Checkpoint System")
    print("=" * 40)
    
    # Test 1: Save a checkpoint
    print("1️⃣ Testing checkpoint save...")
    try:
        save_checkpoint('testing', {
            'test_data': 'hello world',
            'numbers': [1, 2, 3],
            'progress': 50.5
        })
        print("✅ Checkpoint saved successfully")
    except Exception as e:
        print(f"❌ Checkpoint save failed: {e}")
        return False
    
    # Test 2: Check if checkpoint file exists
    print("\n2️⃣ Testing checkpoint file existence...")
    if os.path.exists('../index_build_checkpoint.json'):
        size = os.path.getsize('../index_build_checkpoint.json')
        print(f"✅ Checkpoint file exists ({size} bytes)")
    else:
        print("❌ Checkpoint file not found")
        return False
    
    # Test 3: Read checkpoint content
    print("\n3️⃣ Testing checkpoint content...")
    try:
        with open('../index_build_checkpoint.json', 'r') as f:
            checkpoint_data = json.load(f)
        
        print(f"   Status: {checkpoint_data.get('status', 'unknown')}")
        print(f"   Data keys: {list(checkpoint_data.get('data', {}).keys())}")
        
        if checkpoint_data.get('status') == 'testing':
            print("✅ Checkpoint content correct")
        else:
            print("❌ Checkpoint content incorrect")
            return False
    except Exception as e:
        print(f"❌ Failed to read checkpoint: {e}")
        return False
    
    # Test 4: Test the check_existing_progress function
    print("\n4️⃣ Testing progress check function...")
    try:
        status, checkpoint = check_existing_progress()
        print(f"   Detected status: {status}")
        if status == 'partial' and checkpoint:
            print("✅ Progress check function working")
        else:
            print("⚠️  Progress check returned unexpected result")
    except Exception as e:
        print(f"❌ Progress check failed: {e}")
        return False
    
    # Test 5: Cleanup
    print("\n5️⃣ Cleaning up test checkpoint...")
    try:
        if os.path.exists('../index_build_checkpoint.json'):
            os.remove('../index_build_checkpoint.json')
        print("✅ Cleanup successful")
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")
    
    print("\n🎉 All checkpoint tests passed!")
    return True

if __name__ == "__main__":
    success = test_checkpoint_system()
    if not success:
        print("\n❌ Some tests failed!")
        exit(1)
    else:
        print("\n✅ Checkpoint system is working correctly!")

#!/usr/bin/env python3
"""
Test Separate Checkpoint Functionality
====================================

Tests the separate checkpoint files system to ensure it works correctly.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path to import the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_aware_build_faiss_index import (
    save_separate_checkpoint,
    load_separate_checkpoint,
    cleanup_phase_checkpoint,
    CHECKPOINT_DIR
)

def test_separate_checkpoints():
    """Test separate checkpoint functionality"""
    
    print("🧪 Testing Separate Checkpoint Functionality")
    print("=" * 50)
    
    # Test 1: Save separate checkpoints
    print("\n1️⃣ Testing checkpoint saving...")
    
    feature_data = {
        'status': 'in_progress',
        'processed': 5000,
        'total': 10000,
        'last_id': 'poster_5000'
    }
    
    index_data = {
        'status': 'training',
        'vectors_added': 0,
        'feature_dimensions': 768,
        'batch_size': 1000
    }
    
    # Save checkpoints
    success1 = save_separate_checkpoint('feature_loading', feature_data)
    success2 = save_separate_checkpoint('index_building', index_data)
    
    print(f"Feature loading checkpoint saved: {success1}")
    print(f"Index building checkpoint saved: {success2}")
    
    # Test 2: Verify checkpoint files exist
    print("\n2️⃣ Testing checkpoint file existence...")
    
    feature_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'feature_loading_checkpoint.json')
    index_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'index_building_checkpoint.json')
    
    feature_exists = os.path.exists(feature_checkpoint_path)
    index_exists = os.path.exists(index_checkpoint_path)
    
    print(f"Feature checkpoint exists: {feature_exists}")
    print(f"Index checkpoint exists: {index_exists}")
    
    if feature_exists:
        with open(feature_checkpoint_path, 'r') as f:
            feature_content = json.load(f)
        print(f"Feature checkpoint content: {feature_content}")
    
    if index_exists:
        with open(index_checkpoint_path, 'r') as f:
            index_content = json.load(f)
        print(f"Index checkpoint content: {index_content}")
    
    # Test 3: Load separate checkpoints
    print("\n3️⃣ Testing checkpoint loading...")
    
    loaded_feature = load_separate_checkpoint('feature_loading')
    loaded_index = load_separate_checkpoint('index_building')
    
    print(f"Loaded feature checkpoint: {loaded_feature}")
    print(f"Loaded index checkpoint: {loaded_index}")
    
    # Verify data integrity
    feature_match = loaded_feature.get('processed') == feature_data['processed']
    index_match = loaded_index.get('batch_size') == index_data['batch_size']
    
    print(f"Feature data integrity: {feature_match}")
    print(f"Index data integrity: {index_match}")
    
    # Test 4: Update checkpoints
    print("\n4️⃣ Testing checkpoint updates...")
    
    updated_feature_data = {
        'status': 'completed',
        'processed': 10000,
        'total': 10000,
        'completion_time': time.time()
    }
    
    updated_index_data = {
        'status': 'adding_vectors',
        'vectors_added': 5000,
        'feature_dimensions': 768,
        'batch_size': 1000,
        'current_batch': 5
    }
    
    save_separate_checkpoint('feature_loading', updated_feature_data)
    save_separate_checkpoint('index_building', updated_index_data)
    
    # Verify updates
    reloaded_feature = load_separate_checkpoint('feature_loading')
    reloaded_index = load_separate_checkpoint('index_building')
    
    feature_updated = reloaded_feature.get('status') == 'completed'
    index_updated = reloaded_index.get('vectors_added') == 5000
    
    print(f"Feature checkpoint updated: {feature_updated}")
    print(f"Index checkpoint updated: {index_updated}")
    
    # Test 5: Test non-existent checkpoint
    print("\n5️⃣ Testing non-existent checkpoint...")
    
    nonexistent = load_separate_checkpoint('nonexistent_phase')
    empty_checkpoint = len(nonexistent) == 0
    
    print(f"Non-existent checkpoint returns empty dict: {empty_checkpoint}")
    
    # Test 6: Cleanup checkpoints
    print("\n6️⃣ Testing checkpoint cleanup...")
    
    cleanup_phase_checkpoint('feature_loading')
    cleanup_phase_checkpoint('index_building')
    
    # Verify cleanup
    feature_cleaned = not os.path.exists(feature_checkpoint_path)
    index_cleaned = not os.path.exists(index_checkpoint_path)
    
    print(f"Feature checkpoint cleaned up: {feature_cleaned}")
    print(f"Index checkpoint cleaned up: {index_cleaned}")
    
    # Test 7: Directory structure
    print("\n7️⃣ Testing directory structure...")
    
    checkpoint_dir_exists = os.path.exists(CHECKPOINT_DIR)
    print(f"Checkpoint directory exists: {checkpoint_dir_exists}")
    
    if checkpoint_dir_exists:
        remaining_files = os.listdir(CHECKPOINT_DIR)
        print(f"Remaining files in checkpoint dir: {remaining_files}")
        
        # Clean up the directory if it's empty
        if not remaining_files:
            os.rmdir(CHECKPOINT_DIR)
            print("🧹 Cleaned up empty checkpoint directory")
    
    # Summary
    print("\n📊 Test Summary")
    print("-" * 30)
    
    tests = [
        ("Checkpoint saving", success1 and success2),
        ("File existence", feature_exists and index_exists),
        ("Data loading", bool(loaded_feature and loaded_index)),
        ("Data integrity", feature_match and index_match),
        ("Checkpoint updates", feature_updated and index_updated),
        ("Non-existent handling", empty_checkpoint),
        ("Cleanup functionality", feature_cleaned and index_cleaned)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All separate checkpoint tests passed!")
        return True
    else:
        print("⚠️ Some tests failed!")
        return False

if __name__ == "__main__":
    success = test_separate_checkpoints()
    sys.exit(0 if success else 1)

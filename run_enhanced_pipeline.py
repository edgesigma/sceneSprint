#!/usr/bin/env python3
# run_enhanced_pipeline.py
# ---------------------------------------------
# Complete pipeline to build enhanced matching
# with pose detection + color analysis
# ---------------------------------------------

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Enhanced Movie Poster Matching Pipeline")
    print("=" * 50)
    
    # Check if subset directory exists
    subset_dir = "process_step_1/subset"
    if not os.path.exists(subset_dir) or not os.listdir(subset_dir):
        print(f"❌ Subset directory '{subset_dir}' is empty or missing")
        print("   Please ensure you have movie poster images in the subset directory")
        return False
    
    print(f"📁 Found poster subset directory: {subset_dir}")
    
    # Step 1: Extract enhanced features
    step1_success = run_command(
        "cd process_step_1 && python enhanced_feature_extraction.py",
        "Step 1: Extracting pose + color features from posters"
    )
    
    if not step1_success:
        print("Pipeline failed at Step 1")
        return False
    
    # Step 2: Build enhanced FAISS index
    step2_success = run_command(
        "cd process_step_2 && python enhanced_build_faiss_index.py",
        "Step 2: Building enhanced FAISS index"
    )
    
    if not step2_success:
        print("Pipeline failed at Step 2")
        return False
    
    # Step 3: Test the enhanced server
    print(f"\n🎯 Pipeline completed successfully!")
    print(f"📋 Summary:")
    print(f"   • Enhanced features extracted")
    print(f"   • FAISS index built with pose + color")
    print(f"   • Ready to run enhanced server")
    print(f"\n🚀 To start the enhanced server:")
    print(f"   cd server && python enhanced_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

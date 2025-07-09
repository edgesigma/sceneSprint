#!/usr/bin/env python3
# server_manager.py
# ---------------------------------------------
# Manage switching between original and enhanced servers
# ---------------------------------------------

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import mediapipe
        import faiss
        import cv2
        import flask
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def start_server(server_type='original'):
    """Start server (original, enhanced, robust, or context-aware)"""
    
    if server_type == 'context-aware':
        cmd = "python context_aware_app.py"
        version = "Context-Aware (Intelligent Pose Matching)"
    elif server_type == 'robust':
        cmd = "python robust_app.py"
        version = "Robust (Adaptive Pose + Enhanced Color)"
    elif server_type == 'enhanced':
        cmd = "python enhanced_app.py"
        version = "Enhanced (Pose + Color)"
    else:
        cmd = "python app.py"
        version = "Original (Color Only)"
    
    print(f"🚀 Starting {version} server...")
    print(f"   Command: {cmd}")
    print(f"   Directory: server/")
    print(f"   URL: http://localhost:5000")
    print(f"   Press Ctrl+C to stop")
    
    try:
        os.chdir('server')
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")

def main():
    if len(sys.argv) < 2:
        print("🎬 Movie Poster Match Server Manager")
        print("=" * 50)
        print("Usage:")
        print("  python server_manager.py [original|enhanced|robust|context-aware]")
        print()
        print("Commands:")
        print("  original        - Start original server (color matching only)")
        print("  enhanced        - Start enhanced server (pose + color matching)")
        print("  robust          - Start robust server (adaptive pose + enhanced color)")
        print("  context-aware   - Start context-aware server (intelligent pose matching)")
        print()
        print("📊 Server Comparison:")
        print("  Original: 2x2 grid color histograms only")
        print("  Enhanced: Basic pose detection + 2x2 color grid")
        print("  Robust:   Multi-strategy pose detection + 3x3 color grid + adaptive weighting")
        print("  Context-Aware: Intelligent pose matching with context awareness")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'original':
        if check_dependencies():
            start_server('original')
    
    elif command == 'enhanced':
        if check_dependencies():
            # Check if enhanced index exists
            if not os.path.exists('process_step_2/enhanced_poster_index.faiss'):
                print("❌ Enhanced index not found")
                print("   Run: python run_enhanced_pipeline.py")
                return
            start_server('enhanced')
    
    elif command == 'robust':
        if check_dependencies():
            # Check if robust index exists
            if not os.path.exists('process_step_2/robust_poster_index.faiss'):
                print("❌ Robust index not found")
                print("   Build it with: cd process_step_2 && python robust_build_faiss_index.py")
                return
            start_server('robust')
    
    elif command == 'context-aware':
        if check_dependencies():
            # Check if context-aware index exists
            if not os.path.exists('process_step_2/context_aware_poster_index.faiss'):
                print("❌ Context-aware index not found")
                print("   Build it with: python build_context_aware_pipeline.py")
                return
            start_server('context-aware')
    
    else:
        print(f"❌ Unknown command: {command}")
        print("   Available: original, enhanced, robust, context-aware")

if __name__ == "__main__":
    main()

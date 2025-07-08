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

def start_server(enhanced=False):
    """Start server (original or enhanced)"""
    
    if enhanced:
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
        print("=" * 40)
        print("Usage:")
        print("  python server_manager.py [original|enhanced]")
        print()
        print("Commands:")
        print("  original  - Start original server (color matching only)")
        print("  enhanced  - Start enhanced server (pose + color matching)")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'original':
        if check_dependencies():
            start_server(enhanced=False)
    
    elif command == 'enhanced':
        if check_dependencies():
            # Check if enhanced index exists
            if not os.path.exists('process_step_2/enhanced_poster_index.faiss'):
                print("❌ Enhanced index not found")
                print("   Run: python run_enhanced_pipeline.py")
                return
            start_server(enhanced=True)
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()

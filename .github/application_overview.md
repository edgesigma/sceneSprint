
## Application Overview

**Movie Poster Match** is a web application that uses computer vision and machine learning to find the movie poster that best matches a user's selfie based on color similarity.

## Updated Core Functionality

### Enhanced Version with Pose Detection (process_step_3/search_user.py):
1. **Camera Capture**: Same selfie capture functionality
2. **Pose Detection**: Uses **MediaPipe** to extract 33 body pose keypoints (x,y coordinates)
3. **Combined Analysis**: 
   - **Pose features** (60% weight): 33 keypoints × 2 coordinates = 66 features
   - **Color features** (40% weight): 8×8×8 color histogram = 512 features
4. **Advanced Matching**: Combines both pose and color similarity for more sophisticated matching
5. **Multi-criteria Search**: Finds posters that match both the user's pose/body position AND color palette

The pose detection implementation extracts key body landmarks like shoulders, elbows, wrists, hips, knees, etc., allowing the system to match not just colors but also body positioning and pose similarity with movie poster characters.

This represents a more advanced version of the matching algorithm that considers both **visual similarity** (colors) and **structural similarity** (human pose/positioning), which would provide much more meaningful and accurate poster matches based on how the person is positioned in their selfie.

### Technical Architecture

#### Frontend (index.html)
- **Single Page Application** with embedded CSS and JavaScript
- **Camera Integration**: Uses `getUserMedia()` API for webcam access
- **Canvas Processing**: Captures and processes selfie images
- **AJAX Communication**: Sends images to backend API and displays results
- **Responsive Design**: Mobile-friendly interface with flex layout

#### Backend (app.py)
- **Flask Application** serving both the frontend and API
- **Computer Vision**: Uses OpenCV for image processing
- **Feature Extraction**: Implements grid-based color histogram analysis (2x2 grid, 8 bins per channel)
- **Similarity Search**: FAISS index for fast nearest neighbor search
- **File Management**: Handles image uploads and serves generated composites

#### Data Processing Pipeline

1. **Step 1** (`process_step_1/`): 
   - Extracts features from movie poster collection
   - Creates subset of posters for processing
   - Generates color histogram features

2. **Step 2** (`process_step_2/`):
   - Builds FAISS index from extracted features
   - Creates metadata mapping for poster files

3. **Step 3** (`process_step_3/`):
   - Contains search utilities and samples
   - Final index and metadata files

### Key Features

- **Real-time Processing**: Fast similarity search using FAISS indexing
- **Large Dataset**: Works with 62,000+ movie poster images
- **Social Integration**: Built-in sharing capabilities for Twitter and Facebook
- **Cross-platform**: Works on desktop and mobile browsers with camera support
- **Privacy-focused**: Processes images locally, doesn't permanently store user photos

### File Structure

- **Frontend**: Single HTML file with embedded styles and scripts
- **Backend**: Flask server with image processing capabilities  
- **Data**: Massive collection of movie poster images (covers/)
- **Processing**: Multi-step pipeline for feature extraction and indexing
- **Experiments**: Share intent testing and social media integration

The application demonstrates a practical implementation of computer vision for entertainment purposes, combining web technologies with machine learning to create an engaging user experience.
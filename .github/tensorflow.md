# TensorFlow Usage in Movie Poster Match

This project does not directly import TensorFlow in any of its modules. The only related dependency is **MediaPipe**, which internally relies on TensorFlow Lite models. When the server starts, MediaPipe's pose detector prints TensorFlow initialization messages, which explains the logs you noticed.

## Where MediaPipe Uses TensorFlow
- `process_step_1/context_aware_feature_extraction.py` imports MediaPipe and creates a pose detector:
  ```python
  import mediapipe as mp
  ...
  mp_pose = mp.solutions.pose
  pose_detector = mp_pose.Pose(...)
  ```
- This detector counts people in each poster image so the feature extraction step can include a "person count" component. These features are later loaded by the Flask server (`server/context_aware_app.py`) for similarity search.

## Purpose
TensorFlow is therefore only present through MediaPipe's pose model. It powers the internal inference that estimates whether a person is present in an image. This information becomes part of each poster's feature vector and helps the similarity search consider how many people appear in the scene.

## Alternative Solutions
Other pose detection frameworks could be used instead of MediaPipe (and TensorFlow):
- OpenCV's DNN module with a pre-trained pose model
- PyTorch-based pose estimation libraries (e.g., HRNet, OpenPifPaf)
- A custom TensorFlow Lite or ONNX model

Any of these could replace MediaPipe while still producing person-count features for the existing matching pipeline.

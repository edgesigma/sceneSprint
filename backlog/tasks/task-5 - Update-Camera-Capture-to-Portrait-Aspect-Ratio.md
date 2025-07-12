---
id: task-5
title: Update Camera Capture to Portrait Aspect Ratio
status: Done
assignee: []
created_date: '2025-07-11'
updated_date: '2025-07-12'
labels: []
dependencies: []
---

## Description

To improve consistency between the query image and the portrait-oriented poster dataset, the client-side camera capture should be set to a portrait aspect ratio.

## Acceptance Criteria

- [ ] The web interface is updated to request a portrait aspect ratio from the device camera
- [ ] The captured image stream is consistently in portrait mode
- [ ] This change is tested on common mobile and desktop devices

## Implementation Notes

Updated index.html to request a portrait aspect ratio from the camera, with a fallback for unsupported devices.

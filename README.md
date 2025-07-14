# Movie Poster Match

Movie Poster Match is a web application that uses computer vision to find the movie poster that best matches a user's selfie. The project extracts color and pose information from posters, indexes those features with FAISS and exposes a simple web interface for searching.

## Overview

The application combines several components:

- **Frontend** – a single page (`index.html`) that captures a selfie through the browser, sends it to the server and displays the results.
- **Backend** – a Flask server (`server/context_aware_app.py`) that extracts weighted features from the uploaded image and performs a similarity search.
- **Processing pipeline** – scripts in `process_step_1` and `process_step_2` that generate the feature dataset and FAISS index.

Documentation in `.github/` further explains the approach. The [application overview](.github/application_overview.md) describes the pose enhanced matching pipeline and the [vector embedding](.github/vector_embedding.md) document clarifies how the FAISS index stores the poster embeddings.

## Data Processing Pipeline

The dataset is prepared in two main stages:

1. **Feature Extraction** (`process_step_1/`)
   - Uses a 4×4 HSV color histogram and a binned person count to describe each poster.
   - Feature weights are read from `config.json` allowing the importance of color and pose to be tuned.
2. **Index Building** (`process_step_2/`)
   - Reads the TSV feature file and builds a FAISS index for fast similarity search.
   - Stores poster filenames in `context_aware_poster_metadata.json`.

These stages can be run together using `build_context_aware_pipeline.py` which also validates the outputs.

## Dataset

The poster metadata and movie information come from the
[MovieLens&nbsp;25M dataset](https://www.kaggle.com/datasets/grouplens/movielens-25m)
available on Kaggle. We downloaded this dataset to obtain
movie titles and poster paths, ultimately collecting over
62,000 posters for the `covers/` directory (not included in the repo).
MovieLens&nbsp;25M contains 25 million movie ratings and extensive
metadata, making it a useful source for building a large image collection.

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build features and index
python build_context_aware_pipeline.py

# Start the server
python server_manager.py context-aware
```

By default the server runs on `http://localhost:5000` and serves `index.html` from the project root.

## Directory Structure

```
process_step_1/   Feature extraction logic and tests
process_step_2/   FAISS index builder and tests
server/           Flask application
covers/           Poster image dataset (not included in repo)
.github/          Additional documentation
```

For more details on regional histograms and high-dimensional visualization techniques see [.github/regional-histograms.md](.github/regional-histograms.md) and [.github/visualizing-high-dimensions.md](.github/visualizing-high-dimensions.md).

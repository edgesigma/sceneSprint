# Test Scripts for FAISS Index Builder

This folder contains utility scripts for testing and monitoring the FAISS index building process.

## Scripts

### `check_index_progress.py`
Quick status checker for the FAISS index building progress.

**Usage:**
```bash
cd /var/llm/movieNight/movieNight/process_step_2
python3 tests/check_index_progress.py
```

**What it shows:**
- Current checkpoint status
- Progress percentage
- Vector counts
- File sizes

### `test_resume.py`
Comprehensive test of the resume functionality with recommendations.

**Usage:**
```bash
cd /var/llm/movieNight/movieNight/process_step_2
python3 tests/test_resume.py
```

**What it does:**
- Analyzes all resume-related files
- Shows detailed checkpoint information
- Provides specific recommendations for next steps
- Lists available command options

## Quick Commands

From the `process_step_2` directory:

```bash
# Check current status
python3 tests/check_index_progress.py

# Test resume capability
python3 tests/test_resume.py

# Start/resume building with VS Code friendly output
python3 context_aware_build_faiss_index.py --vscode-friendly

# Clear cache and start fresh
python3 context_aware_build_faiss_index.py --clear-cache --vscode-friendly

# Use smaller batches for stability
python3 context_aware_build_faiss_index.py --vscode-friendly --batch-size 500
```

## Files Monitored

- `index_build_checkpoint.json` - Build progress checkpoint
- `context_aware_poster_index.faiss` - Final index file
- `context_aware_poster_index.faiss.tmp` - Temporary index during building
- `context_aware_poster_metadata.json` - Final metadata
- `features_cache.npz` - Cached feature vectors
- `context_aware_poster_metadata.json.cache` - Cached metadata

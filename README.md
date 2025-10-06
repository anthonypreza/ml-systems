# ML Systems

A collection of machine learning experiments and training pipelines.

## Overview

This repository contains various ML projects and experiments, ranging from computer vision to natural language processing. Each project is organized in its own directory with dedicated documentation and implementation details.

## Projects

- **`sentiment-classification/`** - Text classification experiments with transformer models
- **`video-search/`** - Multimodal video-text retrieval using contrastive learning
- **`similar-listings/`** - Session-based recommendation system using neural embeddings
- **`ad-click-prediction/`** - Click-through rate modeling pipeline using XGBoost with sklearn preprocessing and SHAP diagnostics
- **`recsys/`** - Recommendation experiments spanning Goodreads LTR (XGBoost + neural models) alongside Kaggle dataset
- **`feature-store-lab`** - Example streaming pipeline for an online/offline feature store. 

See individual project directories for specific documentation and usage instructions.

## Quick Start

### Environment Setup

**Using uv (Recommended):**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project environment
uv venv --python 3.12
uv sync
```

**Using pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt  # If available
```

### System Dependencies

Some projects may require additional system dependencies:

- **ffmpeg** (for video processing projects)
- **CUDA** (for GPU acceleration, optional)

## Development

- **Python Version**: 3.12+ (as specified in pyproject.toml)
- **Dependency Management**: Managed with uv for reproducible environments
- **Framework**: Primarily PyTorch with Hugging Face ecosystem

## Common Commands

```bash
# Add dependencies
uv add package-name

# Run scripts in project environment
uv run python script.py

# Update all dependencies
uv lock --upgrade
```

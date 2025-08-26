# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning systems repository containing various ML experiments and research implementations. The repository includes multiple projects ranging from NLP to computer vision, with implementations in both Jupyter notebooks and standalone Python scripts.

### Current Projects:
- **Sentiment Classification**: BERT-based sentiment analysis on IMDB movie reviews
- **Video Search**: Multimodal video-text retrieval using contrastive learning with MSR-VTT dataset

## Development Environment Setup

### Python Environment & Package Management
- **Recommended**: Use uv for fast Python package management
- Python 3.12 is required (as specified in pyproject.toml)
- Virtual environments (`.venv`) are gitignored
- Key dependencies: `torch`, `transformers`, `datasets`, `opencv-python`, `yt-dlp`, `requests`

### Setup Commands
**Using uv (preferred):**
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv --python 3.12
uv sync  # Install dependencies from pyproject.toml
```

**System Dependencies:**
```bash
# Required for video processing (video-search project)
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Dependency Management with uv
```bash
# Add packages
uv add package-name

# Remove packages
uv remove package-name

# Run scripts in environment
uv run python script.py
uv run jupyter notebook

# Update all dependencies
uv lock --upgrade
```

## Architecture Overview

### Sentiment Classification (`sentiment-classification/`)
BERT-based binary sentiment analysis:

1. **TextEncoder**: BERT + projection layer (512-dim embeddings)
2. **TextClassifier**: Linear classification head for binary prediction
3. **Data**: IMDB movie reviews via Hugging Face datasets
4. **Training**: Adam optimizer, CrossEntropyLoss, early stopping

**Migration Notes**: Originally TensorFlow, migrated to PyTorch

### Video Search (`video-search/`)
Multimodal video-text retrieval using contrastive learning:

1. **Video Encoder**: ResNet50 + temporal aggregation → 512-dim embeddings
2. **Text Encoder**: BERT + projection layer → 512-dim embeddings  
3. **Contrastive Learning**: Temperature-scaled cosine similarity with cross-entropy loss
4. **Data Sources**: 
   - **MSR-VTT**: Real YouTube video clips with captions (requires yt-dlp + ffmpeg)
   - **Synthetic**: Generated dummy videos for testing
5. **Search**: Linear search with a precomputed embedding cache (numpy-based)

**Key Components**:
- `video-search/scripts/pipeline.py`: Training script that saves model and search index
- `video-search/scripts/search_interactive.py`: Interactive search interface with rich UI
- `video-search/scripts/search_cli.py`: Command-line search for scripting and single queries
- `video-search/scripts/build_index.py`: Build an index from an already trained model
- `video-search/video_search/core/model.py`: Core contrastive learning implementation
- `video-search/video_search/core/encoders.py`: Encoders for video and text
- `video-search/video_search/core/config.py`: Configuration and save paths
- `video-search/video_search/utils/data.py`: MSR-VTT integration and synthetic data
- `video-search/video_search/utils/video.py`: Video processing and frame extraction

## Common Operations

### Sentiment Classification
**Jupyter Notebook**: `sentiment-classification/imdb_sentiment.ipynb`
- Execute cells sequentially
- Uses 10,000 training examples with 80/20 train/validation split  
- Includes early stopping (patience=3) to prevent overfitting

### Video Search System
**Two-Phase Workflow**: Train once, search many times

**1. Training Phase**:
```bash
# From the video-search/ directory
python scripts/pipeline.py --dataset msrvtt --max-videos 100   # MSR-VTT (recommended)
python scripts/pipeline.py --dataset synthetic                 # Synthetic (quick test)
```
Creates: `./models/video_search_model.pth`, `./models/video_index.pkl` (embedding cache), `./models/video_paths.txt`

**2. Search Phase**:
```bash
# Interactive search with rich UI
python scripts/search_interactive.py

# Command line search
python scripts/search_cli.py "cat playing with toy"

# JSON output for scripting
python scripts/search_cli.py "cooking" --format json
```

**Configuration**: Edit `video-search/video_search/core/config.py` to adjust:
- Training: `max_frames`, `batch_size`, `learning_rate`, `num_epochs`
- Model paths: `model_save_path`, `index_save_path`, `video_paths_save_path`

## Troubleshooting & Tips

### Sentiment Classification Overfitting
Built-in mitigation includes:
- Train/validation split (80/20) with early stopping
- Weight decay and dropout regularization
- Validation monitoring

Additional strategies:
- Freeze BERT layers or use smaller models (distilbert)
- Increase dropout rates (0.3-0.5)
- Reduce training examples or increase weight decay

### Video Search Issues
**Video Download Failures**: Some YouTube videos may be private/unavailable - this is handled gracefully
**Memory Issues**: Reduce `batch_size`, `max_frames`, or `max_videos` in config
**Slow Training**: Use GPU if available, or reduce dataset size for testing
**Missing Model/Index**: Run training phase first (`python scripts/pipeline.py`) before search phase
**Search Performance**: Pre-computed embeddings provide fast search without re-encoding videos

### System Dependencies
- **ffmpeg**: Required for video processing (video-search project)
- **CUDA**: Optional but recommended for GPU acceleration
- **yt-dlp**: Automatically installed via pip for YouTube downloads

## File Structure
```
ml-systems/
├── sentiment-classification/    # BERT sentiment analysis project
│   ├── imdb_sentiment.ipynb   # Main Jupyter notebook
│   └── README.md              # Project-specific documentation
├── video-search/               # Video-text retrieval project
│   ├── scripts/               # Entrypoints (pipeline, search, build_index)
│   │   ├── pipeline.py
│   │   ├── search_interactive.py
│   │   ├── search_cli.py
│   │   └── build_index.py
│   ├── video_search/          # Python package
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── encoders.py
│   │   │   └── model.py
│   │   ├── search/
│   │   │   ├── index.py
│   │   │   └── search.py
│   │   └── utils/
│   │       ├── data.py
│   │       └── video.py
│   └── README.md              # Project-specific documentation
├── pyproject.toml             # Python dependencies
├── README.md                  # Repository overview
└── CLAUDE.md                  # This file
```

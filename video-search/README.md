# Video-Text Search with Contrastive Learning

Multimodal video search system that enables natural language queries over video content using contrastive learning.

## Overview

This project implements a video-text retrieval system that learns joint embeddings for videos and text descriptions. Users can search through video collections using natural language queries like "cat playing with toy" or "person cooking pasta".

## Architecture

```
Video Frames → ResNet50 → Video Encoder (512-dim)
                                          ↘
                                      Contrastive Loss
                                          ↗
Text Query → BERT → Text Encoder (512-dim)
```

**Key Components:**
- **Video Encoder**: ResNet50 backbone + temporal aggregation + projection to 512-dim embeddings
- **Text Encoder**: BERT + projection layer with L2 normalization
- **Contrastive Learning**: Temperature-scaled cosine similarity with cross-entropy loss
- **Training**: 1 positive + N negative text descriptions per video

## Features

- **Real Dataset Support**: MSR-VTT dataset integration with YouTube video downloading
- **Synthetic Data**: Fallback synthetic dataset for testing without downloads
- **Flexible Search**: Both linear search and indexed search options
- **Video Processing**: Automatic frame extraction and preprocessing
- **Robust Downloads**: YouTube clip extraction with precise timing using yt-dlp + ffmpeg

## Dataset Options

### MSR-VTT (Recommended)
- **Source**: Microsoft Research Video to Text dataset
- **Content**: ~7,000 YouTube video clips with multiple captions each
- **Download**: Automatic downloading of video segments with precise start/end times
- **Captions**: ~20 human-annotated captions per video clip

### Synthetic (Testing)
- **Categories**: Cats, dogs, cooking, sports
- **Content**: Generated dummy videos with associated text descriptions
- **Use Case**: Quick testing without video downloads

## Quick Start

### 1. Training Phase
Train the model and create the search index (run from video-search/ directory):

```bash
# MSR-VTT (recommended)
python scripts/pipeline.py --dataset msrvtt --max-videos 100

# Synthetic dataset (quick test)
python scripts/pipeline.py --dataset synthetic
```

This creates:
- `./models/video_search_model.pth` - Trained model
- `./models/video_index.pkl` - Pre-computed video embeddings cache
- `./models/video_paths.txt` - Video file paths

### 2. Search Phase
Use the trained model for interactive or command-line search:

```bash
# Interactive search (recommended)
python scripts/search_interactive.py

# Single query search
python scripts/search_cli.py "cat playing with toy"

# JSON output for scripting
python scripts/search_cli.py "cooking pasta" --format json --top-k 3
```

## System Requirements

- **Python**: 3.12+
- **ffmpeg**: Required for video processing and YouTube downloads
- **yt-dlp**: For downloading YouTube video clips (installed via pip)

### Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Configuration

Edit `video_search/core/config.py` to adjust model parameters:

```python
class VideoSearchConfig:
    max_frames = 8              # Frames per video
    frame_size = (224, 224)     # Frame resolution
    text_embed_dim = 512        # Text embedding dimension
    video_embed_dim = 512       # Video embedding dimension
    temperature = 0.07          # Contrastive learning temperature
    num_negatives = 7           # Negative samples per triplet
    batch_size = 16             # Training batch size
    learning_rate = 1e-4        # Adam learning rate
    num_epochs = 10             # Training epochs
```

## Workflow

### Training Process
1. **Data Loading**: Downloads video clips from YouTube with precise timing
2. **Triplet Creation**: Each video caption becomes a positive query with N negative samples  
3. **Contrastive Training**: Learns to maximize similarity between videos and their captions
4. **Model & Index Saving**: Automatically saves trained model and search index
5. **Evaluation**: Tests retrieval performance with recall@k metrics

### Search Process
1. **Model Loading**: Loads pre-trained model and video index
2. **Query Processing**: Encodes search query using BERT text encoder
3. **Similarity Search**: Uses pre-computed embeddings with dot product similarity
4. **Results Ranking**: Returns top-k videos ranked by similarity score

## Interactive Search Interface

The interactive search provides a rich user experience:

```
🎥 INTERACTIVE VIDEO SEARCH
============================================================
📊 Index contains 145 videos
💡 Type your search query and press Enter
💡 Type 'quit' or 'exit' to stop
💡 Type 'help' for more options
============================================================

🔍 Search query: cat playing with toy

🎯 Top 5 results:
--------------------------------------------------
 1. cats_5.mp4
    Score: 0.847 [████████████████████] 84.7%
    Path:  /tmp/video_search_test/cats_5.mp4

 2. cats_2.mp4
    Score: 0.723 [██████████████▒▒▒▒▒▒] 72.3%
    Path:  /tmp/video_search_test/cats_2.mp4
```

## Project Structure

### Scripts
- `scripts/pipeline.py` - Training pipeline that saves model and index
- `scripts/search_interactive.py` - Interactive search interface with rich UI
- `scripts/search_cli.py` - Command-line search for single queries and scripting
- `scripts/build_index.py` - Build an index from an already trained model

### Package: `video_search`
- `video_search/core/config.py` - Configuration and save paths
- `video_search/core/model.py` - Contrastive learning model and training/eval
- `video_search/core/encoders.py` - Video and text encoder architectures
- `video_search/search/search.py` - Search routines (linear and indexed)
- `video_search/search/index.py` - Pre-computing and caching video embeddings
- `video_search/utils/data.py` - Dataset loading, MSR-VTT integration, synthetic data
- `video_search/utils/video.py` - Video processing and frame extraction

## Performance Notes

- **CPU Training**: Works on CPU but GPU recommended for larger datasets
- **Memory Usage**: Adjust `batch_size` and `max_frames` based on available RAM
- **Video Downloads**: Some YouTube videos may be unavailable (handled gracefully)
- **Search Speed**: Pre-computed embeddings enable fast similarity search without re-encoding videos
- **Model Persistence**: Train once, search many times - no need to retrain for new queries

## Command Line Options

### Training (`scripts/pipeline.py`)
```bash
python scripts/pipeline.py --dataset [synthetic|msrvtt] --max-videos N
```

### Interactive Search (`scripts/search_interactive.py`) 
```bash
python scripts/search_interactive.py [--top-k N] [--model-path PATH] [--index-path PATH]
```

### CLI Search (`scripts/search_cli.py`)
```bash
python scripts/search_cli.py "query" [--top-k N] [--format table|json|simple] [--quiet]
```

## Example Output

### Interactive Search
```
🔍 Search query: cat playing

🎯 Top 3 results:
--------------------------------------------------
 1. cats_5.mp4
    Score: 0.847 [████████████████████] 84.7%
    Path:  /tmp/video_search_test/cats_5.mp4
```

### CLI JSON Output
```bash
python scripts/search_cli.py "cooking" --format json --top-k 2
{
  "query": "cooking",
  "results": [
    {"path": "cooking_3.mp4", "score": 0.923, "filename": "cooking_3.mp4"},
    {"path": "cooking_1.mp4", "score": 0.856, "filename": "cooking_1.mp4"}
  ]
}
```

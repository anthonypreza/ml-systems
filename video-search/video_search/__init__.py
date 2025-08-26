"""
Video-Text Search with Contrastive Learning

A multimodal video search system using contrastive learning to enable
natural language queries over video content.
"""

from .core.config import VideoSearchConfig
from .core.model import (
    VideoTextContrastiveModel,
    train_video_search_model,
    evaluate_retrieval,
)
from .core.encoders import TextEncoder, VideoEncoder
from .utils.data import (
    VideoQueryTriplet,
    VideoQueryDataset,
    load_msrvtt_dataset,
    create_synthetic_dataset,
)
from .utils.video import VideoProcessor
from .search.search import search_videos, search_videos_indexed
from .search.index import build_video_index

__version__ = "1.0.0"
__author__ = "Anthony Preza"

__all__ = [
    # Core
    "VideoSearchConfig",
    "VideoTextContrastiveModel",
    "train_video_search_model",
    "evaluate_retrieval",
    "TextEncoder",
    "VideoEncoder",
    # Data
    "VideoQueryTriplet",
    "VideoQueryDataset",
    "load_msrvtt_dataset",
    "create_synthetic_dataset",
    "VideoProcessor",
    # Search
    "search_videos",
    "search_videos_indexed",
    "build_video_index",
]

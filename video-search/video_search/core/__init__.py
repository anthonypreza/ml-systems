"""Core components for video-text contrastive learning."""

from .config import VideoSearchConfig
from .model import (
    VideoTextContrastiveModel,
    train_video_search_model,
    evaluate_retrieval,
)
from .encoders import TextEncoder, VideoEncoder

__all__ = [
    "VideoSearchConfig",
    "VideoTextContrastiveModel",
    "train_video_search_model",
    "evaluate_retrieval",
    "TextEncoder",
    "VideoEncoder",
]

import torch
import tqdm
import numpy as np

from typing import Any, Dict, List

from ..core.model import VideoTextContrastiveModel


def build_video_index(
    model: VideoTextContrastiveModel, video_paths: List[str], device: str = "cpu"
) -> Dict[str, Any]:
    """Pre-compute embeddings for all bideos and store in index"""
    model.to(device)
    model.eval()

    video_embeddings = []
    valid_paths = []

    with torch.no_grad():
        for video_path in tqdm.tqdm(video_paths, desc="Building video index"):
            try:
                # Compute embeddings once and store
                video_embedding = model.encode_video(video_path)
                video_embeddings.append(video_embedding.cpu().numpy())
                valid_paths.append(video_path)
            except Exception as e:
                print(f"Skipping {video_path}: {e}")

    # Stack into matrix [num_videos, embed_dim]
    video_embeddings = np.vstack(video_embeddings)

    return {
        "embeddings": video_embeddings,
        "paths": valid_paths,
        "index_map": {path: idx for idx, path in enumerate(valid_paths)},
    }

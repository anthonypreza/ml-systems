import torch
import tqdm
import numpy as np

from ..core.model import VideoTextContrastiveModel
from typing import Any, Dict, List, Tuple


def search_videos(
    model: VideoTextContrastiveModel,
    query: str,
    video_paths: List[str],
    device: str = "cpu",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Search for videos using a text query

    Args:
        model: Trained VideoTextContrastiveModel
        query: Text query to search for
        video_paths: List of video file paths to search through
        device: Device for inference
        top_k: Number of top results to return

    Returns:
        List of (video_path, similarity_score) tuples, sorted by similarity
    """
    model.to(device)
    model.eval()

    # Encode query
    query_embedding = model.encode_texts([query]).to(device)  # [1, embed_dim]

    # Encode all videos and compute similarities
    similarities = []

    with torch.no_grad():
        for video_path in tqdm(video_paths, desc=f"Searching for '{query}'"):  # type: ignore
            try:
                # Encode video
                video_frames = model.video_processor.extract_frames(video_path)
                video_frames = video_frames.unsqueeze(0).to(device)  # add batch dim
                video_embedding = model.video_encoder(video_frames)  # [1, embed_dim]

                # compute similarity
                similarity = (
                    torch.matmul(video_embedding, query_embedding.T).squeeze().item()
                )
                similarities.append((video_path, similarity))

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                similarities.append((video_path, -float("inf")))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def search_videos_indexed(
    model: VideoTextContrastiveModel, query: str, video_index: Dict[str, Any], top_k=5
):
    """Fast search using pre-computed video embeddings"""

    # Compute query embedding
    query_embedding = model.encode_texts([query]).cpu().detach().numpy()

    # Batch compute all similarities
    similarities = np.dot(video_index["embeddings"], query_embedding.T).squeeze()

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        video_path = video_index["paths"][idx]
        score = similarities[idx]
        results.append((video_path, float(score)))

    return results

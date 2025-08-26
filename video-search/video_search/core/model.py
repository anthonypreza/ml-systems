"""
Video Search with Contrastive Learning

This module implements a complete video search system using contrastive learning.
The approach uses:
- Video Encoder: Processes videos to create embeddings via frame extraction + image encoding
- Text Encoder: Processes search queries to create embeddings using BERT
- Contrastive Training: For each video, we have 1 positive query + (n-1) negative queries
- Loss: Dot product similarities → softmax → cross-entropy loss

The model learns to maximize similarity between videos and relevant queries while
minimizing similarity with irrelevant queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import List, Tuple, Dict
from tqdm import tqdm
from .config import VideoSearchConfig
from ..utils.data import VideoQueryTriplet
from .encoders import TextEncoder, VideoEncoder
from ..utils.video import VideoProcessor


class VideoTextContrastiveModel(nn.Module):
    """Main contrastive learning model combining video and text encoders"""

    def __init__(
        self,
        video_embed_dim: int = 512,
        text_embed_dim: int = 512,
        text_model_name: str = "bert-base-uncased",
        temperature: float = 0.07,
    ):
        super().__init__()

        self.video_encoder = VideoEncoder(embedding_dim=video_embed_dim)
        self.text_encoder = TextEncoder(
            model_name=text_model_name, embedding_dim=text_embed_dim
        )

        self.temperature = temperature
        self.video_processor = VideoProcessor()

    def encode_video(self, video_path: str) -> torch.Tensor:
        """Encode a single video to embedding"""
        frames = self.video_processor.extract_frames(video_path)
        frames = frames.unsqueeze(0)  # Add batch dimension
        video_embedding = self.video_encoder(frames)
        return video_embedding.squeeze(0)  # Remove batch dimension

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts to embeddings"""
        return self.text_encoder.encode_queries(texts)

    def compute_similarities(
        self, video_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between videos and texts
        Args:
            video_embeddings: [batch_size, embed_dim]
            text_embeddings: [batch_size, num_texts, embed_dim]
        Returns:
            similarities: [batch_size, num_texts]
        """
        # Expand video embeddings to match text dimensions
        video_expanded = video_embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Compute dot product similarities
        similarities = torch.sum(
            video_expanded * text_embeddings, dim=2
        )  # [batch_size, num_texts]

        # Apply temperature scaling
        similarities = similarities / self.temperature

        return similarities

    def contrastive_loss(
        self, similarities: torch.Tensor, target_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        Args:
            similarities: [batch_size, num_texts] - similarity scores
            target_indices: [batch_size] - index of positive text for each video (usually 0)
        Returns:
            loss: scalar tensor
        """
        # Apply softmax to get probabilities
        log_probs = F.log_softmax(similarities, dim=1)

        # Select the log probability of the positive text
        positive_log_probs = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        # Negative log likelihood loss
        loss = -positive_log_probs.mean()

        return loss

    def forward(
        self,
        video_frames: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training
        Args:
            video_frames: [batch_size, num_frames, 3, H, W]
            text_input_ids: [batch_size, num_texts, seq_len]
            text_attention_mask: [batch_size, num_texts, seq_len]
            target_indices: [batch_size] - index of positive text
        Returns:
            loss: scalar tensor
            similarities: [batch_size, num_texts]
        """
        batch_size, num_texts, seq_len = text_input_ids.shape

        # Encode videos
        video_embeddings = self.video_encoder(video_frames)  # [batch_size, embed_dim]

        # Encode texts - reshape for batch processing
        text_input_flat = text_input_ids.view(batch_size * num_texts, seq_len)
        text_mask_flat = text_attention_mask.view(batch_size * num_texts, seq_len)

        text_embeddings_flat = self.text_encoder(
            text_input_flat, text_mask_flat
        )  # [batch_size * num_texts, embed_dim]
        text_embeddings = text_embeddings_flat.view(
            batch_size, num_texts, -1
        )  # [batch_size, num_texts, embed_dim]

        # Compute similarities
        similarities = self.compute_similarities(video_embeddings, text_embeddings)

        # Compute loss
        loss = self.contrastive_loss(similarities, target_indices)

        return loss, similarities


def train_video_search_model(
    model: VideoTextContrastiveModel, dataloader: DataLoader, config: VideoSearchConfig
) -> List[float]:
    """
    Train the video search model

    Args:
        model: The VideoTextContrastiveModel to train
        dataloader: DataLoader with video-query triplets
        config: Configuration object with training parameters

    Returns:
        List of losses per epoch
    """

    # Move model to device
    model.to(config.device)
    model.train()

    # Optimizer - only train the projection layers, freeze pretrained parts for now
    optimizer = torch.optim.Adam(
        [
            {"params": model.video_encoder.projection.parameters()},
            {"params": model.text_encoder.projection.parameters()},
        ],
        lr=config.learning_rate,
    )

    # Track training progress
    epoch_losses = []

    print(f"Training on device: {config.device}")
    print(f"Training for {config.num_epochs} epochs...")

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            video_frames = batch["video_frames"].to(config.device)
            text_input_ids = batch["text_input_ids"].to(config.device)
            text_attention_mask = batch["text_attention_mask"].to(config.device)
            target_indices = batch["target_index"].to(config.device)

            # Forward pass
            optimizer.zero_grad()
            loss, similarities = model(
                video_frames, text_input_ids, text_attention_mask, target_indices
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted_indices = similarities.argmax(dim=1)
            correct = (predicted_indices == target_indices).sum().item()
            correct_predictions += correct
            total_predictions += len(target_indices)

            # Update metrics
            total_loss += loss.item()
            current_accuracy = correct_predictions / total_predictions

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_accuracy:.3f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

        # Calculate epoch metrics
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        epoch_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{config.num_epochs}:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_accuracy:.3f}")
        print(f"  Correct: {correct_predictions}/{total_predictions}")
        print()

    return epoch_losses


def evaluate_retrieval(
    model: VideoTextContrastiveModel,
    test_triplets: List[VideoQueryTriplet],
    device: str = "cpu",
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate video-text retrieval performance

    Args:
        model: Trained VideoTextContrastiveModel
        test_triplets: List of test video-query triplets
        device: Device for inference
        top_k: Number of top results to consider for ranking metrics

    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()

    correct_at_1 = 0
    correct_at_k = 0
    total_queries = 0

    with torch.no_grad():
        for triplet in tqdm(test_triplets, desc="Evaluating"):
            # Encode video
            video_frames = model.video_processor.extract_frames(triplet.video_path)
            video_frames = video_frames.unsqueeze(0).to(device)  # Add batch dim
            video_embedding = model.video_encoder(video_frames)  # [1, embed_dim]

            # Get all queries
            all_queries = triplet.get_all_queries()
            query_embeddings = model.encode_texts(
                all_queries
            )  # [num_queries, embed_dim]
            query_embeddings = query_embeddings.to(device)

            # Compute similarities
            similarities = torch.matmul(video_embedding, query_embeddings.T).squeeze(
                0
            )  # [num_queries]

            # Get rankings
            _, ranked_indices = similarities.sort(descending=True)

            # Check if positive query (index 0) is in top-k
            positive_rank = (ranked_indices == 0).nonzero(as_tuple=True)[0].item()

            if positive_rank == 0:  # Top-1
                correct_at_1 += 1
                correct_at_k += 1
            elif positive_rank < top_k:  # Top-k
                correct_at_k += 1

            total_queries += 1

    metrics = {
        "accuracy_at_1": correct_at_1 / total_queries,
        f"accuracy_at_{top_k}": correct_at_k / total_queries,
        "total_queries": total_queries,
    }

    return metrics

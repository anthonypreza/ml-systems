import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50

from typing import List, Dict
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """BERT-based text encoder for search queries"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 512,
        freeze_bert: bool = False,
        max_length: int = 128,
    ):
        super().__init__()

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1),
        )

    def encode_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and encode a list of texts"""
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through BERT and projection layer"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0]

        # Project to desired dimension
        text_embedding = self.projection(cls_embedding)

        # L2 normalize for cosine similarity
        text_embedding = F.normalize(text_embedding, p=2, dim=1)

        return text_embedding

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Convenience method to encode queries end-to-end"""
        encoding = self.encode_text(queries)
        return self.forward(encoding["input_ids"], encoding["attention_mask"])


class VideoEncoder(nn.Module):
    """ResNet50-based video encoder with temporal aggregation"""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()

        # Use ResNet50 as image encoder for video frames
        self.image_encoder = resnet50(pretrained=pretrained)

        # Remove the final classification layer
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])

        # Add projection layer from ResNet features to desired embedding dim
        resnet_feature_dim = 2048  # ResNet50 feature dimension
        self.projection = nn.Sequential(
            nn.Linear(resnet_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1),
        )

        # Temporal aggregation - simple average pooling for now
        self.temporal_aggregation = "mean"

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to embeddings
        Args:
            frames: [batch_size, num_frames, 3, H, W]
        Returns:
            video_embeddings: [batch_size, embedding_dim]
        """
        batch_size, num_frames, c, h, w = frames.shape

        # Reshape to process all frames at once
        frames_flat = frames.view(batch_size * num_frames, c, h, w)

        # Extract features from all frames
        with torch.no_grad():
            frame_features = self.image_encoder(
                frames_flat
            )  # [batch_size * num_frames, 2048, 1, 1]

        # Remove spatial dimensions and reshape back
        frame_features = frame_features.view(
            batch_size, num_frames, -1
        )  # [batch_size, num_frames, 2048]

        # Apply projection to each frame
        frame_embeddings = self.projection(
            frame_features
        )  # [batch_size, num_frames, embedding_dim]

        # Temporal aggregation - average pooling
        if self.temporal_aggregation == "mean":
            # Mask out padding frames (all zeros)
            frame_mask = (
                frames.sum(dim=[2, 3, 4]) != 0
            ).float()  # [batch_size, num_frames]
            frame_mask = frame_mask.unsqueeze(-1)  # [batch_size, num_frames, 1]

            # Masked average
            masked_embeddings = frame_embeddings * frame_mask
            video_embedding = masked_embeddings.sum(dim=1) / (
                frame_mask.sum(dim=1) + 1e-8
            )
        else:
            # Simple average
            video_embedding = frame_embeddings.mean(dim=1)

        # L2 normalize for cosine similarity
        video_embedding = F.normalize(video_embedding, p=2, dim=1)

        return video_embedding

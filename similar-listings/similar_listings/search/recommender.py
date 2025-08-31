"""Recommendation engine for finding similar listings."""

import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from ..core.model import ListingEmbeddingModel
from ..core.config import Config


class SimilarListingsRecommender:
    """Recommendation engine for finding similar listings."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        mappings_path: Optional[str] = None,
    ):
        """
        Initialize recommender with trained model and embeddings.

        Args:
            model_path: Path to trained PyTorch model
            embeddings_path: Path to precomputed embeddings numpy file
            mappings_path: Path to listing ID mappings JSON file
        """
        if model_path is None:
            model_path = Config.MODEL_FILE
        if embeddings_path is None:
            embeddings_path = Config.EMBEDDINGS_FILE
        if mappings_path is None:
            mappings_path = Config.MAPPINGS_FILE

        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.mappings_path = mappings_path

        # Load model and embeddings
        self._load_model()
        self._load_embeddings()
        self._load_mappings()

    def _load_model(self):
        """Load the trained PyTorch model."""
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

        self.model = ListingEmbeddingModel(
            num_listings=checkpoint["num_listings"],
            embedding_dim=checkpoint["embedding_dim"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _load_embeddings(self):
        """Load precomputed embeddings."""
        self.embeddings = np.load(self.embeddings_path)

    def _load_mappings(self):
        """Load listing ID to index mappings."""
        with open(self.mappings_path, "r") as f:
            mappings = json.load(f)

        self.listing_to_idx = mappings["listing_to_idx"]
        self.idx_to_listing = {int(k): v for k, v in mappings["idx_to_listing"].items()}

    def get_similar_listings(
        self, listing_id: str, top_k: Optional[int] = None, exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get similar listings for a given listing ID.

        Args:
            listing_id: ID of the query listing
            top_k: Number of similar listings to return
            exclude_self: Whether to exclude the query listing from results

        Returns:
            List of similar listings with similarity scores
        """
        if top_k is None:
            top_k = Config.TOP_K

        if listing_id not in self.listing_to_idx:
            raise ValueError(f"Listing {listing_id} not found in trained embeddings")

        # Get query embedding
        query_idx = self.listing_to_idx[listing_id]
        query_embedding = self.embeddings[query_idx]

        # Compute similarities to all listings
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k most similar listings
        if exclude_self:
            # Set self-similarity to negative infinity to exclude from top-k
            similarities[query_idx] = -np.inf

        top_indices = np.argsort(-similarities)[:top_k]

        # Format results
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:  # Skip self if excluded
                continue

            similar_listing_id = self.idx_to_listing[idx]
            similarity_score = float(similarities[idx])

            results.append(
                {"listing_id": similar_listing_id, "similarity_score": similarity_score}
            )

        return results

    def get_recommendations_for_session(
        self, clicked_listing_ids: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on a list of clicked listings (session-based).

        Args:
            clicked_listing_ids: List of listing IDs that were clicked/viewed
            top_k: Number of recommendations to return

        Returns:
            List of recommended listings with similarity scores
        """
        if top_k is None:
            top_k = Config.TOP_K

        # Filter out listings not in our trained embeddings
        valid_clicked_ids = [
            lid for lid in clicked_listing_ids if lid in self.listing_to_idx
        ]

        if not valid_clicked_ids:
            raise ValueError("None of the clicked listings found in trained embeddings")

        # Get embeddings for clicked listings
        clicked_indices = [self.listing_to_idx[lid] for lid in valid_clicked_ids]
        clicked_embeddings = self.embeddings[clicked_indices]

        # Compute average embedding as query
        query_embedding = np.mean(clicked_embeddings, axis=0)

        # Compute similarities to all listings
        similarities = np.dot(self.embeddings, query_embedding)

        # Exclude clicked listings from recommendations
        for idx in clicked_indices:
            similarities[idx] = -np.inf

        # Get top-k recommendations
        top_indices = np.argsort(-similarities)[:top_k]

        # Format results
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:  # Skip excluded listings
                continue

            recommended_listing_id = self.idx_to_listing[idx]
            similarity_score = float(similarities[idx])

            results.append(
                {
                    "listing_id": recommended_listing_id,
                    "similarity_score": similarity_score,
                }
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded model and embeddings."""
        return {
            "num_listings": len(self.listing_to_idx),
            "embedding_dimension": self.embeddings.shape[1],
            "model_path": self.model_path,
            "embeddings_shape": self.embeddings.shape,
        }

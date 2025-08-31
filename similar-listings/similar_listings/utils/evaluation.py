"""Evaluation utilities for ranking metrics."""

import numpy as np
from typing import List, Dict, Any


class RankingEvaluator:
    """Evaluate embeddings using ranking metrics."""

    def __init__(
        self,
        sessions: List[Dict],
        listings: List[Dict],
        listing_to_idx: Dict[str, int],
        idx_to_listing: Dict[int, str],
    ):
        self.sessions = sessions
        self.listings = listings
        self.listing_to_idx = listing_to_idx
        self.idx_to_listing = idx_to_listing

    def evaluate_ranking(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Evaluate embeddings using average rank of booked listing.

        For each session, rank all listings by similarity to clicked listings
        and measure where the booked listing appears in the ranking.
        """
        ranks = []

        for session in self.sessions:
            clicked_ids = session["clicked_listing_ids"]
            booked_id = session["booked_listing_id"]

            if not clicked_ids:
                continue

            # Get embeddings for clicked listings
            clicked_indices = [
                self.listing_to_idx[lid]
                for lid in clicked_ids
                if lid in self.listing_to_idx
            ]
            booked_idx = self.listing_to_idx.get(booked_id)

            if not clicked_indices or booked_idx is None:
                continue

            # Compute average embedding of clicked listings
            clicked_embeddings = embeddings[clicked_indices]
            query_embedding = np.mean(clicked_embeddings, axis=0)

            # Compute similarities to all listings
            similarities = np.dot(embeddings, query_embedding)

            # Rank listings by similarity (descending order)
            ranked_indices = np.argsort(-similarities)

            # Find rank of booked listing (1-indexed)
            booked_rank = np.where(ranked_indices == booked_idx)[0][0] + 1
            ranks.append(booked_rank)

        if not ranks:
            return {"mean_rank": float("inf"), "median_rank": float("inf"), "mrr": 0.0}

        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        mrr = np.mean([1.0 / rank for rank in ranks])  # Mean Reciprocal Rank

        return {
            "mean_rank": float(mean_rank),
            "median_rank": float(median_rank),
            "mrr": float(mrr),
            "total_sessions_evaluated": len(ranks),
        }


def print_triplet_stats(triplets: List[Dict[str, Any]]) -> None:
    """Print statistics about generated triplets."""
    from collections import defaultdict

    if not triplets:
        print("No triplets generated.")
        return

    # Count by type and label
    type_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for triplet in triplets:
        type_counts[triplet["triplet_type"]] += 1
        label_counts[triplet["label"]] += 1

    print(f"\n=== Triplet Statistics ===")
    print(f"Total triplets: {len(triplets)}")
    print(f"Positive samples: {label_counts[1]}")
    print(f"Negative samples: {label_counts[0]}")
    print(f"Positive ratio: {label_counts[1] / len(triplets):.3f}")

    print(f"\nBy triplet type:")
    for triplet_type, count in type_counts.items():
        print(f"  {triplet_type}: {count}")

    # Show sample triplets
    print(f"\n=== Sample Triplets ===")
    for i, triplet in enumerate(triplets[:5]):
        label_str = "POSITIVE" if triplet["label"] == 1 else "NEGATIVE"
        print(f"Triplet {i + 1} [{label_str} - {triplet['triplet_type']}]:")
        print(f"  Central: {triplet['central_listing'][:8]}...")
        print(f"  Context: {triplet['context_listing'][:8]}...")


def print_session_stats(sessions: List[Dict[str, Any]]) -> None:
    """Print statistics about extracted sessions."""
    if not sessions:
        print("No sessions extracted.")
        return

    total_sessions = len(sessions)
    total_clicks = sum(session["num_clicks"] for session in sessions)
    avg_clicks = total_clicks / total_sessions if total_sessions > 0 else 0
    avg_duration = (
        sum(session["session_duration"] for session in sessions) / total_sessions
    )

    # Click distribution
    click_counts = [session["num_clicks"] for session in sessions]
    min_clicks = min(click_counts)
    max_clicks = max(click_counts)

    print(f"\n=== Session Statistics ===")
    print(f"Total sessions: {total_sessions}")
    print(f"Total clicks: {total_clicks}")
    print(f"Average clicks per session: {avg_clicks:.1f}")
    print(f"Click range: {min_clicks} - {max_clicks}")
    print(f"Average session duration: {avg_duration / 60:.1f} minutes")

    # Show sample sessions
    print(f"\n=== Sample Sessions ===")
    for i, session in enumerate(sessions[:3]):
        print(f"Session {i + 1}:")
        print(f"  Clicked: {len(session['clicked_listing_ids'])} listings")
        print(f"  Booked: {session['booked_listing_id'][:8]}...")
        print(f"  Duration: {session['session_duration'] / 60:.1f} minutes")

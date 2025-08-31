#!/usr/bin/env python3
"""Interactive recommendation script using trained embeddings."""

import os
import sys
import argparse
import json
from typing import Optional

# Add the parent directory to Python path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similar_listings.search.recommender import SimilarListingsRecommender
from similar_listings.utils.data import load_listings
from similar_listings.core.config import Config


def format_listing_info(
    listing_id: str, listings_data: list, similarity_score: Optional[float] = None
) -> str:
    """Format listing information for display."""
    # Find listing details
    listing = None
    for ld in listings_data:
        if ld["id"] == listing_id:
            listing = ld
            break

    if listing is None:
        return f"Listing {listing_id[:8]}... (details not found)"

    info = f"Listing {listing_id[:8]}..."
    info += (
        f"\n  Type: {listing['type']}, Beds: {listing['beds']}, City: {listing['city']}"
    )
    info += f"\n  Price: ${listing['price']:.2f}, Sq Ft: {listing['sq_ft']}, Rate: {listing['rate']}"

    if similarity_score is not None:
        info += f"\n  Similarity: {similarity_score:.4f}"

    return info


def main():
    parser = argparse.ArgumentParser(description="Get recommendations for listings")
    parser.add_argument(
        "--listing-id", type=str, help="Listing ID to find similar listings for"
    )
    parser.add_argument(
        "--session-listings",
        type=str,
        nargs="+",
        help="List of listing IDs for session-based recommendations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=Config.TOP_K,
        help="Number of recommendations to return",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained model files",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing data files"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args()

    if not args.listing_id and not args.session_listings:
        parser.error("Must provide either --listing-id or --session-listings")

    # Load recommender
    try:
        model_path = os.path.join(args.model_dir, "best_embedding_model.pth")
        embeddings_path = os.path.join(args.model_dir, "listing_embeddings.npy")
        mappings_path = os.path.join(args.model_dir, "listing_mappings.json")

        recommender = SimilarListingsRecommender(
            model_path=model_path,
            embeddings_path=embeddings_path,
            mappings_path=mappings_path,
        )
        print(
            f"Loaded recommender with {recommender.get_stats()['num_listings']} listings"
        )

    except FileNotFoundError as e:
        print(f"Error loading recommender: {e}")
        print("Make sure you have trained the model first using train_modular.py")
        return

    # Load listings data for display
    try:
        listings_file = os.path.join(args.data_dir, "listings.json")
        listings_data = load_listings(listings_file)
    except FileNotFoundError:
        print("Warning: Could not load listings data for detailed display")
        listings_data = []

    # Get recommendations
    try:
        if args.listing_id:
            # Single listing similarity
            print(f"Finding similar listings to: {args.listing_id[:8]}...")
            if listings_data:
                print(format_listing_info(args.listing_id, listings_data))

            recommendations = recommender.get_similar_listings(
                args.listing_id, top_k=args.top_k
            )

        else:
            # Session-based recommendations
            print(
                f"Getting recommendations for session with {len(args.session_listings)} listings..."
            )
            if listings_data:
                for lid in args.session_listings[:3]:  # Show first 3
                    print(format_listing_info(lid, listings_data))
                if len(args.session_listings) > 3:
                    print(f"... and {len(args.session_listings) - 3} more")

            recommendations = recommender.get_recommendations_for_session(
                args.session_listings, top_k=args.top_k
            )

        # Output results
        if args.format == "json":
            print(json.dumps(recommendations, indent=2))
        else:
            print(f"\n=== Top {len(recommendations)} Recommendations ===")
            for i, rec in enumerate(recommendations, 1):
                print(
                    f"\n{i}. {format_listing_info(rec['listing_id'], listings_data, rec['similarity_score'])}"
                )

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

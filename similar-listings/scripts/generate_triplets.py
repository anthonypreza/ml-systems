#!/usr/bin/env python3
"""Generate training triplets from session data for embedding learning."""

import os
import sys
import argparse

# Add the parent directory to Python path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similar_listings.utils.data import (
    TripletGenerator,
    load_sessions,
    load_listings,
    save_data,
)
from similar_listings.utils.evaluation import print_triplet_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate training triplets from session data"
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="data/sessions.json",
        help="Input sessions JSON file",
    )
    parser.add_argument(
        "--listings",
        type=str,
        default="data/listings.json",
        help="Input listings JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/triplets.json",
        help="Output triplets JSON file",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Sliding window size for context extraction",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=5,
        help="Number of negative samples per positive sample",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generator = TripletGenerator(
        window_size=args.window_size, neg_sampling_ratio=args.neg_ratio, seed=args.seed
    )

    print(f"Loading data...")
    sessions = load_sessions(args.sessions)
    listings = load_listings(args.listings)
    print(f"Loaded {len(sessions)} sessions and {len(listings)} listings")

    print(f"Generating triplets...")
    triplets = generator.generate_triplets(sessions, listings)

    print_triplet_stats(triplets)
    save_data(triplets, args.output)

    print("Triplet generation completed!")


if __name__ == "__main__":
    main()

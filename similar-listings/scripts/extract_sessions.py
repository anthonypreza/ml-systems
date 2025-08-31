#!/usr/bin/env python3
"""Extract search sessions from interaction data for similar listings recommendation."""

import os
import sys
import argparse

# Add the parent directory to Python path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similar_listings.utils.data import SessionExtractor, load_interactions, save_data
from similar_listings.utils.evaluation import print_session_stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract search sessions from interaction data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interactions.json",
        help="Input interactions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sessions.json",
        help="Output sessions JSON file",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=3600,
        help="Maximum time gap (seconds) between interactions in same session",
    )

    args = parser.parse_args()

    print(f"Loading interactions from {args.input}...")
    interactions = load_interactions(args.input)

    print(f"Extracting sessions from {len(interactions)} interactions...")
    extractor = SessionExtractor(max_session_gap=args.max_gap)
    sessions = extractor.extract_sessions(interactions)

    print_session_stats(sessions)
    save_data(sessions, args.output)

    print("Session extraction completed!")


if __name__ == "__main__":
    main()

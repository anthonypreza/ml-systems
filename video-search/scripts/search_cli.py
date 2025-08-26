#!/usr/bin/env python3
"""
Command Line Video Search

Search videos using a single query from the command line.
"""

import os
import pickle
import torch
import argparse
from typing import List, Tuple

import sys
import os

# Ensure the project root (containing the `video_search` package) is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from video_search.core.config import VideoSearchConfig
from video_search.core.model import VideoTextContrastiveModel
from video_search.search.search import search_videos_indexed


def load_model_and_index(
    config: VideoSearchConfig,
) -> Tuple[VideoTextContrastiveModel, dict]:
    """Load trained model and video index from saved files"""

    # Load model
    # Backward-compat: alias legacy module names referenced in old checkpoints
    import types
    try:
        from video_search.core.config import VideoSearchConfig as _VSC
        legacy_cfg_mod = types.ModuleType("config")
        legacy_cfg_mod.VideoSearchConfig = _VSC
        sys.modules.setdefault("config", legacy_cfg_mod)
    except Exception:
        pass

    checkpoint = torch.load(
        config.model_save_path, map_location=config.device, weights_only=False
    )
    model = VideoTextContrastiveModel(
        video_embed_dim=config.video_embed_dim,
        text_embed_dim=config.text_embed_dim,
        text_model_name=config.text_model_name,
        temperature=config.temperature,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    # Load video index
    with open(config.index_save_path, "rb") as f:
        video_index = pickle.load(f)

    return model, video_index


def search_videos_cli(
    query: str, top_k: int = 5, verbose: bool = True
) -> List[Tuple[str, float]]:
    """Search videos using command line query"""

    config = VideoSearchConfig()

    # Check if files exist
    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(
            f"Model not found: {config.model_save_path}. Run training first."
        )
    if not os.path.exists(config.index_save_path):
        raise FileNotFoundError(
            f"Index not found: {config.index_save_path}. Run training first."
        )

    if verbose:
        print(f"Loading model and index...")

    model, video_index = load_model_and_index(config)

    if verbose:
        print(f"Searching for: '{query}'")

    # Perform search
    results = search_videos_indexed(
        model=model, query=query, video_index=video_index, top_k=top_k
    )

    return results


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Command Line Video Search")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "simple"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode - minimal output"
    )

    args = parser.parse_args()

    try:
        # Perform search
        results = search_videos_cli(
            args.query, top_k=args.top_k, verbose=not args.quiet
        )

        if not results:
            if not args.quiet:
                print("No results found.")
            return

        # Format output
        if args.format == "json":
            import json

            output = {
                "query": args.query,
                "results": [
                    {
                        "path": path,
                        "score": float(score),
                        "filename": os.path.basename(path),
                    }
                    for path, score in results
                ],
            }
            print(json.dumps(output, indent=2))

        elif args.format == "simple":
            for path, score in results:
                print(f"{score:.3f}\t{path}")

        else:  # table format
            if not args.quiet:
                print(f"\nResults for '{args.query}':")
                print("-" * 60)

            for i, (path, score) in enumerate(results, 1):
                filename = os.path.basename(path)
                print(f"{i:2d}. {filename:<30} (score: {score:.3f})")
                if not args.quiet:
                    print(f"    {path}")

    except Exception as e:
        if args.quiet:
            print(f"Error: {e}")
        else:
            print(f"❌ Error: {e}")
            print("💡 Make sure you've run training first: python pipeline.py")


if __name__ == "__main__":
    main()

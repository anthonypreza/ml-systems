#!/usr/bin/env python3
"""
Build Video Index from Trained Model

Create a video search index from an already trained model without retraining.
"""

import os
import pickle
import torch
import argparse
import sys

# Ensure the project root (containing the `video_search` package) is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from video_search.core.config import VideoSearchConfig
from video_search.core.model import VideoTextContrastiveModel
from video_search.utils.data import load_msrvtt_dataset, create_synthetic_dataset
from video_search.search.index import build_video_index


def load_trained_model(config: VideoSearchConfig) -> VideoTextContrastiveModel:
    """Load a trained model from saved checkpoint"""

    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(f"Model file not found: {config.model_save_path}")

    print(f"Loading trained model from {config.model_save_path}...")

    checkpoint = torch.load(
        config.model_save_path, map_location=config.device, weights_only=False
    )

    # Initialize model with same configuration
    model = VideoTextContrastiveModel(
        video_embed_dim=config.video_embed_dim,
        text_embed_dim=config.text_embed_dim,
        text_model_name=config.text_model_name,
        temperature=config.temperature,
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    dataset_name = checkpoint.get("dataset_name", "unknown")
    print(f"Model loaded successfully (trained on {dataset_name} dataset)")

    return model


def main():
    """Main function to build index from trained model"""
    parser = argparse.ArgumentParser(description="Build Video Index from Trained Model")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "msrvtt"],
        default="synthetic",
        help="Dataset to build index for (default: synthetic)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=100,
        help="Maximum number of videos from MSR-VTT dataset (default: 100)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom path to trained model (overrides config default)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        help="Custom path to save index (overrides config default)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing index file"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = VideoSearchConfig()

    # Override paths if provided
    if args.model_path:
        config.model_save_path = args.model_path
    if args.index_path:
        config.index_save_path = args.index_path

    try:
        # Check if index already exists
        if os.path.exists(config.index_save_path) and not args.force:
            print(f"❌ Index already exists: {config.index_save_path}")
            print("💡 Use --force to overwrite, or specify different --index-path")
            return

        # Load trained model
        model = load_trained_model(config)

        # Load dataset to get video paths
        print(f"Loading {args.dataset} dataset to get video paths...")
        if args.dataset == "msrvtt":
            triplets = load_msrvtt_dataset(
                data_dir="./data/msrvtt",
                max_videos=args.max_videos,
                num_negatives=7,  # Doesn't matter for index building
            )
            dataset_name = "MSR-VTT"
        else:
            triplets = create_synthetic_dataset(
                num_videos=20,
                num_negatives=7,  # Doesn't matter for index building
            )
            dataset_name = "synthetic"

        # Get unique video paths
        video_paths = list(set(triplet.video_path for triplet in triplets))
        print(f"Found {len(video_paths)} unique videos from {dataset_name} dataset")

        # Build video index
        print("Building video index...")
        video_index = build_video_index(
            model=model, video_paths=video_paths, device=config.device
        )

        # Save video index
        os.makedirs(os.path.dirname(config.index_save_path), exist_ok=True)
        with open(config.index_save_path, "wb") as f:
            pickle.dump(video_index, f)
        print(f"✅ Video index saved to {config.index_save_path}")

        # Save video paths for reference
        with open(config.video_paths_save_path, "w") as f:
            for video_path in video_paths:
                f.write(f"{video_path}\n")
        print(f"✅ Video paths saved to {config.video_paths_save_path}")

        print("\n🎉 Index building completed!")
        print("📊 Indexed {len(video_index['paths'])} videos successfully")
        print("🔍 You can now use search_interactive.py or search_cli.py")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure you've trained a model first:")
        print("   python pipeline.py --dataset msrvtt --max-videos 100")
        print("   python pipeline.py --dataset synthetic")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()

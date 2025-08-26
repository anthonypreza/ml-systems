from torch.utils.data import DataLoader

import os
import argparse

import sys
import os

# Ensure the project root (containing the `video_search` package) is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from video_search.core.config import VideoSearchConfig
from video_search.utils.video import VideoProcessor
from video_search.core.encoders import TextEncoder
from video_search.utils.data import (
    VideoQueryDataset,
    create_synthetic_dataset,
    load_msrvtt_dataset,
)
from video_search.search.index import build_video_index
from video_search.core.model import (
    VideoTextContrastiveModel,
    train_video_search_model,
    evaluate_retrieval,
)
from video_search.search.search import search_videos_indexed


def main(use_msrvtt: bool = False, max_videos: int = 100, skip_training: bool = False):
    """Main function for training video search pipeline

    Args:
        use_msrvtt: Whether to use MSR-VTT dataset (True) or synthetic data (False)
        max_videos: Maximum number of videos to process from MSR-VTT
        skip_training: Whether to skip training and just build index from existing model
    """

    # Initialize configuration
    config = VideoSearchConfig()
    print(f"Configuration: {vars(config)}")
    print()

    # Create dataset
    if use_msrvtt:
        triplets = load_msrvtt_dataset(
            data_dir="./data/msrvtt",
            max_videos=max_videos,
            num_negatives=config.num_negatives,
        )
        dataset_name = "MSR-VTT"
    else:
        print("Creating synthetic dataset...")
        triplets = create_synthetic_dataset(
            num_videos=20, num_negatives=config.num_negatives
        )
        dataset_name = "synthetic"

    print(f"Created {len(triplets)} training triplets from {dataset_name} dataset")
    print("\nExample triplet:")
    example = triplets[0]
    print(f"Video: {os.path.basename(example.video_path)}")
    print(f"Positive: {example.positive_query}")
    print(f"Negatives: {example.negative_queries[:3]}...")
    print()

    # Create dataset and dataloader
    video_processor = VideoProcessor(
        max_frames=config.max_frames, frame_size=config.frame_size
    )

    # Initialize tokenizer (we'll get it from the model's text encoder)
    temp_text_encoder = TextEncoder(
        model_name=config.text_model_name, embedding_dim=config.text_embed_dim
    )
    tokenizer = temp_text_encoder.tokenizer

    dataset = VideoQueryDataset(
        triplets=triplets,
        video_processor=video_processor,
        tokenizer=tokenizer,
        max_text_length=config.max_text_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with videos
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"DataLoader batch size: {config.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print()

    # Initialize model
    model = VideoTextContrastiveModel(
        video_embed_dim=config.video_embed_dim,
        text_embed_dim=config.text_embed_dim,
        text_model_name=config.text_model_name,
        temperature=config.temperature,
    )

    print("VideoTextContrastiveModel initialized")
    print(
        f"Video encoder params: {sum(p.numel() for p in model.video_encoder.parameters()):,}"
    )
    print(
        f"Text encoder params: {sum(p.numel() for p in model.text_encoder.parameters()):,}"
    )
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Handle skip training or normal training flow
    if skip_training:
        # Load existing trained model
        if not os.path.exists(config.model_save_path):
            print(f"❌ No trained model found at {config.model_save_path}")
            print("💡 Train a model first or remove --skip-training flag")
            return

        print("Loading existing trained model...")
        import torch

        checkpoint = torch.load(
            config.model_save_path, map_location=config.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.device)
        model.eval()
        print("✅ Model loaded successfully")
    else:
        # Train the model
        print("Starting training...")
        train_losses = train_video_search_model(model, dataloader, config)

        print("Training completed!")
        print(f"Final loss: {train_losses[-1]:.4f}")
        print()

        # Evaluate the model
        print("Evaluating model...")
        test_triplets = triplets[:5]  # Use first 5 triplets as test set

        eval_metrics = evaluate_retrieval(
            model=model, test_triplets=test_triplets, device=config.device, top_k=3
        )

        print("Evaluation Results:")
        for metric, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        print()

        # Save trained model
        print("Saving trained model...")
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

        # Save model state dict
        import torch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "dataset_name": dataset_name,
            },
            config.model_save_path,
        )
        print(f"Model saved to {config.model_save_path}")

    # Get all video paths
    video_paths = list(set(triplet.video_path for triplet in triplets))

    # Save video paths for later reference
    with open(config.video_paths_save_path, "w") as f:
        for video_path in video_paths:
            f.write(f"{video_path}\n")
    print(f"Video paths saved to {config.video_paths_save_path}")

    # Always create and save video index for search
    print("Creating and saving video index...")
    video_index = build_video_index(
        model=model,
        video_paths=video_paths,
    )

    # Save video index
    import pickle

    with open(config.index_save_path, "wb") as f:
        pickle.dump(video_index, f)
    print(f"Video index saved to {config.index_save_path}")

    # Test search with the saved index
    print("\nTesting search with saved index...")
    search_query = "cat playing"
    search_results = search_videos_indexed(
        model=model, query=search_query, video_index=video_index, top_k=5
    )

    print(f"Search Results for '{search_query}':")
    for i, (video_path, score) in enumerate(search_results, 1):
        video_name = os.path.basename(video_path)
        print(f"  {i}. {video_name} (score: {score:.3f})")

    print(
        f"\nTraining completed! Use search_interactive.py to search with saved model and index."
    )
    print(f"Model: {config.model_save_path}")
    print(f"Index: {config.index_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Search Training Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "msrvtt"],
        default="synthetic",
        help="Dataset to use for training (default: synthetic)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=100,
        help="Maximum number of videos to use from MSR-VTT dataset (default: 100)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and build index from existing trained model",
    )
    args = parser.parse_args()

    use_msrvtt = args.dataset == "msrvtt"
    main(
        use_msrvtt=use_msrvtt,
        max_videos=args.max_videos,
        skip_training=args.skip_training,
    )

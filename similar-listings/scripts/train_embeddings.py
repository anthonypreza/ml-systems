#!/usr/bin/env python3
"""Modular training script using the similar_listings package."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json

# Add the parent directory to Python path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similar_listings.core.config import Config
from similar_listings.core.model import (
    ListingEmbeddingModel,
    TripletDataset,
    train_epoch,
    evaluate,
)
from similar_listings.utils.data import (
    load_all_data,
    create_listing_mappings,
    split_triplets,
)
from similar_listings.utils.evaluation import RankingEvaluator
import random


def main():
    parser = argparse.ArgumentParser(
        description="Train listing embeddings (modular version)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for saved models",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=Config.EMBEDDING_DIM,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=Config.EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--dropout", type=float, default=Config.DROPOUT_RATE, help="Dropout rate"
    )
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data using modular loaders
    print("Loading data...")
    try:
        data = load_all_data(args.data_dir)
        triplets = data["triplets"]
        sessions = data["sessions"]
        listings = data["listings"]
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure you have generated the required data files first.")
        return

    print(
        f"Loaded {len(triplets)} triplets, {len(sessions)} sessions, {len(listings)} listings"
    )

    # Create mappings
    listing_to_idx, idx_to_listing = create_listing_mappings(triplets, listings)
    num_listings = len(listing_to_idx)
    print(f"Total unique listings: {num_listings}")

    # Split data
    train_triplets, val_triplets = split_triplets(triplets, Config.TRAIN_RATIO)
    print(f"Train: {len(train_triplets)}, Validation: {len(val_triplets)}")

    # Create datasets and dataloaders
    train_dataset = TripletDataset(train_triplets, listing_to_idx)
    val_dataset = TripletDataset(val_triplets, listing_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ListingEmbeddingModel(num_listings, args.embedding_dim, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize ranking evaluator
    evaluator = RankingEvaluator(sessions, listings, listing_to_idx, idx_to_listing)

    # Training loop with early stopping
    best_mrr = 0.0
    patience = 5  # Stop if no improvement for 5 epochs
    patience_counter = 0
    min_improvement = 0.001  # Minimum MRR improvement to reset patience

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer)

        # Validate
        val_metrics = evaluate(model, val_loader)

        # Evaluate ranking
        embeddings = model.get_embeddings()
        ranking_metrics = evaluator.evaluate_ranking(embeddings)

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
        )
        print(
            f"Mean Rank: {ranking_metrics['mean_rank']:.2f}, "
            f"Median Rank: {ranking_metrics['median_rank']:.2f}, "
            f"MRR: {ranking_metrics['mrr']:.4f}"
        )

        # Early stopping and model saving logic
        current_mrr = ranking_metrics["mrr"]

        if current_mrr > best_mrr + min_improvement:
            # Significant improvement found
            best_mrr = current_mrr
            patience_counter = 0

            # Save model
            model_path = os.path.join(args.output_dir, "best_embedding_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embeddings": embeddings,
                    "listing_to_idx": listing_to_idx,
                    "idx_to_listing": idx_to_listing,
                    "embedding_dim": args.embedding_dim,
                    "num_listings": num_listings,
                    "epoch": epoch,
                    "mrr": best_mrr,
                },
                model_path,
            )

            # Save embeddings as numpy array
            embeddings_path = os.path.join(args.output_dir, "listing_embeddings.npy")
            np.save(embeddings_path, embeddings)

            # Save mappings
            mappings_path = os.path.join(args.output_dir, "listing_mappings.json")
            with open(mappings_path, "w") as f:
                json.dump(
                    {
                        "listing_to_idx": listing_to_idx,
                        "idx_to_listing": idx_to_listing,
                    },
                    f,
                    indent=2,
                )

            print(f"✓ Saved best model with MRR: {best_mrr:.4f}")
        else:
            # No significant improvement
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

            # Early stopping check
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                print(
                    f"Best MRR: {best_mrr:.4f} (no improvement for {patience} epochs)"
                )
                break

    print(f"\nTraining completed! Best MRR: {best_mrr:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

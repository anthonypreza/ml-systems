"""Runs book recommendation feature engineering pipeline."""

import wandb

from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bookrecsys.feat_eng import run_feature_eng_pipeline
from bookrecsys.logger import get_logger
from bookrecsys.data import (
    load_authors_df,
    load_books_df,
    load_genres_df,
    load_interactions_df,
    save_df_to_csv,
)
from bookrecsys.preprocess import filter_interactions

TRAIN_DATA_FILENAME = "completion_prediction.csv"
USER_BOOK_MAPPING_FILENAME = "user_book_mapping.csv"


logger = get_logger("run")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions-sample", type=int, default=100_000,
                        help="Number of interaction rows to sample (-1 for all)")
    parser.add_argument("--books-sample", type=int, default=100_000,
                        help="Number of books to sample (-1 for all)")
    parser.add_argument("--authors-sample", type=int, default=10_000,
                        help="Number of authors to sample (-1 for all)")
    parser.add_argument("--genres-sample", type=int, default=10_000,
                        help="Number of genres to sample (-1 for all)")
    args = parser.parse_args()
    """Run feature pipeline."""
    # Initialize wandb run
    with wandb.init(
        project="book-recommendation", group="dev", job_type="feat_eng", save_code=True
    ) as run:
        interactions_df = load_interactions_df(sample_size=args.interactions_sample)
        books_df = load_books_df(sample_size=args.books_sample)
        genres_df = load_genres_df(sample_size=args.genres_sample)
        authors_df = load_authors_df(sample_size=args.authors_sample)

        # Log config to wandb for traceability
        run.config.update({
            "interactions_sample": args.interactions_sample,
            "books_sample": args.books_sample,
            "authors_sample": args.authors_sample,
            "genres_sample": args.genres_sample,
        })

        logger.info("=== PREPROCESSING ===")
        logger.info("Interactions columns before: %s", interactions_df.columns.tolist())

        filtered_interactions = filter_interactions(interactions_df)
        train_data, user_book_mapping = run_feature_eng_pipeline(
            filtered_interactions, books_df, genres_df, authors_df
        )

        user_book_mapping_path = save_df_to_csv(
            user_book_mapping, USER_BOOK_MAPPING_FILENAME
        )
        artifact = wandb.Artifact(
            name=USER_BOOK_MAPPING_FILENAME,
            type="supplemental_data",
            description="Mapping of (user_id, book_id) to sample index",
        )
        artifact.add_file(user_book_mapping_path)
        run.log_artifact(artifact)

        logger.info("✅ User-book mapping saved for completion prediction evaluation!")
        logger.info("Mapping shape: %s", user_book_mapping.shape)
        logger.info("Task: Predicting book completion (is_read)")
        logger.info("Clean dataset: No leaky features, no synthetic negatives")

        # Save the final completion prediction dataset
        train_data_path = save_df_to_csv(train_data, TRAIN_DATA_FILENAME)

        artifact = wandb.Artifact(
            name=TRAIN_DATA_FILENAME,
            type="training_data",
            description="Book completion prediction training data",
        )
        artifact.add_file(train_data_path)
        run.log_artifact(artifact)

        # Also export book metadata for books present in the training set
        try:
            unique_book_ids = set(train_data["book_id"].unique())
            # Select commonly used metadata columns if present
            candidate_cols = [
                "book_id",
                "title",
                "publication_year",
                "language_code",
                "format",
                "is_ebook",
                "average_rating",
                "ratings_count",
            ]
            cols = [c for c in candidate_cols if c in books_df.columns]
            book_meta = books_df[cols].copy()
            book_meta = book_meta[book_meta["book_id"].isin(unique_book_ids)]

            # Optionally extract a primary author name if available
            if "authors" in books_df.columns and "title" in book_meta.columns:
                import pandas as _pd

                def _author_name(authors):
                    try:
                        if isinstance(authors, list) and len(authors) > 0:
                            a0 = authors[0]
                            if isinstance(a0, dict) and "name" in a0:
                                return a0.get("name")
                    except Exception:
                        pass
                    return None

                tmp = books_df[["book_id", "authors"]].copy()
                tmp["author_name"] = tmp["authors"].apply(_author_name)
                book_meta = book_meta.merge(
                    tmp[["book_id", "author_name"]], on="book_id", how="left"
                )

            book_meta_path = save_df_to_csv(book_meta, "book_metadata.csv")
            meta_artifact = wandb.Artifact(
                name="book_metadata.csv",
                type="book_metadata",
                description="Book metadata (subset) for serving recommendations",
            )
            meta_artifact.add_file(book_meta_path)
            run.log_artifact(meta_artifact)
            logger.info("✅ Exported book metadata for serving: %s rows", len(book_meta))
        except Exception as e:
            logger.info("⚠️ Skipped exporting book metadata: %s", e)

        # Summary of what we built
        logger.info("\n=== FINAL DATASET SUMMARY ===")
        logger.info("Dataset shape: %s", train_data.shape)
        logger.info("Target: is_read (completion prediction)")
        logger.info("Features: %d", int(train_data.shape[1] - 1))
        logger.info("Completion rate: %.3f", train_data["is_read"].mean())
        logger.info("Clean dataset: No target leakage, no synthetic negatives")
        logger.info("Ready for model training!")


if __name__ == "__main__":
    main()

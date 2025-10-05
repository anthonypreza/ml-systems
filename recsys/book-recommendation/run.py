"""Root orchestrator for feature pipeline and training MLflow projects."""

from __future__ import annotations

from pathlib import Path
import argparse

import mlflow


def main() -> None:
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

    root_path = Path(__file__).resolve().parent

    feature_proj = root_path / "mlflows" / "feature_pipeline"
    train_proj = root_path / "mlflows" / "train"

    print("=== Running feature pipeline MLflow project ===")
    feat_run = mlflow.run(
        str(feature_proj),
        entry_point="main",
        parameters={
            "interactions_sample": args.interactions_sample,
            "books_sample": args.books_sample,
            "authors_sample": args.authors_sample,
            "genres_sample": args.genres_sample,
        },
    )
    print(f"Feature pipeline run finished: {feat_run.run_id}")

    print("=== Running training MLflow project ===")
    train_run = mlflow.run(str(train_proj), entry_point="main")
    print(f"Training run finished: {train_run.run_id}")


if __name__ == "__main__":
    main()

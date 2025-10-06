"""Load datasets."""

import gzip
import random
import pandas as pd
import json
import os

from .logger import get_logger

PWD = os.path.dirname(__file__)
DATA_DIR = f"{PWD}/../data"
MODEL_DIR = f"{PWD}/../models"

logger = get_logger("data")


def sample_json_file(filepath: str, sample_size: int = 100_000) -> pd.DataFrame:
    """Sample a JSON lines file and return a DataFrame.

    - If sample_size == -1, load all lines from the file.
    - Otherwise, probabilistically sample up to sample_size rows.
    """

    sampled_lines = []
    total_lines = 0

    logger.info(
        "Sampling %d lines from %s ...",
        sample_size,
        filepath,
    )

    # Load entire file if requested
    if sample_size == -1:
        logger.info("Loading all lines from %s ...", filepath)
        with gzip.open(filepath, "rt") as f:
            for line in f:
                try:
                    sampled_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info("Loaded %d lines (full file).", len(sampled_lines))
        return pd.DataFrame(sampled_lines)

    logger.info("Estimating total number of lines in the file...")
    with gzip.open(filepath, "rt") as f:
        for i, _ in enumerate(f):
            total_lines += 1
            if i > 1_000_000:  # Limit to first million lines for speed
                break

    # Calculate sampling probability
    sample_prob = 1.0 if total_lines <= sample_size else (sample_size / total_lines)

    logger.info(
        "Total lines estimated: %d. Sampling probability: %.6f",
        total_lines,
        sample_prob,
    )

    # Collect sampled
    with gzip.open(filepath, "rt") as f:
        for line in f:
            try:
                if random.random() < sample_prob:
                    sampled_lines.append(json.loads(line))
                    if len(sampled_lines) >= sample_size:
                        break
            except json.JSONDecodeError:
                continue

    logger.info("Sampled %d lines.", len(sampled_lines))
    return pd.DataFrame(sampled_lines)


def load_interactions_df(sample_size: int = 100_000):
    interactions_records = sample_json_file(
        f"{DATA_DIR}/goodreads_interactions_dedup.json.gz", sample_size=sample_size
    )
    return pd.DataFrame(interactions_records)


def load_books_df(sample_size: int = 100_000):
    books_records = sample_json_file(
        f"{DATA_DIR}/goodreads_books.json.gz", sample_size=sample_size
    )
    return pd.DataFrame(books_records)


def load_genres_df(sample_size: int = 10_000):
    genres_records = sample_json_file(
        f"{DATA_DIR}/goodreads_book_genres_initial.json.gz", sample_size=sample_size
    )
    return pd.DataFrame(genres_records)


def load_authors_df(sample_size: int = 10_000):
    author_records = sample_json_file(
        f"{DATA_DIR}/goodreads_book_authors.json.gz", sample_size=sample_size
    )
    return pd.DataFrame(author_records)


def save_df_to_csv(df: pd.DataFrame, filename):
    path = f"{DATA_DIR}/{filename}"
    df.to_csv(path, index=False)
    logger.info("Saved data to %s", filename)
    return path


def save_to_file(text, filename):
    path = f"{DATA_DIR}/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

        print(f"Detailed report saved to: {filename}")
    return path


def save_model(model, filename, onnx=False):
    model_path = f"{MODEL_DIR}/{filename}"
    if onnx:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        return model_path

    import torch

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to: %s", model_path)

    return model_path

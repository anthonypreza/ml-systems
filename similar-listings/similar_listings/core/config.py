"""Configuration settings for similar listings recommendation system."""

import os


class Config:
    """Configuration class for the recommendation system."""

    # Data paths
    DATA_DIR = "data"
    MODELS_DIR = "models"

    # File paths
    USERS_FILE = os.path.join(DATA_DIR, "users.json")
    LISTINGS_FILE = os.path.join(DATA_DIR, "listings.json")
    INTERACTIONS_FILE = os.path.join(DATA_DIR, "interactions.json")
    SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
    TRIPLETS_FILE = os.path.join(DATA_DIR, "triplets.json")

    # Model paths
    MODEL_FILE = os.path.join(MODELS_DIR, "best_embedding_model.pth")
    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "listing_embeddings.npy")
    MAPPINGS_FILE = os.path.join(MODELS_DIR, "listing_mappings.json")

    # Model parameters
    EMBEDDING_DIM = 256
    DROPOUT_RATE = 0.1

    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    TRAIN_RATIO = 0.8

    # Triplet generation parameters
    WINDOW_SIZE = 5
    NEG_SAMPLING_RATIO = 10
    MAX_SESSION_GAP = 3600  # seconds

    # Search parameters
    TOP_K = 10

    # Random seed
    SEED = 42

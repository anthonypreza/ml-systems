# Similar Listings Recommendation System

A session-based recommendation system that learns listing embeddings from user browsing co-occurrences to recommend similar listings.

## Overview

This system builds listing embeddings by analyzing which items frequently appear together in users' browsing sessions. Items that co-occur often will have similar embeddings, enabling effective similarity-based recommendations using neural embedding techniques.

### Key Features

- **Session-based Learning**: Learns from realistic user browsing patterns and booking behavior
- **Triplet Training**: Uses positive/negative triplets with sliding window contexts and hard negatives
- **Neural Embeddings**: Shallow neural network learns 128-dimensional listing representations
- **Ranking Evaluation**: Measures performance using Mean Reciprocal Rank (MRR) of booked listings
- **Flexible Recommendations**: Supports both single-listing similarity and session-based recommendations

## Approach

1. **Synthetic Data Generation**: Create realistic users, listings, and browsing interactions
2. **Session Extraction**: Parse interactions into coherent browsing sessions ending with bookings
3. **Triplet Generation**: Create training triplets using sliding windows and negative sampling
4. **Embedding Training**: Train neural network to learn listing embeddings via dot-product similarity
5. **Recommendation**: Find similar listings using nearest neighbor search in embedding space

## Quick Start

```bash
# Setup environment
uv venv --python 3.12
uv sync

# 1. Generate synthetic data (1K users, 5K listings, 50K interactions)
python scripts/generate_synthetic_data.py --users 1000 --listings 5000 --interactions 50000

# 2. Extract browsing sessions (sequences ending with bookings)
python scripts/extract_sessions.py

# 3. Generate training triplets (positive/negative pairs with hard negatives)
python scripts/generate_triplets.py --window-size 5 --neg-ratio 3

# 4. Train embeddings (neural network with ranking evaluation)
python scripts/train_embeddings.py --epochs 10 --embedding-dim 128

# 5. Get recommendations
# Single listing similarity:
python scripts/recommend.py --listing-id [LISTING_ID] --top-k 5

# Session-based recommendations:
python scripts/recommend.py --session-listings [ID1] [ID2] [ID3] --top-k 5

# JSON output for integration:
python scripts/recommend.py --listing-id [LISTING_ID] --format json
```

## Data Pipeline

### 1. Synthetic Data Generation

Creates realistic datasets with geographical distribution and browsing patterns:

- **Users**: Demographics, location, preferences
- **Listings**: Property details, pricing, host assignments
- **Interactions**: Realistic browsing sessions with views, clicks, and bookings

### 2. Session Extraction

Converts raw interactions into meaningful browsing sessions:

- Groups interactions by user and timestamp
- Identifies sessions ending with bookings
- Filters out incomplete sessions

### 3. Triplet Generation

Creates training data using multiple strategies:

- **Sliding window**: Listings within browsing context windows
- **Hard positives**: Clicked listings → booked listing connections
- **Hard negatives**: Same city listings outside context (trickier to distinguish)
- **Random negatives**: Random listings for easier discrimination

### 4. Model Training

Neural embedding model with ranking optimization:

- **Architecture**: Embedding layer + dot product similarity
- **Loss**: Binary cross-entropy on positive/negative pairs
- **Evaluation**: Mean Reciprocal Rank (MRR) of actually booked listings
- **Early stopping**: Saves best model based on MRR performance

## Model Architecture

```
Training Pipeline:

Triplet Input:
┌─────────────────────────────────────────────┐
│ Central Listing ID: "abc123..."             │
│ Context Listing ID: "def456..."             │
│ Label: 1 (positive) or 0 (negative)         │
└─────────────────────────────────────────────┘
                    │
                    ▼
ID to Index Mapping:
┌─────────────────────────────────────────────┐
│ "abc123..." → 42   "def456..." → 157        │
└─────────────────────────────────────────────┘
                    │
                    ▼
Neural Network:
┌─────────────────────────────────────────────┐
│  Embedding Layer (2000 × 128)               │
│  ┌─────────────┐    ┌─────────────┐         │
│  │ Listing 42  │    │ Listing 157 │         │
│  │ Embedding   │    │ Embedding   │         │
│  │ [128 dims]  │    │ [128 dims]  │         │
│  └─────────────┘    └─────────────┘         │
│         │                   │               │
│         └───────┬───────────┘               │
│                 ▼                           │
│      Dot Product Similarity                 │
│         (element-wise multiply + sum)       │
│                 │                           │
│                 ▼                           │
│        Similarity Score (logit)             │
└─────────────────────────────────────────────┘
                    │
                    ▼
Loss Calculation:
┌─────────────────────────────────────────────┐
│ BCE Loss = BCEWithLogitsLoss(logit, label)  │
│                                             │
│ If label = 1: push score higher            │
│ If label = 0: push score lower             │
└─────────────────────────────────────────────┘
                    │
                    ▼
Backpropagation Updates Embeddings
```

**Key Components:**

- **Embedding Layer**: `nn.Embedding(2000, 128)` - learnable lookup table
- **Similarity Function**: `dot_product(emb1, emb2) = sum(emb1 * emb2)`
- **Loss Function**: `BCEWithLogitsLoss` (sigmoid + cross-entropy)
- **Training**: Adam optimizer updates embedding weights based on positive/negative pairs

**At Inference:**

```
Query Listing ID → Embedding Vector → Similarity Search → Top-K Recommendations
```

## Architecture

### Scripts (`scripts/`)

- `generate_synthetic_data.py`: Create realistic synthetic datasets
- `extract_sessions.py`: Extract browsing sessions from interactions
- `generate_triplets.py`: Generate positive/negative training triplets
- `train_embeddings.py`: Train listing embeddings with neural network
- `recommend.py`: Get recommendations from trained embeddings

### Python Package (`similar_listings/`)

- **`core/`**: Core models and configuration
  - `config.py`: Centralized configuration (paths, hyperparameters)
  - `model.py`: Neural network models and training logic
- **`search/`**: Recommendation engine
  - `recommender.py`: Main recommendation interface with similarity search
- **`utils/`**: Utilities and helpers
  - `data.py`: Data loading, session processing, and triplet generation
  - `evaluation.py`: Ranking metrics (MRR, mean rank) and statistics

### Generated Assets

- **`data/`**: JSON datasets
  - `users.json`: User profiles with demographics
  - `listings.json`: Property details and metadata
  - `interactions.json`: Raw user-listing interaction events
  - `sessions.json`: Extracted browsing sessions with bookings
  - `triplets.json`: Training triplets with labels and types
- **`models/`**: Trained models and embeddings
  - `best_embedding_model.pth`: PyTorch model checkpoint
  - `listing_embeddings.npy`: Pre-computed embedding vectors
  - `listing_mappings.json`: ID-to-index mappings

## Data Schemas

### Sessions

```json
{
  "session_id": "uuid",
  "clicked_listing_ids": ["uuid1", "uuid2"],
  "booked_listing_id": "uuid3",
  "session_start": 1234567890,
  "session_end": 1234567920,
  "num_clicks": 2,
  "session_duration": 30
}
```

### Triplets

```json
{
  "central_listing": "uuid1",
  "context_listing": "uuid2",
  "label": 1,
  "session_id": "session_uuid",
  "triplet_type": "positive|random_negative|hard_negative"
}
```

## Model Performance

The system measures success using **Mean Reciprocal Rank (MRR)**: given a user's clicked listings, where does the actually booked listing rank among all possible recommendations?

**Typical Results** (after 10 epochs):

- Mean Rank: ~100-200 (out of 5000 listings)
- Median Rank: ~10-50
- MRR: ~0.1-0.2
- Validation Accuracy: ~65% on triplet classification

## Usage Examples

### Single Listing Similarity

```bash
python scripts/recommend.py --listing-id 9393f85c-46dd-47b9-ba36-1fbf34c5c8ce --top-k 3

# Output:
# Listing 9393f85c... (townhouse, 4 beds, Chicago, $269.74)
#
# === Top 3 Recommendations ===
# 1. Listing 35ef12ff... (townhouse, 5 beds, Berlin, $125.28) - Similarity: 0.4133
# 2. Listing d4dfb64e... (loft, 4 beds, Tokyo, $1049.23) - Similarity: 0.3976
# 3. Listing d4d9774c... (apartment, 3 beds, Berlin, $74.15) - Similarity: 0.3872
```

### Session-based Recommendations

```bash
python scripts/recommend.py --session-listings id1 id2 id3 --top-k 5

# Finds listings similar to the average embedding of the input listings
# Useful for "users who viewed these also liked..." scenarios
```

**How it works**: Session recommendations create a single query vector by averaging the embeddings of all clicked listings. This "centroid" represents the user's overall browsing preferences rather than similarity to just one listing, capturing their broader intent across the session.

## Configuration

All hyperparameters are centralized in `similar_listings/core/config.py`:

- **Model**: Embedding dimension (256), dropout (0.1), learning rate (0.001)
- **Training**: Batch size (256), epochs (20), train/val split (80/20)
- **Triplets**: Window size (5), negative sampling ratio (10:1)
- **Data**: File paths, session gap threshold (1 hour)

## Model Tuning

### Quick Wins (High Impact)

1. **More Training Data**: Double dataset size for 0.05-0.15 MRR improvement

   ```bash
   python scripts/generate_synthetic_data.py --users 2000 --listings 10000 --interactions 100000
   ```

2. **Longer Training**: Increase epochs or let early stopping find optimal point

   ```bash
   python scripts/train_embeddings.py --epochs 30
   ```

3. **Larger Embeddings**: Try 512-dimensional embeddings for complex patterns
   ```bash
   python scripts/train_embeddings.py --embedding-dim 512
   ```

### Hyperparameter Tuning

- **Learning Rate**: Try `0.01` (faster) or `0.0001` (more stable)
- **Negative Sampling**: Increase `NEG_SAMPLING_RATIO` in config for harder training
- **Window Size**: Smaller (3) for tight context, larger (7) for broader patterns
- **Batch Size**: Larger batches (512) for stable gradients, smaller (128) for regularization

### Advanced Techniques

- **Learning Rate Scheduling**: Decay LR every few epochs
- **Weighted Loss**: Handle class imbalance between positive/negative samples
- **Dropout Tuning**: Increase (0.2-0.3) if overfitting, decrease (0.05) if underfitting

## Technical Details

- **Framework**: PyTorch for neural network training
- **Embeddings**: 128-dimensional learned representations
- **Similarity**: Dot product between embedding vectors
- **Training**: Binary cross-entropy loss with Adam optimizer
- **Evaluation**: Custom ranking evaluator measures MRR on held-out sessions
- **Hardware**: CPU training (typically 5-10 minutes for full pipeline)

## Current Status

> **⚠️ Note**: This system has been developed and tested only on **synthetic data**. While the synthetic data generator creates realistic browsing patterns, the system needs validation on real-world datasets to prove its effectiveness.

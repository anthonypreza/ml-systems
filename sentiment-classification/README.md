# IMDB Sentiment Analysis

Binary sentiment classification on movie reviews using BERT and PyTorch.

## Overview

This project implements a sentiment classifier that predicts whether movie reviews are positive or negative. The model uses a BERT backbone with custom projection layers and is trained on the IMDB movie review dataset.

## Features

- **BERT-based Architecture**: Uses `bert-base-uncased` with custom projection layers
- **PyTorch Implementation**: Native PyTorch training loop with device-agnostic code
- **Overfitting Prevention**: Includes weight decay, validation split, and early stopping
- **Interactive Notebook**: Jupyter notebook with step-by-step implementation
- **Hugging Face Integration**: Uses `datasets` library for IMDB data loading

## Architecture

```
Input Text → BERT Tokenizer → BERT Encoder → TextEncoder (512-dim) → TextClassifier → Binary Output
```

**Components:**
- **TextEncoder**: BERT + linear projection layer with L2 normalization
- **TextClassifier**: Linear classification head for binary sentiment prediction
- **Training**: Adam optimizer with CrossEntropyLoss

## Dataset

- **Source**: IMDB Movie Review Dataset (via Hugging Face)
- **Size**: 25,000 training examples, 25,000 test examples
- **Classes**: Binary (positive/negative sentiment)
- **Current Setup**: 10,000 samples with 80/20 train/validation split

## Quick Start

1. **Open the notebook**: `imdb_sentiment.ipynb`
2. **Install dependencies** (handled in notebook):
   ```python
   %pip install torch transformers datasets scikit-learn tqdm --quiet
   ```
3. **Run cells sequentially** to train and evaluate the model

## Training Configuration

- **Model**: `bert-base-uncased`
- **Embedding Dimension**: 512
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Max Sequence Length**: 512
- **Early Stopping**: Patience of 3 epochs

## Results

The model achieves competitive performance on the IMDB dataset with proper overfitting mitigation through:
- Train/validation splits
- Early stopping monitoring
- Weight decay regularization

## Development Notes

- **Migration**: Originally implemented in TensorFlow, migrated to PyTorch
- **Device Support**: Automatically detects and uses GPU if available
- **Monitoring**: Includes progress tracking and accuracy metrics
- **Validation**: Proper train/test separation to prevent overfitting

## Files

- `imdb_sentiment.ipynb` - Main implementation notebook
- Example outputs and training logs are preserved in notebook cells
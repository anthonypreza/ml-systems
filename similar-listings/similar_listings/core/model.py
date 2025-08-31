"""Neural network models for learning listing embeddings."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional


class TripletDataset(Dataset):
    """Dataset for loading triplet training data."""

    def __init__(self, triplets: List[Dict[str, Any]], listing_to_idx: Dict[str, int]):
        self.triplets = triplets
        self.listing_to_idx = listing_to_idx

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        central_idx = self.listing_to_idx[triplet["central_listing"]]
        context_idx = self.listing_to_idx[triplet["context_listing"]]
        label = float(triplet["label"])

        return torch.tensor(central_idx), torch.tensor(context_idx), torch.tensor(label)


class ListingEmbeddingModel(nn.Module):
    """Shallow neural network for learning listing embeddings."""

    def __init__(
        self, num_listings: int, embedding_dim: int = 128, dropout: float = 0.1
    ):
        super(ListingEmbeddingModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_listings = num_listings

        # Embedding layer
        self.embeddings = nn.Embedding(num_listings, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings with small random values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.1)

    def forward(self, central_idx, context_idx):
        """
        Forward pass: compute similarity score between central and context listings.

        Returns:
            logits: Raw dot product scores
        """
        # Get embeddings
        central_emb = self.embeddings(central_idx)  # (batch_size, embedding_dim)
        context_emb = self.embeddings(context_idx)  # (batch_size, embedding_dim)

        # Apply dropout
        central_emb = self.dropout(central_emb)
        context_emb = self.dropout(context_emb)

        # Compute dot product similarity
        logits = (central_emb * context_emb).sum(dim=1)  # (batch_size,)

        return logits

    def get_embeddings(self):
        """Return all learned embeddings."""
        return self.embeddings.weight.detach().cpu().numpy()

    def get_embedding(self, listing_idx: int):
        """Get embedding for a single listing."""
        return self.embeddings(torch.tensor(listing_idx)).detach().cpu().numpy()


def train_epoch(
    model: ListingEmbeddingModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    if not criterion:
        criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (central_idx, context_idx, labels) in enumerate(dataloader):
        central_idx = central_idx.to(device)
        context_idx = context_idx.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(central_idx, context_idx)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def evaluate(
    model: ListingEmbeddingModel,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if not criterion:
        criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for central_idx, context_idx, labels in dataloader:
            central_idx = central_idx.to(device)
            context_idx = context_idx.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(central_idx, context_idx)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bookrecsys.logger import get_logger
from bookrecsys.data import save_to_file, save_model, save_df_to_csv


PERFORMANCE_REPORT_FILENAME = "model_performance_report_nn.txt"
MODEL_FILENAME = "nn_recommender_model.pt"
TEST_RESULTS_FILENAME = "test_results_nn.csv"
RANKING_SUMMARY_FILENAME = "ranking_metrics_summary_nn.csv"
ONNX_FILENAME = "nn_recommender_model.onnx"

logger = get_logger("train")

# Initialize wandb run
# Expanded parameter dictionary for production-style training visibility
param = {
    "hidden_layers": (256, 128),
    "blocks_per_layer": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "epochs": 100,
    "early_stopping_rounds": 10,
}


# Ranking evaluation functions (module-level)
def compute_precision_at_k(user_data, k):
    """Compute Precision@k for a single user"""
    top_k = user_data.head(k)
    return top_k["true_completion"].sum() / k


def compute_recall_at_k(user_data, k):
    """Compute Recall@k for a single user"""
    top_k = user_data.head(k)
    total_relevant = user_data["true_completion"].sum()
    if total_relevant == 0:
        return 0.0
    return top_k["true_completion"].sum() / total_relevant


def compute_ap_at_k(user_data, k):
    """Compute Average Precision@k for a single user"""
    top_k = user_data.head(k)
    y_true = top_k["true_completion"].values
    y_scores = top_k["prediction_score"].values

    if y_true.sum() == 0:
        return 0.0

    return average_precision_score(y_true, y_scores)


def compute_ndcg_at_k(user_data, k):
    """Compute NDCG@k for a single user"""
    top_k = user_data.head(k)
    y_true = top_k["true_completion"].values.reshape(1, -1)
    y_scores = top_k["prediction_score"].values.reshape(1, -1)

    if len(y_true[0]) == 0:
        return 0.0

    return ndcg_score(y_true, y_scores, k=k)


def evaluate_ranking_metrics(test_mapping, k_values=None):
    """Compute comprehensive ranking metrics across users for provided k-values."""
    if k_values is None:
        k_values = [5, 10, 20, 50]

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"map@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    users_evaluated = 0

    for user_id in test_mapping["user_id"].unique():
        user_data = test_mapping[test_mapping["user_id"] == user_id]

        # Skip users with no positive items in test
        if user_data["true_completion"].sum() == 0:
            continue

        # Sort by prediction score (highest first)
        user_data = user_data.sort_values("prediction_score", ascending=False)
        users_evaluated += 1

        # Compute metrics for each k
        for k in k_values:
            if len(user_data) >= k:  # Only compute if user has enough items
                metrics[f"precision@{k}"].append(compute_precision_at_k(user_data, k))
                metrics[f"recall@{k}"].append(compute_recall_at_k(user_data, k))
                metrics[f"map@{k}"].append(compute_ap_at_k(user_data, k))
                metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(user_data, k))

    # Average across users
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = float(np.mean(values)) if values else 0.0

    results["users_evaluated"] = users_evaluated
    return results


# Torch dataset/dataloader
class InteractionsDataset(Dataset):
    def __init__(self, X_dense, y, user_ids, book_ids):
        self.X = X_dense.astype(np.float32)
        self.y = y.to_numpy().astype(np.float32)
        self.u = user_ids.astype(np.int64)
        self.b = book_ids.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.u[i], self.b[i], self.y[i]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class UserBookRecommenderModel(nn.Module):
    """Neural network with residual MLP blocks for pointwise book recommendations."""

    def __init__(
        self,
        num_users: int,
        num_books: int,
        input_dim: int,
        user_cols: list[int],
        book_cols: list[int],
        blocks_per_layer: int = 2,
        hidden_layers=(256, 128),
        emb_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.book_emb = nn.Embedding(num_books, emb_dim)

        self.user_idx = user_cols
        self.book_idx = book_cols

        # crossed features = same length as user/book shelf vectors
        cross_dim = len(self.user_idx)
        total_in = input_dim + cross_dim + 2 * emb_dim

        layers = []
        prev = total_in

        for h in hidden_layers:
            # projection into new hidden dim
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU()]
            # add several residual blocks at this dimension
            for _ in range(blocks_per_layer):
                layers.append(ResidualBlock(h, dropout=dropout))
            prev = h

        # output layer
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, user_ids, book_ids):
        u_shelves = x[:, self.user_idx]
        b_shelves = x[:, self.book_idx]
        cross = u_shelves * b_shelves

        # look up embeddings
        u = self.user_emb(user_ids)
        b = self.book_emb(book_ids)

        x_in = torch.cat([x, cross, u, b], dim=1)
        return self.net(x_in).squeeze(-1)


def main():
    """Run training pipeline."""
    with wandb.init(
        project="book-recommendation",
        group="dev",
        job_type="train",
        save_code=True,
        config=param,
    ) as run:
        artifact = run.use_artifact(
            "book-recommendation/completion_prediction.csv:latest"
        )
        artifact_path = artifact.file()

        # Try preferred mapping artifact name; fallback to alternate
        try:
            user_book_mapping_artifact = run.use_artifact(
                "book-recommendation/completion_user_book_mapping.csv:latest"
            )
        except Exception:
            user_book_mapping_artifact = run.use_artifact(
                "book-recommendation/user_book_mapping.csv:latest"
            )
        user_book_mapping_path = user_book_mapping_artifact.file()

        df = pd.read_csv(artifact_path)
        user_book_mapping = pd.read_csv(user_book_mapping_path)

        logger.info("Completion prediction model")
        logger.info("Task: Will user complete books they interact with?")
        logger.info("Dataset shape: %s", df.shape)
        logger.info("Target: is_read (completion)")
        logger.info("Features: %s", df.shape[1] - 1)

        # Check target distribution
        logger.info("\nTarget distribution:")
        logger.info("%s", df["is_read"].value_counts())
        completion_rate = df["is_read"].mean()
        logger.info("Overall completion rate: %s", f"{completion_rate:.3f}")

        # If train data already includes user_id/book_id, skip merge. Otherwise merge by row order
        if not {"user_id", "book_id"}.issubset(df.columns):
            df = df.merge(user_book_mapping, left_index=True, right_on="sample_index")

        user_enc = LabelEncoder().fit(df["user_id"])
        book_enc = LabelEncoder().fit(df["book_id"])

        df["user_id_idx"] = user_enc.transform(df["user_id"])
        df["book_id_idx"] = book_enc.transform(df["book_id"])

        df.head()

        # Prepare data
        TARGET = "is_read"
        ID_USER = "user_id_idx"
        ID_BOOK = "book_id_idx"

        dense_cols = [
            c
            for c in df.columns
            if c not in [TARGET, "user_id", "book_id", ID_USER, ID_BOOK]
        ]

        skewed = [
            "rating_count",
            "rating_count_book",
            "book_id_nunique",
            "author_ratings_count",
            "author_text_reviews_count",
            "user_id_nunique",
            "user_rating_count_historical",
            "user_author_interaction_count",
        ]
        for col in skewed:
            if col in dense_cols:  # guard
                df[col] = np.log1p(df[col])

        test_size = 0.2
        n = len(df)
        split_idx = int(n * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        X_train = train_df[dense_cols]
        y_train = train_df[TARGET]
        X_test = test_df[dense_cols]
        y_test = test_df[TARGET]

        X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
            X_train,
            y_train,
            train_df.index,
            test_size=0.2,
            stratify=y_train,
            random_state=42,
        )

        # Standardize features for NN
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Align ID tensors
        u_tr = train_df.loc[idx_tr, ID_USER].to_numpy()  # type: ignore
        b_tr = train_df.loc[idx_tr, ID_BOOK].to_numpy()  # type: ignore
        u_val = train_df.loc[idx_val, ID_USER].to_numpy()  # type: ignore
        b_val = train_df.loc[idx_val, ID_BOOK].to_numpy()  # type: ignore
        u_test = test_df[ID_USER].to_numpy()
        b_test = test_df[ID_BOOK].to_numpy()

        batch_size = int(param.get("batch_size", 512))
        train_ds = InteractionsDataset(X_tr, y_tr, u_tr, b_tr)
        val_ds = InteractionsDataset(X_val, y_val, u_val, b_val)

        pin_mem = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_mem,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_mem,
        )

        num_users = int(df[ID_USER].max() + 1)
        num_books = int(df[ID_BOOK].max() + 1)

        dense_index = {c: i for i, c in enumerate(dense_cols)}
        u_cols = [dense_index[c] for c in dense_cols if c.endswith("_user")]
        b_cols = [
            dense_index[c]
            for c in dense_cols
            if c.endswith("_book")
            and c not in ("rating_mean_book", "rating_std_book", "rating_count_book")
        ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UserBookRecommenderModel(
            num_users=num_users,
            num_books=num_books,
            input_dim=len(dense_cols),
            user_cols=u_cols,
            book_cols=b_cols,
            hidden_layers=param.get("hidden_layers", (256, 128)),
            blocks_per_layer=param.get("blocks_per_layer", 2),
            emb_dim=int(param.get("emb_dim", 32)),
            dropout=float(param.get("dropout", 0.2)),
        ).to(device)

        model.to(device)
        wandb.watch(model, log="all", log_freq=50)

        # Class imbalance handling: pos_weight = neg/pos on training split
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimization
        lr = float(param.get("lr", 1e-3))
        weight_decay = float(param.get("weight_decay", 1e-4))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(param.get("lr", 1e-3)),
            weight_decay=float(param.get("weight_decay", 1e-5)),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(param.get("lr", 1e-3)),
            steps_per_epoch=len(train_loader),
            epochs=int(param.get("epochs", 30)),
        )

        epochs = int(param.get("epochs", 30))
        patience = int(param.get("early_stopping_rounds", 5))
        best_val = -float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, ub, bb, yb in train_loader:
                xb = torch.as_tensor(xb, device=device)
                ub = torch.as_tensor(ub, device=device)
                bb = torch.as_tensor(bb, device=device)
                yb = torch.as_tensor(yb, device=device)

                optimizer.zero_grad()
                logits = model(xb, ub.long(), bb.long())
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_ds)

            # validation
            model.eval()
            val_probs = []
            val_targets = []
            with torch.no_grad():
                for xb, ub, bb, yb in val_loader:
                    xb = xb.to(device)
                    ub = ub.to(device)
                    bb = bb.to(device)
                    logits = model(xb, ub.long(), bb.long())
                    val_probs.append(torch.sigmoid(logits).cpu())
                    val_targets.append(yb)

            val_probs = np.concatenate([v.numpy() for v in val_probs])
            val_targets = np.concatenate([t.numpy() for t in val_targets])
            val_ap = float(average_precision_score(val_targets, val_probs))

            run.log(
                {"nn/train_loss": train_loss, "nn/val_aucpr": val_ap, "epoch": epoch}
            )

            if val_ap > best_val + 1e-6:
                best_val = val_ap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %s (best val AP %s)",
                        epoch,
                        f"{best_val:.4f}",
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Inference on test
        test_ds = InteractionsDataset(X_test, y_test, u_test, b_test)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model.eval()
        probs = []
        with torch.no_grad():
            for xb, ub, bb, yb in test_loader:
                xb = xb.to(device)
                ub = ub.to(device)
                bb = bb.to(device)
                logits = model(xb, ub.long(), bb.long())
                probs.append(torch.sigmoid(logits).cpu())
        y_proba = torch.cat(probs).numpy().ravel()
        preds = (y_proba >= 0.5).astype(int)

        logger.info("Basic classification metrics")
        logger.info("Test Accuracy: %s", f"{accuracy_score(y_test, preds):.4f}")
        logger.info("Precision: %s", f"{precision_score(y_test, preds):.4f}")
        logger.info("Recall: %s", f"{recall_score(y_test, preds):.4f}")
        logger.info("F1-Score: %s", f"{f1_score(y_test, preds):.4f}")

        # Get prediction probabilities for ranking
        # y_proba was computed by the NN above
        logger.info("ROC-AUC: %s", f"{roc_auc_score(y_test, y_proba):.4f}")
        logger.info("PR-AUC: %s", f"{average_precision_score(y_test, y_proba):.4f}")

        logger.info("\nClassification Report:")
        logger.info("%s", classification_report(y_test, preds))

        # Confusion matrix with plotly
        cm = confusion_matrix(y_test, preds)
        logger.info("\nConfusion Matrix:")
        logger.info("%s", cm)

        # Create confusion matrix heatmap
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                showscale=True,
            )
        )

        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            width=500,
            height=400,
        )

        run.log({"confusion_matrix": fig_cm})

        # ROC and precision-recall curves

        # Calculate curves
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # Create subplots for ROC and PR curves
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["ROC Curve", "Precision-Recall Curve"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.3f})",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Diagonal line for ROC
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="red", width=1, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Precision-recall curve
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"PR Curve (AUC = {pr_auc:.3f})",
                line=dict(color="green", width=2),
            ),
            row=1,
            col=2,
        )

        # Baseline for PR curve
        baseline = y_test.sum() / len(y_test)  # Positive rate
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode="lines",
                name=f"Random Baseline ({baseline:.3f})",
                line=dict(color="red", width=1, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)

        fig.update_layout(
            title="Model Performance Curves", width=1000, height=400, showlegend=True
        )

        run.log({"roc_pr_curves": fig})

        logger.info("Model performance summary:")
        logger.info(
            "ROC-AUC: %s (%s)",
            f"{roc_auc:.4f}",
            (
                "Excellent"
                if roc_auc > 0.9
                else "Good"
                if roc_auc > 0.8
                else "Fair"
                if roc_auc > 0.7
                else "Poor"
            ),
        )
        logger.info(
            "PR-AUC: %s (%s)",
            f"{pr_auc:.4f}",
            (
                "Excellent"
                if pr_auc > 0.8
                else "Good"
                if pr_auc > 0.6
                else "Fair"
                if pr_auc > 0.4
                else "Poor"
            ),
        )
        logger.info("Baseline (random): %s", f"{baseline:.4f}")

        # Prediction distribution analysis

        # Create prediction distribution plots
        fig_dist = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Prediction Probability Distribution",
                "Prediction Probabilities by Class",
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Overall distribution
        fig_dist.add_trace(
            go.Histogram(
                x=y_proba,
                nbinsx=50,
                name="All Predictions",
                opacity=0.7,
                marker_color="lightblue",
            ),
            row=1,
            col=1,
        )

        # Distribution by class
        fig_dist.add_trace(
            go.Histogram(
                x=y_proba[y_test == 0],
                nbinsx=30,
                name="Negative Class (0)",
                opacity=0.7,
                marker_color="red",
            ),
            row=1,
            col=2,
        )

        fig_dist.add_trace(
            go.Histogram(
                x=y_proba[y_test == 1],
                nbinsx=30,
                name="Positive Class (1)",
                opacity=0.7,
                marker_color="green",
            ),
            row=1,
            col=2,
        )

        fig_dist.update_xaxes(title_text="Prediction Probability", row=1, col=1)
        fig_dist.update_yaxes(title_text="Count", row=1, col=1)
        fig_dist.update_xaxes(title_text="Prediction Probability", row=1, col=2)
        fig_dist.update_yaxes(title_text="Count", row=1, col=2)

        fig_dist.update_layout(
            title="Prediction Probability Distributions",
            width=1000,
            height=400,
            showlegend=True,
            barmode="overlay",  # For overlapping histograms in the second subplot
        )

        run.log({"prediction_distributions": fig_dist})

        # Calculate separation metrics
        mean_pos = y_proba[y_test == 1].mean()
        mean_neg = y_proba[y_test == 0].mean()
        separation = abs(mean_pos - mean_neg)

        logger.info("Prediction analysis:")
        logger.info("Mean probability for positive class: %s", f"{mean_pos:.4f}")
        logger.info("Mean probability for negative class: %s", f"{mean_neg:.4f}")
        logger.info(
            "Class separation: %s (%s)",
            f"{separation:.4f}",
            (
                "Good"
                if separation > 0.3
                else "Moderate"
                if separation > 0.1
                else "Poor"
            ),
        )
        logger.info("Optimal threshold (balanced): 0.5")

        logger.info("=== Completion Prediction Evaluation ===")
        logger.info("Task: Predicting if users complete books they interact with")
        logger.info("Mapping shape: %s", user_book_mapping.shape)

        # Create test mapping for completion prediction
        test_indices = test_df.index
        if {"user_id", "book_id"}.issubset(df.columns):
            test_mapping = df.loc[test_indices, ["user_id", "book_id"]].copy()
        else:
            test_mapping = user_book_mapping.iloc[test_indices].copy()
        test_mapping["prediction_score"] = y_proba
        test_mapping["true_completion"] = y_test.values  # is_read target

        logger.info("Test mapping created: %s samples", len(test_mapping))
        logger.info("Users in test: %s", test_mapping["user_id"].nunique())
        logger.info(
            "Books completed in test: %s", test_mapping["true_completion"].sum()
        )

        test_mapping.head()

        # Ranking evaluation functions moved to module level

        logger.info("Computing ranking metrics")
        ranking_results = evaluate_ranking_metrics(
            test_mapping, k_values=[5, 10, 20, 50]
        )

        logger.info("Users evaluated: %s", ranking_results["users_evaluated"])
        logger.info("\nRanking metrics:")
        for metric, value in ranking_results.items():
            if metric != "users_evaluated":
                logger.info("%s: %s", metric.upper(), f"{value:.4f}")
        # Log ranking metrics to Weights & Biases
        _rank_metrics = {
            "ranking/users_evaluated": ranking_results.get("users_evaluated", 0)
        }
        for _k in [5, 10, 20, 50]:
            _rank_metrics.update(
                {
                    f"ranking/precision@{_k}": float(
                        ranking_results.get(f"precision@{_k}", 0.0)
                    ),
                    f"ranking/recall@{_k}": float(
                        ranking_results.get(f"recall@{_k}", 0.0)
                    ),
                    f"ranking/map@{_k}": float(ranking_results.get(f"map@{_k}", 0.0)),
                    f"ranking/ndcg@{_k}": float(ranking_results.get(f"ndcg@{_k}", 0.0)),
                }
            )
        run.log(_rank_metrics)

        # Also log a compact table
        rank_table = wandb.Table(columns=["k", "precision", "recall", "map", "ndcg"])
        for _k in [5, 10, 20, 50]:
            rank_table.add_data(
                _k,
                float(ranking_results.get(f"precision@{_k}", 0.0)),
                float(ranking_results.get(f"recall@{_k}", 0.0)),
                float(ranking_results.get(f"map@{_k}", 0.0)),
                float(ranking_results.get(f"ndcg@{_k}", 0.0)),
            )
        run.log({"ranking/metrics_table": rank_table})

        # Promote commonly tracked ones to summary
        for _key in ["precision@20", "recall@20", "map@20", "ndcg@20"]:
            run.summary[f"ranking/{_key}"] = float(ranking_results.get(_key, 0.0))

        # Performance report

        logger.info("=" * 80)
        logger.info("BOOK RECOMMENDATION SYSTEM - PERFORMANCE REPORT")
        logger.info("=" * 80)

        logger.info(
            "%s",
            f"""
        DATASET OVERVIEW:
        • Total samples: {len(df):,}
        • Training samples: {len(X_train):,}
        • Test samples: {len(X_test):,}
        • Features: {X_train.shape[1]}
        • Unique users in test: {test_mapping["user_id"].nunique():,}
        • Users evaluated for ranking: {ranking_results["users_evaluated"]:,}

        CLASSIFICATION PERFORMANCE:
        • Accuracy: {accuracy_score(y_test, preds):.4f}
        • Precision: {precision_score(y_test, preds):.4f}
        • Recall: {recall_score(y_test, preds):.4f}
        • F1-Score: {f1_score(y_test, preds):.4f}
        • ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}

        RANKING PERFORMANCE:""",
        )

        # Format ranking metrics nicely
        metrics_by_k = {}
        for k in [5, 10, 20, 50]:
            metrics_by_k[k] = {
                "Precision": ranking_results[f"precision@{k}"],
                "Recall": ranking_results[f"recall@{k}"],
                "mAP": ranking_results[f"map@{k}"],
                "NDCG": ranking_results[f"ndcg@{k}"],
            }

        logger.info("   ┌─────────┬──────────┬─────────┬─────────┬─────────┐")
        logger.info("   │    k    │ Precision│  Recall │   mAP   │  NDCG   │")
        logger.info("   ├─────────┼──────────┼─────────┼─────────┼─────────┤")
        for k in [5, 10, 20, 50]:
            metrics = metrics_by_k[k]
            logger.info(
                "   │   @%s   │  %s  │ %s  │ %s  │ %s  │",
                f"{k:2d}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['mAP']:.4f}",
                f"{metrics['NDCG']:.4f}",
            )
        logger.info("   └─────────┴──────────┴─────────┴─────────┴─────────┘")

        logger.info(
            "%s",
            f"""
        RECOMMENDATION QUALITY INSIGHTS:
        • mAP@20 of {ranking_results["map@20"]:.4f} indicates {"excellent" if ranking_results["map@20"] > 0.3 else "good" if ranking_results["map@20"] > 0.1 else "moderate"} ranking quality
        • Precision@20 of {ranking_results["precision@20"]:.4f} means {ranking_results["precision@20"] * 100:.1f}% of top-20 recommendations are relevant
        • Model successfully learns from {X_train.shape[1]} engineered features

        BUSINESS IMPACT:
        • For every 20 books recommended, ~{ranking_results["precision@20"] * 20:.0f} will be relevant to the user
        • {ranking_results["recall@20"] * 100:.1f}% of relevant books are captured in top-20 recommendations
        • Strong classification performance (AUC: {roc_auc_score(y_test, y_proba):.3f}) enables confident ranking
        """,
        )

        logger.info("=" * 80)

        # Save report to file
        report_text = f"""Book Recommendation System Performance Report
        Generated: {pd.Timestamp.now()}

        Dataset Overview:
        - Total samples: {len(df):,}
        - Training samples: {len(X_train):,}
        - Test samples: {len(X_test):,}
        - Features: {X_train.shape[1]}
        - Users evaluated: {ranking_results["users_evaluated"]:,}

        Classification Metrics:
        - Accuracy: {accuracy_score(y_test, preds):.4f}
        - Precision: {precision_score(y_test, preds):.4f}
        - Recall: {recall_score(y_test, preds):.4f}
        - F1-Score: {f1_score(y_test, preds):.4f}
        - ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}

        Ranking Metrics:
        """

        for k in [5, 10, 20, 50]:
            report_text += f"@{k}: Precision={ranking_results[f'precision@{k}']:.4f}, Recall={ranking_results[f'recall@{k}']:.4f}, mAP={ranking_results[f'map@{k}']:.4f}, NDCG={ranking_results[f'ndcg@{k}']:.4f}\n"

        # Save to file
        performance_report_path = save_to_file(report_text, PERFORMANCE_REPORT_FILENAME)
        artifact = wandb.Artifact(
            name=PERFORMANCE_REPORT_FILENAME,
            type="performance_report",
            description="Performance report for book recommendation model training",
        )
        artifact.add_file(performance_report_path)
        run.log_artifact(artifact)

        logger.info("Saving model and artifacts")

        # Save the trained PyTorch model (state dict)
        model_path = save_model(model, MODEL_FILENAME)
        artifact = wandb.Artifact(
            name=MODEL_FILENAME,
            type="model",
            description="Model file for book recommender",
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)

        # Save test results for further analysis
        test_results_path = save_df_to_csv(test_mapping, TEST_RESULTS_FILENAME)
        artifact = wandb.Artifact(
            name=TEST_RESULTS_FILENAME,
            type="test_result",
            description="Test results for book recommendation model",
        )
        artifact.add_file(test_results_path)
        run.log_artifact(artifact)

        # Save ranking metrics summary
        ranking_summary = pd.DataFrame([ranking_results]).T
        ranking_summary.columns = ["value"]
        ranking_summary_path = save_df_to_csv(ranking_summary, RANKING_SUMMARY_FILENAME)

        artifact = wandb.Artifact(
            name=RANKING_SUMMARY_FILENAME,
            type="ranking_summary",
            description="Ranking summary for book recommendation model",
        )
        artifact.add_file(ranking_summary_path)
        run.log_artifact(artifact)

        # Prepare example inputs matching the model's forward signature
        model.eval()
        example_x = torch.tensor(X_tr[:1], dtype=torch.float32, device=device)
        example_u = torch.tensor(u_tr[:1], dtype=torch.long, device=device)
        example_b = torch.tensor(b_tr[:1], dtype=torch.long, device=device)

        example_inputs = (example_x, example_u, example_b)
        onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
        onnx_path = save_model(onnx_program, ONNX_FILENAME, onnx=True)

        artifact = wandb.Artifact(
            name=ONNX_FILENAME,
            type="model",
            description="ONNX model file for book recommender",
        )
        artifact.add_file(onnx_path)
        run.log_artifact(artifact)


if __name__ == "__main__":
    main()

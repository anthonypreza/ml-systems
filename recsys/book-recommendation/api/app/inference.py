import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class RecommenderService:
    def __init__(self,
                 cache_dir: str,
                 model_path: str,
                 data_path: str,
                 book_meta_path: Optional[str] = None,
                 mapping_path: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.book_meta_path = Path(book_meta_path) if book_meta_path else None
        self.mapping_path = Path(mapping_path) if mapping_path else None

        self.df: Optional[pd.DataFrame] = None
        self.book_meta: Optional[pd.DataFrame] = None

        self.TARGET = "is_read"
        self.ID_USER = "user_id"
        self.ID_BOOK = "book_id"

        self.user_enc = LabelEncoder()
        self.book_enc = LabelEncoder()
        self.scaler = StandardScaler()

        self.ort_session: Optional[ort.InferenceSession] = None

        # Populated during prepare()
        self.dense_cols: List[str] = []
        self.skewed_cols: List[str] = [
            "rating_count",
            "rating_count_book",
            "book_id_nunique",
            "author_ratings_count",
            "author_text_reviews_count",
            "user_id_nunique",
            "user_rating_count_historical",
            "user_author_interaction_count",
        ]

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def load_artifacts(self) -> None:
        self.df = pd.read_csv(self.data_path)
        print(f"[svc] Loaded data: {self.data_path} -> {self.df.shape} columns={list(self.df.columns)[:6]}…")
        if self.book_meta_path and self.book_meta_path.exists():
            self.book_meta = pd.read_csv(self.book_meta_path)
            print(f"[svc] Loaded book metadata: {self.book_meta_path} -> {self.book_meta.shape} cols={list(self.book_meta.columns)}")
        
        # If IDs are missing, try to recover via mapping artifact
        missing_user = self.ID_USER not in self.df.columns
        missing_book = self.ID_BOOK not in self.df.columns
        if (missing_user or missing_book) and self.mapping_path and self.mapping_path.exists():
            mapping = pd.read_csv(self.mapping_path)
            # Align via a sample_index column; if not present, synthesize from order
            if "sample_index" not in mapping.columns:
                if "index" in mapping.columns:
                    mapping = mapping.rename(columns={"index": "sample_index"})
                else:
                    mapping["sample_index"] = range(len(mapping))
            # Attach sample_index to df as its current row order
            tmp = self.df.copy().reset_index().rename(columns={"index": "sample_index"})
            self.df = tmp.merge(mapping, on="sample_index", how="left")
            print(f"[svc] Merged mapping: {self.mapping_path} -> df cols now include user_id/book_id? user_id={self.ID_USER in self.df.columns}, book_id={self.ID_BOOK in self.df.columns}")

        # Normalize ID dtypes and known columns
        if self.df is not None:
            if self.ID_USER in self.df.columns:
                self.df[self.ID_USER] = self.df[self.ID_USER].astype(str)
            if self.ID_BOOK in self.df.columns:
                self.df[self.ID_BOOK] = self.df[self.ID_BOOK].astype(str)
            if self.TARGET in self.df.columns:
                self.df[self.TARGET] = self.df[self.TARGET].fillna(0).astype(int)

        if self.book_meta is not None:
            if self.ID_BOOK in self.book_meta.columns:
                self.book_meta[self.ID_BOOK] = self.book_meta[self.ID_BOOK].astype(str)

    def _build_preprocessing(self) -> None:
        assert self.df is not None
        df = self.df.copy()

        # features available for the model
        drop_cols = {self.TARGET, self.ID_USER, self.ID_BOOK, "sample_index"}
        self.dense_cols = [c for c in df.columns if c not in drop_cols]

        # log1p transform for skewed cols when available
        for col in self.skewed_cols:
            if col in self.dense_cols:
                df[col] = np.log1p(df[col])

        # Fit encoders deterministically on full dataset
        self.user_enc.fit(df[self.ID_USER].astype(str))
        self.book_enc.fit(df[self.ID_BOOK].astype(str))

        # Fit scaler on first 80% to mimic training roughly
        n = len(df)
        split_idx = int(n * 0.8)
        X = df[self.dense_cols].astype(np.float32)
        self.scaler.fit(X.iloc[:split_idx])

    def _transform_batch(self, batch_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensure same transforms as in _build_preprocessing
        X = batch_df[self.dense_cols].copy()
        for col in self.skewed_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        X = X.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        u_idx = self.user_enc.transform(batch_df[self.ID_USER].astype(str)).astype(np.int64)
        b_idx = self.book_enc.transform(batch_df[self.ID_BOOK].astype(str)).astype(np.int64)
        return X_scaled, u_idx, b_idx

    def prepare(self) -> None:
        self.load_artifacts()
        self._build_preprocessing()
        providers = ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(str(self.model_path), providers=providers)
        # Capture expected input shapes if statically encoded in the model
        try:
            ishape = self.ort_session.get_inputs()[0].shape
            # Typical shape is [1, features] when exported with batch=1
            self._exp_batch = ishape[0] if len(ishape) > 0 else None
            self._exp_feat = ishape[1] if len(ishape) > 1 else None
            # Normalize possible None/str values
            self._exp_batch = int(self._exp_batch) if isinstance(self._exp_batch, (int,)) else self._exp_batch
            self._exp_feat = int(self._exp_feat) if isinstance(self._exp_feat, (int,)) else self._exp_feat
        except Exception:
            self._exp_batch, self._exp_feat = None, None

    def _align_features(self, X: np.ndarray) -> np.ndarray:
        exp_f = getattr(self, "_exp_feat", None)
        if isinstance(exp_f, int) and exp_f > 0:
            if X.shape[1] > exp_f:
                return X[:, :exp_f]
            if X.shape[1] < exp_f:
                pad = np.zeros((X.shape[0], exp_f - X.shape[1]), dtype=X.dtype)
                return np.concatenate([X, pad], axis=1)
        return X

    def run_onnx(self, X: np.ndarray, U: np.ndarray, B: np.ndarray) -> np.ndarray:
        assert self.ort_session is not None
        X = self._align_features(X)

        # If model was exported with a fixed batch size (=1), run row-wise
        exp_b = getattr(self, "_exp_batch", None)
        if isinstance(exp_b, int) and exp_b == 1 and X.shape[0] != 1:
            outs = []
            for i in range(X.shape[0]):
                xi = X[i : i + 1]
                ui = U[i : i + 1]
                bi = B[i : i + 1]
                feed = {inp.name: val for inp, val in zip(self.ort_session.get_inputs(), [xi, ui, bi])}
                out = self.ort_session.run(None, feed)[0]
                outs.append(out.reshape(-1))
            logits = np.concatenate(outs, axis=0).astype(np.float32)
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            return self.sigmoid(logits)

        # Normal batched path
        inputs_np = [X, U, B]
        feed = {inp.name: val for inp, val in zip(self.ort_session.get_inputs(), inputs_np)}
        outputs = self.ort_session.run(None, feed)[0]
        logits = outputs.astype(np.float32).reshape(-1)
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return self.sigmoid(logits)

    def recommend_for_user(self,
                            user_id: str,
                            k: int = 10,
                            exclude_book_ids: Optional[List[str]] = None,
                            filter_seen: bool = True) -> pd.DataFrame:
        assert self.df is not None
        exclude_set = set(exclude_book_ids or [])

        # Select rows for this user from our dataset (fake predictions over observed pairs)
        if user_id is None or str(user_id).lower() in {"", "none", "nan", "undefined"}:
            return pd.DataFrame(columns=[self.ID_BOOK, "score"])  # invalid user

        user_rows = self.df[self.df[self.ID_USER].astype(str) == str(user_id)].copy()
        if filter_seen and self.TARGET in user_rows.columns:
            # keep rows (books) not read yet
            user_rows = user_rows[user_rows[self.TARGET] == 0]

        if len(exclude_set) > 0:
            user_rows = user_rows[~user_rows[self.ID_BOOK].astype(str).isin(exclude_set)]

        if len(user_rows) == 0:
            return pd.DataFrame(columns=[self.ID_BOOK, "score"])  # nothing to score

        # Exclude books already interacted (any interaction) when requested
        if filter_seen:
            hist_books = set(
                self.df[self.df[self.ID_USER] == str(user_id)][self.ID_BOOK].astype(str).unique()
            )
            user_rows = user_rows[~user_rows[self.ID_BOOK].astype(str).isin(hist_books)]

        if len(user_rows) == 0:
            return pd.DataFrame(columns=[self.ID_BOOK, "score"])  # nothing to score

        X, U, B = self._transform_batch(user_rows)
        scores = self.run_onnx(X, U, B)
        out = user_rows[[self.ID_BOOK]].copy()
        out["score"] = scores
        out = out.sort_values("score", ascending=False).drop_duplicates(self.ID_BOOK)
        if k:
            out = out.head(k)

        # Attach metadata if available
        if self.book_meta is not None:
            out[self.ID_BOOK] = out[self.ID_BOOK].astype(str)
            self.book_meta[self.ID_BOOK] = self.book_meta[self.ID_BOOK].astype(str)
            out = out.merge(self.book_meta, on=self.ID_BOOK, how="left")
        return out.reset_index(drop=True)

    def user_history(self, user_id: str, limit: int = 50) -> pd.DataFrame:
        assert self.df is not None
        hist = self.df[self.df[self.ID_USER].astype(str) == str(user_id)].copy()
        # Sort by an available temporal proxy
        if "date_added" in hist.columns:
            try:
                hist["date_added"] = pd.to_datetime(hist["date_added"], errors="coerce")
                hist = hist.sort_values("date_added", ascending=False)
            except Exception:
                pass
        if self.book_meta is not None:
            hist[self.ID_BOOK] = hist[self.ID_BOOK].astype(str)
            self.book_meta[self.ID_BOOK] = self.book_meta[self.ID_BOOK].astype(str)
            hist = hist.merge(self.book_meta, on=self.ID_BOOK, how="left")
        return hist.head(limit)

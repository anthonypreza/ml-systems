#!/usr/bin/env python3
"""
Build & Push to Weaviate (HNSW + cosine) in one shot.

- Loads your trained VideoTextContrastiveModel
- Builds embeddings for a dataset (msrvtt | synthetic)
- Ensures Weaviate schema exists (VideoClip)
- Upserts vectors + metadata in batches

Usage examples:
  python scripts/build_and_push_weaviate.py --dataset msrvtt --max-videos 100
  python scripts/build_and_push_weaviate.py --dataset synthetic --normalize
"""

import os
import sys
import json
import uuid
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import torch

# --- Ensure project root on sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Project imports ---
from video_search.core.config import VideoSearchConfig
from video_search.core.model import VideoTextContrastiveModel
from video_search.utils.data import load_msrvtt_dataset, create_synthetic_dataset
from video_search.search.index import build_video_index


# -----------------------
# Utilities
# -----------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def load_trained_model(config: VideoSearchConfig) -> VideoTextContrastiveModel:
    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(f"Model file not found: {config.model_save_path}")
    print(f"📦 Loading trained model from {config.model_save_path} ...")
    ckpt = torch.load(
        config.model_save_path, map_location=config.device, weights_only=False
    )

    model = VideoTextContrastiveModel(
        video_embed_dim=config.video_embed_dim,
        text_embed_dim=config.text_embed_dim,
        text_model_name=config.text_model_name,
        temperature=config.temperature,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device).eval()
    print("✅ Model loaded.")
    return model


def ensure_schema(
    weaviate_url: str,
    class_name: str = "VideoClip",
    distance: str = "cosine",
    ef_construction: int = 128,
    max_connections: int = 16,
    api_key: Optional[str] = None,
) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # check if class exists
    r = requests.get(
        f"{weaviate_url}/v1/schema/{class_name}", headers=headers, timeout=20
    )
    if r.status_code == 200:
        print(f"ℹ️  Schema class '{class_name}' already exists.")
        return
    elif r.status_code != 404:
        print("⚠️  Unexpected schema GET response:", r.text)
        r.raise_for_status()

    schema = {
        "classes": [
            {
                "class": class_name,
                "vectorizer": "none",
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": distance,
                    "efConstruction": ef_construction,
                    "maxConnections": max_connections,
                },
                "properties": [
                    {"name": "video_id", "dataType": ["text"]},
                    {"name": "start_s", "dataType": ["number"]},
                    {"name": "end_s", "dataType": ["number"]},
                    {"name": "title", "dataType": ["text"]},
                    {"name": "channel", "dataType": ["text"]},
                    {"name": "url", "dataType": ["text"]},
                    {"name": "caption", "dataType": ["text"]},
                    {"name": "tags", "dataType": ["text[]"]},
                ],
            }
        ]
    }
    print(f"🛠️  Creating schema class '{class_name}' ...")
    r = requests.post(
        f"{weaviate_url}/v1/schema", headers=headers, json=schema, timeout=30
    )
    try:
        r.raise_for_status()
    except Exception:
        print("❌ Schema creation failed:", r.text)
        raise
    print("✅ Schema created.")


def path_to_properties(path: str, channel: str) -> Dict:
    base = os.path.basename(path)
    video_id = os.path.splitext(base)[0]
    return {
        "video_id": video_id,
        "start_s": 0.0,
        "end_s": 0.0,
        "title": base,
        "channel": channel,
        "url": f"file://{path}",  # replace with real URL if you have one
        "caption": "",
        "tags": [],
    }


def batch_upsert(
    weaviate_url: str,
    class_name: str,
    vectors: np.ndarray,
    paths: List[str],
    channel: str,
    api_key: Optional[str] = None,
    batch_size: int = 256,
    timeout: int = 120,
) -> None:
    assert len(vectors) == len(paths)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{weaviate_url}/v1/batch/objects"
    total = len(paths)
    sent = 0
    t0 = time.time()

    while sent < total:
        chunk_paths = paths[sent : sent + batch_size]
        chunk_vecs = vectors[sent : sent + batch_size]
        objects = []
        for p, v in zip(chunk_paths, chunk_vecs):
            objects.append(
                {
                    "class": class_name,
                    "id": str(uuid.uuid4()),
                    "properties": path_to_properties(p, channel=channel),
                    "vector": v.tolist(),  # float32 list
                }
            )
        payload = {"objects": objects}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            r.raise_for_status()
        except Exception:
            print("❌ Batch upsert failed:", r.text)
            raise
        sent += len(objects)
        print(f"⬆️  Upserted {sent}/{total}")

    print(f"✅ Completed upsert of {total} objects in {time.time() - t0:.1f}s")


# -----------------------
# Main flow
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build embeddings and push to Weaviate in one shot."
    )
    ap.add_argument("--dataset", choices=["msrvtt", "synthetic"], default="msrvtt")
    ap.add_argument("--max-videos", type=int, default=100)
    ap.add_argument(
        "--model-path", type=str, default=None, help="Override model path in config"
    )
    ap.add_argument(
        "--weaviate-url",
        type=str,
        default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"),
    )
    ap.add_argument(
        "--weaviate-api-key", type=str, default=os.environ.get("WEAVIATE_API_KEY")
    )
    ap.add_argument("--class-name", type=str, default="VideoClip")
    ap.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Channel metadata value; defaults to dataset name",
    )
    ap.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("WEAVIATE_BATCH", "256"))
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize vectors before upsert (safe even if already normalized)",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building; load existing pickle index instead",
    )
    ap.add_argument(
        "--index-path",
        type=str,
        default="models/video_index.pkl",
        help="Pickle index path (read if --skip-build; also written if built)",
    )
    # HNSW knobs for initial schema creation (ignored if class already exists)
    ap.add_argument("--ef-construction", type=int, default=128)
    ap.add_argument("--max-connections", type=int, default=16)
    args = ap.parse_args()

    config = VideoSearchConfig()
    if args.model_path:
        config.model_save_path = args.model_path

    # ---- Ensure Weaviate up and schema present ----
    print(f"🔗 Weaviate: {args.weaviate_url}")
    ensure_schema(
        weaviate_url=args.weaviate_url,
        class_name=args.class_name,
        distance="cosine",
        ef_construction=args.ef_construction,
        max_connections=args.max_connections,
        api_key=args.weaviate_api_key,
    )

    # ---- Build or load index ----
    if not args.skip_build:
        print("🧠 Loading model and building video index ...")
        model = load_trained_model(config)

        if args.dataset == "msrvtt":
            triplets = load_msrvtt_dataset(
                data_dir="./data/msrvtt", max_videos=args.max_videos, num_negatives=7
            )
            channel = args.channel or "msrvtt"
        else:
            triplets = create_synthetic_dataset(num_videos=20, num_negatives=7)
            channel = args.channel or "synthetic"

        video_paths = sorted(set(t.video_path for t in triplets))
        print(f"📼 Found {len(video_paths)} videos.")

        # Build embeddings using your existing helper
        idx = build_video_index(
            model=model, video_paths=video_paths, device=config.device
        )  # {"embeddings": [N,D], "paths": [...]}

        # Save pickle for later reuse
        Path(os.path.dirname(args.index_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.index_path, "wb") as f:
            pickle.dump(idx, f)
        print(f"💾 Saved pickle index → {args.index_path}")

    else:
        print(f"📥 Loading existing pickle index from {args.index_path}")
        with open(args.index_path, "rb") as f:
            idx = pickle.load(f)
        channel = args.channel or (
            "msrvtt" if "msrvtt" in args.index_path.lower() else "synthetic"
        )

    vectors = np.asarray(idx["embeddings"], dtype=np.float32)
    paths = list(idx["paths"])
    if args.normalize:
        vectors = l2_normalize(vectors)

    print(
        f"🚚 Pushing {len(paths)} vectors to Weaviate (class={args.class_name}, channel={channel}) ..."
    )
    batch_upsert(
        weaviate_url=args.weaviate_url,
        class_name=args.class_name,
        vectors=vectors,
        paths=paths,
        channel=channel,
        api_key=args.weaviate_api_key,
        batch_size=args.batch_size,
    )
    print("🎉 Done. You can now query with GraphQL nearVector.")


if __name__ == "__main__":
    main()

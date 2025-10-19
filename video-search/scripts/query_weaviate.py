#!/usr/bin/env python3
"""
Query Weaviate with a text search query:
- Tokenize with Hugging Face
- Encode with ONNX Runtime (text encoder)
- nearVector GraphQL against Weaviate (cosine)

Usage:
  python scripts/query_weaviate.py "julienne carrots" --top-k 10

Env vars (optional):
  WEAVIATE_URL (default: http://localhost:8080)
  WEAVIATE_API_KEY (if auth enabled)
"""

import os
import json
import argparse
from typing import List, Optional

import numpy as np
import requests
import onnxruntime as ort
from transformers import AutoTokenizer


def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def load_onnx_session(model_path: str) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    sess = ort.InferenceSession(model_path, providers=providers)
    return sess


def embed_text(
    sess: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int = 128,
) -> np.ndarray:
    """Returns (1, D) float32 embedding (L2-normalized)."""
    enc = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    # ONNX inputs must match names used at export
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    # Export names text output as "text_embedding"
    (emb,) = sess.run(["text_embedding"], inputs)  # (1, D) float32
    # Your model already normalizes, but re-normalize defensively (idempotent)
    emb = l2_normalize(emb, axis=1)
    return emb.astype(np.float32)


def weaviate_near_vector(
    weaviate_url: str,
    class_name: str,
    vector: np.ndarray,
    top_k: int = 10,
    ef: Optional[int] = None,
    channel_filter: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Calls Weaviate GraphQL nearVector and returns JSON."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    vec_list = ", ".join(f"{x:.8f}" for x in vector.flatten().tolist())
    near_vec = (
        f"nearVector: {{ vector: [{vec_list}]{', ef: ' + str(ef) if ef else ''} }}"
    )
    where = (
        f'where: {{ operator: Equal, path: ["channel"], valueText: "{channel_filter}" }}'
        if channel_filter
        else ""
    )

    gql = f"""
{{
  Get {{
    {class_name}(
      {near_vec}
      limit: {top_k}
      {where}
    ) {{
      video_id
      title
      url
      start_s
      end_s
      _additional {{ distance score }}
    }}
  }}
}}
""".strip()

    payload = {"query": gql}
    resp = requests.post(
        f"{weaviate_url}/v1/graphql", headers=headers, json=payload, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def print_results(data: dict, class_name: str, query: str, fmt: str):
    try:
        hits = data["data"]["Get"][class_name]
    except Exception:
        print(json.dumps(data, indent=2))
        return

    rows = []
    for h in hits:
        rows.append(
            {
                "video_id": h.get("video_id"),
                "title": h.get("title"),
                "url": h.get("url"),
                "start_s": h.get("start_s"),
                "end_s": h.get("end_s"),
                # Weaviate's "score" is often 1 - distance for cosine; distance is reliable.
                "distance": h.get("_additional", {}).get("distance"),
                "score": h.get("_additional", {}).get("score"),
            }
        )

    if fmt == "json":
        print(json.dumps({"query": query, "results": rows}, indent=2))
        return

    if fmt == "simple":
        for r in rows:
            print(f"{r['distance']:.6f}\t{r['video_id']}\t{r['title']}\t{r['url']}")
        return

    # table
    print(f"\nResults for: {query!r}")
    print("-" * 80)
    for i, r in enumerate(rows, 1):
        dist = r["distance"]
        print(f"{i:2d}. {r['title'] or r['video_id']}  (distance: {dist:.6f})")
        print(f"    url:  {r['url']}")
        print(f"    span: [{r['start_s']}, {r['end_s']}]")
    if not rows:
        print("(no results)")


def main():
    ap = argparse.ArgumentParser(
        description="Text→Vector→Weaviate search (cosine nearVector)."
    )
    ap.add_argument("query", type=str, help="Text query")
    ap.add_argument(
        "--onnx",
        type=str,
        default="models/text_encoder.onnx",
        help="Path to text ONNX model",
    )
    ap.add_argument(
        "--hf-model",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer HF model name",
    )
    ap.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    ap.add_argument(
        "--weaviate-url",
        type=str,
        default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"),
    )
    ap.add_argument("--class-name", type=str, default="VideoClip")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--ef", type=int, default=None, help="HNSW ef (query-time) override"
    )
    ap.add_argument(
        "--channel",
        type=str,
        default=None,
        help='Optional where filter: channel == "<value>"',
    )
    ap.add_argument("--api-key", type=str, default=os.environ.get("WEAVIATE_API_KEY"))
    ap.add_argument("--format", choices=["table", "json", "simple"], default="table")
    args = ap.parse_args()

    # Load tokenizer + ONNX session
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    sess = load_onnx_session(args.onnx)

    # Encode query
    emb = embed_text(sess, tokenizer, args.query, max_length=args.max_length)

    # Query Weaviate
    data = weaviate_near_vector(
        weaviate_url=args.weaviate_url,
        class_name=args.class_name,
        vector=emb,
        top_k=args.top_k,
        ef=args.ef,
        channel_filter=args.channel,
        api_key=args.api_key,
    )

    # Print results
    print_results(data, args.class_name, args.query, args.format)


if __name__ == "__main__":
    main()

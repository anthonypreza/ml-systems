API service for Book Recommender

Overview

- FastAPI app serving ONNX model via onnxruntime
- Downloads artifacts (ONNX, completion_prediction.csv, book_metadata.csv) from Weights & Biases
- Simple web UI to pick a user and get recommendations, with re-ranking to avoid repeats

Config

- Env vars:
  - WANDB_ENTITY: your W&B entity (user or org)
  - WANDB_PROJECT: defaults to book-recommendation
  - MODEL_ARTIFACT: defaults to nn_recommender_model.onnx:latest
  - DATA_ARTIFACT: defaults to completion_prediction.csv:latest
  - BOOK_META_ARTIFACT: defaults to book_metadata.csv:latest

Run locally

- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 8000

Docker

- docker build -t bookrec-api .
- docker run -p 8000:8000 \
  -e WANDB_ENTITY=your_entity \
  -e WANDB_PROJECT=book-recommendation \
  bookrec-api


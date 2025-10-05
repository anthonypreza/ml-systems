import os
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .inference import RecommenderService


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
STATIC_DIR = APP_DIR / "static"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _download_artifact(
    name_with_alias: str, out_dir: Path, expected_glob: str | None = None
) -> Path:
    # Non-interactive login (uses WANDB_API_KEY env var)
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    api = wandb.Api()
    # Allow shorthand like "book-recommendation/foo.csv:latest" by inferring entity
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT", "book-recommendation")
    if "/" not in name_with_alias:
        if entity:
            name_with_alias = f"{entity}/{project}/{name_with_alias}"
        else:
            name_with_alias = f"{project}/{name_with_alias}"

    # Create a unique subfolder per artifact to avoid mixing files
    safe = name_with_alias.replace("/", "_").replace(":", "_")
    dest_dir = out_dir / safe
    dest_dir.mkdir(parents=True, exist_ok=True)

    art = api.artifact(name_with_alias)
    art.download(root=str(dest_dir))

    # Find best candidate file within dest_dir
    pattern = expected_glob or "*"
    files = [f for f in dest_dir.rglob(pattern) if f.is_file()]
    if not files:
        # fallback: any file
        files = [f for f in dest_dir.rglob("*") if f.is_file()]
    if not files:
        # ultimately return directory if nothing found
        return dest_dir
    # pick the first file deterministically (sorted)
    return sorted(files)[0]


def bootstrap_service() -> RecommenderService:
    model_art = os.getenv("MODEL_ARTIFACT", "nn_recommender_model.onnx:latest")
    data_art = os.getenv("DATA_ARTIFACT", "completion_prediction.csv:latest")
    book_meta_art = os.getenv("BOOK_META_ARTIFACT", "book_metadata.csv:latest")
    user_book_map_art = os.getenv(
        "USER_BOOK_MAPPING_ARTIFACT", "completion_user_book_mapping.csv:latest"
    )

    model_path = _download_artifact(model_art, CACHE_DIR, expected_glob="*.onnx")
    data_path = _download_artifact(data_art, CACHE_DIR, expected_glob="*.csv")
    try:
        book_meta_path = _download_artifact(
            book_meta_art, CACHE_DIR, expected_glob="*.csv"
        )
    except Exception:
        book_meta_path = None
    # Try to download mapping; if the configured name fails, try alternate name
    try:
        mapping_path = _download_artifact(
            user_book_map_art, CACHE_DIR, expected_glob="*.csv"
        )
    except Exception:
        try:
            mapping_path = _download_artifact(
                "user_book_mapping.csv:latest", CACHE_DIR, expected_glob="*.csv"
            )
        except Exception:
            mapping_path = None

    svc = RecommenderService(
        cache_dir=str(CACHE_DIR),
        model_path=str(model_path),
        data_path=str(data_path),
        book_meta_path=str(book_meta_path) if book_meta_path else None,
        mapping_path=str(mapping_path) if mapping_path else None,
    )
    svc.prepare()
    return svc


app = FastAPI(title="Book Recommender API", version="0.1.0")
service: Optional[RecommenderService] = None


@app.on_event("startup")
def _startup():
    global service
    service = bootstrap_service()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/users")
def list_users(limit: int = 200):
    assert service is not None
    df = service.df
    if df is None:
        raise HTTPException(500, "Dataframe not loaded")
    series = df[service.ID_USER]
    series = series[~series.isna()]
    vc = series.astype(str).value_counts()
    users = (
        vc.reset_index(name="count").rename(columns={"index": "user_id"}).head(limit)
    )
    return {"users": users.to_dict(orient="records")}


@app.get("/user/{user_id}/history")
def user_history(user_id: str, limit: int = 50):
    assert service is not None
    if user_id.lower() in {"", "none", "nan", "undefined"}:
        raise HTTPException(400, "Invalid user_id")
    hist = service.user_history(user_id, limit=limit)
    cols = [
        c
        for c in [
            service.ID_BOOK,
            "title",
            "author_name",
            "publication_year",
            "language_code",
            "format",
            "is_ebook",
            service.TARGET,
        ]
        if c in hist.columns
    ]
    df_obj = hist[cols].astype(object)
    hist_json = df_obj.where(pd.notna(df_obj), None).to_dict(orient="records")
    return {"history": hist_json}


@app.get("/recommendations")
def recommendations(
    user_id: str,
    k: int = 10,
    exclude: Optional[str] = Query(
        None, description="Comma-separated book_ids to exclude"
    ),
    filter_seen: bool = True,
):
    assert service is not None
    exclude_ids: list[str] = []
    if exclude:
        exclude_ids = [x for x in exclude.split(",") if x]
    if user_id.lower() in {"", "none", "nan", "undefined"}:
        return {"recommendations": []}
    recs = service.recommend_for_user(
        user_id=user_id, k=k, exclude_book_ids=exclude_ids, filter_seen=filter_seen
    )
    cols = [
        c
        for c in [
            service.ID_BOOK,
            "score",
            "title",
            "author_name",
            "publication_year",
            "language_code",
            "format",
            "is_ebook",
        ]
        if c in recs.columns
    ]
    df_obj = recs[cols].astype(object)
    recs_json = df_obj.where(pd.notna(df_obj), None).to_dict(orient="records")
    return {"recommendations": recs_json}


# Static UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(index_html)

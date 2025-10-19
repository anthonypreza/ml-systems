use std::{env, path::PathBuf, sync::Arc};

use anyhow::{anyhow, Context, Result};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use ndarray::{Axis, Ix2};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    weaviate_url: String,
    class_name: String,
    client: Client,
    tokenizer: Arc<Tokenizer>,
    ort_session: Arc<tokio::sync::Mutex<Session>>,
}

#[derive(Deserialize)]
struct QueryReq {
    q: String,
    #[serde(default = "default_topk")]
    top_k: usize,
    #[serde(default)]
    ef_search: Option<i32>,
    #[serde(default)]
    channel: Option<String>,
}

fn default_topk() -> usize {
    10
}

#[derive(Serialize)]
struct Hit {
    video_id: Option<String>,
    title: Option<String>,
    url: Option<String>,
    start_s: Option<f64>,
    end_s: Option<f64>,
    distance: Option<f64>,
    score: Option<f64>,
}

#[derive(Serialize)]
struct QueryResp {
    results: Vec<Hit>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let weaviate_url = env::var("WEAVIATE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "http://weaviate:8080".into());
    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "./models/text_encoder.onnx".into());
    let tok_path = env::var("TOKENIZER_PATH").unwrap_or_else(|_| "./models/tokenizer.json".into());
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8088);

    info!(%weaviate_url, %model_path, %tok_path, port, "starting API (ort)");

    // load tokenizer
    let tokenizer = Arc::new(
        Tokenizer::from_file(PathBuf::from(&tok_path))
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {}", e))?,
    );

    // build ORT session (CPU; add .with_intra_threads / EP config as needed)
    let ort_session = Arc::new(tokio::sync::Mutex::new(
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load ONNX model at {}", model_path))?,
    ));

    let client = Client::new();
    let class_name = "VideoClip".to_string();
    ensure_weaviate_schema(&client, &weaviate_url, &class_name).await?;

    let state = AppState {
        weaviate_url,
        class_name,
        client,
        tokenizer,
        ort_session,
    };

    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/query/text", post(query_text))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    info!(?listener, "listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn query_text(
    State(state): State<AppState>,
    Json(req): Json<QueryReq>,
) -> Result<Json<QueryResp>, axum::http::StatusCode> {
    match handle_query(&state, req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(err) => {
            error!(error=?err, "query failed");
            Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_query(state: &AppState, req: QueryReq) -> Result<QueryResp> {
    // --- 1) Tokenize (fast tokenizer, no padding required—dynamic length is fine) ---
    // Use encode with add_special_tokens=true to mirror training.
    let enc = state
        .tokenizer
        .encode(req.q.clone(), true)
        .map_err(|e| anyhow::anyhow!("tokenize error: {}", e))?;

    let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
    let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&m| m as i64).collect();

    let n_tokens = ids.len();
    // If your export assumed a fixed max length, you can pad/truncate here instead.

    // Shape [1, L]
    let input_ids = TensorRef::from_array_view(([1usize, n_tokens], &*ids))?;
    let attn_mask = TensorRef::from_array_view(([1usize, n_tokens], &*mask))?;

    // --- 2) ONNX forward with `ort` ---
    // IMPORTANT: Inputs order must match export order: (input_ids, attention_mask)
    let mut session = state.ort_session.lock().await;
    let outputs = session.run(ort::inputs![input_ids, attn_mask])?;

    // You exported a single output named "text_embedding". `ort` returns by position.
    // Usually it's index 0. If your model yields multiple outputs, adjust the index.
    let emb = outputs[0]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix2>()
        .expect("embedding should be 2D [B,D]");

    // --- 3) L2-normalize defensively (idempotent if already normalized) ---
    let mut v = emb.index_axis(Axis(0), 0).to_owned(); // 1 x D -> D
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }

    // --- 4) Build GraphQL (nearVector) ---
    let top_k = req.top_k;
    let ef = req.ef_search;
    let channel = req.channel;

    let vec_str = v
        .iter()
        .map(|x| format!("{:.8}", x))
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        tokens = n_tokens,
        dim = v.len(),
        "query embedding generated"
    );

    if let Some(efv) = ef {
        warn!(ef = efv, "ignoring ef override; GraphQL nearVector does not support it on this Weaviate version");
    }

    let near_vec = format!("nearVector: {{ vector: [{}] }}", vec_str);

    let where_clause = if let Some(ch) = channel {
        format!(
            "where: {{ operator: Equal, path: [\"channel\"], valueText: \"{}\" }}",
            ch
        )
    } else {
        String::new()
    };

    let gql = format!(
        r#"{{
            Get {{
                {class_name}(
                {near_vec}
                limit: {top_k}
                {where_clause}
                ) {{
                video_id title url start_s end_s
                _additional {{ distance score }}
                }}
            }}
        }}"#,
        class_name = state.class_name,
        near_vec = near_vec,
        top_k = top_k,
        where_clause = where_clause
    );

    #[derive(Serialize)]
    struct GqlPayload<'a> {
        query: &'a str,
    }
    let payload = GqlPayload { query: &gql };

    let resp = state
        .client
        .post(format!("{}/v1/graphql", state.weaviate_url))
        .json(&payload)
        .send()
        .await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("weaviate graphql error {}: {}", status, body);
    }

    let json: serde_json::Value = serde_json::from_str(&body)?;
    if let Some(errors) = json.get("errors") {
        warn!(?errors, "weaviate returned graphql errors");
        return Ok(QueryResp { results: Vec::new() });
    }

    // --- 5) Map to response ---
    let mut results = Vec::<Hit>::new();
    if let Some(arr) = json["data"]["Get"][&state.class_name].as_array() {
        for h in arr {
            results.push(Hit {
                video_id: h
                    .get("video_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                title: h
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                url: h.get("url").and_then(|v| v.as_str()).map(|s| s.to_string()),
                start_s: h.get("start_s").and_then(|v| v.as_f64()),
                end_s: h.get("end_s").and_then(|v| v.as_f64()),
                distance: h
                    .get("_additional")
                    .and_then(|a| a.get("distance"))
                    .and_then(|v| v.as_f64()),
                score: h
                    .get("_additional")
                    .and_then(|a| a.get("score"))
                    .and_then(|v| v.as_f64()),
            });
        }
    }

    Ok(QueryResp { results })
}

async fn ensure_weaviate_schema(client: &Client, base_url: &str, class_name: &str) -> Result<()> {
    let schema_url = format!("{}/v1/schema", base_url);
    let schema = fetch_schema_with_retry(client, &schema_url).await?;
    let already_exists = schema["classes"].as_array().map_or(false, |classes| {
        classes
            .iter()
            .any(|cls| cls["class"].as_str() == Some(class_name))
    });

    if already_exists {
        return Ok(());
    }

    info!(class = class_name, "creating weaviate class");
    let class_def = json!({
        "class": class_name,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "cosine",
            "efConstruction": 128,
            "maxConnections": 16
        },
        "properties": [
            {"name": "video_id", "dataType": ["text"]},
            {"name": "start_s", "dataType": ["number"]},
            {"name": "end_s", "dataType": ["number"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "channel", "dataType": ["text"]},
            {"name": "url", "dataType": ["text"]},
            {"name": "caption", "dataType": ["text"]},
            {"name": "tags", "dataType": ["text[]"]}
        ]
    });

    let create_resp = client.post(&schema_url).json(&class_def).send().await?;
    let create_status = create_resp.status();
    let create_body = create_resp.text().await?;
    if create_status == StatusCode::CONFLICT {
        info!(class = class_name, "weaviate class already exists");
        return Ok(());
    }

    if !create_status.is_success() {
        anyhow::bail!(
            "failed to create weaviate class {} {}: {}",
            class_name,
            create_status,
            create_body
        );
    }

    Ok(())
}

async fn fetch_schema_with_retry(client: &Client, url: &str) -> Result<serde_json::Value> {
    const MAX_ATTEMPTS: usize = 20;
    let mut delay = Duration::from_millis(500);
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=MAX_ATTEMPTS {
        match client.get(url).send().await {
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await?;
                if status.is_success() {
                    return Ok(serde_json::from_str(&body)?);
                }
                last_err = Some(anyhow!("schema fetch {}: {}", status, body));
            }
            Err(err) => {
                last_err = Some(anyhow!(err));
            }
        }

        warn!(attempt, "waiting for Weaviate at {}", url);
        sleep(delay).await;
        delay = (delay * 2).min(Duration::from_secs(5));
    }

    Err(last_err.unwrap_or_else(|| anyhow!("failed to reach weaviate at {}", url)))
}

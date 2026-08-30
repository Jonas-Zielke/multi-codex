//! Discovery for OpenAI-compatible inference endpoints.
//!
//! A "fleet" is the set of endpoints a session can hand work to: Nebius Token
//! Factory for the large model, and whatever is serving smaller models nearby.
//! Rather than make people hand-write provider blocks, this module asks each
//! candidate endpoint what it is, which models it holds, and which wire
//! protocol it speaks.

use codex_http_client::ClientRouteClass;
use codex_http_client::HttpClientFactory;
use codex_http_client::RouteAwareClientPool;
use codex_model_provider_info::WireApi;
use serde_json::Value;
use serde_json::json;
use std::time::Duration;

/// Local probes should fail fast: an unused port is the common case.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// A locally servable inference runtime, identified by its own side-channel API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runtime {
    Ollama,
    LmStudio,
    Vllm,
    /// llama.cpp's server, which is also what Unsloth Studio serves through.
    LlamaCpp,
    /// Speaks the OpenAI API but did not identify itself.
    Unknown,
}

impl Runtime {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::LmStudio => "LM Studio",
            Self::Vllm => "vLLM",
            Self::LlamaCpp => "llama.cpp / Unsloth Studio",
            Self::Unknown => "OpenAI-compatible server",
        }
    }

    /// Short, stable token used when naming a generated provider id.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "lmstudio",
            Self::Vllm => "vllm",
            Self::LlamaCpp => "llamacpp",
            Self::Unknown => "openai-compat",
        }
    }

    /// Port the runtime listens on out of the box.
    pub fn default_port(self) -> Option<u16> {
        match self {
            Self::Ollama => Some(11434),
            Self::LmStudio => Some(1234),
            Self::Vllm => Some(8000),
            Self::LlamaCpp => Some(8080),
            Self::Unknown => None,
        }
    }
}

/// Ports worth trying when no explicit endpoint is given.
pub fn default_ports() -> Vec<u16> {
    [
        Runtime::Ollama,
        Runtime::LmStudio,
        Runtime::Vllm,
        Runtime::LlamaCpp,
    ]
    .into_iter()
    .filter_map(Runtime::default_port)
    .collect()
}

/// What a reachable endpoint turned out to be.
#[derive(Clone, Debug)]
pub struct Endpoint {
    pub base_url: String,
    pub runtime: Runtime,
    pub models: Vec<String>,
    pub wire_api: WireApi,
}

pub fn client_pool(http_client_factory: HttpClientFactory) -> RouteAwareClientPool {
    RouteAwareClientPool::new(http_client_factory, ClientRouteClass::Api)
}

/// Returns the server root for an OpenAI-style base URL (`…/v1` → `…`).
fn host_root(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

/// Asks one endpoint what it is.
///
/// Returns `None` when nothing answers, which is the expected outcome for most
/// candidate ports.
pub async fn probe(
    pool: &RouteAwareClientPool,
    base_url: &str,
    api_key: Option<&str>,
) -> Option<Endpoint> {
    let models = list_models(pool, base_url, api_key).await?;
    let runtime = identify_runtime(pool, base_url).await;
    let wire_api = detect_wire_api(pool, base_url, api_key).await;

    Some(Endpoint {
        base_url: base_url.trim_end_matches('/').to_string(),
        runtime,
        models,
        wire_api,
    })
}

async fn list_models(
    pool: &RouteAwareClientPool,
    base_url: &str,
    api_key: Option<&str>,
) -> Option<Vec<String>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut request = pool.get(&url).timeout(PROBE_TIMEOUT);
    if let Some(api_key) = api_key {
        request = request.header("authorization", format!("Bearer {api_key}"));
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }

    let body: Value = response.json().await.ok()?;
    let models = body
        .get("data")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.get("id").and_then(Value::as_str))
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(models)
}

/// Identifies the server from the private endpoint each runtime exposes
/// alongside its OpenAI-compatible surface.
async fn identify_runtime(pool: &RouteAwareClientPool, base_url: &str) -> Runtime {
    let root = host_root(base_url);
    for (path, runtime) in [
        ("/api/tags", Runtime::Ollama),
        ("/api/v0/models", Runtime::LmStudio),
        ("/version", Runtime::Vllm),
        ("/props", Runtime::LlamaCpp),
    ] {
        let url = format!("{root}{path}");
        if let Ok(response) = pool.get(&url).timeout(PROBE_TIMEOUT).send().await
            && response.status().is_success()
        {
            return runtime;
        }
    }
    Runtime::Unknown
}

/// Decides which wire protocol to configure for an endpoint.
///
/// Posting a deliberately incomplete body distinguishes "no such route" from
/// "route exists and rejected this request", without running inference.
async fn detect_wire_api(
    pool: &RouteAwareClientPool,
    base_url: &str,
    api_key: Option<&str>,
) -> WireApi {
    let url = format!("{}/responses", base_url.trim_end_matches('/'));
    let mut request = pool.post(&url).timeout(PROBE_TIMEOUT).json(&json!({}));
    if let Some(api_key) = api_key {
        request = request.header("authorization", format!("Bearer {api_key}"));
    }
    match request.send().await {
        Ok(response) if responses_route_missing(response.status().as_u16()) => WireApi::Chat,
        Ok(_) => WireApi::Responses,
        // If the probe itself fails, chat completions is the safer assumption:
        // every OpenAI-compatible server implements it.
        Err(_) => WireApi::Chat,
    }
}

fn responses_route_missing(status: u16) -> bool {
    matches!(status, 404 | 405 | 501)
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod probe_tests;

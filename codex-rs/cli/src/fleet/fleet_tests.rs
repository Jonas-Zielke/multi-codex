use super::*;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::OLLAMA_OSS_PROVIDER_ID;
use pretty_assertions::assert_eq;

fn endpoint(base_url: &str, runtime: Runtime) -> Endpoint {
    Endpoint {
        base_url: base_url.to_string(),
        runtime,
        models: vec!["nvidia/nemotron-3-nano-30b-a3b".to_string()],
        wire_api: WireApi::Chat,
    }
}

/// Flattens edits into `("dotted.path", "rendered value")` pairs.
///
/// The real writer builds proper TOML tables; these tests care about which
/// keys are written and what they are set to.
fn written(edits: &[ConfigEdit]) -> Vec<(String, String)> {
    edits
        .iter()
        .map(|edit| {
            let ConfigEdit::SetPath { segments, value } = edit else {
                panic!("fleet only writes SetPath edits");
            };
            (
                segments.join("."),
                value
                    .as_value()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
            )
        })
        .collect()
}

#[test]
fn provider_edits_describe_the_endpoint() {
    let edits = provider_edits(
        "dgx-spark",
        "http://10.0.0.5:8000/v1",
        Runtime::Vllm.label(),
        WireApi::Chat,
        /*api_key_env*/ None,
    );

    assert_eq!(
        vec![
            (
                "model_providers.dgx-spark.name".to_string(),
                r#""vLLM""#.to_string()
            ),
            (
                "model_providers.dgx-spark.base_url".to_string(),
                r#""http://10.0.0.5:8000/v1""#.to_string()
            ),
            (
                "model_providers.dgx-spark.wire_api".to_string(),
                r#""chat""#.to_string()
            ),
        ],
        written(&edits)
    );
}

#[test]
fn an_api_key_env_is_recorded_when_given() {
    let edits = provider_edits(
        "remote",
        "https://gpu.example/v1",
        Runtime::Unknown.label(),
        WireApi::Responses,
        Some("GPU_API_KEY"),
    );
    let written = written(&edits);

    assert!(
        written.contains(&(
            "model_providers.remote.env_key".to_string(),
            r#""GPU_API_KEY""#.to_string()
        )),
        "{written:?}"
    );
    assert!(
        written.contains(&(
            "model_providers.remote.wire_api".to_string(),
            r#""responses""#.to_string()
        )),
        "{written:?}"
    );
}

#[test]
fn the_allowlist_keeps_what_was_already_authorized() {
    let existing = BTreeSet::from(["nebius".to_string()]);
    let edits = [allowlist_edit(
        &existing,
        &["vllm".to_string(), "nebius".to_string()],
    )];

    assert_eq!(
        vec![(
            "agents.allowed_model_providers".to_string(),
            r#"["nebius", "vllm"]"#.to_string()
        )],
        written(&edits)
    );
}

#[tokio::test]
async fn a_runtime_on_its_default_port_reuses_the_built_in_provider() {
    let config = test_config().await;
    let built_in_base_url = config.model_providers[OLLAMA_OSS_PROVIDER_ID]
        .base_url
        .clone()
        .expect("built-in oss provider has a base url");

    let assignments =
        assign_provider_ids(&config, &[endpoint(&built_in_base_url, Runtime::Ollama)]);

    // Built-in providers cannot be redefined, so writing a duplicate id would
    // produce config that never takes effect.
    assert_eq!(OLLAMA_OSS_PROVIDER_ID, assignments[0].0);
}

#[tokio::test]
async fn an_unclaimed_runtime_name_is_used_as_is() {
    let config = test_config().await;

    let assignments = assign_provider_ids(
        &config,
        &[endpoint("http://127.0.0.1:8000/v1", Runtime::Vllm)],
    );

    assert_eq!("vllm", assignments[0].0);
}

#[tokio::test]
async fn a_second_instance_of_a_runtime_is_qualified_by_port() {
    let mut config = test_config().await;
    config.model_providers.insert(
        "vllm".to_string(),
        ModelProviderInfo {
            base_url: Some("http://10.0.0.5:8000/v1".to_string()),
            ..Default::default()
        },
    );

    let assignments = assign_provider_ids(
        &config,
        &[endpoint("http://127.0.0.1:8001/v1", Runtime::Vllm)],
    );

    assert_eq!("vllm-8001", assignments[0].0);
}

#[test]
fn provider_ids_are_restricted_to_config_safe_characters() {
    assert!(validate_provider_id("dgx-spark_2").is_ok());
    assert!(validate_provider_id("").is_err());
    assert!(validate_provider_id("has space").is_err());
    assert!(validate_provider_id("quote\"").is_err());
}

#[test]
fn only_local_endpoints_are_probed_for_reachability() {
    assert!(is_loopback_url("http://127.0.0.1:8000/v1"));
    assert!(is_loopback_url("http://localhost:1234/v1"));
    assert!(!is_loopback_url("https://api.tokenfactory.nebius.com/v1"));
    assert!(!is_loopback_url("http://10.0.0.5:8000/v1"));
}

async fn test_config() -> Config {
    let home = tempfile::tempdir().expect("create temp dir");
    let home_path = home.path().to_path_buf();
    codex_core::config::ConfigBuilder::default()
        .codex_home(home_path.clone())
        .fallback_cwd(Some(home_path))
        .build()
        .await
        .expect("load test config")
}

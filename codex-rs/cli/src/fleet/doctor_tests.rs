use super::*;
use codex_model_provider_info::WireApi;
use pretty_assertions::assert_eq;

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

fn local_provider(base_url: &str) -> ModelProviderInfo {
    ModelProviderInfo {
        base_url: Some(base_url.to_string()),
        wire_api: WireApi::Chat,
        ..Default::default()
    }
}

#[tokio::test]
async fn a_declared_but_unauthorized_endpoint_routes_nothing() {
    let mut config = test_config().await;
    config
        .model_providers
        .insert("vllm".to_string(), local_provider("http://127.0.0.1:1/v1"));

    let checks = routing_checks(&config);

    // Declared but never authorized: nothing routes to it.
    assert_eq!(
        vec![Check::Warn {
            note: "no endpoint is authorized for agent routing".to_string(),
            fix: "Every agent runs on the session's endpoint. Run `codex fleet scan --write`."
                .to_string(),
        }],
        checks
    );
}

#[tokio::test]
async fn an_authorized_but_undeclared_endpoint_is_a_failure() {
    let mut config = test_config().await;
    config
        .agent_allowed_model_providers
        .insert("ghost".to_string());

    let checks = routing_checks(&config);

    assert!(checks.iter().any(Check::is_failure), "{checks:?}");
}

#[tokio::test]
async fn an_authorized_and_declared_endpoint_passes() {
    let mut config = test_config().await;
    config
        .model_providers
        .insert("vllm".to_string(), local_provider("http://127.0.0.1:1/v1"));
    config
        .agent_allowed_model_providers
        .insert("vllm".to_string());

    let checks = routing_checks(&config);

    assert_eq!(
        vec![Check::Ok("`vllm` is authorized for routing".to_string())],
        checks
    );
}

#[tokio::test]
async fn a_flat_tree_is_a_warning_not_a_failure() {
    let mut config = test_config().await;
    config.agent_max_depth = 1;
    let _ = config.features.enable(Feature::MultiAgentV2);

    let checks = team_checks(&config);

    // Depth 1 is the default and works; it just is not an org.
    assert!(!checks.iter().any(Check::is_failure), "{checks:?}");
    assert!(
        checks
            .iter()
            .any(|check| matches!(check, Check::Warn { note, .. } if note.contains("max_depth"))),
        "{checks:?}"
    );
}

#[tokio::test]
async fn a_disabled_backend_is_a_failure() {
    let mut config = test_config().await;
    config.agent_max_depth = 2;
    let _ = config.features.disable(Feature::MultiAgentV2);

    let checks = team_checks(&config);

    assert!(checks.iter().any(Check::is_failure), "{checks:?}");
}

#[tokio::test]
async fn a_role_routed_to_an_unauthorized_endpoint_fails() {
    let mut config = test_config().await;
    config
        .model_providers
        .insert("vllm".to_string(), local_provider("http://127.0.0.1:1/v1"));
    let pool = probe::client_pool(config.http_client_factory());

    let checks = routed_role_checks(&config, &pool, "worker", "vllm", "nemotron").await;

    // This is the failure that would otherwise only appear when the role is
    // used, as a spawn that refuses to run.
    assert!(checks.iter().any(Check::is_failure), "{checks:?}");
}

#[tokio::test]
async fn a_routed_role_without_a_model_is_flagged() {
    let mut config = test_config().await;
    config
        .model_providers
        .insert("vllm".to_string(), local_provider("http://127.0.0.1:1/v1"));
    config
        .agent_allowed_model_providers
        .insert("vllm".to_string());
    let pool = probe::client_pool(config.http_client_factory());

    let checks = routed_role_checks(&config, &pool, "worker", "vllm", /*model*/ "").await;

    assert!(
        checks.iter().any(
            |check| matches!(check, Check::Warn { note, .. } if note.contains("names no model"))
        ),
        "{checks:?}"
    );
}

#[tokio::test]
async fn an_unreachable_endpoint_is_reported_with_its_url() {
    let mut config = test_config().await;
    // Port 1 is reserved and never has a listener.
    config
        .model_providers
        .insert("vllm".to_string(), local_provider("http://127.0.0.1:1/v1"));
    config
        .agent_allowed_model_providers
        .insert("vllm".to_string());
    let pool = probe::client_pool(config.http_client_factory());

    let checks = routed_role_checks(&config, &pool, "worker", "vllm", "nemotron").await;

    assert!(
        checks.iter().any(|check| matches!(
            check,
            Check::Fail { note, .. } if note.contains("http://127.0.0.1:1/v1")
        )),
        "{checks:?}"
    );
}

#[test]
fn a_missing_credential_names_the_variable() {
    let provider = ModelProviderInfo {
        env_key: Some("DEFINITELY_UNSET_FLEET_TEST_KEY".to_string()),
        ..Default::default()
    };

    let check = credential_check(&provider).expect_err("an unset key should fail");

    assert!(
        matches!(&check, Check::Fail { note, .. } if note.contains("DEFINITELY_UNSET_FLEET_TEST_KEY")),
        "{check:?}"
    );
}

#[test]
fn a_provider_without_a_credential_needs_none() {
    assert_eq!(Ok(None), credential_check(&ModelProviderInfo::default()));
}

#[test]
fn the_report_shows_failures_and_their_fixes() {
    let mut report = Report::default();
    report.section(
        "Routing",
        vec![
            Check::Ok("`vllm` is authorized".to_string()),
            Check::Fail {
                note: "`ghost` is not declared".to_string(),
                fix: "Add it to config.toml.".to_string(),
            },
        ],
    );

    assert!(report.failed());
    let rendered = report.render();
    assert!(
        rendered.contains("  ok    `vllm` is authorized"),
        "{rendered}"
    );
    assert!(
        rendered.contains("  FAIL  `ghost` is not declared"),
        "{rendered}"
    );
    assert!(
        rendered.contains("          Add it to config.toml."),
        "{rendered}"
    );
}

#[test]
fn a_report_of_warnings_alone_does_not_fail() {
    let mut report = Report::default();
    report.section(
        "Teams",
        vec![Check::Warn {
            note: "flat".to_string(),
            fix: "raise the depth".to_string(),
        }],
    );

    assert!(!report.failed());
}

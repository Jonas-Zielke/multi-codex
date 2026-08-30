use super::*;
use codex_http_client::OutboundProxyPolicy;
use pretty_assertions::assert_eq;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

async fn models_endpoint(server: &MockServer, ids: &[&str]) {
    let data: Vec<Value> = ids.iter().map(|id| json!({"id": id})).collect();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": data})))
        .mount(server)
        .await;
}

#[tokio::test]
async fn reports_models_and_identifies_the_runtime() {
    let server = MockServer::start().await;
    models_endpoint(&server, &["nvidia/nemotron-3-nano-30b-a3b"]).await;
    Mock::given(method("GET"))
        .and(path("/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version": "0.11.0"})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let base_url = format!("{}/v1", server.uri());
    let endpoint = probe(&test_pool(), &base_url, /*api_key*/ None)
        .await
        .expect("server should answer");

    assert_eq!(Runtime::Vllm, endpoint.runtime);
    assert_eq!(vec!["nvidia/nemotron-3-nano-30b-a3b"], endpoint.models);
    assert_eq!(WireApi::Chat, endpoint.wire_api);
}

#[tokio::test]
async fn a_served_responses_route_selects_the_responses_wire() {
    let server = MockServer::start().await;
    models_endpoint(&server, &["nvidia/Nemotron-3-Ultra-550b-a55b"]).await;
    // A served route rejects the empty probe body rather than 404ing.
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": "missing model"})))
        .mount(&server)
        .await;

    let base_url = format!("{}/v1", server.uri());
    let endpoint = probe(&test_pool(), &base_url, /*api_key*/ None)
        .await
        .expect("server should answer");

    assert_eq!(WireApi::Responses, endpoint.wire_api);
    // Nothing identified the server, but it still speaks the OpenAI API.
    assert_eq!(Runtime::Unknown, endpoint.runtime);
}

#[tokio::test]
async fn an_endpoint_without_a_model_list_is_not_reported() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let base_url = format!("{}/v1", server.uri());
    assert!(
        probe(&test_pool(), &base_url, /*api_key*/ None)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn nothing_listening_is_not_an_error() {
    // Port 1 is reserved and never has a listener.
    assert!(
        probe(&test_pool(), "http://127.0.0.1:1/v1", /*api_key*/ None)
            .await
            .is_none()
    );
}

#[test]
fn host_root_strips_the_openai_suffix() {
    assert_eq!(
        "http://localhost:8000",
        host_root("http://localhost:8000/v1")
    );
    assert_eq!(
        "http://localhost:8000",
        host_root("http://localhost:8000/v1/")
    );
    assert_eq!("http://localhost:8000", host_root("http://localhost:8000"));
}

#[test]
fn default_ports_cover_every_known_runtime() {
    assert_eq!(vec![11434, 1234, 8000, 8080], default_ports());
}

fn test_pool() -> RouteAwareClientPool {
    client_pool(HttpClientFactory::new(OutboundProxyPolicy::ReqwestDefault))
}

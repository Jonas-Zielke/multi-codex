//! End-to-end coverage for the Chat Completions wire protocol: what leaves the
//! client on the wire, and what a provider's SSE reply decodes back into.
#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::ChatCompletionsClient;
use codex_api::ChatCompletionsOptions;
use codex_api::ChatCompletionsRequestOptions;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::ResponsesApiRequest;
use codex_api::ResponsesApiTools;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use serde_json::value::RawValue;

/// Transport that records the outgoing request and replays a canned SSE body.
#[derive(Clone)]
struct ScriptedTransport {
    body: Arc<String>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl ScriptedTransport {
    fn new(body: String) -> Self {
        Self {
            body: Arc::new(body),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn take_requests(&self) -> Vec<Request> {
        let mut guard = self.requests.lock().expect("requests mutex");
        std::mem::take(&mut *guard)
    }
}

impl HttpTransport for ScriptedTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.requests.lock().expect("requests mutex").push(req);
        let chunk: Result<Bytes, TransportError> = Ok(Bytes::from(self.body.as_str().to_owned()));
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(futures::stream::iter(vec![chunk])),
        })
    }
}

#[derive(Clone)]
struct BearerAuth(&'static str);

impl AuthProvider for BearerAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        if let Ok(header) = HeaderValue::from_str(&format!("Bearer {}", self.0)) {
            headers.insert(http::header::AUTHORIZATION, header);
        }
    }
}

fn provider() -> Provider {
    Provider {
        name: "nebius".to_string(),
        base_url: "https://api.tokenfactory.nebius.com/v1".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: codex_api::RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: true,
        },
        stream_idle_timeout: Duration::from_secs(1),
    }
}

fn request() -> ResponsesApiRequest {
    let tools = json!([{
        "type": "function",
        "name": "shell",
        "description": "run a command",
        "strict": false,
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "array", "items": {"type": "string"}}},
            "required": ["command"],
        },
    }]);
    let raw = RawValue::from_string(tools.to_string()).expect("valid tool JSON");

    ResponsesApiRequest {
        model: "nvidia/nemotron-3-super-120b-a12b".to_string(),
        instructions: "you are codex".to_string(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "list the files".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }],
        tools: Some(ResponsesApiTools::from(Arc::from(raw))),
        tool_choice: "auto".to_string(),
        parallel_tool_calls: true,
        reasoning: None,
        store: false,
        stream: true,
        stream_options: None,
        include: Vec::new(),
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
        access_programs: None,
    }
}

fn sse_body(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn request_body_json(request: &Request) -> Value {
    let Some(RequestBody::EncodedJson(body)) = request.body.as_ref() else {
        panic!("expected a prepared request body");
    };
    serde_json::from_slice(body.as_bytes()).expect("request body should be JSON")
}

#[tokio::test]
async fn posts_a_chat_shaped_body_to_the_chat_completions_path() -> Result<()> {
    let transport = ScriptedTransport::new(sse_body(&[json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
    })]));
    let client = ChatCompletionsClient::new(
        transport.clone(),
        provider(),
        Arc::new(BearerAuth("nebius-key")),
    );

    let mut stream = client
        .stream_request(request(), ChatCompletionsRequestOptions::default())
        .await?;
    while stream.next().await.is_some() {}

    let requests = transport.take_requests();
    assert_eq!(1, requests.len());
    let request = &requests[0];
    assert_eq!(
        "https://api.tokenfactory.nebius.com/v1/chat/completions",
        request.url
    );
    assert_eq!(
        Some(&HeaderValue::from_static("Bearer nebius-key")),
        request.headers.get(http::header::AUTHORIZATION)
    );

    let body = request_body_json(request);
    assert_eq!("nvidia/nemotron-3-super-120b-a12b", body["model"]);
    assert_eq!(
        json!([
            {"role": "system", "content": "you are codex"},
            {"role": "user", "content": "list the files"},
        ]),
        body["messages"]
    );
    // Responses declares tools flat; chat nests them under `function`.
    assert_eq!("shell", body["tools"][0]["function"]["name"]);
    assert_eq!("auto", body["tool_choice"]);
    assert_eq!(json!(true), body["stream"]);
    Ok(())
}

#[tokio::test]
async fn decodes_a_tool_call_reply_into_response_items() -> Result<()> {
    let transport = ScriptedTransport::new(sse_body(&[
        json!({
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": "{\"command\":[\"ls\"]}"},
            }]}}],
        }),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        json!({
            "choices": [],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        }),
    ]));
    let client =
        ChatCompletionsClient::new(transport, provider(), Arc::new(BearerAuth("nebius-key")));

    let mut stream = client
        .stream_request(request(), ChatCompletionsRequestOptions::default())
        .await?;
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }

    let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
        name,
        arguments,
        call_id,
        ..
    }) = &events[1]
    else {
        panic!("expected a function call, got {:?}", events[1]);
    };
    assert_eq!("shell", name);
    assert_eq!(r#"{"command":["ls"]}"#, arguments);
    assert_eq!("call_1", call_id);

    let ResponseEvent::Completed { token_usage, .. } = &events[2] else {
        panic!("expected completed, got {:?}", events[2]);
    };
    let usage = token_usage.as_ref().expect("usage should be reported");
    assert_eq!(42, usage.input_tokens);
    assert_eq!(7, usage.output_tokens);
    Ok(())
}

#[tokio::test]
async fn provider_options_reach_the_request_body() -> Result<()> {
    let transport = ScriptedTransport::new(sse_body(&[]));
    let client = ChatCompletionsClient::new(
        transport.clone(),
        provider(),
        Arc::new(BearerAuth("nebius-key")),
    )
    .with_options(ChatCompletionsOptions {
        send_reasoning_effort: false,
        extra_body: [(
            "chat_template_kwargs".to_string(),
            json!({"thinking": true}),
        )]
        .into_iter()
        .collect(),
    });

    let mut stream = client
        .stream_request(request(), ChatCompletionsRequestOptions::default())
        .await?;
    while stream.next().await.is_some() {}

    let requests = transport.take_requests();
    let body = request_body_json(&requests[0]);
    assert_eq!(json!({"thinking": true}), body["chat_template_kwargs"]);
    Ok(())
}

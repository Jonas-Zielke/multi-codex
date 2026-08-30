//! Client for OpenAI-compatible `/v1/chat/completions` endpoints.
//!
//! Codex builds every turn in the Responses API's shape. This client accepts
//! that same [`ResponsesApiRequest`], rewrites it for the chat surface, and
//! decodes the reply back into [`ResponseEvent`]s — so providers that only
//! speak chat completions (Nebius Token Factory models without Responses
//! support, llama.cpp, Unsloth Studio, older vLLM builds) work with no changes
//! anywhere else in the client.

use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::chat_completions::ChatCompletionsOptions;
use crate::requests::chat_completions::build_chat_completions_payload;
use crate::requests::headers::build_session_headers;
use crate::sse::chat_completions::spawn_chat_completions_stream;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use std::sync::Arc;
use tracing::instrument;

/// Provider-relative path for the chat completions surface.
pub const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";

pub struct ChatCompletionsClient<T: HttpTransport> {
    session: EndpointSession<T>,
    options: ChatCompletionsOptions,
}

#[derive(Default)]
pub struct ChatCompletionsRequestOptions {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub extra_headers: HeaderMap,
}

impl<T: HttpTransport> ChatCompletionsClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            options: ChatCompletionsOptions::default(),
        }
    }

    /// Applies provider-specific request tuning, such as whether
    /// `reasoning_effort` is safe to send.
    pub fn with_options(mut self, options: ChatCompletionsOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_telemetry(self, request: Option<Arc<dyn RequestTelemetry>>) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            options: self.options,
        }
    }

    #[instrument(
        name = "chat_completions.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "chat_completions_http",
            http.method = "POST",
            api.path = CHAT_COMPLETIONS_PATH
        )
    )]
    pub async fn stream_request(
        &self,
        request: ResponsesApiRequest,
        options: ChatCompletionsRequestOptions,
    ) -> Result<ResponseStream, ApiError> {
        let ChatCompletionsRequestOptions {
            session_id,
            thread_id,
            extra_headers,
        } = options;

        let payload = build_chat_completions_payload(&request, &self.options).map_err(|e| {
            ApiError::Stream(format!("failed to build chat completions request: {e}"))
        })?;
        let body = EncodedJsonBody::encode(&payload.body).map_err(|e| {
            ApiError::Stream(format!("failed to encode chat completions request: {e}"))
        })?;

        let mut headers = extra_headers;
        headers.extend(build_session_headers(session_id, thread_id));

        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                CHAT_COMPLETIONS_PATH,
                headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                },
            )
            .await?;

        Ok(spawn_chat_completions_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            payload.tool_bindings,
        ))
    }
}

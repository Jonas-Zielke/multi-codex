//! Decodes an OpenAI-compatible Chat Completions SSE stream into the
//! [`ResponseEvent`]s the rest of Codex is written against.
//!
//! Chat Completions streams deltas rather than items: text arrives as
//! `choices[].delta.content`, reasoning as `delta.reasoning_content`, and tool
//! calls as fragments keyed by `index` that have to be reassembled. This module
//! buffers those fragments and emits the item lifecycle Codex expects — an
//! `OutputItemAdded` before any text delta, matching `OutputItemDone`s, and a
//! final `Completed` carrying token usage.

use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::rate_limits::has_rate_limit_data;
use crate::rate_limits::parse_all_rate_limits;
use crate::requests::chat_completions::ChatToolBinding;
use crate::requests::chat_completions::FREEFORM_INPUT_PARAM;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::debug;

const REQUEST_ID_HEADER: &str = "x-request-id";

pub(crate) fn spawn_chat_completions_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    tool_bindings: HashMap<String, ChatToolBinding>,
) -> ResponseStream {
    // Third-party endpoints rarely send rate-limit headers, and the shared
    // parser synthesizes an empty default snapshot when they are absent. Only
    // forward snapshots that actually carry a limit.
    let rate_limit_snapshots: Vec<_> = parse_all_rate_limits(&stream_response.headers)
        .into_iter()
        .filter(has_rate_limit_data)
        .collect();
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        for snapshot in rate_limit_snapshots {
            let _ = tx_event.send(Ok(ResponseEvent::RateLimits(snapshot))).await;
        }
        process_chat_sse(stream_response.bytes, tx_event, idle_timeout, tool_bindings).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

/// A tool call reassembled from `delta.tool_calls` fragments.
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Default)]
struct ToolCallBuffer {
    calls: Vec<ToolCallAccumulator>,
    /// Wire `index` values seen so far, mapped to positions in `calls`.
    positions: HashMap<i64, usize>,
}

impl ToolCallBuffer {
    /// Resolves the slot a fragment belongs to.
    ///
    /// Well-behaved servers key fragments by `index`. Some send whole tool calls
    /// with no index at all, so fall back to matching on `id` and finally to
    /// extending the most recent call.
    fn slot(&mut self, index: Option<i64>, id: Option<&str>) -> usize {
        if let Some(index) = index {
            if let Some(position) = self.positions.get(&index) {
                return *position;
            }
            let position = self.calls.len();
            self.calls.push(ToolCallAccumulator::default());
            self.positions.insert(index, position);
            return position;
        }

        if let Some(id) = id {
            if let Some(position) = self
                .calls
                .iter()
                .position(|call| call.id.as_deref() == Some(id))
            {
                return position;
            }
            let position = self.calls.len();
            self.calls.push(ToolCallAccumulator::default());
            return position;
        }

        if self.calls.is_empty() {
            self.calls.push(ToolCallAccumulator::default());
        }
        self.calls.len() - 1
    }

    fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

#[derive(Default)]
struct StreamState {
    response_id: Option<String>,
    text: String,
    /// True between the `OutputItemAdded` and `OutputItemDone` of a text item.
    message_open: bool,
    reasoning: String,
    reasoning_open: bool,
    tool_calls: ToolCallBuffer,
    usage: Option<TokenUsage>,
    finish_reason: Option<String>,
}

async fn process_chat_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    tool_bindings: HashMap<String, ChatToolBinding>,
) {
    let mut stream = stream.eventsource();
    let mut state = StreamState::default();
    let mut created_sent = false;

    loop {
        let event = match timeout(idle_timeout, stream.next()).await {
            Ok(Some(Ok(event))) => event,
            Ok(Some(Err(err))) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!("chat stream error: {err}"))))
                    .await;
                return;
            }
            // Upstream closed the stream. Emit whatever the turn produced so a
            // truncated response still lands in history.
            Ok(None) => break,
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "chat stream idle for more than {idle_timeout:?}"
                    ))))
                    .await;
                return;
            }
        };

        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }

        let chunk: Value = match serde_json::from_str(data) {
            Ok(chunk) => chunk,
            Err(err) => {
                debug!("failed to parse chat completions chunk: {err}");
                continue;
            }
        };

        if let Some(error) = chunk.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error from chat completions endpoint");
            let _ = tx_event
                .send(Err(ApiError::Stream(message.to_string())))
                .await;
            return;
        }

        if !created_sent {
            created_sent = true;
            if tx_event.send(Ok(ResponseEvent::Created)).await.is_err() {
                return;
            }
        }

        if state.response_id.is_none()
            && let Some(id) = chunk.get("id").and_then(Value::as_str)
        {
            state.response_id = Some(id.to_string());
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            state.usage = parse_usage(usage);
        }

        let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for choice in choices {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                state.finish_reason = Some(reason.to_string());
            }
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if handle_delta(delta, &mut state, &tx_event).await.is_err() {
                return;
            }
        }
    }

    finalize(state, &tx_event, &tool_bindings).await;
}

/// Applies one `choices[].delta` to the accumulated stream state.
///
/// Returns `Err(())` once the receiver is gone, so the caller can stop early.
async fn handle_delta(
    delta: &Value,
    state: &mut StreamState,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> Result<(), ()> {
    // Reasoning models expose their scratchpad under one of two field names
    // depending on the server; accept both.
    let reasoning = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty());
    if let Some(reasoning) = reasoning {
        if !state.message_open {
            open_message_item(state, tx_event).await?;
        }
        state.reasoning.push_str(reasoning);
        state.reasoning_open = true;
        send(
            tx_event,
            ResponseEvent::ReasoningContentDelta {
                delta: reasoning.to_string(),
                content_index: 0,
            },
        )
        .await?;
    }

    if let Some(content) = delta
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        if !state.message_open {
            open_message_item(state, tx_event).await?;
        }
        state.text.push_str(content);
        send(
            tx_event,
            ResponseEvent::OutputTextDelta(content.to_string()),
        )
        .await?;
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for fragment in tool_calls {
            apply_tool_call_fragment(fragment, state);
        }
    }

    Ok(())
}

/// Opens a streaming assistant item.
///
/// Codex requires an active item before it will accept text or reasoning
/// deltas, so every delta path funnels through here first.
async fn open_message_item(
    state: &mut StreamState,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
) -> Result<(), ()> {
    state.message_open = true;
    send(
        tx_event,
        ResponseEvent::OutputItemAdded(ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: String::new(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }),
    )
    .await
}

fn apply_tool_call_fragment(fragment: &Value, state: &mut StreamState) {
    let index = fragment.get("index").and_then(Value::as_i64);
    let id = fragment.get("id").and_then(Value::as_str);
    let slot = state.tool_calls.slot(index, id);
    let Some(call) = state.tool_calls.calls.get_mut(slot) else {
        return;
    };

    if let Some(id) = id {
        call.id = Some(id.to_string());
    }
    let Some(function) = fragment.get("function") else {
        return;
    };
    if let Some(name) = function.get("name").and_then(Value::as_str)
        && !name.is_empty()
    {
        call.name = Some(name.to_string());
    }
    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
        call.arguments.push_str(arguments);
    }
}

async fn finalize(
    state: StreamState,
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    tool_bindings: &HashMap<String, ChatToolBinding>,
) {
    let StreamState {
        response_id,
        text,
        message_open,
        reasoning,
        reasoning_open,
        tool_calls,
        usage,
        finish_reason,
    } = state;

    if reasoning_open && !reasoning.is_empty() {
        let item = ResponseItem::Reasoning {
            id: None,
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: reasoning,
            }]),
            encrypted_content: None,
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, ResponseEvent::OutputItemDone(item))
            .await
            .is_err()
        {
            return;
        }
    }

    if message_open {
        let item = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText { text }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        };
        if send(tx_event, ResponseEvent::OutputItemDone(item))
            .await
            .is_err()
        {
            return;
        }
    }

    if !tool_calls.is_empty() {
        for (position, call) in tool_calls.calls.into_iter().enumerate() {
            let Some(item) = tool_call_item(call, position, tool_bindings) else {
                continue;
            };
            if send(tx_event, ResponseEvent::OutputItemDone(item))
                .await
                .is_err()
            {
                return;
            }
        }
    }

    let _ = tx_event
        .send(Ok(ResponseEvent::Completed {
            response_id: response_id.unwrap_or_default(),
            token_usage: usage,
            usage_metadata: None,
            end_turn: finish_reason.as_deref().map(|reason| reason == "stop"),
        }))
        .await;
}

/// Rebuilds a Codex tool call from an accumulated fragment set, undoing the
/// namespace flattening applied when the tools were declared.
fn tool_call_item(
    call: ToolCallAccumulator,
    position: usize,
    tool_bindings: &HashMap<String, ChatToolBinding>,
) -> Option<ResponseItem> {
    let wire_name = call.name?;
    // Servers occasionally omit tool call ids; the id only has to be unique
    // within the turn for the follow-up `tool` message to match.
    let call_id = call
        .id
        .unwrap_or_else(|| format!("call_{position}_{wire_name}"));

    let binding = tool_bindings.get(&wire_name);
    let (namespace, name, freeform) = match binding {
        Some(binding) => (
            binding.namespace.clone(),
            binding.name.clone(),
            binding.freeform,
        ),
        None => (None, wire_name, false),
    };

    if freeform {
        let input = serde_json::from_str::<Value>(&call.arguments)
            .ok()
            .and_then(|args| {
                args.get(FREEFORM_INPUT_PARAM)
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or(call.arguments);
        return Some(ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id,
            name,
            namespace,
            input,
            internal_chat_message_metadata_passthrough: None,
        });
    }

    Some(ResponseItem::FunctionCall {
        id: None,
        name,
        namespace,
        // Models sometimes emit no arguments at all for zero-argument tools;
        // downstream parsing expects a JSON object either way.
        arguments: if call.arguments.trim().is_empty() {
            "{}".to_string()
        } else {
            call.arguments
        },
        encrypted_function_args: None,
        call_id,
        internal_chat_message_metadata_passthrough: None,
    })
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let field = |name: &str| usage.get(name).and_then(Value::as_i64).unwrap_or(0);
    let nested = |parent: &str, name: &str| {
        usage
            .get(parent)
            .and_then(|details| details.get(name))
            .and_then(Value::as_i64)
            .unwrap_or(0)
    };

    let input_tokens = field("prompt_tokens");
    let output_tokens = field("completion_tokens");
    let total_tokens = match field("total_tokens") {
        0 => input_tokens + output_tokens,
        total => total,
    };

    Some(TokenUsage {
        input_tokens,
        cached_input_tokens: nested("prompt_tokens_details", "cached_tokens"),
        cache_write_input_tokens: 0,
        output_tokens,
        reasoning_output_tokens: nested("completion_tokens_details", "reasoning_tokens"),
        total_tokens,
        codex_rollout_budget_units: None,
    })
}

async fn send(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    event: ResponseEvent,
) -> Result<(), ()> {
    tx_event.send(Ok(event)).await.map_err(|_| ())
}

#[cfg(test)]
#[path = "chat_completions_tests.rs"]
mod chat_completions_tests;

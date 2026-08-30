//! Translation from Codex's Responses-API request shape into the
//! OpenAI-compatible Chat Completions wire format.
//!
//! Codex models a turn as a list of [`ResponseItem`]s plus Responses-style tool
//! declarations. Nebius Token Factory and every local runtime Codex can drive
//! (vLLM, Ollama, LM Studio, llama.cpp) expose `/v1/chat/completions` instead,
//! so a turn has to be rewritten on the way out:
//!
//! * response items become chat messages, with consecutive tool calls merged
//!   into a single assistant message so `tool` replies line up with the
//!   assistant turn that requested them;
//! * namespaced tools are flattened to plain function names, because Chat
//!   Completions has no notion of a namespace;
//! * freeform ("custom") tools are projected onto a one-string-argument
//!   function, which is the closest thing the chat surface offers.
//!
//! [`ChatCompletionsPayload::tool_bindings`] carries what the stream decoder
//! needs to undo the flattening when tool calls come back.

use crate::common::ResponsesApiRequest;
use codex_protocol::models::ContentItem;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::ResponseItem;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;

/// Separator used to flatten tool namespaces into the flat function names Chat
/// Completions supports. Codex tool names use single underscores, so a double
/// underscore never collides with a real name.
pub(crate) const NAMESPACE_SEPARATOR: &str = "__";

/// Parameter name a freeform tool's single string argument is carried under.
pub(crate) const FREEFORM_INPUT_PARAM: &str = "input";

/// How a wire-level function name maps back onto a Codex tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatToolBinding {
    /// Namespace the tool was declared in, if it was namespaced.
    pub(crate) namespace: Option<String>,
    /// Tool name as Codex knows it, without the namespace prefix.
    pub(crate) name: String,
    /// Freeform tools take a raw string rather than a JSON argument object.
    pub(crate) freeform: bool,
}

/// A request body ready to POST, plus the mapping needed to decode its replies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatCompletionsPayload {
    pub(crate) body: Value,
    pub(crate) tool_bindings: HashMap<String, ChatToolBinding>,
}

/// Provider-specific knobs for the Chat Completions wire protocol.
///
/// OpenAI-compatible servers disagree about which optional request fields they
/// tolerate, so anything beyond the common core is opt-in per provider. These
/// are surfaced in `config.toml` under `model_providers.<id>.chat`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ChatCompletionsOptions {
    /// Send `reasoning_effort` when the turn selects one. Off by default: some
    /// OpenAI-compatible servers reject request fields they do not recognize.
    #[serde(default)]
    pub send_reasoning_effort: bool,
    /// Extra top-level fields merged into every request body, for
    /// server-specific switches such as vLLM's `chat_template_kwargs`.
    #[serde(default)]
    pub extra_body: HashMap<String, Value>,
}

pub(crate) fn build_chat_completions_payload(
    request: &ResponsesApiRequest,
    options: &ChatCompletionsOptions,
) -> Result<ChatCompletionsPayload, serde_json::Error> {
    let mut body = Map::new();
    body.insert("model".to_string(), json!(request.model));
    body.insert(
        "messages".to_string(),
        Value::Array(convert_messages(&request.instructions, &request.input)),
    );

    let mut tool_bindings = HashMap::new();
    if let Some(tools) = request.tools.as_ref() {
        let declared: Vec<Value> = serde_json::from_str(tools.as_raw_value().get())?;
        let converted = convert_tools(&declared, &mut tool_bindings);
        if !converted.is_empty() {
            body.insert("tools".to_string(), Value::Array(converted));
            body.insert(
                "tool_choice".to_string(),
                json!(chat_tool_choice(&request.tool_choice)),
            );
            body.insert(
                "parallel_tool_calls".to_string(),
                json!(request.parallel_tool_calls),
            );
        }
    }

    body.insert("stream".to_string(), json!(request.stream));
    if request.stream {
        // Without this, most OpenAI-compatible servers omit usage on streamed
        // responses and Codex cannot account for the turn's tokens.
        body.insert("stream_options".to_string(), json!({"include_usage": true}));
    }

    if let Some(format) = request.text.as_ref().and_then(|text| text.format.as_ref()) {
        body.insert(
            "response_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": format.name,
                    "strict": format.strict,
                    "schema": format.schema,
                },
            }),
        );
    }

    if options.send_reasoning_effort
        && let Some(effort) = request.reasoning.as_ref().and_then(|r| r.effort.as_ref())
    {
        body.insert("reasoning_effort".to_string(), json!(effort.to_string()));
    }

    for (key, value) in &options.extra_body {
        body.insert(key.clone(), value.clone());
    }

    Ok(ChatCompletionsPayload {
        body: Value::Object(body),
        tool_bindings,
    })
}

/// Maps a Responses `tool_choice` onto its Chat Completions spelling.
fn chat_tool_choice(tool_choice: &str) -> &str {
    match tool_choice {
        "none" => "none",
        "required" => "required",
        _ => "auto",
    }
}

fn convert_tools(
    declared: &[Value],
    bindings: &mut HashMap<String, ChatToolBinding>,
) -> Vec<Value> {
    let mut tools = Vec::new();
    for tool in declared {
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => {
                if let Some(converted) = convert_function_tool(tool, None, bindings) {
                    tools.push(converted);
                }
            }
            Some("custom") => {
                if let Some(converted) = convert_freeform_tool(tool, None, bindings) {
                    tools.push(converted);
                }
            }
            Some("namespace") => {
                let namespace = tool.get("name").and_then(Value::as_str);
                let inner = tool.get("tools").and_then(Value::as_array);
                let (Some(namespace), Some(inner)) = (namespace, inner) else {
                    continue;
                };
                for nested in inner {
                    let converted = match nested.get("type").and_then(Value::as_str) {
                        Some("custom") => convert_freeform_tool(nested, Some(namespace), bindings),
                        _ => convert_function_tool(nested, Some(namespace), bindings),
                    };
                    if let Some(converted) = converted {
                        tools.push(converted);
                    }
                }
            }
            // Server-side tools (`web_search`, `tool_search`) have no
            // Chat Completions equivalent; drop them rather than send
            // something the provider will reject.
            _ => {}
        }
    }
    tools
}

/// Flattens `namespace` + `name` into the single name sent on the wire.
fn wire_tool_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(namespace) if !namespace.is_empty() => {
            format!("{namespace}{NAMESPACE_SEPARATOR}{name}")
        }
        _ => name.to_string(),
    }
}

fn convert_function_tool(
    tool: &Value,
    namespace: Option<&str>,
    bindings: &mut HashMap<String, ChatToolBinding>,
) -> Option<Value> {
    let name = tool.get("name").and_then(Value::as_str)?;
    let wire_name = wire_tool_name(namespace, name);
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let parameters = tool
        .get("parameters")
        .cloned()
        .unwrap_or_else(empty_parameters);

    bindings.insert(
        wire_name.clone(),
        ChatToolBinding {
            namespace: namespace.map(str::to_string),
            name: name.to_string(),
            freeform: false,
        },
    );

    // `strict` is deliberately not forwarded: it means guided decoding with
    // OpenAI's schema restrictions, which several OpenAI-compatible servers
    // reject outright on schemas Codex happily sends to the Responses API.
    Some(json!({
        "type": "function",
        "function": {
            "name": wire_name,
            "description": description,
            "parameters": parameters,
        },
    }))
}

fn convert_freeform_tool(
    tool: &Value,
    namespace: Option<&str>,
    bindings: &mut HashMap<String, ChatToolBinding>,
) -> Option<Value> {
    let name = tool.get("name").and_then(Value::as_str)?;
    let wire_name = wire_tool_name(namespace, name);
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();

    bindings.insert(
        wire_name.clone(),
        ChatToolBinding {
            namespace: namespace.map(str::to_string),
            name: name.to_string(),
            freeform: true,
        },
    );

    Some(json!({
        "type": "function",
        "function": {
            "name": wire_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    FREEFORM_INPUT_PARAM: {
                        "type": "string",
                        "description": "Raw tool input, passed through verbatim.",
                    },
                },
                "required": [FREEFORM_INPUT_PARAM],
                "additionalProperties": false,
            },
        },
    }))
}

fn empty_parameters() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

/// Accumulates tool calls so that a run of them lands in one assistant message.
#[derive(Default)]
struct PendingToolCalls(Vec<Value>);

impl PendingToolCalls {
    fn push(&mut self, call_id: &str, name: &str, arguments: String) {
        self.0.push(json!({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }));
    }

    fn flush(&mut self, messages: &mut Vec<Value>) {
        if self.0.is_empty() {
            return;
        }
        messages.push(json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": Value::Array(std::mem::take(&mut self.0)),
        }));
    }
}

fn convert_messages(instructions: &str, input: &[ResponseItem]) -> Vec<Value> {
    let mut messages = Vec::with_capacity(input.len() + 1);
    if !instructions.is_empty() {
        messages.push(json!({"role": "system", "content": instructions}));
    }

    let mut pending = PendingToolCalls::default();
    for item in input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                pending.flush(&mut messages);
                if let Some(message) = convert_message(role, content) {
                    messages.push(message);
                }
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                let wire_name = wire_tool_name(namespace.as_deref(), name);
                pending.push(call_id, &wire_name, arguments.clone());
            }
            ResponseItem::CustomToolCall {
                name,
                namespace,
                input,
                call_id,
                ..
            } => {
                let wire_name = wire_tool_name(namespace.as_deref(), name);
                let arguments = json!({ FREEFORM_INPUT_PARAM: input }).to_string();
                pending.push(call_id, &wire_name, arguments);
            }
            ResponseItem::LocalShellCall {
                call_id: Some(call_id),
                action,
                ..
            } => {
                let LocalShellAction::Exec(exec) = action;
                let arguments = serde_json::to_string(exec).unwrap_or_else(|_| "{}".to_string());
                pending.push(call_id, "local_shell", arguments);
            }
            ResponseItem::FunctionCallOutput {
                call_id: Some(call_id),
                output,
                ..
            } => {
                pending.flush(&mut messages);
                messages.push(tool_message(call_id, output.body.to_text()));
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                pending.flush(&mut messages);
                messages.push(tool_message(call_id, output.body.to_text()));
            }
            // Reasoning items are re-supplied by the server on the Responses
            // API; Chat Completions has no input slot for them. Everything else
            // here is a Responses-only item with no chat counterpart.
            _ => {}
        }
    }
    pending.flush(&mut messages);

    messages
}

fn tool_message(call_id: &str, content: Option<String>) -> Value {
    json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": content.unwrap_or_default(),
    })
}

fn convert_message(role: &str, content: &[ContentItem]) -> Option<Value> {
    // `developer` is Responses-only; older OpenAI-compatible servers reject it.
    let role = if role == "developer" { "system" } else { role };
    let is_input_role = matches!(role, "user" | "system" | "tool");

    let mut text_parts = Vec::new();
    let mut images = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                text_parts.push(text.as_str());
            }
            ContentItem::InputImage { image_url, .. } => {
                images.push(image_url.as_str());
            }
            ContentItem::InputAudio { .. } => {}
        }
    }

    if text_parts.is_empty() && images.is_empty() {
        return None;
    }

    // Multimodal content parts are only valid on input roles; an assistant turn
    // has to collapse to plain text.
    if images.is_empty() || !is_input_role {
        return Some(json!({"role": role, "content": text_parts.join("\n")}));
    }

    let mut parts = Vec::with_capacity(text_parts.len() + images.len());
    for text in text_parts {
        parts.push(json!({"type": "text", "text": text}));
    }
    for image_url in images {
        parts.push(json!({"type": "image_url", "image_url": {"url": image_url}}));
    }
    Some(json!({"role": role, "content": Value::Array(parts)}))
}

#[cfg(test)]
#[path = "chat_completions_tests.rs"]
mod chat_completions_tests;

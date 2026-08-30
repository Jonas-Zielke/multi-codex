use super::*;
use codex_client::TransportError;
use futures::TryStreamExt;
use pretty_assertions::assert_eq;
use serde_json::json;
use tokio_util::io::ReaderStream;

fn idle_timeout() -> Duration {
    Duration::from_millis(1000)
}

/// Feeds `chunks` through the decoder as `data:` frames followed by `[DONE]`.
async fn run_chunks(
    chunks: Vec<Value>,
    tool_bindings: HashMap<String, ChatToolBinding>,
) -> Vec<ResponseEvent> {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");

    let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(64);
    let stream = ReaderStream::new(std::io::Cursor::new(body))
        .map_err(|err| TransportError::Network(err.to_string()));
    tokio::spawn(process_chat_sse(
        Box::pin(stream),
        tx,
        idle_timeout(),
        tool_bindings,
    ));

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event.expect("stream error"));
    }
    events
}

fn text_delta(content: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {"content": content}}],
    })
}

#[tokio::test]
async fn text_is_opened_streamed_and_closed() {
    let events = run_chunks(
        vec![
            text_delta("Hel"),
            text_delta("lo"),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            json!({
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            }),
        ],
        HashMap::new(),
    )
    .await;

    assert!(matches!(events[0], ResponseEvent::Created));
    // Codex refuses text deltas without an active item, so the decoder has to
    // open one before the first delta lands.
    assert!(matches!(events[1], ResponseEvent::OutputItemAdded(_)));
    assert!(matches!(
        (&events[2], &events[3]),
        (
            ResponseEvent::OutputTextDelta(first),
            ResponseEvent::OutputTextDelta(second),
        ) if first == "Hel" && second == "lo"
    ));

    let ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) = &events[4] else {
        panic!("expected assistant message, got {:?}", events[4]);
    };
    assert_eq!(
        vec![ContentItem::OutputText {
            text: "Hello".to_string()
        }],
        *content
    );

    let ResponseEvent::Completed {
        response_id,
        token_usage,
        end_turn,
        ..
    } = &events[5]
    else {
        panic!("expected completed, got {:?}", events[5]);
    };
    assert_eq!("chatcmpl-1", response_id);
    assert_eq!(Some(true), *end_turn);
    let usage = token_usage.as_ref().expect("usage");
    assert_eq!(11, usage.input_tokens);
    assert_eq!(2, usage.output_tokens);
    assert_eq!(13, usage.total_tokens);
}

#[tokio::test]
async fn tool_call_fragments_are_reassembled_by_index() {
    let events = run_chunks(
        vec![
            json!({
                "id": "chatcmpl-2",
                "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{\"command\":"},
                }]}}],
            }),
            json!({
                "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "[\"ls\"]}"},
                }]}}],
            }),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ],
        HashMap::new(),
    )
    .await;

    let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
        name,
        arguments,
        call_id,
        ..
    }) = &events[1]
    else {
        panic!("expected function call, got {:?}", events[1]);
    };
    assert_eq!("shell", name);
    assert_eq!(r#"{"command":["ls"]}"#, arguments);
    assert_eq!("call_1", call_id);

    let ResponseEvent::Completed { end_turn, .. } = &events[2] else {
        panic!("expected completed, got {:?}", events[2]);
    };
    // A turn that ends in tool calls has not affirmatively ended.
    assert_eq!(Some(false), *end_turn);
}

#[tokio::test]
async fn parallel_tool_calls_stay_separate() {
    let events = run_chunks(
        vec![json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "one", "arguments": "{}"}},
                {"index": 1, "id": "b", "function": {"name": "two", "arguments": "{}"}},
            ]}}],
        })],
        HashMap::new(),
    )
    .await;

    let names: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { name, .. }) => {
                Some(name.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(vec!["one", "two"], names);
}

#[tokio::test]
async fn flattened_names_are_restored_to_their_namespace() {
    let bindings = HashMap::from([(
        "collaboration__spawn_agent".to_string(),
        ChatToolBinding {
            namespace: Some("collaboration".to_string()),
            name: "spawn_agent".to_string(),
            freeform: false,
        },
    )]);
    let events = run_chunks(
        vec![json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {"name": "collaboration__spawn_agent", "arguments": "{}"},
            }]}}],
        })],
        bindings,
    )
    .await;

    let ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
        name, namespace, ..
    }) = &events[1]
    else {
        panic!("expected function call, got {:?}", events[1]);
    };
    assert_eq!("spawn_agent", name);
    assert_eq!(Some("collaboration".to_string()), *namespace);
}

#[tokio::test]
async fn freeform_tool_input_is_unwrapped() {
    let bindings = HashMap::from([(
        "apply_patch".to_string(),
        ChatToolBinding {
            namespace: None,
            name: "apply_patch".to_string(),
            freeform: true,
        },
    )]);
    let events = run_chunks(
        vec![json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {
                    "name": "apply_patch",
                    "arguments": r#"{"input":"*** Begin Patch"}"#,
                },
            }]}}],
        })],
        bindings,
    )
    .await;

    let ResponseEvent::OutputItemDone(ResponseItem::CustomToolCall { name, input, .. }) =
        &events[1]
    else {
        panic!("expected custom tool call, got {:?}", events[1]);
    };
    assert_eq!("apply_patch", name);
    assert_eq!("*** Begin Patch", input);
}

#[tokio::test]
async fn reasoning_content_is_streamed_and_kept() {
    let events = run_chunks(
        vec![
            json!({"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]}),
            text_delta("answer"),
        ],
        HashMap::new(),
    )
    .await;

    assert!(matches!(events[1], ResponseEvent::OutputItemAdded(_)));
    assert!(matches!(
        &events[2],
        ResponseEvent::ReasoningContentDelta { delta, .. } if delta == "thinking"
    ));

    let ResponseEvent::OutputItemDone(ResponseItem::Reasoning { content, .. }) = &events[4] else {
        panic!("expected reasoning item, got {:?}", events[4]);
    };
    assert_eq!(
        Some(vec![ReasoningItemContent::ReasoningText {
            text: "thinking".to_string()
        }]),
        *content
    );
}

#[tokio::test]
async fn an_error_chunk_fails_the_stream() {
    let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
    let body = "data: {\"error\":{\"message\":\"model not found\"}}\n\n".to_string();
    let stream = ReaderStream::new(std::io::Cursor::new(body))
        .map_err(|err| TransportError::Network(err.to_string()));
    tokio::spawn(process_chat_sse(
        Box::pin(stream),
        tx,
        idle_timeout(),
        HashMap::new(),
    ));

    let first = rx.recv().await.expect("event");
    assert!(
        matches!(&first, Err(ApiError::Stream(message)) if message.contains("model not found")),
        "{first:?}"
    );
}

#[tokio::test]
async fn a_truncated_stream_still_reports_what_arrived() {
    // No `[DONE]`, no finish_reason: the connection just ends.
    let body = format!("data: {}\n\n", text_delta("partial"));
    let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(16);
    let stream = ReaderStream::new(std::io::Cursor::new(body))
        .map_err(|err| TransportError::Network(err.to_string()));
    tokio::spawn(process_chat_sse(
        Box::pin(stream),
        tx,
        idle_timeout(),
        HashMap::new(),
    ));

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event.expect("stream error"));
    }

    let ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) = &events[3] else {
        panic!("expected assistant message, got {:?}", events[3]);
    };
    assert_eq!(
        vec![ContentItem::OutputText {
            text: "partial".to_string()
        }],
        *content
    );
    assert!(matches!(events[4], ResponseEvent::Completed { .. }));
}

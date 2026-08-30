use super::*;
use codex_protocol::models::FunctionCallOutputPayload;
use pretty_assertions::assert_eq;
use std::sync::Arc;

fn request(input: Vec<ResponseItem>, tools: Option<Value>) -> ResponsesApiRequest {
    ResponsesApiRequest {
        model: "nvidia/nemotron-3-super-120b-a12b".to_string(),
        instructions: "be helpful".to_string(),
        input,
        tools: tools.map(|tools| {
            let raw = serde_json::value::to_raw_value(&tools).expect("serialize tools");
            crate::common::ResponsesApiTools::from(Arc::from(raw))
        }),
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

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_call(call_id: &str, name: &str, namespace: Option<&str>) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: name.to_string(),
        namespace: namespace.map(str::to_string),
        arguments: r#"{"a":1}"#.to_string(),
        encrypted_function_args: None,
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_output(call_id: &str, text: &str) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        id: None,
        call_id: Some(call_id.to_string()),
        name: None,
        namespace: None,
        output: FunctionCallOutputPayload::from_text(text.to_string()),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn build(request: &ResponsesApiRequest) -> ChatCompletionsPayload {
    build_chat_completions_payload(request, &ChatCompletionsOptions::default())
        .expect("build payload")
}

#[test]
fn instructions_become_a_leading_system_message() {
    let payload = build(&request(vec![user_message("hi")], None));
    let messages = payload.body["messages"].as_array().expect("messages");

    assert_eq!(
        vec![
            json!({"role": "system", "content": "be helpful"}),
            json!({"role": "user", "content": "hi"}),
        ],
        *messages
    );
}

#[test]
fn consecutive_tool_calls_share_one_assistant_message() {
    let payload = build(&request(
        vec![
            user_message("run both"),
            function_call("call_a", "shell", None),
            function_call("call_b", "read_file", None),
            function_output("call_a", "ok a"),
            function_output("call_b", "ok b"),
        ],
        None,
    ));
    let messages = payload.body["messages"].as_array().expect("messages");

    // One assistant message carrying both calls, then one tool reply each: the
    // shape Chat Completions requires for parallel tool calls.
    assert_eq!(5, messages.len(), "{messages:#?}");
    let tool_calls = messages[2]["tool_calls"].as_array().expect("tool_calls");
    assert_eq!(2, tool_calls.len());
    assert_eq!("call_a", tool_calls[0]["id"]);
    assert_eq!("shell", tool_calls[0]["function"]["name"]);
    assert_eq!("call_b", tool_calls[1]["id"]);
    assert_eq!(
        json!({"role": "tool", "tool_call_id": "call_a", "content": "ok a"}),
        messages[3]
    );
}

#[test]
fn namespaced_tool_calls_round_trip_through_the_flattened_name() {
    let payload = build(&request(
        vec![function_call(
            "call_a",
            "spawn_agent",
            Some("collaboration"),
        )],
        None,
    ));
    let messages = payload.body["messages"].as_array().expect("messages");
    let tool_calls = messages[1]["tool_calls"].as_array().expect("tool_calls");

    assert_eq!(
        "collaboration__spawn_agent",
        tool_calls[0]["function"]["name"]
    );
}

#[test]
fn namespaced_tools_are_flattened_and_bound() {
    let tools = json!([
        {
            "type": "namespace",
            "name": "collaboration",
            "description": "team tools",
            "tools": [{
                "type": "function",
                "name": "spawn_agent",
                "description": "spawn one",
                "strict": false,
                "parameters": {"type": "object", "properties": {}},
            }],
        },
        {
            "type": "function",
            "name": "shell",
            "description": "run a command",
            "strict": false,
            "parameters": {"type": "object", "properties": {}},
        },
    ]);
    let payload = build(&request(vec![user_message("hi")], Some(tools)));

    let declared = payload.body["tools"].as_array().expect("tools");
    let names: Vec<&str> = declared
        .iter()
        .map(|tool| tool["function"]["name"].as_str().expect("name"))
        .collect();
    assert_eq!(vec!["collaboration__spawn_agent", "shell"], names);

    assert_eq!(
        Some(&ChatToolBinding {
            namespace: Some("collaboration".to_string()),
            name: "spawn_agent".to_string(),
            freeform: false,
        }),
        payload.tool_bindings.get("collaboration__spawn_agent")
    );
    assert_eq!(
        Some(&ChatToolBinding {
            namespace: None,
            name: "shell".to_string(),
            freeform: false,
        }),
        payload.tool_bindings.get("shell")
    );
}

#[test]
fn freeform_tools_are_projected_onto_a_single_string_parameter() {
    let tools = json!([{
        "type": "custom",
        "name": "apply_patch",
        "description": "edit files",
        "format": {"type": "grammar"},
    }]);
    let payload = build(&request(vec![user_message("hi")], Some(tools)));

    let declared = payload.body["tools"].as_array().expect("tools");
    assert_eq!("apply_patch", declared[0]["function"]["name"]);
    assert_eq!(
        json!(["input"]),
        declared[0]["function"]["parameters"]["required"]
    );
    assert!(
        payload
            .tool_bindings
            .get("apply_patch")
            .is_some_and(|binding| binding.freeform)
    );
}

#[test]
fn server_side_tools_without_a_chat_equivalent_are_dropped() {
    let tools = json!([
        {"type": "web_search"},
        {"type": "tool_search", "execution": "client", "description": "", "parameters": {}},
    ]);
    let payload = build(&request(vec![user_message("hi")], Some(tools)));

    assert!(payload.body.get("tools").is_none());
    assert!(payload.tool_bindings.is_empty());
}

#[test]
fn streaming_requests_ask_for_usage() {
    let payload = build(&request(vec![user_message("hi")], None));

    assert_eq!(
        json!({"include_usage": true}),
        payload.body["stream_options"]
    );
}

#[test]
fn images_ride_along_as_content_parts_on_user_messages() {
    let item = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![
            ContentItem::InputText {
                text: "what is this".to_string(),
            },
            ContentItem::InputImage {
                image_url: "data:image/png;base64,AAAA".to_string(),
                detail: None,
            },
        ],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    };
    let payload = build(&request(vec![item], None));
    let parts = payload.body["messages"][1]["content"]
        .as_array()
        .expect("content parts");

    assert_eq!(2, parts.len());
    assert_eq!("text", parts[0]["type"]);
    assert_eq!("data:image/png;base64,AAAA", parts[1]["image_url"]["url"]);
}

#[test]
fn developer_messages_fall_back_to_system() {
    let item = ResponseItem::Message {
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: "policy".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    };
    let payload = build(&request(vec![item], None));

    assert_eq!("system", payload.body["messages"][1]["role"]);
}

#[test]
fn reasoning_effort_is_opt_in_per_provider() {
    let mut req = request(vec![user_message("hi")], None);
    req.reasoning = Some(crate::common::Reasoning {
        effort: Some(codex_protocol::openai_models::ReasoningEffort::High),
        summary: None,
        context: None,
    });

    let without = build(&req);
    assert!(without.body.get("reasoning_effort").is_none());

    let with = build_chat_completions_payload(
        &req,
        &ChatCompletionsOptions {
            send_reasoning_effort: true,
            extra_body: HashMap::new(),
        },
    )
    .expect("build payload");
    assert_eq!("high", with.body["reasoning_effort"]);
}

#[test]
fn extra_body_lets_a_provider_add_server_specific_switches() {
    let options = ChatCompletionsOptions {
        send_reasoning_effort: false,
        extra_body: HashMap::from([(
            "chat_template_kwargs".to_string(),
            json!({"thinking": true}),
        )]),
    };
    let payload =
        build_chat_completions_payload(&request(vec![user_message("hi")], None), &options)
            .expect("build payload");

    assert_eq!(
        json!({"thinking": true}),
        payload.body["chat_template_kwargs"]
    );
}

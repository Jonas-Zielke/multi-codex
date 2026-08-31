use super::*;
use pretty_assertions::assert_eq;

fn args(depth: i32) -> TeamArgs {
    TeamArgs {
        local_provider: None,
        local_model: None,
        depth,
        write: false,
        force: false,
    }
}

fn written(edits: &[ConfigEdit]) -> Vec<(String, String)> {
    edits
        .iter()
        .map(|edit| {
            let ConfigEdit::SetPath { segments, value } = edit else {
                panic!("team only writes SetPath edits");
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
fn roles_are_declared_with_paths_relative_to_codex_home() {
    let codex_home = Path::new("/home/dev/.codex");
    let edits = team_edits(&args(2), &codex_home.join(ROLES_DIR), codex_home);
    let written = written(&edits);

    // A relative path keeps a copied CODEX_HOME working on another machine.
    assert!(
        written.contains(&(
            "agents.lead.config_file".to_string(),
            r#""agents/lead.toml""#.to_string()
        )),
        "{written:?}"
    );
    assert!(
        written.contains(&(
            "agents.worker.config_file".to_string(),
            r#""agents/worker.toml""#.to_string()
        )),
        "{written:?}"
    );
}

#[test]
fn the_depth_limit_is_raised_so_leads_can_run_teams() {
    let codex_home = Path::new("/home/dev/.codex");
    let edits = team_edits(&args(3), &codex_home.join(ROLES_DIR), codex_home);

    assert!(
        written(&edits).contains(&("agents.max_depth".to_string(), "3".to_string())),
        "{:?}",
        written(&edits)
    );
}

#[test]
fn a_role_file_carries_its_endpoint_and_instructions() {
    let contents = role_file_contents(
        WORKER_ROLE,
        WORKER_INSTRUCTIONS,
        Some("nvidia/nemotron-3-nano-30b-a3b"),
        Some("vllm"),
    );

    assert!(contents.contains(r#"name = "worker""#), "{contents}");
    assert!(
        contents.contains(r#"model = "nvidia/nemotron-3-nano-30b-a3b""#),
        "{contents}"
    );
    assert!(
        contents.contains(r#"model_provider = "vllm""#),
        "{contents}"
    );
    assert!(
        contents.contains("You are one worker on a team."),
        "{contents}"
    );
    // The file has to parse as the role loader reads it.
    let parsed: toml::Value = toml::from_str(&contents).expect("role file should be valid TOML");
    assert_eq!(Some("worker"), parsed["name"].as_str());
}

#[test]
fn a_lead_inherits_the_session_endpoint() {
    let contents = role_file_contents(LEAD_ROLE, LEAD_INSTRUCTIONS, None, None);

    // Leads run on whatever the session runs on, which is the whole point of
    // splitting the hierarchy across endpoints.
    assert!(!contents.contains("model ="), "{contents}");
    assert!(!contents.contains("model_provider ="), "{contents}");
}

#[test]
fn quotes_in_an_endpoint_name_cannot_break_out_of_the_role_file() {
    let contents = role_file_contents(WORKER_ROLE, "be brief\n", Some(r#"evil"model"#), None);

    let parsed: toml::Value = toml::from_str(&contents).expect("role file should be valid TOML");
    assert_eq!(Some(r#"evil"model"#), parsed["model"].as_str());
}

#[test]
fn a_single_served_model_is_chosen_without_asking() {
    assert_eq!(
        Some("qwen3-coder".to_string()),
        choose_model(&["qwen3-coder".to_string()])
    );
}

#[test]
fn the_nemotron_family_wins_when_several_models_are_served() {
    let models = vec![
        "qwen3-coder".to_string(),
        "nvidia/Nemotron-3-Nano-30b-a3b".to_string(),
    ];

    assert_eq!(
        Some("nvidia/Nemotron-3-Nano-30b-a3b".to_string()),
        choose_model(&models)
    );
}

#[test]
fn an_ambiguous_endpoint_asks_rather_than_guessing() {
    let models = vec!["qwen3-coder".to_string(), "llama-3.3-70b".to_string()];

    assert_eq!(None, choose_model(&models));
    assert_eq!(None, choose_model(&[]));
}

#[test]
fn generated_role_files_satisfy_the_role_loader() {
    // Locks the generated format to what Codex actually accepts, so a drift in
    // either one fails here rather than at spawn time.
    let base = tempfile::tempdir().expect("create temp dir");
    for (name, instructions, model, provider) in [
        (LEAD_ROLE, LEAD_INSTRUCTIONS, None, None),
        (
            WORKER_ROLE,
            WORKER_INSTRUCTIONS,
            Some("nvidia/nemotron-3-nano-30b-a3b"),
            Some("vllm"),
        ),
    ] {
        let contents = role_file_contents(name, instructions, model, provider);
        let label = base.path().join(format!("{name}.toml"));
        let parsed = codex_agent_roles::parse_agent_role_file_contents(
            &contents,
            &label,
            base.path(),
            /*role_name_hint*/ None,
        )
        .unwrap_or_else(|err| panic!("{name} role should load: {err}"));

        assert_eq!(name, parsed.role_name);
        assert_eq!(
            model.map(str::to_string),
            parsed
                .config
                .get("model")
                .and_then(|model| model.as_str().map(str::to_string))
        );
    }
}

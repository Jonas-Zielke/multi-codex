//! `codex fleet team` — install the roles that turn subagents into an org.
//!
//! Depth alone does not make a hierarchy useful. What makes it work is two
//! roles with different jobs and different hardware: a lead that keeps the
//! whole picture and delegates, and workers that each finish one self-contained
//! piece on a small local model. This command writes both, points the workers
//! at a local endpoint, and raises the depth limit so leads may run teams.

use super::probe;
use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use codex_core::config::Config;
use codex_core::config::edit::ConfigEdit;
use codex_core::config::edit::ConfigEditsBuilder;
use std::path::Path;
use toml_edit::value;

/// Directory under `CODEX_HOME` that holds generated role files.
const ROLES_DIR: &str = "agents";
const LEAD_ROLE: &str = "lead";
const WORKER_ROLE: &str = "worker";

/// Model family this is built around, preferred when a local endpoint serves
/// several models and the caller did not name one.
const PREFERRED_MODEL_SUBSTRING: &str = "nemotron";

const LEAD_DESCRIPTION: &str = "Runs a team. Give it one domain — frontend, backend, tests, docs — \
and it breaks that domain down, spawns its own workers, reviews their results, and reports back. \
Runs on your model.";

const WORKER_DESCRIPTION: &str = "Does one self-contained piece of work and reports what it did. \
Runs on a small local model, so brief it fully and expect no memory of your conversation.";

const LEAD_INSTRUCTIONS: &str = "\
You lead a team. You were given one domain to own.

Break it into pieces small enough that one worker can finish a piece in a
single sitting, then spawn a worker per piece with agent_type \"worker\". Work
the pieces in parallel where they do not touch the same files.

Your workers run on a small local model and cannot see your conversation, so
each brief has to stand on its own: which files to touch, what done looks like,
and how to verify it. Point at the project's own checks rather than describing
them.

Review what comes back before you pass it up. When a result is wrong or
incomplete, say what is wrong and send it back rather than fixing it yourself —
you are the only one holding the whole picture, and your context is the scarce
resource here.

Report to whoever spawned you: what changed, what you verified, what is still
open. Keep it short; they are coordinating several of you.
";

const WORKER_INSTRUCTIONS: &str = "\
You are one worker on a team. You were handed a self-contained piece of work.
Finish that piece and nothing else.

Read the code before changing it — you do not have the context your lead has,
so guessing is expensive. Run the project's own checks on what you touched.

If the brief turns out to be wrong, or you are blocked, say so in your report
instead of widening the task. Your lead can re-scope; you cannot see enough to
do it safely.

Report back three things: what you changed, what you ran and what it said, and
anything you could not do.
";

#[derive(Debug, Parser)]
pub struct TeamArgs {
    /// Provider the workers run on. Defaults to the authorized endpoint that answers.
    #[arg(long, value_name = "ID")]
    local_provider: Option<String>,

    /// Model the workers run. Defaults to what the endpoint serves.
    #[arg(long, value_name = "MODEL")]
    local_model: Option<String>,

    /// How many levels of agents may spawn further agents. 2 gives you leads.
    #[arg(long, default_value_t = 2)]
    depth: i32,

    /// Write the roles and configuration instead of only describing them.
    #[arg(long, default_value_t = false)]
    write: bool,

    /// Overwrite role files that already exist.
    #[arg(long, default_value_t = false)]
    force: bool,
}

pub async fn run_team(config: &Config, args: TeamArgs) -> Result<()> {
    if args.depth < 2 {
        bail!("--depth must be at least 2 for leads to run teams of their own");
    }

    let (provider_id, model) = resolve_worker_runtime(config, &args).await?;
    let roles_dir = config.codex_home.as_path().join(ROLES_DIR);
    let lead_path = roles_dir.join(format!("{LEAD_ROLE}.toml"));
    let worker_path = roles_dir.join(format!("{WORKER_ROLE}.toml"));

    println!("lead    your model, inherited from the session");
    println!("worker  {model} on `{provider_id}`");
    println!(
        "depth   {} — leads may run teams of their own\n",
        args.depth
    );

    if !args.write {
        println!("Would write:");
        println!("  {}", lead_path.display());
        println!("  {}", worker_path.display());
        println!("  agents.lead, agents.worker, agents.max_depth in config.toml\n");
        println!("Re-run with --write to install them.");
        return Ok(());
    }

    for path in [&lead_path, &worker_path] {
        if path.exists() && !args.force {
            bail!(
                "{} already exists; edit it directly, or pass --force to replace it",
                path.display()
            );
        }
    }

    tokio::fs::create_dir_all(&roles_dir)
        .await
        .with_context(|| format!("failed to create {}", roles_dir.display()))?;
    write_role_file(&lead_path, LEAD_ROLE, LEAD_INSTRUCTIONS, None, None).await?;
    write_role_file(
        &worker_path,
        WORKER_ROLE,
        WORKER_INSTRUCTIONS,
        Some(&model),
        Some(&provider_id),
    )
    .await?;

    ConfigEditsBuilder::for_config(config)
        .with_edits(team_edits(&args, &roles_dir, config.codex_home.as_path()))
        // The org only exists under the V2 agent backend.
        .set_feature_enabled("multi_agent_v2", /*enabled*/ true)
        .apply()
        .await
        .context("failed to write config.toml")?;

    println!("Installed the lead and worker roles.\n");
    println!(
        "Ask for something big enough to split, and the agent you are talking to can hand \
domains to leads:\n  \"Add dark mode: one lead for the UI, one for the theme plumbing.\""
    );
    Ok(())
}

/// Picks the endpoint and model the workers will run on.
async fn resolve_worker_runtime(config: &Config, args: &TeamArgs) -> Result<(String, String)> {
    let provider_id = match args.local_provider.as_deref() {
        Some(id) => {
            if !config.agent_allowed_model_providers.contains(id) {
                bail!(
                    "`{id}` is not authorized for agent routing; run `codex fleet add {id} …` \
or add it to agents.allowed_model_providers"
                );
            }
            id.to_string()
        }
        None => sole_authorized_provider(config)?,
    };

    let Some(provider) = config.model_providers.get(&provider_id) else {
        bail!("`{provider_id}` is authorized for routing but not declared under model_providers");
    };
    let Some(base_url) = provider.base_url.as_deref() else {
        bail!("`{provider_id}` has no base_url, so its models cannot be listed");
    };

    if let Some(model) = args.local_model.clone() {
        return Ok((provider_id, model));
    }

    let pool = probe::client_pool(config.http_client_factory());
    let Some(endpoint) = probe::probe(&pool, base_url, /*api_key*/ None).await else {
        bail!(
            "{base_url} did not answer, so its models are unknown. \
Start it, or name the model with --local-model."
        );
    };
    let model = choose_model(&endpoint.models).ok_or_else(|| {
        anyhow::anyhow!(
            "`{provider_id}` serves {count} models; pick one with --local-model",
            count = endpoint.models.len()
        )
    })?;
    Ok((provider_id, model))
}

fn sole_authorized_provider(config: &Config) -> Result<String> {
    let mut authorized = config.agent_allowed_model_providers.iter();
    let (Some(only), None) = (authorized.next(), authorized.next()) else {
        if config.agent_allowed_model_providers.is_empty() {
            bail!(
                "no endpoint is authorized for agent routing yet. \
Run `codex fleet scan --write` first."
            );
        }
        bail!("several endpoints are authorized; choose one with --local-provider");
    };
    Ok(only.clone())
}

/// Chooses among the models an endpoint serves.
///
/// A single model is unambiguous. Otherwise prefer the family this is built
/// around rather than making the caller name it every time.
fn choose_model(models: &[String]) -> Option<String> {
    match models {
        [] => None,
        [only] => Some(only.clone()),
        models => models
            .iter()
            .find(|model| {
                model
                    .to_ascii_lowercase()
                    .contains(PREFERRED_MODEL_SUBSTRING)
            })
            .cloned(),
    }
}

/// Renders a role file.
///
/// A role that names no model or provider inherits its parent's, which is how
/// leads end up on the session's endpoint while workers move to a local one.
fn role_file_contents(
    name: &str,
    instructions: &str,
    model: Option<&str>,
    model_provider: Option<&str>,
) -> String {
    let mut contents = String::from("# Generated by `codex fleet team`. Edit freely.\n");
    contents.push_str(&format!("name = {}\n", toml_string(name)));
    if let Some(model) = model {
        contents.push_str(&format!("model = {}\n", toml_string(model)));
    }
    if let Some(model_provider) = model_provider {
        contents.push_str(&format!(
            "model_provider = {}\n",
            toml_string(model_provider)
        ));
    }
    contents.push_str(&format!(
        "\ndeveloper_instructions = {}\n",
        toml_string(instructions)
    ));
    contents
}

async fn write_role_file(
    path: &Path,
    name: &str,
    instructions: &str,
    model: Option<&str>,
    model_provider: Option<&str>,
) -> Result<()> {
    let contents = role_file_contents(name, instructions, model, model_provider);
    tokio::fs::write(path, contents)
        .await
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Renders a TOML string literal, so an endpoint or model name containing
/// quotes cannot break out of the generated file.
fn toml_string(raw: &str) -> String {
    value(raw).to_string().trim().to_string()
}

fn team_edits(args: &TeamArgs, roles_dir: &Path, codex_home: &Path) -> Vec<ConfigEdit> {
    let role_path = |role: &str| {
        // Relative paths resolve against the config.toml that declares them,
        // which keeps a copied CODEX_HOME working.
        let file = roles_dir.join(format!("{role}.toml"));
        file.strip_prefix(codex_home)
            .map(Path::to_path_buf)
            .unwrap_or(file)
    };
    let agents_field = |role: &str, key: &str, item: toml_edit::Item| ConfigEdit::SetPath {
        segments: vec!["agents".to_string(), role.to_string(), key.to_string()],
        value: item,
    };

    vec![
        agents_field(LEAD_ROLE, "description", value(LEAD_DESCRIPTION)),
        agents_field(
            LEAD_ROLE,
            "config_file",
            value(path_for_config(&role_path(LEAD_ROLE))),
        ),
        agents_field(WORKER_ROLE, "description", value(WORKER_DESCRIPTION)),
        agents_field(
            WORKER_ROLE,
            "config_file",
            value(path_for_config(&role_path(WORKER_ROLE))),
        ),
        ConfigEdit::SetPath {
            segments: vec!["agents".to_string(), "max_depth".to_string()],
            value: value(i64::from(args.depth)),
        },
    ]
}

fn path_for_config(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
#[path = "team_tests.rs"]
mod team_tests;

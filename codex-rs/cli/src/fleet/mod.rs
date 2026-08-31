//! `codex fleet` — set up the endpoints a hierarchy of agents runs on.
//!
//! The point of the fleet is that different levels of an agent hierarchy run on
//! different hardware: a large model on Nebius Token Factory coordinating,
//! small models on nearby GPUs doing the legwork. Wiring that up by hand means
//! writing provider blocks, guessing wire protocols, and remembering to
//! authorize each endpoint for agent routing. These commands find the endpoints
//! and write that configuration instead.

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use codex_core::config::Config;
use codex_core::config::edit::ConfigEdit;
use codex_core::config::edit::ConfigEditsBuilder;
use codex_model_provider_info::NEBIUS_CHAT_PROVIDER_ID;
use codex_model_provider_info::NEBIUS_PROVIDER_ID;
use codex_model_provider_info::WireApi;
use codex_model_provider_info::built_in_model_providers;
use codex_utils_cli::CliConfigOverrides;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use toml_edit::Array;
use toml_edit::Item as TomlItem;
use toml_edit::value;

mod doctor;
mod probe;
mod team;

use probe::Endpoint;
use probe::Runtime;

#[derive(Debug, Parser)]
pub struct FleetCommand {
    #[command(subcommand)]
    action: FleetAction,
}

#[derive(Debug, Subcommand)]
enum FleetAction {
    /// Find inference endpoints running nearby and configure them.
    Scan(ScanArgs),
    /// Configure one endpoint by URL, for a GPU box that is not on this host.
    Add(AddArgs),
    /// Show the configured endpoints and whether they are answering.
    List,
    /// Install the lead and worker roles that make a hierarchy an org.
    Team(team::TeamArgs),
    /// Check that the configured fleet will actually run.
    Doctor,
}

#[derive(Debug, Parser)]
struct ScanArgs {
    /// Host to scan. Point this at a DGX Spark or workstation on your network.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to try. Repeatable; defaults to the ports Ollama, LM Studio, vLLM
    /// and llama.cpp use.
    #[arg(long = "port", value_name = "PORT")]
    ports: Vec<u16>,

    /// Write the discovered endpoints to config.toml instead of only listing them.
    #[arg(long, default_value_t = false)]
    write: bool,
}

#[derive(Debug, Parser)]
struct AddArgs {
    /// Provider id to write, as roles and `--profile` will refer to it.
    #[arg(value_name = "ID")]
    id: String,

    /// OpenAI-compatible base URL, including the `/v1` suffix.
    #[arg(long, value_name = "URL")]
    url: String,

    /// Wire protocol to configure. Probes the endpoint by default.
    #[arg(long, value_enum, default_value_t = WireChoice::Auto)]
    wire: WireChoice,

    /// Environment variable holding the API key, for endpoints that need one.
    #[arg(long, value_name = "ENV")]
    api_key_env: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WireChoice {
    /// Ask the endpoint which protocols it serves.
    Auto,
    Chat,
    Responses,
}

pub async fn run_main(command: FleetCommand, config_overrides: CliConfigOverrides) -> Result<()> {
    let overrides = config_overrides
        .parse_overrides()
        .map_err(anyhow::Error::msg)?;
    let config = Config::load_with_cli_overrides(overrides)
        .await
        .context("failed to load configuration")?;

    match command.action {
        FleetAction::Scan(args) => run_scan(&config, args).await,
        FleetAction::Add(args) => run_add(&config, args).await,
        FleetAction::List => run_list(&config).await,
        FleetAction::Team(args) => team::run_team(&config, args).await,
        FleetAction::Doctor => {
            // A non-zero exit lets this gate a script or a CI step.
            if doctor::run_doctor(&config).await? {
                Ok(())
            } else {
                std::process::exit(1);
            }
        }
    }
}

async fn run_scan(config: &Config, args: ScanArgs) -> Result<()> {
    let ports = if args.ports.is_empty() {
        probe::default_ports()
    } else {
        args.ports.clone()
    };
    let pool = probe::client_pool(config.http_client_factory());

    println!(
        "Scanning {host} on {count} port(s)…\n",
        host = args.host,
        count = ports.len()
    );
    let probes = ports.iter().map(|port| {
        let base_url = format!("http://{host}:{port}/v1", host = args.host);
        let pool = &pool;
        async move {
            probe::probe(pool, &base_url, /*api_key*/ None).await
        }
    });
    let found: Vec<Endpoint> = futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .collect();

    if found.is_empty() {
        println!("No inference endpoints answered.");
        println!("Start one, or point at another machine with --host.");
        return Ok(());
    }

    let assignments = assign_provider_ids(config, &found);
    for (id, endpoint) in &assignments {
        print_endpoint(id, endpoint);
        // A built-in provider carries a fixed wire protocol and cannot be
        // overridden, so a server that disagrees would fail at request time
        // with nothing pointing at the cause.
        if let Some(configured) = config.model_providers.get(id)
            && configured.wire_api != endpoint.wire_api
        {
            println!(
                "    warning: `{id}` is configured for {configured_wire}, but this server \
answers {detected_wire}. Configure it under another id with `codex fleet add`.",
                configured_wire = configured.wire_api,
                detected_wire = endpoint.wire_api,
            );
        }
    }

    if !args.write {
        println!("Re-run with --write to add these to config.toml.");
        return Ok(());
    }

    let existing_allowlist = config.agent_allowed_model_providers.clone();
    let mut edits = Vec::new();
    let mut written = Vec::new();
    for (id, endpoint) in &assignments {
        if !config.model_providers.contains_key(id) {
            edits.extend(provider_edits(
                id,
                &endpoint.base_url,
                endpoint.runtime.label(),
                endpoint.wire_api,
                /*api_key_env*/ None,
            ));
        }
        written.push(id.clone());
    }
    edits.push(allowlist_edit(&existing_allowlist, &written));

    ConfigEditsBuilder::for_config(config)
        .with_edits(edits)
        .apply()
        .await
        .context("failed to write config.toml")?;

    println!(
        "Wrote {count} endpoint(s) to config.toml and authorized them for agent routing.",
        count = written.len()
    );
    print_next_steps(&written);
    Ok(())
}

async fn run_add(config: &Config, args: AddArgs) -> Result<()> {
    validate_provider_id(&args.id)?;
    if built_in_model_providers(/*openai_base_url*/ None).contains_key(&args.id)
        && !matches!(
            args.id.as_str(),
            NEBIUS_PROVIDER_ID | NEBIUS_CHAT_PROVIDER_ID
        )
    {
        bail!(
            "`{id}` is a built-in provider and cannot be redefined; choose another id",
            id = args.id
        );
    }

    let base_url = args.url.trim_end_matches('/').to_string();
    let api_key = args
        .api_key_env
        .as_deref()
        .and_then(|env| std::env::var(env).ok());

    let pool = probe::client_pool(config.http_client_factory());
    let endpoint = probe::probe(&pool, &base_url, api_key.as_deref()).await;
    let wire_api = match args.wire {
        WireChoice::Chat => WireApi::Chat,
        WireChoice::Responses => WireApi::Responses,
        WireChoice::Auto => match endpoint.as_ref() {
            Some(endpoint) => endpoint.wire_api,
            None => bail!(
                "{base_url} did not answer, so its wire protocol is unknown. \
Start the server, or pass --wire chat|responses to configure it anyway."
            ),
        },
    };

    match endpoint.as_ref() {
        Some(endpoint) => print_endpoint(&args.id, endpoint),
        None => println!("{id}  {base_url}  (not answering)", id = args.id),
    }

    let label = endpoint
        .as_ref()
        .map_or(Runtime::Unknown, |endpoint| endpoint.runtime)
        .label();
    let mut edits = provider_edits(
        &args.id,
        &base_url,
        label,
        wire_api,
        args.api_key_env.as_deref(),
    );
    edits.push(allowlist_edit(
        &config.agent_allowed_model_providers,
        std::slice::from_ref(&args.id),
    ));

    ConfigEditsBuilder::for_config(config)
        .with_edits(edits)
        .apply()
        .await
        .context("failed to write config.toml")?;

    println!(
        "Added `{id}` and authorized it for agent routing.",
        id = args.id
    );
    print_next_steps(std::slice::from_ref(&args.id));
    Ok(())
}

async fn run_list(config: &Config) -> Result<()> {
    let pool = probe::client_pool(config.http_client_factory());
    let mut providers: Vec<_> = config.model_providers.iter().collect();
    providers.sort_by_key(|(id, _)| id.as_str());

    let id_width = providers
        .iter()
        .map(|(id, _)| id.len())
        .chain(std::iter::once("PROVIDER".len()))
        .max()
        .unwrap_or_default();
    println!(
        "{:<id_width$} {:<10} {:<9} ENDPOINT",
        "PROVIDER", "WIRE", "ROUTABLE"
    );
    for (id, provider) in providers {
        // A provider with no configured URL still resolves one at request time.
        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or("(built-in endpoint)");
        let routable = if config.agent_allowed_model_providers.contains(id) {
            "yes"
        } else {
            "no"
        };
        // Only local endpoints are probed: a hosted one needs credentials, and
        // an unauthenticated request would just report a misleading failure.
        let reachability = if is_loopback_url(base_url) {
            match probe::probe(&pool, base_url, /*api_key*/ None).await {
                Some(endpoint) => match endpoint.models.len() {
                    1 => "  (1 model)".to_string(),
                    count => format!("  ({count} models)"),
                },
                None => "  (not answering)".to_string(),
            }
        } else {
            String::new()
        };
        println!(
            "{id:<id_width$} {wire:<10} {routable:<9} {base_url}{reachability}",
            wire = provider.wire_api.to_string(),
        );
    }

    if config.agent_allowed_model_providers.is_empty() {
        println!(
            "\nNo provider is authorized for agent routing yet, so every agent runs on the \
session's own endpoint."
        );
    }
    Ok(())
}

/// Chooses the config id to write for each discovered endpoint.
///
/// A runtime found where a built-in provider already points is reported under
/// that built-in id rather than duplicated, since the built-in cannot be
/// overridden anyway.
fn assign_provider_ids(config: &Config, found: &[Endpoint]) -> Vec<(String, Endpoint)> {
    let built_ins = built_in_model_providers(/*openai_base_url*/ None);
    let mut by_base_url: BTreeMap<String, &str> = BTreeMap::new();
    for (id, provider) in &built_ins {
        if let Some(base_url) = provider.base_url.as_deref() {
            by_base_url.insert(normalize_base_url(base_url), id.as_str());
        }
    }

    found
        .iter()
        .map(|endpoint| {
            let id = by_base_url
                .get(&normalize_base_url(&endpoint.base_url))
                .map(|id| (*id).to_string())
                .unwrap_or_else(|| generated_provider_id(config, endpoint));
            (id, endpoint.clone())
        })
        .collect()
}

/// Normalizes a base URL for comparison.
///
/// `localhost`, `127.0.0.1` and `[::1]` name the same endpoint, and which one
/// appears depends on whether it was typed, scanned, or built in. Comparing
/// them literally would report an already-configured server as a new one.
fn normalize_base_url(base_url: &str) -> String {
    let normalized = base_url.trim_end_matches('/').to_ascii_lowercase();
    for alias in ["//localhost", "//[::1]"] {
        let Some(index) = normalized.find(alias) else {
            continue;
        };
        let (prefix, rest) = normalized.split_at(index);
        let tail = &rest[alias.len()..];
        // Only a whole host component is an alias: `localhost.example.com` is
        // an ordinary hostname that happens to start with one.
        if !tail.is_empty() && !tail.starts_with(':') && !tail.starts_with('/') {
            continue;
        }
        return format!("{prefix}//127.0.0.1{tail}");
    }
    normalized
}

fn generated_provider_id(config: &Config, endpoint: &Endpoint) -> String {
    let slug = endpoint.runtime.slug();
    let port = endpoint
        .base_url
        .rsplit(':')
        .next()
        .and_then(|tail| tail.split('/').next())
        .unwrap_or_default();

    // The bare runtime name reads best, so only qualify it when it is taken by
    // something pointing somewhere else.
    let bare = slug.to_string();
    let claimed_by_another_endpoint = config.model_providers.get(&bare).is_some_and(|existing| {
        existing.base_url.as_deref().map(normalize_base_url)
            != Some(normalize_base_url(&endpoint.base_url))
    });
    if claimed_by_another_endpoint {
        format!("{slug}-{port}")
    } else {
        bare
    }
}

fn provider_edits(
    id: &str,
    base_url: &str,
    name: &str,
    wire_api: WireApi,
    api_key_env: Option<&str>,
) -> Vec<ConfigEdit> {
    let field = |key: &str, item: TomlItem| ConfigEdit::SetPath {
        segments: vec![
            "model_providers".to_string(),
            id.to_string(),
            key.to_string(),
        ],
        value: item,
    };

    let mut edits = vec![
        field("name", value(name)),
        field("base_url", value(base_url)),
        field("wire_api", value(wire_api.to_string())),
    ];
    if let Some(env) = api_key_env {
        edits.push(field("env_key", value(env)));
    }
    edits
}

/// Produces the edit that authorizes `additions` for role-based routing,
/// preserving whatever was already authorized.
fn allowlist_edit(existing: &BTreeSet<String>, additions: &[String]) -> ConfigEdit {
    let mut allowed: BTreeSet<String> = existing.clone();
    allowed.extend(additions.iter().cloned());

    let mut array = Array::new();
    for id in &allowed {
        array.push(id.as_str());
    }
    ConfigEdit::SetPath {
        segments: vec!["agents".to_string(), "allowed_model_providers".to_string()],
        value: value(array),
    }
}

fn validate_provider_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("provider id must be non-empty and use only letters, digits, `-` and `_`");
    }
    Ok(())
}

fn is_loopback_url(base_url: &str) -> bool {
    base_url.contains("//127.0.0.1")
        || base_url.contains("//localhost")
        || base_url.contains("//[::1]")
}

fn print_endpoint(id: &str, endpoint: &Endpoint) {
    let models = if endpoint.models.is_empty() {
        "no models loaded".to_string()
    } else {
        let shown = endpoint
            .models
            .iter()
            .take(3)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        match endpoint.models.len() {
            0..=3 => shown,
            total => format!("{shown}, +{} more", total - 3),
        }
    };
    println!(
        "{id}  {label}  {base_url}  wire={wire}\n    {models}",
        label = endpoint.runtime.label(),
        base_url = endpoint.base_url,
        wire = endpoint.wire_api,
    );
}

fn print_next_steps(ids: &[String]) {
    let Some(first) = ids.first() else {
        return;
    };
    println!(
        "\nGive a role its own endpoint by adding `model_provider = \"{first}\"` to its file \
under ~/.codex/agents/, then spawn agents with that role."
    );
}

#[cfg(test)]
#[path = "fleet_tests.rs"]
mod fleet_tests;

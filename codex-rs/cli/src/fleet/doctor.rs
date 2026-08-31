//! `codex fleet doctor` — check that a configured fleet will actually run.
//!
//! A hierarchy spread across endpoints has several ways to be almost right: a
//! key that is not exported, a local server that is not up, a role pointing at
//! an endpoint that was never authorized, a depth limit that quietly keeps
//! leads from having teams. Each of those surfaces much later, as an agent
//! behaving oddly rather than as a configuration error. This checks them up
//! front and says what to do about each one.

use super::probe;
use anyhow::Result;
use codex_core::config::Config;
use codex_features::Feature;
use codex_http_client::RouteAwareClientPool;
use codex_model_provider_info::ModelProviderInfo;
use std::fmt::Write as _;

/// One line of the report.
#[derive(Debug, PartialEq, Eq)]
enum Check {
    Ok(String),
    /// Works, but not the way the fleet is meant to be set up.
    Warn {
        note: String,
        fix: String,
    },
    /// Will not work as configured.
    Fail {
        note: String,
        fix: String,
    },
}

impl Check {
    fn render(&self, into: &mut String) {
        match self {
            Self::Ok(note) => {
                let _ = writeln!(into, "  ok    {note}");
            }
            Self::Warn { note, fix } => {
                let _ = writeln!(into, "  warn  {note}\n          {fix}");
            }
            Self::Fail { note, fix } => {
                let _ = writeln!(into, "  FAIL  {note}\n          {fix}");
            }
        }
    }

    fn is_failure(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }
}

#[derive(Default)]
struct Report {
    sections: Vec<(String, Vec<Check>)>,
}

impl Report {
    fn section(&mut self, title: &str, checks: Vec<Check>) {
        self.sections.push((title.to_string(), checks));
    }

    fn failed(&self) -> bool {
        self.sections
            .iter()
            .any(|(_, checks)| checks.iter().any(Check::is_failure))
    }

    fn render(&self) -> String {
        let mut out = String::new();
        for (title, checks) in &self.sections {
            let _ = writeln!(out, "{title}");
            for check in checks {
                check.render(&mut out);
            }
            out.push('\n');
        }
        out
    }
}

pub async fn run_doctor(config: &Config) -> Result<bool> {
    let pool = probe::client_pool(config.http_client_factory());
    let mut report = Report::default();

    report.section("Session", session_checks(config, &pool).await);
    report.section("Routing", routing_checks(config));
    report.section("Roles", role_checks(config, &pool).await);
    report.section("Teams", team_checks(config));

    print!("{}", report.render());
    let ok = !report.failed();
    if ok {
        println!("The fleet is ready.");
    } else {
        println!("Fix the failures above, then run `codex fleet doctor` again.");
    }
    Ok(ok)
}

/// Checks the endpoint the session itself talks to.
async fn session_checks(config: &Config, pool: &RouteAwareClientPool) -> Vec<Check> {
    let provider_id = config.model_provider_id.as_str();
    let provider = &config.model_provider;
    let model = config.model.as_deref().unwrap_or_default();
    let mut checks = vec![Check::Ok(format!(
        "provider `{provider_id}` → {endpoint}",
        endpoint = provider
            .base_url
            .as_deref()
            .unwrap_or("its built-in endpoint")
    ))];

    let api_key = match credential_check(provider) {
        Ok(api_key) => {
            if let Some(env_key) = provider.env_key.as_deref() {
                checks.push(Check::Ok(format!("{env_key} is set")));
            }
            api_key
        }
        Err(check) => {
            checks.push(check);
            return checks;
        }
    };

    if model.is_empty() {
        checks.push(Check::Ok(
            "no model pinned — the session default applies".to_string(),
        ));
    }
    checks.push(model_availability(pool, provider, &api_key, model, "session").await);
    checks
}

/// Confirms the credential a provider declares is actually available.
fn credential_check(provider: &ModelProviderInfo) -> Result<Option<String>, Check> {
    let Some(env_key) = provider.env_key.as_deref() else {
        return Ok(None);
    };
    match std::env::var(env_key) {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        _ => Err(Check::Fail {
            note: format!("{env_key} is not set"),
            fix: format!("export {env_key}=… and run this again."),
        }),
    }
}

/// Asks an endpoint whether it serves the model something is configured to use.
async fn model_availability(
    pool: &RouteAwareClientPool,
    provider: &ModelProviderInfo,
    api_key: &Option<String>,
    model: &str,
    label: &str,
) -> Check {
    let Some(base_url) = provider.base_url.as_deref() else {
        // Providers that resolve their endpoint at request time have nothing
        // to ask, so there is nothing to verify here.
        return Check::Ok("the endpoint is resolved at request time".to_string());
    };
    let Some(endpoint) = probe::probe(pool, base_url, api_key.as_deref()).await else {
        return Check::Fail {
            note: format!("{label}: {base_url} is not answering"),
            fix: "Start the server, or re-point it with `codex fleet add`.".to_string(),
        };
    };

    if model.is_empty() {
        return Check::Ok(format!(
            "{label} endpoint answers with {count} model(s)",
            count = endpoint.models.len()
        ));
    }
    if endpoint.models.iter().any(|served| served == model) {
        return Check::Ok(format!("serves {model}"));
    }
    // A model list is advisory: some gateways serve models they do not
    // enumerate, so this is worth flagging without calling it broken.
    Check::Warn {
        note: format!("{label}: {base_url} does not list {model}"),
        fix: format!(
            "It may still serve it. Otherwise pick one it lists: {}.",
            sample_models(&endpoint.models)
        ),
    }
}

fn sample_models(models: &[String]) -> String {
    if models.is_empty() {
        return "none listed".to_string();
    }
    models
        .iter()
        .take(3)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

fn routing_checks(config: &Config) -> Vec<Check> {
    if config.agent_allowed_model_providers.is_empty() {
        return vec![Check::Warn {
            note: "no endpoint is authorized for agent routing".to_string(),
            fix: "Every agent runs on the session's endpoint. Run `codex fleet scan --write`."
                .to_string(),
        }];
    }

    config
        .agent_allowed_model_providers
        .iter()
        .map(|id| {
            if config.model_providers.contains_key(id) {
                Check::Ok(format!("`{id}` is authorized for routing"))
            } else {
                Check::Fail {
                    note: format!("`{id}` is authorized but not declared"),
                    fix: format!("Add [model_providers.{id}], or drop it from the allowlist."),
                }
            }
        })
        .collect()
}

async fn role_checks(config: &Config, pool: &RouteAwareClientPool) -> Vec<Check> {
    if config.agent_roles.is_empty() {
        return vec![Check::Warn {
            note: "no roles are declared".to_string(),
            fix: "Run `codex fleet team --write` to install a lead and worker.".to_string(),
        }];
    }

    let mut checks = Vec::new();
    for (name, role) in &config.agent_roles {
        let Some(path) = role.config_file.as_ref() else {
            checks.push(Check::Ok(format!("{name} declares no config file")));
            continue;
        };
        let contents = match tokio::fs::read_to_string(path).await {
            Ok(contents) => contents,
            Err(err) => {
                checks.push(Check::Fail {
                    note: format!("{name} cannot read {}: {err}", path.display()),
                    fix: "Fix the path under [agents] in config.toml.".to_string(),
                });
                continue;
            }
        };
        let base_dir = path.parent().unwrap_or(config.codex_home.as_path());
        let parsed = match codex_agent_roles::parse_agent_role_file_contents(
            &contents,
            path,
            base_dir,
            Some(name),
        ) {
            Ok(parsed) => parsed,
            Err(err) => {
                checks.push(Check::Fail {
                    note: format!("{name} does not parse: {err}"),
                    fix: format!("Fix {}.", path.display()),
                });
                continue;
            }
        };

        let model = parsed
            .config
            .get("model")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let Some(provider_id) = parsed
            .config
            .get("model_provider")
            .and_then(|value| value.as_str())
        else {
            checks.push(Check::Ok(format!("{name} inherits the session endpoint")));
            continue;
        };

        checks.extend(routed_role_checks(config, pool, name, provider_id, model).await);
    }
    checks
}

/// Checks a role that names its own endpoint.
async fn routed_role_checks(
    config: &Config,
    pool: &RouteAwareClientPool,
    name: &str,
    provider_id: &str,
    model: &str,
) -> Vec<Check> {
    if !config.agent_allowed_model_providers.contains(provider_id) {
        return vec![Check::Fail {
            note: format!("{name} routes to `{provider_id}`, which is not authorized"),
            fix: format!(
                "Add `{provider_id}` to agents.allowed_model_providers. \
Spawning with this role fails until then."
            ),
        }];
    }
    let Some(provider) = config.model_providers.get(provider_id) else {
        return vec![Check::Fail {
            note: format!("{name} routes to `{provider_id}`, which is not declared"),
            fix: format!("Add [model_providers.{provider_id}] to config.toml."),
        }];
    };
    if model.is_empty() {
        // Without its own model the role keeps the parent's slug, which the
        // other endpoint almost certainly does not serve.
        return vec![Check::Warn {
            note: format!("{name} routes to `{provider_id}` but names no model"),
            fix: "Add `model = \"…\"` to the role file; it otherwise inherits a slug \
the other endpoint does not serve."
                .to_string(),
        }];
    }

    let api_key = match credential_check(provider) {
        Ok(api_key) => api_key,
        Err(check) => return vec![check],
    };
    vec![
        match model_availability(pool, provider, &api_key, model, name).await {
            Check::Ok(_) => Check::Ok(format!("{name} runs {model} on `{provider_id}`")),
            other => other,
        },
    ]
}

fn team_checks(config: &Config) -> Vec<Check> {
    let mut checks = Vec::new();
    if config.features.enabled(Feature::MultiAgentV2) {
        checks.push(Check::Ok("the multi-agent backend is enabled".to_string()));
    } else {
        checks.push(Check::Fail {
            note: "the multi-agent backend is disabled".to_string(),
            fix: "Set features.multi_agent_v2 = true, or run `codex fleet team --write`."
                .to_string(),
        });
    }

    if config.agent_max_depth >= 2 {
        checks.push(Check::Ok(format!(
            "agents.max_depth = {} — leads may run teams",
            config.agent_max_depth
        )));
    } else {
        checks.push(Check::Warn {
            note: format!(
                "agents.max_depth = {} — spawned agents cannot spawn",
                config.agent_max_depth
            ),
            fix: "Set agents.max_depth = 2 for team leads.".to_string(),
        });
    }
    checks
}

#[cfg(test)]
#[path = "doctor_tests.rs"]
mod doctor_tests;

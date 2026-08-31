# multi-codex

**A coding agent that runs as an org chart across a frontier model in the cloud
and small models on your own hardware.**

A fork of [OpenAI Codex CLI](https://github.com/openai/codex) that adds three
things: it can talk to [Nebius Token Factory](https://tokenfactory.nebius.com)
and to any OpenAI-compatible server on your network, it can put different agents
on different endpoints, and it can nest them into teams.

The agent you talk to runs on a large model — NVIDIA Nemotron 3 Ultra or Super
on Nebius. It does not do the work. It hands a domain to a **lead**, which
breaks that domain into self-contained pieces and spawns **workers** that run
Nemotron 3 Nano on your own GPU. Workers report to their lead, the lead reviews
before passing anything up, and you only ever talk to the top.

```
                    you
                     │
             ┌───────┴────────┐        Nemotron 3 Ultra · Nebius Token Factory
             │  coordinator   │
             └───────┬────────┘
          ┌──────────┴──────────┐
     ┌────┴─────┐         ┌─────┴────┐              Nemotron 3 Super · Nebius
     │ UI lead  │         │ API lead │
     └────┬─────┘         └─────┬────┘
      ┌───┴───┐             ┌───┴───┐
    ┌─┴─┐   ┌─┴─┐         ┌─┴─┐   ┌─┴─┐        Nemotron 3 Nano · your GPU
    │ w │   │ w │         │ w │   │ w │
    └───┘   └───┘         └───┘   └───┘
```

## Why

Most of what a coding agent does is not hard, there is just a lot of it: reading
files to answer one question, applying a change that has already been decided,
running the test suite and reporting what broke, writing the third variation of
something that exists twice. Sending all of that to a frontier endpoint is what
makes agent runs slow and expensive.

Splitting it is not new. Doing it *across endpoints*, so the volume work never
leaves your hardware, is what this fork adds — and once agents can sit on
different endpoints, nesting them into teams is what keeps the expensive model's
context free for the decisions only it can make.

It also means the code that stays local, stays local.

## Quickstart

```shell
# Build
cd codex-rs && cargo build --release -p codex-cli --bin codex

# 1. The cloud model
export NEBIUS_API_KEY=...

# 2. Find whatever is serving models nearby (Ollama, LM Studio, vLLM, llama.cpp)
codex fleet scan --write

# 3. Install the lead and worker roles
codex fleet team --write

# 4. Check it before you rely on it
codex fleet doctor

# 5. Go
codex -c model_provider=nebius -c model="nvidia/Nemotron-3-Ultra-550b-a55b"
```

Then ask for something big enough to split:

> Add dark mode: one lead for the UI, one for the theme plumbing.

See [docs/fleet.md](./docs/fleet.md) for the full setup, including endpoints on
another machine.

## What this fork adds

**A chat-completions wire protocol.** Codex speaks the OpenAI Responses API.
Nebius serves it for some models but not all, and several local runtimes —
llama.cpp, Unsloth Studio, older vLLM builds — do not serve it at all.
`wire_api = "chat"` is a first-class transport rather than a translating
sidecar: it rewrites a turn into chat messages on the way out and
reassembles response items on the way in, keeping tool calls, streamed
reasoning, and token accounting intact. It inherits the client's existing auth,
retry, and telemetry.

**Per-agent endpoint routing.** A role may name the provider its agents run on,
within an allowlist you declare. That allowlist is the trust boundary: a role
file that could name any endpoint would be a way to redirect model traffic, so
roles select from what you sanctioned and cannot introduce endpoints of their
own.

**Hierarchical teams.** `agents.max_depth` decides how many levels may spawn
further agents. The default of `1` keeps the tree flat, as it was; `2` gives you
leads that run teams of their own.

**`codex fleet`.** Finds inference endpoints by asking each candidate port what
it is, works out which wire protocol it speaks, and writes the configuration —
providers, allowlist, roles, depth — so none of the above has to be assembled by
hand. `codex fleet doctor` then checks the result end to end and exits non-zero
when something will not work.

## Configuration

Everything lives in `~/.codex/config.toml`, and `codex fleet` writes it for you:

```toml
model_provider = "nebius"
model = "nvidia/Nemotron-3-Ultra-550b-a55b"

[model_providers.vllm]
base_url = "http://127.0.0.1:8000/v1"
wire_api = "chat"

[agents]
# Endpoints a role may route its agents to.
allowed_model_providers = ["nebius", "vllm"]
# How many levels may spawn further agents. 2 gives you team leads.
max_depth = 2
```

## Docs

- [Fleets and teams](./docs/fleet.md) — setup, routing, and what to run where
- [Codex configuration](https://github.com/openai/codex/blob/main/docs/config.md)
- [Codex documentation](https://developers.openai.com/codex)

## Upstream

This is a fork of [openai/codex](https://github.com/openai/codex), licensed
under [Apache-2.0](./LICENSE). Everything the upstream CLI does, this does —
sandboxing, approvals, MCP servers, skills, the TUI. The additions above sit
alongside it and are off by default: without a configured fleet, this behaves
exactly like upstream Codex.

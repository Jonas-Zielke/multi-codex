# Fleets and teams

A *fleet* is the set of inference endpoints one Codex session can hand work to.
A *team* is the org chart it runs across them: the agent you talk to delegates
domains to leads, and each lead runs its own workers.

The point of putting these together is that the two levels want different
hardware. Planning and review need a model that holds a lot of context at once.
Reading files, writing boilerplate, and running tests do not — and there are far
more of those. Sending both to the same frontier endpoint is what makes agent
runs slow and expensive.

## Setting up

### 1. The cloud model

[Nebius Token Factory](https://tokenfactory.nebius.com) serves NVIDIA's open
Nemotron models. Create an API key and export it:

```shell
export NEBIUS_API_KEY=...
```

Two built-in providers point at it. Which one you want depends on the model:
the Token Factory console shows **Responses API: Available** on the model cards
that serve it.

| Provider      | Wire protocol     | Use when                                     |
| ------------- | ----------------- | -------------------------------------------- |
| `nebius`      | Responses API     | The model card lists Responses API support   |
| `nebius-chat` | Chat completions  | It does not                                  |

Select one for your session, with the routing key from the model card:

```shell
codex -c model_provider=nebius -c model="nvidia/Nemotron-3-Ultra-550b-a55b"
```

Or put it in `~/.codex/config.toml` so you do not repeat it:

```toml
model_provider = "nebius"
model = "nvidia/Nemotron-3-Ultra-550b-a55b"
```

Nebius also serves the same catalog from regional hosts. Point at one by
overriding just the endpoint — the provider itself stays built in:

```toml
[model_providers.nebius]
base_url = "https://api.tokenfactory.us-central1.nebius.com/v1"
```

### 2. The local models

Start any OpenAI-compatible server holding a smaller model — Nemotron 3 Nano is
the one this is built around:

```shell
# vLLM
vllm serve nvidia/nemotron-3-nano-30b-a3b --port 8000

# or Ollama
ollama serve
```

Then let Codex find it:

```shell
$ codex fleet scan --write
Scanning 127.0.0.1 on 4 port(s)…

vllm  vLLM  http://127.0.0.1:8000/v1  wire=chat
    nvidia/nemotron-3-nano-30b-a3b
Wrote 1 endpoint(s) to config.toml and authorized them for agent routing.
```

`scan` asks each candidate port what it is: the model list comes from
`/v1/models`, the runtime from the side-channel API each server exposes, and the
wire protocol from whether `/v1/responses` is routed at all. It recognizes
Ollama, LM Studio, vLLM, and llama.cpp — which is also what Unsloth Studio
serves through.

For a machine that is not this one, a DGX Spark or a workstation on your
network:

```shell
codex fleet add dgx-spark --url http://10.0.0.5:8000/v1
```

`codex fleet list` shows what is configured, what is answering, and what is
authorized for agent routing.

### 3. The team

```shell
$ codex fleet team --write
lead    your model, inherited from the session
worker  nvidia/nemotron-3-nano-30b-a3b on `vllm`
depth   2 — leads may run teams of their own

Installed the lead and worker roles.
```

That writes two role files under `~/.codex/agents/`, raises the depth limit so
leads may spawn their own workers, and enables the V2 agent backend. Edit the
files freely afterwards — they are yours.

### 4. Check it

```shell
$ codex fleet doctor
Session
  ok    provider `nebius` → https://api.tokenfactory.nebius.com/v1
  ok    NEBIUS_API_KEY is set
  ok    serves nvidia/Nemotron-3-Ultra-550b-a55b

Routing
  ok    `vllm` is authorized for routing

Roles
  ok    lead inherits the session endpoint
  ok    worker runs nvidia/nemotron-3-nano-30b-a3b on `vllm`

Teams
  ok    the multi-agent backend is enabled
  ok    agents.max_depth = 2 — leads may run teams

The fleet is ready.
```

A fleet has several ways to be almost right: a key that is not exported, a
local server that is not up, a role pointing at an endpoint that was never
authorized. Each of those otherwise surfaces much later, as an agent behaving
oddly rather than as a configuration error. `doctor` exits non-zero when
something will not work, so it can gate a script.

Now ask for something big enough to split:

> Add dark mode: one lead for the UI, one for the theme plumbing.

The agent you are talking to spawns a lead per domain on your cloud model. Each
lead breaks its domain into self-contained pieces and spawns workers on the
local endpoint. Results flow back up: workers report to their lead, the lead
reviews before passing anything on, and you only talk to the top.

## How the routing works

Three pieces have to line up.

**A wire protocol per endpoint.** Codex speaks the Responses API. Most
OpenAI-compatible servers — and some Nebius models — only offer chat
completions, so `wire_api = "chat"` translates a turn into chat messages on the
way out and back into response items on the way in. Tool calls, streamed
reasoning, and token usage all survive the round trip.

**A provider per role.** A role file may name the endpoint its agents run on:

```toml
# ~/.codex/agents/worker.toml
name = "worker"
model = "nvidia/nemotron-3-nano-30b-a3b"
model_provider = "vllm"
```

This only takes effect for providers you have authorized:

```toml
[agents]
allowed_model_providers = ["nebius", "vllm"]
```

The allowlist exists because a role file that could name any endpoint would be a
way to redirect model traffic — including the source code in your prompts —
somewhere you did not choose. Roles select from what you sanctioned in
`config.toml`; they cannot introduce endpoints of their own. A role naming an
unauthorized provider is rejected outright rather than quietly falling back to
the session's endpoint, so a routing mistake fails visibly instead of silently
running the expensive model.

**A depth limit.** `agents.max_depth` decides how many levels may spawn further
agents:

```toml
[agents]
max_depth = 2
```

`1`, the default, keeps the tree flat: the agent you talk to manages a pool of
workers directly. `2` gives you leads. Higher values nest further, at the cost
of every level between you and the work.

## Tuning an endpoint

OpenAI-compatible servers disagree about which optional request fields they
accept, so anything beyond the common core is opt-in per provider:

```toml
[model_providers.vllm.chat]
# Forward the turn's reasoning effort. Off by default: some servers reject
# request fields they do not recognize.
send_reasoning_effort = true

# Merged into every request body, for server-specific switches.
extra_body = { chat_template_kwargs = { thinking = true } }
```

## Choosing what runs where

The split that pays off is judgement versus volume, not hard versus easy.

Keep on the large model: deciding what to build, reviewing a worker's diff,
resolving a conflict between two workers, and anything where being wrong is
expensive to discover later.

Move to local models: reading a file to answer a specific question, applying a
change that has already been specified, running tests and reporting what broke,
writing the third variation of something that already exists twice.

A lead is worth its cost when it removes more work from your model than it adds.
A domain with two or three pieces usually does not need one; the top-level agent
can hold that itself.

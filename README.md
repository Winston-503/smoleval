# smoleval 🧪🤖🦀

Minimal and lightweight evaluation framework for AI agents written in Rust. Define test cases in YAML and get structured pass/fail reports.

## Features

- YAML-driven evaluation datasets
- CLI tool for running evals against HTTP agent endpoints
- Registry-based check system with pluggable built-in validators
- Structured reports with per-test scores and aggregate metrics
- Pluggable `Agent` trait — back it with an HTTP API, local model, or a mock

## Quick start

### Define a dataset

```yaml
# eval.yaml
name: "Weather Agent Eval"
tests:
  - name: basicLookup
    prompt: "What's the weather in Paris?"
    checks:
      - kind: responseContainsAny
        values: ["Paris"]
        caseSensitive: true
      - kind: toolUsedAtLeast
        name: "get_weather"
        times: 1
```

### Run with the CLI

```bash
smoleval-cli --dataset eval.yaml --endpoint http://localhost:8080/chat
```

### Run programmatically

See the [`smoleval`](crates/smoleval/README.md) crate README for the library API.

## Built-in checks

| Check                  | Description                                                    |
|------------------------|----------------------------------------------------------------|
| `responseContainsAll`  | Response contains all specified values                         |
| `responseContainsAny`  | Response contains at least one of the specified values         |
| `responseNotContains`  | Response does not contain any of the specified values          |
| `responseExactMatch`   | Response exactly matches the expected value                    |
| `toolUsedAtLeast`      | Tool was used at least `N` times (optional parameter matching) |
| `toolUsedAtMost`       | Tool was used at most `N` times (optional parameter matching)  |
| `toolUsedExactly`      | Tool was used exactly `N` times (optional parameter matching)  |
| `toolsUsedInOrder`     | Tools were used in a specific order, with gaps allowed         |

## Workspace crates

| Crate                  | Description                                  |
|------------------------|----------------------------------------------|
| `smoleval`             | Core evaluation engine                       |
| `smoleval-cli`         | CLI for running evals against HTTP endpoints |
| `smoleval-example`     | Example with a mock agent and custom checks  |
| `smoleval-cli-example` | Example HTTP agent servers                   |

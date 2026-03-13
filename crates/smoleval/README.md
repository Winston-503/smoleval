# smoleval

A minimal evaluation framework for AI agents in Rust.

Define test cases in YAML, implement the `Agent` trait for your system, and get structured pass/fail reports.

## Features

- YAML-driven evaluation datasets
- Pluggable `Agent` trait — back it with an HTTP API, local model, or a mock
- Registry-based check system with pluggable built-in validators
- Structured reports with per-test scores (0.0–1.0) and aggregate metrics
- Optional HTTP agent client (enabled by default via `http` feature)

## Quick start

```toml
[dependencies]
smoleval = "0.1"
tokio = { version = "1", features = ["full"] }
```

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
```

### Run an evaluation

```rust
use smoleval::{Agent, AgentResponse, CheckRegistry, EvalDataset, evaluate};

struct MyAgent;

impl Agent for MyAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
        // Call your AI agent here
        todo!()
    }
}

#[tokio::main]
async fn main() {
    let dataset = EvalDataset::from_file("eval.yaml".as_ref()).unwrap();
    let registry = CheckRegistry::default(); // built-in checks
    let agent = MyAgent;

    let report = evaluate(&agent, &dataset, &registry).await.unwrap();
    println!("Score: {:.0}% ({}/{})",
        report.mean_score() * 100.0,
        report.passed_count(),
        report.total_count(),
    );
}
```

## Built-in checks

| Check                  | Description                                                   |
|------------------------|---------------------------------------------------------------|
| `responseContainsAll`  | Response contains all specified values                        |
| `responseContainsAny`  | Response contains at least one of the specified values        |
| `responseNotContains`  | Response does not contain any of the specified values         |
| `responseExactMatch`   | Response exactly matches the expected value                   |
| `toolUsedAtLeast`      | Tool was used at least N times (optional parameter matching)  |
| `toolUsedAtMost`       | Tool was used at most N times (optional parameter matching)   |
| `toolUsedExactly`      | Tool was used exactly N times (optional parameter matching)   |
| `toolsUsedInOrder`     | Tools were used in a specific order (allows gaps)             |

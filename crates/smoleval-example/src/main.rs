use serde::Deserialize;
use smoleval::check::parse_config;
use smoleval::{
    Agent, AgentResponse, Check, CheckRegistry, CheckResult, EvalDataset, EvalOptions, ToolCall, evaluate_with_options,
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// A mock agent that echoes its input after delay and always calls "echo_tool"
// ---------------------------------------------------------------------------

struct MockAgent {
    delay: Duration,
}

impl MockAgent {
    pub fn new(delay: Duration) -> Self {
        MockAgent { delay }
    }
}

impl Agent for MockAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
        tokio::time::sleep(self.delay).await;

        Ok(AgentResponse::new(
            prompt,
            vec![ToolCall::new("echo_tool", serde_json::Value::Null)],
        ))
    }
}

// ---------------------------------------------------------------------------
// CustomCheck registration example
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomCheckConfig {
    custom_value: String,
}

pub struct CustomCheck {
    custom_value: String,
}

impl CustomCheck {
    fn from_config(config: &serde_json::Value) -> smoleval::Result<Box<dyn Check>> {
        let cfg: CustomCheckConfig = parse_config(config)?;
        Ok(Box::new(Self {
            custom_value: cfg.custom_value,
        }))
    }
}

impl Check for CustomCheck {
    fn run(&self, _response: &AgentResponse) -> CheckResult {
        let reason = format!("This check always passes; customValue: {}", self.custom_value);
        CheckResult::pass(reason)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mock_eval_dataset.yaml");

    let dataset = EvalDataset::from_file(&path)?;
    let mut registry = CheckRegistry::with_builtins();
    registry.register("customCheck", Box::new(CustomCheck::from_config));

    println!("=== {} ===\n", dataset.name());

    let agent = MockAgent::new(Duration::from_millis(500));
    let options = EvalOptions::new().with_print_on_result();

    let report = evaluate_with_options(&agent, &dataset, &registry, &options).await?;

    println!(
        "Results: {}/{} passed | Mean score: {:.2}",
        report.passed_count(),
        report.total_count(),
        report.mean_score()
    );

    Ok(())
}

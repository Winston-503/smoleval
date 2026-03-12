use smoleval::{Agent, AgentResponse, CheckRegistry, EvalDataset, EvalOptions, ToolCall, evaluate_with_options};

/// A mock agent that echoes its input after 1 sec delay and always calls an "echo_tool"
struct MockAgent;

impl Agent for MockAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        Ok(AgentResponse {
            text: prompt.to_string(),
            tool_calls: vec![ToolCall {
                name: "echo_tool".to_string(),
                arguments: serde_json::Value::Null,
            }],
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let yaml =
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mock_eval_dataset.yaml"))?;

    let dataset = EvalDataset::from_yaml(&yaml)?;
    let registry = CheckRegistry::with_builtins();

    println!("=== {} ===\n", dataset.name);

    let options = EvalOptions {
        on_result: Some(Box::new(|result| {
            println!("[{}] {} ({:.2})", result.label(), result.test_case.name, result.score);
            for (check, check_result) in result.test_case.checks.iter().zip(&result.check_results) {
                println!("  [{}] {}: {}", check_result.label(), check.kind, check_result.reason());
            }
            println!();
        })),
        ..Default::default()
    };

    let report = evaluate_with_options(&MockAgent, &dataset, &registry, &options).await?;

    println!(
        "Results: {}/{} passed | Mean score: {:.2}",
        report.passed_count(),
        report.total_count(),
        report.mean_score()
    );

    Ok(())
}

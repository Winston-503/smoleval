use smoleval::{Agent, AgentResponse, CheckRegistry, EvalDataset, ToolCall, evaluate};

/// A mock agent that echoes its input and always calls an "echo_tool".
struct MockAgent;

impl Agent for MockAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
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
    let yaml = r#"
name: "Mock Agent Eval"
description: "Demo evaluation with a mock echo agent"
tests:
  - name: echoGreeting
    prompt: "Hello, world!"
    checks:
      - type: containsAll
        values: ["Hello", "world"]
      - type: exactMatch
        expected: "Hello, world!"

  - name: checkToolUsage
    prompt: "Do something with tools"
    checks:
      - type: containsAny
        values: ["tools", "something"]
      - type: toolsUsed
        tools: ["echo_tool"]
        strictness: atLeast

  - name: noHallucination
    prompt: "Simple response"
    checks:
      - type: notContains
        values: ["hallucinated", "made up"]
"#;

    let dataset = EvalDataset::from_yaml(yaml)?;
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&MockAgent, &dataset, &registry).await?;

    // Print results
    println!("=== {} ===\n", report.dataset_name);
    for result in &report.results {
        let status = if result.score == 1.0 { "PASS" } else { "FAIL" };
        println!("[{status}] {} ({:.2})", result.test_case.name, result.score);
        for (def, cr) in result.test_case.checks.iter().zip(&result.check_results) {
            let icon = if cr.passed() { "OK" } else { "FAIL" };
            println!("  [{icon}] {}: {}", def.check_type, cr.reason());
        }
        println!();
    }
    println!(
        "Results: {}/{} passed | Mean score: {:.2}",
        report.passed_count(),
        report.total_count(),
        report.mean_score()
    );

    Ok(())
}

use smoleval::{Agent, AgentResponse, CheckRegistry, EvalDataset, ToolCall, evaluate};

/// Mock agent that echoes the prompt and optionally uses tools.
struct EchoAgent;

impl Agent for EchoAgent {
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

#[tokio::test]
async fn load_yaml_and_evaluate() {
    let dataset =
        EvalDataset::from_file(std::path::Path::new("tests/fixtures/sample_eval.yaml")).unwrap();

    assert_eq!(dataset.name, "Echo Agent Eval");
    assert_eq!(dataset.tests.len(), 4);

    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry).await.unwrap();

    assert_eq!(report.total_count(), 4);
    // echoBasic: containsAll + exactMatch on "Hello, world!" -> both pass
    assert_eq!(report.results[0].score, 1.0);
    // echoKeywords: containsAny ["Paris", "sunny"] on "The weather in Paris is sunny" -> pass
    assert_eq!(report.results[1].score, 1.0);
    // echoNoHallucination: notContains ["hallucinated", "made up"] -> pass
    assert_eq!(report.results[2].score, 1.0);
    // echoWithTools: toolsUsed ["echo_tool"] atLeast -> pass
    assert_eq!(report.results[3].score, 1.0);

    assert_eq!(report.mean_score(), 1.0);
    assert_eq!(report.passed_count(), 4);
    assert_eq!(report.failed_count(), 0);
}

#[tokio::test]
async fn parse_yaml_from_string() {
    let yaml = r#"
name: "Inline Test"
tests:
  - name: failCase
    prompt: "hello"
    checks:
      - type: containsAll
        values: ["missing_keyword"]
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry).await.unwrap();

    assert_eq!(report.total_count(), 1);
    assert_eq!(report.failed_count(), 1);
    assert_eq!(report.results[0].score, 0.0);
}

#[tokio::test]
async fn custom_check_registration() {
    use smoleval::check::{Check, CheckResult};

    /// A custom check that always passes.
    struct AlwaysPass;

    impl Check for AlwaysPass {
        fn run(&self, _response: &AgentResponse) -> CheckResult {
            CheckResult::pass("always passes")
        }
    }

    let yaml = r#"
name: "Custom Check Test"
tests:
  - name: customTest
    prompt: "anything"
    checks:
      - type: alwaysPass
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let mut registry = CheckRegistry::with_builtins();
    registry.register("alwaysPass", Box::new(|_config| Ok(Box::new(AlwaysPass))));

    let report = evaluate(&EchoAgent, &dataset, &registry).await.unwrap();
    assert_eq!(report.passed_count(), 1);
}

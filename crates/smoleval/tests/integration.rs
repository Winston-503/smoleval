use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smoleval::error::SmolError;
use smoleval::{Agent, AgentResponse, CheckRegistry, EvalDataset, EvalOptions, ToolCall, evaluate};

/// Mock agent that echoes the prompt and optionally uses tools.
struct EchoAgent;

impl Agent for EchoAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
        Ok(AgentResponse::new(
            prompt,
            vec![ToolCall::new("echo_tool", serde_json::Value::Null)],
        ))
    }
}

#[tokio::test]
async fn load_yaml_and_evaluate() {
    let dataset = EvalDataset::from_file(std::path::Path::new("tests/fixtures/sample_eval.yaml")).unwrap();

    assert_eq!(dataset.name(), "Echo Agent Eval");
    assert_eq!(dataset.tests().len(), 4);

    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.total_count(), 4);
    // echoBasic: responseContainsAll + responseExactMatch on "Hello, world!" -> both pass
    assert_eq!(report.results()[0].score(), 1.0);
    // echoKeywords: responseContainsAny ["Paris", "sunny"] on "The weather in Paris is sunny" -> pass
    assert_eq!(report.results()[1].score(), 1.0);
    // echoNoHallucination: responseNotContains ["hallucinated", "made up"] -> pass
    assert_eq!(report.results()[2].score(), 1.0);
    // echoWithTools: toolUsedAtLeast "echo_tool" (times: 1 default) -> pass
    assert_eq!(report.results()[3].score(), 1.0);

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
      - kind: responseContainsAll
        values: ["missing_keyword"]
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.total_count(), 1);
    assert_eq!(report.failed_count(), 1);
    assert_eq!(report.results()[0].score(), 0.0);
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
      - kind: alwaysPass
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let mut registry = CheckRegistry::with_builtins();
    registry.register("alwaysPass", Box::new(|_config| Ok(Box::new(AlwaysPass))));

    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();
    assert_eq!(report.passed_count(), 1);
}

// ---------------------------------------------------------------------------
// Agent that fails on specific prompts
// ---------------------------------------------------------------------------

struct SelectiveFailAgent;

impl Agent for SelectiveFailAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
        if prompt.contains("FAIL") {
            Err(SmolError::AgentError("selective failure".into()))
        } else {
            Ok(AgentResponse::new(prompt, vec![]))
        }
    }
}

// ---------------------------------------------------------------------------
// Concurrent evaluation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn evaluate_concurrent_produces_same_results() {
    let yaml = r#"
name: "Concurrent Test"
tests:
  - name: t1
    prompt: "hello world"
    checks:
      - kind: responseContainsAll
        values: ["hello", "world"]
  - name: t2
    prompt: "foo bar"
    checks:
      - kind: responseContainsAll
        values: ["foo", "bar"]
  - name: t3
    prompt: "alpha beta"
    checks:
      - kind: responseExactMatch
        value: "alpha beta"
  - name: t4
    prompt: "gamma"
    checks:
      - kind: responseContainsAny
        values: ["gamma", "delta"]
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();

    // Run sequentially
    let seq_opts = EvalOptions::new().with_concurrency(1);
    let seq_report = evaluate(&EchoAgent, &dataset, &registry, &seq_opts).await.unwrap();

    // Run concurrently
    let conc_opts = EvalOptions::new().with_concurrency(4);
    let conc_report = evaluate(&EchoAgent, &dataset, &registry, &conc_opts).await.unwrap();

    assert_eq!(seq_report.total_count(), conc_report.total_count());
    assert_eq!(seq_report.passed_count(), conc_report.passed_count());
    assert_eq!(seq_report.mean_score(), conc_report.mean_score());
    for (s, c) in seq_report.results().iter().zip(conc_report.results().iter()) {
        assert_eq!(s.score(), c.score());
        assert_eq!(s.test_case().name(), c.test_case().name());
    }
}

// ---------------------------------------------------------------------------
// Fail-fast behavior
// ---------------------------------------------------------------------------

#[tokio::test]
async fn evaluate_fail_fast_aborts_on_agent_error() {
    let yaml = r#"
name: "Fail Fast"
tests:
  - name: ok1
    prompt: "good"
    checks: []
  - name: fail1
    prompt: "FAIL here"
    checks: []
  - name: ok2
    prompt: "also good"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();

    let opts = EvalOptions::new().with_fail_fast(true);
    let result = evaluate(&SelectiveFailAgent, &dataset, &registry, &opts).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("selective failure"));
}

#[tokio::test]
async fn evaluate_no_fail_fast_captures_errors() {
    let yaml = r#"
name: "No Fail Fast"
tests:
  - name: ok1
    prompt: "good"
    checks: []
  - name: fail1
    prompt: "FAIL here"
    checks: []
  - name: ok2
    prompt: "also good"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let opts = EvalOptions::new().with_fail_fast(false);

    let report = evaluate(&SelectiveFailAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(report.total_count(), 3);
    // First and third pass (no checks, agent succeeds)
    assert_eq!(report.results()[0].score(), 1.0);
    assert!(!report.results()[0].outcome().is_error());
    // Second errors but is captured
    assert_eq!(report.results()[1].score(), 0.0);
    assert!(report.results()[1].outcome().is_error());
    assert!(
        report.results()[1]
            .outcome()
            .error()
            .unwrap()
            .contains("selective failure")
    );
    // Third still runs
    assert_eq!(report.results()[2].score(), 1.0);

    assert_eq!(report.passed_count(), 2);
    assert_eq!(report.failed_count(), 1);
    assert_eq!(report.errored_count(), 1);
}

// ---------------------------------------------------------------------------
// Concurrent evaluation captures agent errors without aborting
// ---------------------------------------------------------------------------

#[tokio::test]
async fn evaluate_concurrent_captures_agent_errors() {
    let yaml = r#"
name: "Concurrent Errors"
tests:
  - name: ok1
    prompt: "good"
    checks: []
  - name: fail1
    prompt: "FAIL"
    checks: []
  - name: ok2
    prompt: "fine"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let opts = EvalOptions::new().with_concurrency(3);

    let report = evaluate(&SelectiveFailAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(report.total_count(), 3);
    assert_eq!(report.errored_count(), 1);
    assert_eq!(report.passed_count(), 2);
}

// ---------------------------------------------------------------------------
// on_result callback
// ---------------------------------------------------------------------------

#[tokio::test]
async fn on_result_callback_invoked_for_each_test() {
    let yaml = r#"
name: "Callback Test"
tests:
  - name: t1
    prompt: "a"
    checks: []
  - name: t2
    prompt: "b"
    checks: []
  - name: t3
    prompt: "c"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let opts = EvalOptions::new().with_on_result(move |_result| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    evaluate(&EchoAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn on_result_callback_invoked_concurrently() {
    let yaml = r#"
name: "Concurrent Callback"
tests:
  - name: t1
    prompt: "a"
    checks: []
  - name: t2
    prompt: "b"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let opts = EvalOptions::new().with_concurrency(2).with_on_result(move |_result| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    evaluate(&EchoAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

// ---------------------------------------------------------------------------
// Partial scores (mixed checks on a single test case)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn partial_score_with_mixed_checks() {
    let yaml = r#"
name: "Partial Score"
tests:
  - name: mixedChecks
    prompt: "hello world"
    checks:
      - kind: responseContainsAll
        values: ["hello"]
      - kind: responseExactMatch
        value: "wrong value"
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.total_count(), 1);
    let result = &report.results()[0];
    // One check passes (1.0), one fails (0.0) → mean = 0.5
    assert!((result.score() - 0.5).abs() < f64::EPSILON);
    assert_eq!(result.check_results().len(), 2);
    assert!(result.check_results()[0].passed());
    assert!(!result.check_results()[1].passed());
}

// ---------------------------------------------------------------------------
// Unknown check kind in evaluation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn evaluate_unknown_check_fails_fast() {
    let yaml = r#"
name: "Unknown Check"
tests:
  - name: t1
    prompt: "hello"
    checks:
      - kind: nonExistentCheck
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let result = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default()).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("nonExistentCheck"));
}

#[tokio::test]
async fn evaluate_unknown_check_captured_without_fail_fast() {
    let yaml = r#"
name: "Unknown Check No Fail Fast"
tests:
  - name: badCheck
    prompt: "hello"
    checks:
      - kind: nonExistentCheck
  - name: goodCheck
    prompt: "world"
    checks:
      - kind: responseExactMatch
        value: "world"
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let opts = EvalOptions::new().with_fail_fast(false).with_skip_preflight(true);

    let report = evaluate(&EchoAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(report.total_count(), 2);
    assert_eq!(report.results()[0].score(), 0.0);
    assert!(report.results()[0].outcome().is_error());
    assert_eq!(report.results()[1].score(), 1.0);
}

// ---------------------------------------------------------------------------
// tool check variants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_used_exactly_pass() {
    let yaml = r#"
name: "Tool Used Exactly"
tests:
  - name: exactTool
    prompt: "use tools"
    checks:
      - kind: toolUsedExactly
        name: "echo_tool"
        times: 1
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.results()[0].score(), 1.0);
}

#[tokio::test]
async fn tool_used_exactly_fail_wrong_tool() {
    // EchoAgent always returns ["echo_tool"], so requiring "other_tool" exactly 1 time should fail
    let yaml = r#"
name: "Tool Used Exactly Fail"
tests:
  - name: wrongTool
    prompt: "use tools"
    checks:
      - kind: toolUsedExactly
        name: "other_tool"
        times: 1
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.results()[0].score(), 0.0);
}

#[tokio::test]
async fn tool_used_at_least_with_zero_times() {
    // EchoAgent returns ["echo_tool"], requiring atLeast 0 times for any tool should pass
    let yaml = r#"
name: "Tool AtLeast Zero"
tests:
  - name: atLeastZero
    prompt: "use tools"
    checks:
      - kind: toolUsedAtLeast
        name: "nonexistent_tool"
        times: 0
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.results()[0].score(), 1.0);
}

// ---------------------------------------------------------------------------
// Case sensitivity for containsAll
// ---------------------------------------------------------------------------

#[tokio::test]
async fn contains_all_case_sensitive() {
    let yaml = r#"
name: "Case Sensitive"
tests:
  - name: caseSensitivePass
    prompt: "Hello World"
    checks:
      - kind: responseContainsAll
        values: ["Hello", "World"]
        caseSensitive: true
  - name: caseSensitiveFail
    prompt: "Hello World"
    checks:
      - kind: responseContainsAll
        values: ["hello", "world"]
        caseSensitive: true
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let opts = EvalOptions::new().with_fail_fast(false);
    let report = evaluate(&EchoAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(report.results()[0].score(), 1.0);
    assert_eq!(report.results()[1].score(), 0.0);
}

// ---------------------------------------------------------------------------
// EvalReport metrics with errored tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn report_metrics_with_mixed_outcomes() {
    let yaml = r#"
name: "Mixed Outcomes"
tests:
  - name: pass1
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "hello"
  - name: fail1
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "wrong"
  - name: error1
    prompt: "FAIL please"
    checks: []
  - name: pass2
    prompt: "world"
    checks:
      - kind: responseContainsAny
        values: ["world"]
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let opts = EvalOptions::new().with_fail_fast(false);

    let report = evaluate(&SelectiveFailAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(report.total_count(), 4);
    assert_eq!(report.passed_count(), 2);
    assert_eq!(report.failed_count(), 2);
    assert_eq!(report.errored_count(), 1);
    // Mean: (1.0 + 0.0 + 0.0 + 1.0) / 4 = 0.5
    assert!((report.mean_score() - 0.5).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// responseNotContains check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn response_not_contains_fails_when_present() {
    let yaml = r#"
name: "NotContains Fail"
tests:
  - name: hasForbidden
    prompt: "this is a secret word"
    checks:
      - kind: responseNotContains
        values: ["secret"]
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.results()[0].score(), 0.0);
}

// ---------------------------------------------------------------------------
// responseContainsAny with no matches
// ---------------------------------------------------------------------------

#[tokio::test]
async fn contains_any_fails_when_none_match() {
    let yaml = r#"
name: "ContainsAny Fail"
tests:
  - name: noMatch
    prompt: "hello world"
    checks:
      - kind: responseContainsAny
        values: ["foo", "bar", "baz"]
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.results()[0].score(), 0.0);
}

// ---------------------------------------------------------------------------
// Multiple custom checks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multiple_custom_checks_in_one_test() {
    use smoleval::check::{Check, CheckResult};

    struct LengthCheck {
        min_len: usize,
    }

    impl Check for LengthCheck {
        fn run(&self, response: &AgentResponse) -> CheckResult {
            if response.text().len() >= self.min_len {
                CheckResult::pass("long enough")
            } else {
                CheckResult::fail("too short")
            }
        }
    }

    let yaml = r#"
name: "Multi Custom"
tests:
  - name: multiCheck
    prompt: "hello world"
    checks:
      - kind: minLength
        minLen: 5
      - kind: responseContainsAll
        values: ["hello"]
      - kind: minLength
        minLen: 100
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let mut registry = CheckRegistry::with_builtins();
    registry.register(
        "minLength",
        Box::new(|config| {
            let min_len = config["minLen"].as_u64().unwrap_or(0) as usize;
            Ok(Box::new(LengthCheck { min_len }))
        }),
    );

    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();
    let result = &report.results()[0];

    assert_eq!(result.check_results().len(), 3);
    assert!(result.check_results()[0].passed()); // len >= 5
    assert!(result.check_results()[1].passed()); // contains "hello"
    assert!(!result.check_results()[2].passed()); // len < 100
    // Score: 2/3
    assert!((result.score() - 2.0 / 3.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Evaluation preserves test case ordering and metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn evaluation_preserves_test_case_metadata() {
    let yaml = r#"
name: "Metadata Test"
description: "Verify metadata flows through"
tests:
  - name: first
    description: "first test"
    prompt: "one"
    checks: []
  - name: second
    description: "second test"
    prompt: "two"
    checks: []
  - name: third
    prompt: "three"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert_eq!(report.dataset_name(), "Metadata Test");
    assert_eq!(report.results()[0].test_case().name(), "first");
    assert_eq!(report.results()[0].test_case().description(), "first test");
    assert_eq!(report.results()[1].test_case().name(), "second");
    assert_eq!(report.results()[2].test_case().name(), "third");
    assert_eq!(report.results()[2].test_case().description(), "");

    // Verify response text matches prompt (echo agent)
    for result in report.results() {
        let response = result.outcome().response().unwrap();
        assert_eq!(response.text(), result.test_case().prompt());
    }
}

// ---------------------------------------------------------------------------
// Report duration is non-zero
// ---------------------------------------------------------------------------

#[tokio::test]
async fn report_has_nonzero_duration() {
    let yaml = r#"
name: "Duration"
tests:
  - name: t1
    prompt: "hello"
    checks: []
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();

    assert!(!report.duration().is_zero());
    assert!(!report.results()[0].agent_duration().is_zero());
}

// ---------------------------------------------------------------------------
// TestCaseLabel values
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_case_labels_match_scores() {
    use smoleval::TestCaseLabel;

    let yaml = r#"
name: "Labels"
tests:
  - name: passing
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "hello"
  - name: failing
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "nope"
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let opts = EvalOptions::new().with_fail_fast(false);
    let report = evaluate(&EchoAgent, &dataset, &registry, &opts).await.unwrap();

    assert_eq!(report.results()[0].label(), TestCaseLabel::Pass);
    assert_eq!(report.results()[1].label(), TestCaseLabel::Fail);
}

// ---------------------------------------------------------------------------
// Preflight check validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn preflight_catches_unknown_check_before_agent_runs() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    struct CountingAgent(Arc<AtomicUsize>);
    impl Agent for CountingAgent {
        async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(AgentResponse::new(prompt, vec![]))
        }
    }

    let yaml = r#"
name: "Preflight Test"
tests:
  - name: goodTest
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "hello"
  - name: badTest
    prompt: "world"
    checks:
      - kind: completelyBogusCheck
        foo: bar
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();
    let agent = CountingAgent(call_count_clone);

    let result = evaluate(&agent, &dataset, &registry, &EvalOptions::default()).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("preflight"));
    assert!(msg.contains("completelyBogusCheck"));
    assert!(msg.contains("badTest"));
    // Agent should NOT have been called
    assert_eq!(call_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn preflight_collects_multiple_errors() {
    let yaml = r#"
name: "Multi Error Preflight"
tests:
  - name: t1
    prompt: "hello"
    checks:
      - kind: fakeCheckA
  - name: t2
    prompt: "world"
    checks:
      - kind: responseExactMatch
        value: "world"
      - kind: fakeCheckB
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();

    let result = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default()).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("fakeCheckA"));
    assert!(msg.contains("fakeCheckB"));
    assert!(msg.contains("t1"));
    assert!(msg.contains("t2"));
}

#[tokio::test]
async fn preflight_invalid_config_caught() {
    let yaml = r#"
name: "Bad Config"
tests:
  - name: badConfig
    prompt: "hello"
    checks:
      - kind: responseContainsAll
        wrongField: 123
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();

    let result = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default()).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("badConfig"));
    assert!(msg.contains("responseContainsAll"));
}

#[tokio::test]
async fn preflight_valid_dataset_runs_normally() {
    let yaml = r#"
name: "Valid Dataset"
tests:
  - name: t1
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "hello"
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();

    let report = evaluate(&EchoAgent, &dataset, &registry, &EvalOptions::default())
        .await
        .unwrap();
    assert_eq!(report.passed_count(), 1);
}

#[tokio::test]
async fn validate_dataset_standalone() {
    let yaml = r#"
name: "Standalone Validation"
tests:
  - name: t1
    prompt: "hello"
    checks:
      - kind: responseExactMatch
        value: "hello"
  - name: t2
    prompt: "world"
    checks:
      - kind: bogusCheck
"#;

    let dataset = EvalDataset::from_yaml(yaml).unwrap();
    let registry = CheckRegistry::with_builtins();

    let result = registry.validate_dataset(&dataset);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("bogusCheck"));
}

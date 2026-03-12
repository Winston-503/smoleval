use std::fmt;
use std::time::{Duration, Instant};

use futures::stream::{self, StreamExt};

use crate::Result;
use crate::agent::{Agent, AgentOutcome, AgentResponse};
use crate::check::{CheckRegistry, CheckResult};
use crate::dataset::{EvalDataset, TestCase};

// ---------------------------------------------------------------------------
// TestCaseLabel — display label for test case results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCaseLabel {
    Pass,
    Fail,
}

impl fmt::Display for TestCaseLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestCaseLabel::Pass => write!(f, "PASS"),
            TestCaseLabel::Fail => write!(f, "FAIL"),
        }
    }
}

/// Options for controlling evaluation behavior.
#[derive(Debug, Clone)]
pub struct EvalOptions {
    concurrency: usize,
    fail_fast: bool,
}

impl EvalOptions {
    /// Create a new `EvalOptions` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of test cases to run concurrently.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// If true, abort the evaluation on the first error (only effective with concurrency=1).
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Maximum number of test cases to run concurrently.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Whether to abort on the first error.
    pub fn fail_fast(&self) -> bool {
        self.fail_fast
    }
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            concurrency: 1,
            fail_fast: false,
        }
    }
}

/// Result for a single test case.
#[derive(Debug, Clone)]
pub struct TestCaseResult {
    test_case: TestCase,
    outcome: AgentOutcome,
    check_results: Vec<CheckResult>,
    score: f64,
    duration: Duration,
}

impl TestCaseResult {
    /// The test case that was evaluated.
    pub fn test_case(&self) -> &TestCase {
        &self.test_case
    }

    /// The agent's outcome — either a successful response or an error.
    pub fn outcome(&self) -> &AgentOutcome {
        &self.outcome
    }

    /// Results for each check (same order as `test_case.checks`).
    pub fn check_results(&self) -> &[CheckResult] {
        &self.check_results
    }

    /// Overall score (mean of check scores).
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Wall-clock duration for this test case.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn label(&self) -> TestCaseLabel {
        if self.score == 1.0 {
            TestCaseLabel::Pass
        } else {
            TestCaseLabel::Fail
        }
    }
}

/// Report for an entire evaluation run.
#[derive(Debug, Clone)]
pub struct EvalReport {
    dataset_name: String,
    results: Vec<TestCaseResult>,
    duration: Duration,
}

impl EvalReport {
    /// Name of the dataset.
    pub fn dataset_name(&self) -> &str {
        &self.dataset_name
    }

    /// Per-test-case results.
    pub fn results(&self) -> &[TestCaseResult] {
        &self.results
    }

    /// Total wall-clock duration of the evaluation run.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Mean score across all test cases.
    pub fn mean_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.score).sum();
        sum / self.results.len() as f64
    }

    /// Number of test cases that scored 1.0.
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.score == 1.0).count()
    }

    /// Number of test cases that scored below 1.0.
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| r.score < 1.0).count()
    }

    /// Number of test cases that errored (agent failure, etc.).
    pub fn errored_count(&self) -> usize {
        self.results.iter().filter(|r| r.outcome.is_error()).count()
    }

    /// Total number of test cases.
    pub fn total_count(&self) -> usize {
        self.results.len()
    }
}

/// Run a full evaluation (backward-compatible, sequential, fail-fast).
///
/// For each test case: sends the prompt to the agent, runs all checks from
/// the registry, and collects results into an [`EvalReport`].
pub async fn evaluate<A: Agent>(agent: &A, dataset: &EvalDataset, registry: &CheckRegistry) -> Result<EvalReport> {
    let options = EvalOptions {
        concurrency: 1,
        fail_fast: true,
    };
    evaluate_with_options(agent, dataset, registry, &options).await
}

/// Run a full evaluation with configurable options.
///
/// When `fail_fast` is false, agent errors and check-creation errors are
/// captured per-test-case (score 0.0) instead of aborting the run.
/// When `concurrency` > 1, test cases run in parallel (fail_fast is ignored).
pub async fn evaluate_with_options<A: Agent>(
    agent: &A,
    dataset: &EvalDataset,
    registry: &CheckRegistry,
    options: &EvalOptions,
) -> Result<EvalReport> {
    let start = Instant::now();
    let concurrency = options.concurrency.max(1);

    let results = if concurrency <= 1 {
        evaluate_sequential(agent, dataset, registry, options.fail_fast).await?
    } else {
        evaluate_concurrent(agent, dataset, registry, concurrency).await
    };

    Ok(EvalReport {
        dataset_name: dataset.name().to_owned(),
        results,
        duration: start.elapsed(),
    })
}

async fn evaluate_sequential<A: Agent>(
    agent: &A,
    dataset: &EvalDataset,
    registry: &CheckRegistry,
    fail_fast: bool,
) -> Result<Vec<TestCaseResult>> {
    let mut results = Vec::with_capacity(dataset.tests().len());

    for test_case in dataset.tests() {
        let start = Instant::now();
        match run_single(agent, test_case, registry).await {
            Ok((response, check_results, score)) => {
                results.push(TestCaseResult {
                    test_case: test_case.clone(),
                    outcome: AgentOutcome::Response(response),
                    check_results,
                    score,
                    duration: start.elapsed(),
                });
            }
            Err(e) => {
                if fail_fast {
                    return Err(e);
                }
                results.push(TestCaseResult {
                    test_case: test_case.clone(),
                    outcome: AgentOutcome::Error(e.to_string()),
                    check_results: vec![],
                    score: 0.0,
                    duration: start.elapsed(),
                });
            }
        }
    }

    Ok(results)
}

async fn evaluate_concurrent<A: Agent>(
    agent: &A,
    dataset: &EvalDataset,
    registry: &CheckRegistry,
    concurrency: usize,
) -> Vec<TestCaseResult> {
    stream::iter(dataset.tests())
        .map(|test_case| async move {
            let start = Instant::now();
            let result = run_single(agent, test_case, registry).await;
            let duration = start.elapsed();
            match result {
                Ok((response, check_results, score)) => TestCaseResult {
                    test_case: test_case.clone(),
                    outcome: AgentOutcome::Response(response),
                    check_results,
                    score,
                    duration,
                },
                Err(e) => TestCaseResult {
                    test_case: test_case.clone(),
                    outcome: AgentOutcome::Error(e.to_string()),
                    check_results: vec![],
                    score: 0.0,
                    duration,
                },
            }
        })
        .buffered(concurrency)
        .collect()
        .await
}

async fn run_single<A: Agent>(
    agent: &A,
    test_case: &TestCase,
    registry: &CheckRegistry,
) -> Result<(AgentResponse, Vec<CheckResult>, f64)> {
    let response = agent.run(test_case.prompt()).await?;
    let check_results = run_checks(test_case, &response, registry)?;
    let score = mean_score(&check_results);
    Ok((response, check_results, score))
}

fn run_checks(test_case: &TestCase, response: &AgentResponse, registry: &CheckRegistry) -> Result<Vec<CheckResult>> {
    test_case
        .checks()
        .iter()
        .map(|def| {
            let check = registry.create(def)?;
            Ok(check.run(response))
        })
        .collect()
}

fn mean_score(results: &[CheckResult]) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    let sum: f64 = results.iter().map(|r| r.score()).sum();
    sum / results.len() as f64
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::check::CheckSpec;
    use crate::error::SmolError;

    struct EchoAgent;

    impl Agent for EchoAgent {
        async fn run(&self, prompt: &str) -> crate::Result<AgentResponse> {
            Ok(AgentResponse::new(prompt, vec![]))
        }
    }

    struct FailAgent;

    impl Agent for FailAgent {
        async fn run(&self, _prompt: &str) -> crate::Result<AgentResponse> {
            Err(SmolError::AgentError("agent crashed".into()))
        }
    }

    fn make_dataset(name: &str, tests: Vec<TestCase>) -> EvalDataset {
        EvalDataset::new(name, "", tests)
    }

    fn make_test_case(name: &str, prompt: &str, checks: Vec<CheckSpec>) -> TestCase {
        TestCase::new(name, prompt, checks)
    }

    // -- mean_score tests --

    #[test]
    fn mean_score_empty() {
        assert_eq!(mean_score(&[]), 1.0);
    }

    #[test]
    fn mean_score_single_pass() {
        let results = vec![CheckResult::pass("ok")];
        assert_eq!(mean_score(&results), 1.0);
    }

    #[test]
    fn mean_score_single_fail() {
        let results = vec![CheckResult::fail("nope")];
        assert_eq!(mean_score(&results), 0.0);
    }

    #[test]
    fn mean_score_mixed() {
        let results = vec![CheckResult::pass("ok"), CheckResult::fail("nope")];
        assert_eq!(mean_score(&results), 0.5);
    }

    #[test]
    fn mean_score_partial() {
        let results = vec![CheckResult::build(0.5, "half").unwrap(), CheckResult::pass("ok")];
        assert_eq!(mean_score(&results), 0.75);
    }

    // -- EvalReport tests --

    #[test]
    fn eval_report_empty() {
        let report = EvalReport {
            dataset_name: "empty".into(),
            results: vec![],
            duration: Duration::ZERO,
        };
        assert_eq!(report.mean_score(), 0.0);
        assert_eq!(report.passed_count(), 0);
        assert_eq!(report.failed_count(), 0);
        assert_eq!(report.total_count(), 0);
    }

    #[test]
    fn eval_report_all_pass() {
        let report = EvalReport {
            dataset_name: "test".into(),
            results: vec![
                TestCaseResult {
                    test_case: make_test_case("a", "p", vec![]),
                    outcome: AgentOutcome::Response(AgentResponse::new("p", vec![])),
                    check_results: vec![],
                    score: 1.0,
                    duration: Duration::ZERO,
                },
                TestCaseResult {
                    test_case: make_test_case("b", "q", vec![]),
                    outcome: AgentOutcome::Response(AgentResponse::new("q", vec![])),
                    check_results: vec![],
                    score: 1.0,
                    duration: Duration::ZERO,
                },
            ],
            duration: Duration::ZERO,
        };
        assert_eq!(report.mean_score(), 1.0);
        assert_eq!(report.passed_count(), 2);
        assert_eq!(report.failed_count(), 0);
        assert_eq!(report.total_count(), 2);
    }

    #[test]
    fn eval_report_mixed() {
        let report = EvalReport {
            dataset_name: "test".into(),
            results: vec![
                TestCaseResult {
                    test_case: make_test_case("a", "p", vec![]),
                    outcome: AgentOutcome::Response(AgentResponse::new("p", vec![])),
                    check_results: vec![],
                    score: 1.0,
                    duration: Duration::ZERO,
                },
                TestCaseResult {
                    test_case: make_test_case("b", "q", vec![]),
                    outcome: AgentOutcome::Response(AgentResponse::new("q", vec![])),
                    check_results: vec![],
                    score: 0.0,
                    duration: Duration::ZERO,
                },
                TestCaseResult {
                    test_case: make_test_case("c", "r", vec![]),
                    outcome: AgentOutcome::Response(AgentResponse::new("r", vec![])),
                    check_results: vec![],
                    score: 0.5,
                    duration: Duration::ZERO,
                },
            ],
            duration: Duration::ZERO,
        };
        assert_eq!(report.passed_count(), 1);
        assert_eq!(report.failed_count(), 2);
        assert_eq!(report.total_count(), 3);
        assert!((report.mean_score() - 0.5).abs() < f64::EPSILON);
    }

    // -- run_checks tests --

    #[test]
    fn run_checks_no_checks() {
        let tc = make_test_case("t", "hello", vec![]);
        let response = AgentResponse::new("hello", vec![]);
        let registry = CheckRegistry::with_builtins();
        let results = run_checks(&tc, &response, &registry).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn run_checks_single_passing() {
        let tc = make_test_case(
            "t",
            "hello",
            vec![CheckSpec::new("responseExactMatch", serde_json::json!({"value": "hello"}))],
        );
        let response = AgentResponse::new("hello", vec![]);
        let registry = CheckRegistry::with_builtins();
        let results = run_checks(&tc, &response, &registry).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].passed());
    }

    #[test]
    fn run_checks_multiple_mixed() {
        let tc = make_test_case(
            "t",
            "hello world",
            vec![
                CheckSpec::new("responseContainsAll", serde_json::json!({"values": ["hello"]})),
                CheckSpec::new("responseExactMatch", serde_json::json!({"value": "wrong"})),
            ],
        );
        let response = AgentResponse::new("hello world", vec![]);
        let registry = CheckRegistry::with_builtins();
        let results = run_checks(&tc, &response, &registry).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].passed());
        assert!(!results[1].passed());
    }

    #[test]
    fn run_checks_unknown_type_errors() {
        let tc = make_test_case(
            "t",
            "hello",
            vec![CheckSpec::new("doesNotExist", serde_json::json!({}))],
        );
        let response = AgentResponse::new("hello", vec![]);
        let registry = CheckRegistry::with_builtins();
        assert!(run_checks(&tc, &response, &registry).is_err());
    }

    // -- evaluate tests --

    #[tokio::test]
    async fn evaluate_empty_dataset() {
        let dataset = make_dataset("empty", vec![]);
        let registry = CheckRegistry::with_builtins();
        let report = evaluate(&EchoAgent, &dataset, &registry).await.unwrap();
        assert_eq!(report.total_count(), 0);
        assert_eq!(report.mean_score(), 0.0);
    }

    #[tokio::test]
    async fn evaluate_no_checks_scores_one() {
        let dataset = make_dataset("test", vec![make_test_case("t1", "hello", vec![])]);
        let registry = CheckRegistry::with_builtins();
        let report = evaluate(&EchoAgent, &dataset, &registry).await.unwrap();
        assert_eq!(report.results()[0].score(), 1.0);
    }

    #[tokio::test]
    async fn evaluate_agent_error_propagates() {
        let dataset = make_dataset("test", vec![make_test_case("t1", "hello", vec![])]);
        let registry = CheckRegistry::with_builtins();
        let result = evaluate(&FailAgent, &dataset, &registry).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn evaluate_preserves_response() {
        let dataset = make_dataset("test", vec![make_test_case("t1", "my prompt", vec![])]);
        let registry = CheckRegistry::with_builtins();
        let report = evaluate(&EchoAgent, &dataset, &registry).await.unwrap();
        assert_eq!(report.results()[0].outcome().response().unwrap().text(), "my prompt");
    }
}

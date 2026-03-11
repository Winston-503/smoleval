use std::time::{Duration, Instant};

use futures::stream::{self, StreamExt};

use crate::Result;
use crate::agent::{Agent, AgentResponse};
use crate::check::{CheckRegistry, CheckResult};
use crate::dataset::{EvalDataset, TestCase};

/// Options for controlling evaluation behavior.
#[derive(Debug, Clone)]
pub struct EvalOptions {
    /// Maximum number of test cases to run concurrently (default: 1).
    pub concurrency: usize,
    /// If true, abort the evaluation on the first error (default: false).
    /// Only effective when `concurrency` is 1.
    pub fail_fast: bool,
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
    /// The test case that was evaluated.
    pub test_case: TestCase,
    /// The agent's response.
    pub response: AgentResponse,
    /// Results for each check (same order as `test_case.checks`).
    pub check_results: Vec<CheckResult>,
    /// Overall score (mean of check scores).
    pub score: f64,
    /// Wall-clock duration for this test case.
    pub duration: Duration,
    /// Error message if the agent call or check creation failed.
    pub error: Option<String>,
}

/// Report for an entire evaluation run.
#[derive(Debug, Clone)]
pub struct EvalReport {
    /// Name of the dataset.
    pub dataset_name: String,
    /// Per-test-case results.
    pub results: Vec<TestCaseResult>,
    /// Total wall-clock duration of the evaluation run.
    pub duration: Duration,
}

impl EvalReport {
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
        self.results.iter().filter(|r| r.error.is_some()).count()
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
pub async fn evaluate<A: Agent>(
    agent: &A,
    dataset: &EvalDataset,
    registry: &CheckRegistry,
) -> Result<EvalReport> {
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
        dataset_name: dataset.name.clone(),
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
    let mut results = Vec::with_capacity(dataset.tests.len());

    for test_case in &dataset.tests {
        let start = Instant::now();
        match run_single(agent, test_case, registry).await {
            Ok((response, check_results, score)) => {
                results.push(TestCaseResult {
                    test_case: test_case.clone(),
                    response,
                    check_results,
                    score,
                    duration: start.elapsed(),
                    error: None,
                });
            }
            Err(e) => {
                if fail_fast {
                    return Err(e);
                }
                results.push(TestCaseResult {
                    test_case: test_case.clone(),
                    response: AgentResponse {
                        text: String::new(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 0.0,
                    duration: start.elapsed(),
                    error: Some(e.to_string()),
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
    stream::iter(&dataset.tests)
        .map(|test_case| async move {
            let start = Instant::now();
            let result = run_single(agent, test_case, registry).await;
            let duration = start.elapsed();
            match result {
                Ok((response, check_results, score)) => TestCaseResult {
                    test_case: test_case.clone(),
                    response,
                    check_results,
                    score,
                    duration,
                    error: None,
                },
                Err(e) => TestCaseResult {
                    test_case: test_case.clone(),
                    response: AgentResponse {
                        text: String::new(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 0.0,
                    duration,
                    error: Some(e.to_string()),
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
    let response = agent.run(&test_case.prompt).await?;
    let check_results = run_checks(test_case, &response, registry)?;
    let score = mean_score(&check_results);
    Ok((response, check_results, score))
}

fn run_checks(
    test_case: &TestCase,
    response: &AgentResponse,
    registry: &CheckRegistry,
) -> Result<Vec<CheckResult>> {
    test_case
        .checks
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

use crate::Result;
use crate::agent::{Agent, AgentResponse};
use crate::check::{CheckRegistry, CheckResult};
use crate::dataset::{EvalDataset, TestCase};

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
}

/// Report for an entire evaluation run.
#[derive(Debug, Clone)]
pub struct EvalReport {
    /// Name of the dataset.
    pub dataset_name: String,
    /// Per-test-case results.
    pub results: Vec<TestCaseResult>,
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

    /// Total number of test cases.
    pub fn total_count(&self) -> usize {
        self.results.len()
    }
}

/// Run a full evaluation.
///
/// For each test case: sends the prompt to the agent, runs all checks from
/// the registry, and collects results into an [`EvalReport`].
pub async fn evaluate<A: Agent>(
    agent: &A,
    dataset: &EvalDataset,
    registry: &CheckRegistry,
) -> Result<EvalReport> {
    let mut results = Vec::with_capacity(dataset.tests.len());

    for test_case in &dataset.tests {
        let response = agent.run(&test_case.prompt).await?;
        let check_results = run_checks(test_case, &response, registry)?;
        let score = mean_score(&check_results);

        results.push(TestCaseResult {
            test_case: test_case.clone(),
            response,
            check_results,
            score,
        });
    }

    Ok(EvalReport {
        dataset_name: dataset.name.clone(),
        results,
    })
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

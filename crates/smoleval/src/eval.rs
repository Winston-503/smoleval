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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SmolError;

    struct EchoAgent;

    impl Agent for EchoAgent {
        async fn run(&self, prompt: &str) -> crate::Result<AgentResponse> {
            Ok(AgentResponse {
                text: prompt.to_string(),
                tool_calls: vec![],
            })
        }
    }

    struct FailAgent;

    impl Agent for FailAgent {
        async fn run(&self, _prompt: &str) -> crate::Result<AgentResponse> {
            Err(SmolError::AgentError("agent crashed".into()))
        }
    }

    fn make_dataset(name: &str, tests: Vec<TestCase>) -> EvalDataset {
        EvalDataset {
            name: name.into(),
            description: String::new(),
            tests,
        }
    }

    fn make_test_case(name: &str, prompt: &str, checks: Vec<crate::check::CheckDef>) -> TestCase {
        TestCase {
            name: name.into(),
            description: String::new(),
            prompt: prompt.into(),
            checks,
        }
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
        let results = vec![
            CheckResult::build(0.5, "half").unwrap(),
            CheckResult::pass("ok"),
        ];
        assert_eq!(mean_score(&results), 0.75);
    }

    // -- EvalReport tests --

    #[test]
    fn eval_report_empty() {
        let report = EvalReport {
            dataset_name: "empty".into(),
            results: vec![],
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
                    response: AgentResponse {
                        text: "p".into(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 1.0,
                },
                TestCaseResult {
                    test_case: make_test_case("b", "q", vec![]),
                    response: AgentResponse {
                        text: "q".into(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 1.0,
                },
            ],
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
                    response: AgentResponse {
                        text: "p".into(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 1.0,
                },
                TestCaseResult {
                    test_case: make_test_case("b", "q", vec![]),
                    response: AgentResponse {
                        text: "q".into(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 0.0,
                },
                TestCaseResult {
                    test_case: make_test_case("c", "r", vec![]),
                    response: AgentResponse {
                        text: "r".into(),
                        tool_calls: vec![],
                    },
                    check_results: vec![],
                    score: 0.5,
                },
            ],
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
        let response = AgentResponse {
            text: "hello".into(),
            tool_calls: vec![],
        };
        let registry = CheckRegistry::with_builtins();
        let results = run_checks(&tc, &response, &registry).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn run_checks_single_passing() {
        let tc = make_test_case(
            "t",
            "hello",
            vec![crate::check::CheckDef {
                check_type: "exactMatch".into(),
                config: serde_json::json!({"expected": "hello"}),
            }],
        );
        let response = AgentResponse {
            text: "hello".into(),
            tool_calls: vec![],
        };
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
                crate::check::CheckDef {
                    check_type: "containsAll".into(),
                    config: serde_json::json!({"values": ["hello"]}),
                },
                crate::check::CheckDef {
                    check_type: "exactMatch".into(),
                    config: serde_json::json!({"expected": "wrong"}),
                },
            ],
        );
        let response = AgentResponse {
            text: "hello world".into(),
            tool_calls: vec![],
        };
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
            vec![crate::check::CheckDef {
                check_type: "doesNotExist".into(),
                config: serde_json::json!({}),
            }],
        );
        let response = AgentResponse {
            text: "hello".into(),
            tool_calls: vec![],
        };
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
        assert_eq!(report.results[0].score, 1.0);
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
        assert_eq!(report.results[0].response.text, "my prompt");
    }
}

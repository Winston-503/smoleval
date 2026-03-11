use std::fmt;

/// A single evaluation sample: input for the agent plus expected output.
#[derive(Debug, Clone)]
pub struct Sample {
    pub input: String,
    pub expected: String,
}

/// Result produced by scoring one sample.
#[derive(Debug, Clone)]
pub struct SampleResult {
    pub sample: Sample,
    pub actual: String,
    pub score: f64,
}

/// Summary of an entire evaluation run.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub results: Vec<SampleResult>,
}

impl EvalReport {
    pub fn mean_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.score).sum();
        sum / self.results.len() as f64
    }
}

impl fmt::Display for EvalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, r) in self.results.iter().enumerate() {
            writeln!(
                f,
                "[{i}] score={:.2}  expected={:?}  actual={:?}",
                r.score, r.sample.expected, r.actual,
            )?;
        }
        write!(f, "mean score: {:.2}", self.mean_score())
    }
}

/// Trait that an AI agent must implement so it can be evaluated.
pub trait Agent {
    fn run(&self, input: &str) -> String;
}

/// Trait for scoring agent output against expected output.
pub trait Scorer {
    fn score(&self, expected: &str, actual: &str) -> f64;
}

/// Simple exact-match scorer: 1.0 if equal, 0.0 otherwise.
#[derive(Debug, Clone, Copy)]
pub struct ExactMatch;

impl Scorer for ExactMatch {
    fn score(&self, expected: &str, actual: &str) -> f64 {
        if expected == actual { 1.0 } else { 0.0 }
    }
}

/// Run an evaluation: feed each sample through the agent, score it, and return a report.
pub fn evaluate(agent: &dyn Agent, scorer: &dyn Scorer, samples: &[Sample]) -> EvalReport {
    let results = samples
        .iter()
        .map(|sample| {
            let actual = agent.run(&sample.input);
            let score = scorer.score(&sample.expected, &actual);
            SampleResult {
                sample: sample.clone(),
                actual,
                score,
            }
        })
        .collect();
    EvalReport { results }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoAgent;

    impl Agent for EchoAgent {
        fn run(&self, input: &str) -> String {
            input.to_string()
        }
    }

    #[test]
    fn exact_match_all_correct() {
        let samples = vec![
            Sample {
                input: "hello".into(),
                expected: "hello".into(),
            },
            Sample {
                input: "world".into(),
                expected: "world".into(),
            },
        ];
        let report = evaluate(&EchoAgent, &ExactMatch, &samples);
        assert_eq!(report.mean_score(), 1.0);
    }

    #[test]
    fn exact_match_partial() {
        let samples = vec![
            Sample {
                input: "hello".into(),
                expected: "hello".into(),
            },
            Sample {
                input: "world".into(),
                expected: "wrong".into(),
            },
        ];
        let report = evaluate(&EchoAgent, &ExactMatch, &samples);
        assert_eq!(report.mean_score(), 0.5);
    }
}

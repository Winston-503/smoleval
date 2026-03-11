use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::check::CheckDef;

/// A full evaluation dataset loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalDataset {
    /// Human-readable name for this eval suite.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: String,
    /// The test cases.
    pub tests: Vec<TestCase>,
}

/// A single test case within an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TestCase {
    /// Unique name for this test case.
    pub name: String,
    /// What this test case is checking.
    #[serde(default)]
    pub description: String,
    /// The prompt sent to the agent.
    pub prompt: String,
    /// Checks to run against the agent's response.
    pub checks: Vec<CheckDef>,
}

impl EvalDataset {
    /// Load a dataset from a YAML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_yaml(&contents)
    }

    /// Parse a dataset from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let dataset: EvalDataset = serde_yaml::from_str(yaml)?;
        Ok(dataset)
    }
}

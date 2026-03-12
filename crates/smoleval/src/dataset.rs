use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::check::CheckSpec;
use crate::error::SmolError;

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
    /// Optional description of the test case.
    #[serde(default)]
    pub description: String,
    /// The prompt sent to the agent.
    pub prompt: String,
    /// Checks to run against the agent's response.
    pub checks: Vec<CheckSpec>,
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
        let mut seen = HashSet::new();
        for test in &dataset.tests {
            if !seen.insert(&test.name) {
                return Err(SmolError::DuplicateTestName(test.name.clone()));
            }
        }
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_dataset() {
        let yaml = r#"
name: minimal
tests:
  - name: t1
    prompt: hello
    checks: []
"#;
        let ds = EvalDataset::from_yaml(yaml).unwrap();
        assert_eq!(ds.name, "minimal");
        assert_eq!(ds.description, "");
        assert_eq!(ds.tests.len(), 1);
        assert_eq!(ds.tests[0].description, "");
    }

    #[test]
    fn parse_full_dataset() {
        let yaml = r#"
name: full
description: a full dataset
tests:
  - name: t1
    description: first test
    prompt: say hi
    checks:
      - kind: responseExactMatch
        value: hi
  - name: t2
    prompt: say bye
    checks:
      - kind: responseContainsAll
        values: ["bye"]
"#;
        let ds = EvalDataset::from_yaml(yaml).unwrap();
        assert_eq!(ds.name, "full");
        assert_eq!(ds.description, "a full dataset");
        assert_eq!(ds.tests.len(), 2);
        assert_eq!(ds.tests[0].name, "t1");
        assert_eq!(ds.tests[0].description, "first test");
        assert_eq!(ds.tests[0].checks.len(), 1);
        assert_eq!(ds.tests[0].checks[0].kind, "responseExactMatch");
        assert_eq!(ds.tests[1].checks[0].kind, "responseContainsAll");
    }

    #[test]
    fn parse_empty_tests_list() {
        let yaml = r#"
name: empty
tests: []
"#;
        let ds = EvalDataset::from_yaml(yaml).unwrap();
        assert!(ds.tests.is_empty());
    }

    #[test]
    fn parse_invalid_yaml() {
        let yaml = "not: [valid: yaml: {{";
        assert!(EvalDataset::from_yaml(yaml).is_err());
    }

    #[test]
    fn parse_missing_required_field_name() {
        let yaml = r#"
tests:
  - name: t1
    prompt: hello
    checks: []
"#;
        assert!(EvalDataset::from_yaml(yaml).is_err());
    }

    #[test]
    fn parse_missing_required_field_prompt() {
        let yaml = r#"
name: test
tests:
  - name: t1
    checks: []
"#;
        assert!(EvalDataset::from_yaml(yaml).is_err());
    }

    #[test]
    fn parse_missing_required_field_tests() {
        let yaml = "name: orphan\n";
        assert!(EvalDataset::from_yaml(yaml).is_err());
    }

    #[test]
    fn from_file_nonexistent() {
        let result = EvalDataset::from_file(std::path::Path::new("/nonexistent/file.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn serialize_roundtrip() {
        let yaml = r#"
name: roundtrip
tests:
  - name: t1
    prompt: hello
    checks:
      - kind: responseExactMatch
        value: hello
"#;
        let ds = EvalDataset::from_yaml(yaml).unwrap();
        let serialized = serde_yaml::to_string(&ds).unwrap();
        let ds2 = EvalDataset::from_yaml(&serialized).unwrap();
        assert_eq!(ds.name, ds2.name);
        assert_eq!(ds.tests.len(), ds2.tests.len());
        assert_eq!(ds.tests[0].prompt, ds2.tests[0].prompt);
    }

    #[test]
    fn parse_duplicate_test_names() {
        let yaml = r#"
name: dupes
tests:
  - name: t1
    prompt: hello
    checks: []
  - name: t1
    prompt: world
    checks: []
"#;
        let err = EvalDataset::from_yaml(yaml).unwrap_err();
        assert!(err.to_string().contains("duplicate test case name: t1"));
    }

    #[test]
    fn check_spec_preserves_config() {
        let yaml = r#"
name: cfg
tests:
  - name: t1
    prompt: test
    checks:
      - kind: responseContainsAll
        values: ["a", "b"]
        caseSensitive: true
"#;
        let ds = EvalDataset::from_yaml(yaml).unwrap();
        let check = &ds.tests[0].checks[0];
        assert_eq!(check.kind, "responseContainsAll");
        let values = check.config["values"].as_array().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(check.config["caseSensitive"], true);
    }
}

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::agent::AgentResponse;
use crate::error::SmolError;

// ---------------------------------------------------------------------------
// CheckLabel — display label for check results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckLabel {
    Ok,
    Fail,
}

impl fmt::Display for CheckLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckLabel::Ok => write!(f, "OK"),
            CheckLabel::Fail => write!(f, "FAIL"),
        }
    }
}

// ---------------------------------------------------------------------------
// CheckSpec — raw spec from YAML
// ---------------------------------------------------------------------------

/// A check specification as deserialized from YAML.
///
/// The `kind` field selects the [`Check`] implementation from the [`CheckRegistry`];
/// the remaining fields are passed as config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckSpec {
    kind: String,
    #[serde(flatten)]
    config: serde_json::Value,
}

impl CheckSpec {
    /// Create a new check specification.
    pub fn new(kind: impl Into<String>, config: serde_json::Value) -> Self {
        Self {
            kind: kind.into(),
            config,
        }
    }

    /// The check kind name.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The config passed to the check factory.
    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// CheckResult — outcome of running a check
// ---------------------------------------------------------------------------

/// The result of running a single check against an agent response.
///
/// Construct via [`CheckResult::pass`], [`CheckResult::fail`], or [`CheckResult::build`]
/// to guarantee the score is in `[0.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct CheckResult {
    score: f64,
    reason: String,
}

impl CheckResult {
    /// Create a passing result (score = 1.0).
    pub fn pass(reason: impl Into<String>) -> Self {
        Self {
            score: 1.0,
            reason: reason.into(),
        }
    }

    /// Create a failing result (score = 0.0).
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            score: 0.0,
            reason: reason.into(),
        }
    }

    /// Create a result with a custom score.
    ///
    /// Returns `Err` if the score is not in `[0.0, 1.0]`.
    pub fn build(score: f64, reason: impl Into<String>) -> Result<Self> {
        if !(0.0..=1.0).contains(&score) {
            return Err(SmolError::InvalidScore(score));
        }

        Ok(Self {
            score,
            reason: reason.into(),
        })
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn reason(&self) -> &str {
        self.reason.as_str()
    }

    pub fn passed(&self) -> bool {
        self.score == 1.0
    }

    pub fn label(&self) -> CheckLabel {
        if self.passed() {
            CheckLabel::Ok
        } else {
            CheckLabel::Fail
        }
    }
}

// ---------------------------------------------------------------------------
// Check trait — implement this for custom checks
// ---------------------------------------------------------------------------

/// Trait for check implementations.
///
/// Implement this trait and register it with [`CheckRegistry`] to add custom checks.
pub trait Check: Send + Sync {
    /// Run this check against an agent response.
    fn run(&self, response: &AgentResponse) -> CheckResult;
}

// ---------------------------------------------------------------------------
// CheckRegistry — maps type names to factory functions
// ---------------------------------------------------------------------------

/// Factory function: takes YAML config JSON, returns a boxed [`Check`].
type CheckFactory = Box<dyn Fn(&serde_json::Value) -> Result<Box<dyn Check>> + Send + Sync>;

/// Registry that maps check type names to factory functions.
///
/// Use [`CheckRegistry::with_builtins`] to get a registry pre-loaded with all built-in check types.
pub struct CheckRegistry {
    factories: HashMap<String, CheckFactory>,
}

impl CheckRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Create a registry preloaded with all built-in checks.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register("responseContainsAll", Box::new(ContainsAll::from_config));
        registry.register("responseContainsAny", Box::new(ContainsAny::from_config));
        registry.register("responseNotContains", Box::new(NotContains::from_config));
        registry.register("responseExactMatch", Box::new(ExactMatch::from_config));
        registry.register("toolUsedAtLeast", Box::new(ToolUsedAtLeast::from_config));
        registry.register("toolUsedAtMost", Box::new(ToolUsedAtMost::from_config));
        registry.register("toolUsedExactly", Box::new(ToolUsedExactly::from_config));
        registry.register("toolsUsedInOrder", Box::new(ToolsUsedInOrder::from_config));
        registry
    }

    /// Register a custom check type.
    pub fn register(&mut self, name: &str, factory: CheckFactory) {
        self.factories.insert(name.to_string(), factory);
    }

    /// Create a [`Check`] from a [`CheckSpec`].
    ///
    /// Looks up the `kind` in the registry and passes the config to the factory function.
    pub fn create(&self, def: &CheckSpec) -> Result<Box<dyn Check>> {
        let factory = self
            .factories
            .get(&def.kind)
            .ok_or_else(|| SmolError::UnknownCheck(def.kind.clone()))?;
        factory(&def.config)
    }
}

impl Default for CheckRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: parse config JSON into a typed struct
// ---------------------------------------------------------------------------

pub fn parse_config<T: serde::de::DeserializeOwned>(config: &serde_json::Value) -> Result<T> {
    serde_json::from_value(config.clone()).map_err(|e| SmolError::CheckConfig(e.to_string()))
}

// ---------------------------------------------------------------------------
// Built-in checks
// ---------------------------------------------------------------------------

// --- ContainsAll ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContainsAllConfig {
    values: Vec<String>,
    #[serde(default)]
    case_sensitive: bool,
}

/// Check that the response text contains **all** of the given values.
pub struct ContainsAll {
    values: Vec<String>,
    case_sensitive: bool,
}

impl ContainsAll {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ContainsAllConfig = parse_config(config)?;
        Ok(Box::new(Self {
            values: cfg.values,
            case_sensitive: cfg.case_sensitive,
        }))
    }
}

impl Check for ContainsAll {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let text = if self.case_sensitive {
            response.text().to_owned()
        } else {
            response.text().to_lowercase()
        };

        let missing: Vec<&str> = self
            .values
            .iter()
            .filter(|v| {
                let needle = if self.case_sensitive {
                    v.to_string()
                } else {
                    v.to_lowercase()
                };
                !text.contains(&needle)
            })
            .map(|v| v.as_str())
            .collect();

        let case_suffix = if self.case_sensitive {
            "(case-sensitive)"
        } else {
            "(case-insensitive)"
        };

        if missing.is_empty() {
            CheckResult::pass(format!("found all of {:?} in response {case_suffix}", self.values))
        } else {
            CheckResult::fail(format!("missing {:?} in response {case_suffix}", missing))
        }
    }
}

// --- ContainsAny ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContainsAnyConfig {
    values: Vec<String>,
    #[serde(default)]
    case_sensitive: bool,
}

/// Check that the response text contains **at least one** of the given values.
pub struct ContainsAny {
    values: Vec<String>,
    case_sensitive: bool,
}

impl ContainsAny {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ContainsAnyConfig = parse_config(config)?;
        Ok(Box::new(Self {
            values: cfg.values,
            case_sensitive: cfg.case_sensitive,
        }))
    }
}

impl Check for ContainsAny {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let text = if self.case_sensitive {
            response.text().to_owned()
        } else {
            response.text().to_lowercase()
        };

        let found = self.values.iter().find(|v| {
            let needle = if self.case_sensitive {
                v.to_string()
            } else {
                v.to_lowercase()
            };
            text.contains(&needle)
        });

        let case_suffix = if self.case_sensitive {
            "(case-sensitive)"
        } else {
            "(case-insensitive)"
        };

        match found {
            Some(v) => CheckResult::pass(format!("found {:?} in response {case_suffix}", v)),
            None => CheckResult::fail(format!("none of {:?} found in response {case_suffix}", self.values)),
        }
    }
}

// --- NotContains ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NotContainsConfig {
    values: Vec<String>,
    #[serde(default)]
    case_sensitive: bool,
}

/// Check that the response text does **not** contain any of the given values.
pub struct NotContains {
    values: Vec<String>,
    case_sensitive: bool,
}

impl NotContains {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: NotContainsConfig = parse_config(config)?;
        Ok(Box::new(Self {
            values: cfg.values,
            case_sensitive: cfg.case_sensitive,
        }))
    }
}

impl Check for NotContains {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let text = if self.case_sensitive {
            response.text().to_owned()
        } else {
            response.text().to_lowercase()
        };

        let found: Vec<&str> = self
            .values
            .iter()
            .filter(|v| {
                let needle = if self.case_sensitive {
                    v.to_string()
                } else {
                    v.to_lowercase()
                };
                text.contains(&needle)
            })
            .map(|v| v.as_str())
            .collect();

        let case_suffix = if self.case_sensitive {
            "(case-sensitive)"
        } else {
            "(case-insensitive)"
        };

        if found.is_empty() {
            CheckResult::pass(format!("none of {:?} found in response {case_suffix}", self.values))
        } else {
            CheckResult::fail(format!("found {:?} in response {case_suffix}", found))
        }
    }
}

// --- ExactMatch ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExactMatchConfig {
    value: String,
}

/// Check that the response text exactly matches the expected string.
pub struct ExactMatch {
    value: String,
}

impl ExactMatch {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ExactMatchConfig = parse_config(config)?;
        Ok(Box::new(Self { value: cfg.value }))
    }
}

/// Truncate a string to `max` characters, appending `…` if truncated.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

impl Check for ExactMatch {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let text = response.text().trim();
        if text == self.value.trim() {
            CheckResult::pass(format!("found exact match {:?}", truncate(text, 80)))
        } else {
            CheckResult::fail(format!(
                "expected {:?}, got {:?}",
                truncate(self.value.trim(), 80),
                truncate(text, 80)
            ))
        }
    }
}

// --- ToolMatcher (shared helper for count-based tool checks) ---

use crate::agent::ToolCall;

/// Matches a tool call by name and optionally by a subset of its parameters.
#[derive(Debug)]
struct ToolMatcher {
    name: String,
    /// When set, each key/value must appear in the tool call's arguments.
    parameters: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ToolMatcher {
    fn matches(&self, tc: &ToolCall) -> bool {
        if tc.name() != self.name {
            return false;
        }
        if let Some(ref expected) = self.parameters {
            let actual = tc.arguments();
            for (key, val) in expected {
                match actual.get(key) {
                    Some(actual_val) if actual_val == val => {}
                    _ => return false,
                }
            }
        }
        true
    }

    fn count_matches(&self, tool_calls: &[ToolCall]) -> usize {
        tool_calls.iter().filter(|tc| self.matches(tc)).count()
    }
}

fn default_one() -> usize {
    1
}

// --- ToolUsedAtLeast ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolUsedAtLeastConfig {
    name: String,
    #[serde(default)]
    parameters: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default = "default_one")]
    times: usize,
}

/// Check that a tool was used at least N times.
pub struct ToolUsedAtLeast {
    matcher: ToolMatcher,
    times: usize,
}

impl ToolUsedAtLeast {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ToolUsedAtLeastConfig = parse_config(config)?;
        Ok(Box::new(Self {
            matcher: ToolMatcher {
                name: cfg.name,
                parameters: cfg.parameters,
            },
            times: cfg.times,
        }))
    }
}

impl Check for ToolUsedAtLeast {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let count = self.matcher.count_matches(response.tool_calls());
        if count >= self.times {
            CheckResult::pass(format!(
                "tool {:?} used {} time(s) (>= {})",
                self.matcher.name, count, self.times
            ))
        } else {
            CheckResult::fail(format!(
                "tool {:?} used {} time(s), expected at least {}",
                self.matcher.name, count, self.times
            ))
        }
    }
}

// --- ToolUsedAtMost ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolUsedAtMostConfig {
    name: String,
    #[serde(default)]
    parameters: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default = "default_one")]
    times: usize,
}

/// Check that a tool was used at most N times.
pub struct ToolUsedAtMost {
    matcher: ToolMatcher,
    times: usize,
}

impl ToolUsedAtMost {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ToolUsedAtMostConfig = parse_config(config)?;
        Ok(Box::new(Self {
            matcher: ToolMatcher {
                name: cfg.name,
                parameters: cfg.parameters,
            },
            times: cfg.times,
        }))
    }
}

impl Check for ToolUsedAtMost {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let count = self.matcher.count_matches(response.tool_calls());
        if count <= self.times {
            CheckResult::pass(format!(
                "tool {:?} used {} time(s) (<= {})",
                self.matcher.name, count, self.times
            ))
        } else {
            CheckResult::fail(format!(
                "tool {:?} used {} time(s), expected at most {}",
                self.matcher.name, count, self.times
            ))
        }
    }
}

// --- ToolUsedExactly ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolUsedExactlyConfig {
    name: String,
    #[serde(default)]
    parameters: Option<serde_json::Map<String, serde_json::Value>>,
    times: usize, // required — no default
}

/// Check that a tool was used exactly N times.
pub struct ToolUsedExactly {
    matcher: ToolMatcher,
    times: usize,
}

impl ToolUsedExactly {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ToolUsedExactlyConfig = parse_config(config)?;
        Ok(Box::new(Self {
            matcher: ToolMatcher {
                name: cfg.name,
                parameters: cfg.parameters,
            },
            times: cfg.times,
        }))
    }
}

impl Check for ToolUsedExactly {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let count = self.matcher.count_matches(response.tool_calls());
        if count == self.times {
            CheckResult::pass(format!(
                "tool {:?} used exactly {} time(s)",
                self.matcher.name, self.times
            ))
        } else {
            CheckResult::fail(format!(
                "tool {:?} used {} time(s), expected exactly {}",
                self.matcher.name, count, self.times
            ))
        }
    }
}

// --- ToolsUsedInOrder ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolsUsedInOrderConfig {
    tools: Vec<String>,
}

/// Check that tools were used in a specific order (allows other tools in between).
pub struct ToolsUsedInOrder {
    tools: Vec<String>,
}

impl ToolsUsedInOrder {
    fn from_config(config: &serde_json::Value) -> Result<Box<dyn Check>> {
        let cfg: ToolsUsedInOrderConfig = parse_config(config)?;
        Ok(Box::new(Self { tools: cfg.tools }))
    }
}

impl Check for ToolsUsedInOrder {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let actual = response.tool_calls();
        let mut cursor = 0;
        for tc in actual {
            if cursor < self.tools.len() && tc.name() == self.tools[cursor] {
                cursor += 1;
            }
        }
        if cursor == self.tools.len() {
            CheckResult::pass(format!("tools used in order: {:?}", self.tools))
        } else {
            let actual_names: Vec<&str> = actual.iter().map(|tc| tc.name()).collect();
            CheckResult::fail(format!(
                "expected tools in order {:?}, got {:?} (matched {}/{})",
                self.tools,
                actual_names,
                cursor,
                self.tools.len()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ToolCall;

    fn text_response(text: &str) -> AgentResponse {
        AgentResponse::new(text, vec![])
    }

    fn response_with_tools(text: &str, tools: &[&str]) -> AgentResponse {
        AgentResponse::new(
            text,
            tools
                .iter()
                .map(|name| ToolCall::new(*name, serde_json::Value::Null))
                .collect(),
        )
    }

    // -- CheckResult tests --

    #[test]
    fn check_result_pass() {
        let r = CheckResult::pass("ok");
        assert_eq!(r.score(), 1.0);
        assert!(r.passed());
    }

    #[test]
    fn check_result_fail() {
        let r = CheckResult::fail("nope");
        assert_eq!(r.score(), 0.0);
        assert!(!r.passed());
    }

    #[test]
    fn check_result_build_valid() {
        let r = CheckResult::build(0.5, "partial").unwrap();
        assert_eq!(r.score(), 0.5);
        assert!(!r.passed());
    }

    #[test]
    fn check_result_build_invalid() {
        assert!(CheckResult::build(1.5, "bad").is_err());
        assert!(CheckResult::build(-0.1, "bad").is_err());
    }

    // -- ContainsAll tests --

    #[test]
    fn contains_all_pass() {
        let check = ContainsAll {
            values: vec!["hello".into(), "world".into()],
            case_sensitive: false,
        };
        let r = check.run(&text_response("Hello World!"));
        assert!(r.passed());
    }

    #[test]
    fn contains_all_fail() {
        let check = ContainsAll {
            values: vec!["hello".into(), "missing".into()],
            case_sensitive: false,
        };
        let r = check.run(&text_response("Hello World!"));
        assert!(!r.passed());
    }

    #[test]
    fn contains_all_case_sensitive() {
        let check = ContainsAll {
            values: vec!["Hello".into()],
            case_sensitive: true,
        };
        assert!(check.run(&text_response("Hello")).passed());
        assert!(!check.run(&text_response("hello")).passed());
    }

    // -- ContainsAny tests --

    #[test]
    fn contains_any_pass() {
        let check = ContainsAny {
            values: vec!["missing".into(), "world".into()],
            case_sensitive: false,
        };
        let r = check.run(&text_response("Hello World!"));
        assert!(r.passed());
    }

    #[test]
    fn contains_any_fail() {
        let check = ContainsAny {
            values: vec!["missing".into(), "absent".into()],
            case_sensitive: false,
        };
        let r = check.run(&text_response("Hello World!"));
        assert!(!r.passed());
    }

    // -- NotContains tests --

    #[test]
    fn not_contains_pass() {
        let check = NotContains {
            values: vec!["missing".into()],
            case_sensitive: false,
        };
        let r = check.run(&text_response("Hello World!"));
        assert!(r.passed());
    }

    #[test]
    fn not_contains_fail() {
        let check = NotContains {
            values: vec!["hello".into()],
            case_sensitive: false,
        };
        let r = check.run(&text_response("Hello World!"));
        assert!(!r.passed());
    }

    // -- ExactMatch tests --

    #[test]
    fn exact_match_pass() {
        let check = ExactMatch { value: "hello".into() };
        assert!(check.run(&text_response("hello")).passed());
    }

    #[test]
    fn exact_match_fail() {
        let check = ExactMatch { value: "hello".into() };
        assert!(!check.run(&text_response("Hello")).passed());
    }

    // -- ToolUsedAtLeast tests --

    fn response_with_tool_calls(text: &str, calls: Vec<ToolCall>) -> AgentResponse {
        AgentResponse::new(text, calls)
    }

    #[test]
    fn tool_used_at_least_pass_default_times() {
        let check = ToolUsedAtLeast {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 1,
        };
        assert!(check.run(&response_with_tools("ok", &["search", "other"])).passed());
    }

    #[test]
    fn tool_used_at_least_pass_multiple() {
        let check = ToolUsedAtLeast {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 2,
        };
        assert!(check.run(&response_with_tools("ok", &["search", "search", "search"])).passed());
    }

    #[test]
    fn tool_used_at_least_fail_not_present() {
        let check = ToolUsedAtLeast {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 1,
        };
        assert!(!check.run(&response_with_tools("ok", &["other"])).passed());
    }

    #[test]
    fn tool_used_at_least_fail_insufficient() {
        let check = ToolUsedAtLeast {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 3,
        };
        assert!(!check.run(&response_with_tools("ok", &["search", "search"])).passed());
    }

    #[test]
    fn tool_used_at_least_with_params_pass() {
        let mut params = serde_json::Map::new();
        params.insert("city".into(), serde_json::json!("Paris"));
        let check = ToolUsedAtLeast {
            matcher: ToolMatcher { name: "weather".into(), parameters: Some(params) },
            times: 1,
        };
        let calls = vec![
            ToolCall::new("weather", serde_json::json!({"city": "Paris", "units": "metric"})),
        ];
        assert!(check.run(&response_with_tool_calls("ok", calls)).passed());
    }

    #[test]
    fn tool_used_at_least_with_params_fail() {
        let mut params = serde_json::Map::new();
        params.insert("city".into(), serde_json::json!("Paris"));
        let check = ToolUsedAtLeast {
            matcher: ToolMatcher { name: "weather".into(), parameters: Some(params) },
            times: 1,
        };
        let calls = vec![
            ToolCall::new("weather", serde_json::json!({"city": "London"})),
        ];
        assert!(!check.run(&response_with_tool_calls("ok", calls)).passed());
    }

    // -- ToolUsedAtMost tests --

    #[test]
    fn tool_used_at_most_pass_zero_calls() {
        let check = ToolUsedAtMost {
            matcher: ToolMatcher { name: "delete".into(), parameters: None },
            times: 1,
        };
        assert!(check.run(&response_with_tools("ok", &["other"])).passed());
    }

    #[test]
    fn tool_used_at_most_pass_exact() {
        let check = ToolUsedAtMost {
            matcher: ToolMatcher { name: "delete".into(), parameters: None },
            times: 1,
        };
        assert!(check.run(&response_with_tools("ok", &["delete"])).passed());
    }

    #[test]
    fn tool_used_at_most_fail() {
        let check = ToolUsedAtMost {
            matcher: ToolMatcher { name: "delete".into(), parameters: None },
            times: 1,
        };
        assert!(!check.run(&response_with_tools("ok", &["delete", "delete"])).passed());
    }

    #[test]
    fn tool_used_at_most_with_params() {
        let mut params = serde_json::Map::new();
        params.insert("force".into(), serde_json::json!(true));
        let check = ToolUsedAtMost {
            matcher: ToolMatcher { name: "delete".into(), parameters: Some(params) },
            times: 0,
        };
        // delete called twice, but only one matches params — that's 1 > 0, so fail
        let calls = vec![
            ToolCall::new("delete", serde_json::json!({"force": true})),
            ToolCall::new("delete", serde_json::json!({"force": false})),
        ];
        assert!(!check.run(&response_with_tool_calls("ok", calls)).passed());
    }

    // -- ToolUsedExactly tests --

    #[test]
    fn tool_used_exactly_pass() {
        let check = ToolUsedExactly {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 2,
        };
        assert!(check.run(&response_with_tools("ok", &["search", "other", "search"])).passed());
    }

    #[test]
    fn tool_used_exactly_fail_too_few() {
        let check = ToolUsedExactly {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 3,
        };
        assert!(!check.run(&response_with_tools("ok", &["search", "search"])).passed());
    }

    #[test]
    fn tool_used_exactly_fail_too_many() {
        let check = ToolUsedExactly {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 1,
        };
        assert!(!check.run(&response_with_tools("ok", &["search", "search"])).passed());
    }

    #[test]
    fn tool_used_exactly_zero_pass() {
        let check = ToolUsedExactly {
            matcher: ToolMatcher { name: "search".into(), parameters: None },
            times: 0,
        };
        assert!(check.run(&response_with_tools("ok", &["other"])).passed());
    }

    #[test]
    fn tool_used_exactly_config_requires_times() {
        let registry = CheckRegistry::with_builtins();
        let def = CheckSpec::new("toolUsedExactly", serde_json::json!({"name": "search"}));
        assert!(registry.create(&def).is_err());
    }

    // -- ToolsUsedInOrder tests --

    #[test]
    fn tools_used_in_order_pass_exact() {
        let check = ToolsUsedInOrder { tools: vec!["a".into(), "b".into(), "c".into()] };
        assert!(check.run(&response_with_tools("ok", &["a", "b", "c"])).passed());
    }

    #[test]
    fn tools_used_in_order_pass_with_extras() {
        let check = ToolsUsedInOrder { tools: vec!["a".into(), "c".into()] };
        assert!(check.run(&response_with_tools("ok", &["a", "b", "c", "d"])).passed());
    }

    #[test]
    fn tools_used_in_order_fail_wrong_order() {
        let check = ToolsUsedInOrder { tools: vec!["b".into(), "a".into()] };
        assert!(!check.run(&response_with_tools("ok", &["a", "b"])).passed());
    }

    #[test]
    fn tools_used_in_order_fail_missing() {
        let check = ToolsUsedInOrder { tools: vec!["a".into(), "z".into()] };
        assert!(!check.run(&response_with_tools("ok", &["a", "b"])).passed());
    }

    #[test]
    fn tools_used_in_order_empty_passes() {
        let check = ToolsUsedInOrder { tools: vec![] };
        assert!(check.run(&response_with_tools("ok", &["a"])).passed());
    }

    #[test]
    fn tools_used_in_order_repeated() {
        let check = ToolsUsedInOrder { tools: vec!["a".into(), "a".into()] };
        assert!(check.run(&response_with_tools("ok", &["a", "b", "a"])).passed());
        assert!(!check.run(&response_with_tools("ok", &["a", "b"])).passed());
    }

    // -- CheckRegistry tests --

    #[test]
    fn registry_builtins_resolve() {
        let registry = CheckRegistry::with_builtins();
        let def = CheckSpec::new("responseContainsAll", serde_json::json!({"values": ["hello"]}));
        let check = registry.create(&def).unwrap();
        let r = check.run(&text_response("hello world"));
        assert!(r.passed());
    }

    #[test]
    fn registry_unknown_type() {
        let registry = CheckRegistry::with_builtins();
        let def = CheckSpec::new("nonExistent", serde_json::json!({}));
        assert!(registry.create(&def).is_err());
    }

    // -- Additional edge-case tests --

    #[test]
    fn check_result_build_boundary_zero() {
        let r = CheckResult::build(0.0, "zero").unwrap();
        assert_eq!(r.score(), 0.0);
        assert!(!r.passed());
    }

    #[test]
    fn check_result_build_boundary_one() {
        let r = CheckResult::build(1.0, "one").unwrap();
        assert_eq!(r.score(), 1.0);
        assert!(r.passed());
    }

    #[test]
    fn check_result_reason_preserved() {
        let r = CheckResult::pass("detailed reason");
        assert_eq!(r.reason(), "detailed reason");
    }

    #[test]
    fn contains_any_case_sensitive() {
        let check = ContainsAny {
            values: vec!["Hello".into()],
            case_sensitive: true,
        };
        assert!(check.run(&text_response("Hello World")).passed());
        assert!(!check.run(&text_response("hello world")).passed());
    }

    #[test]
    fn not_contains_case_sensitive() {
        let check = NotContains {
            values: vec!["Hello".into()],
            case_sensitive: true,
        };
        // "hello" doesn't match "Hello" case-sensitively
        assert!(check.run(&text_response("hello world")).passed());
        assert!(!check.run(&text_response("Hello world")).passed());
    }

    #[test]
    fn contains_all_empty_values() {
        let check = ContainsAll {
            values: vec![],
            case_sensitive: false,
        };
        assert!(check.run(&text_response("anything")).passed());
    }

    #[test]
    fn contains_any_empty_values() {
        let check = ContainsAny {
            values: vec![],
            case_sensitive: false,
        };
        assert!(!check.run(&text_response("anything")).passed());
    }

    #[test]
    fn not_contains_empty_values() {
        let check = NotContains {
            values: vec![],
            case_sensitive: false,
        };
        assert!(check.run(&text_response("anything")).passed());
    }

    // -- Registry tests for new tool checks --

    #[test]
    fn registry_tool_used_at_least_from_config() {
        let registry = CheckRegistry::with_builtins();
        let def = CheckSpec::new("toolUsedAtLeast", serde_json::json!({"name": "search"}));
        let check = registry.create(&def).unwrap();
        assert!(check.run(&response_with_tools("ok", &["search"])).passed());
    }

    #[test]
    fn registry_tools_used_in_order_from_config() {
        let registry = CheckRegistry::with_builtins();
        let def = CheckSpec::new("toolsUsedInOrder", serde_json::json!({"tools": ["a", "b"]}));
        let check = registry.create(&def).unwrap();
        assert!(check.run(&response_with_tools("ok", &["a", "x", "b"])).passed());
    }

    #[test]
    fn registry_empty_cannot_create() {
        let registry = CheckRegistry::new();
        let def = CheckSpec::new("responseContainsAll", serde_json::json!({"values": ["hi"]}));
        assert!(registry.create(&def).is_err());
    }

    #[test]
    fn registry_default_is_empty() {
        let registry = CheckRegistry::default();
        let def = CheckSpec::new("responseContainsAll", serde_json::json!({"values": ["hi"]}));
        assert!(registry.create(&def).is_err());
    }

    #[test]
    fn registry_custom_check() {
        struct AlwaysFail;
        impl Check for AlwaysFail {
            fn run(&self, _response: &AgentResponse) -> CheckResult {
                CheckResult::fail("always fails")
            }
        }

        let mut registry = CheckRegistry::new();
        registry.register("alwaysFail", Box::new(|_| Ok(Box::new(AlwaysFail))));
        let def = CheckSpec::new("alwaysFail", serde_json::json!({}));
        let check = registry.create(&def).unwrap();
        assert!(!check.run(&text_response("anything")).passed());
    }

    #[test]
    fn contains_all_invalid_config() {
        let registry = CheckRegistry::with_builtins();
        let def = CheckSpec::new("responseContainsAll", serde_json::json!({"wrong_field": 123}));
        assert!(registry.create(&def).is_err());
    }

    #[test]
    fn exact_match_empty_string() {
        let check = ExactMatch { value: "".into() };
        assert!(check.run(&text_response("")).passed());
        // whitespace-only trimmed to "" matches "" after trim
        assert!(check.run(&text_response(" ")).passed());
    }

    #[test]
    fn check_spec_deserialize() {
        let json = r#"{"kind": "responseContainsAll", "values": ["a"]}"#;
        let def: CheckSpec = serde_json::from_str(json).unwrap();
        assert_eq!(def.kind(), "responseContainsAll");
        assert_eq!(def.config()["values"][0], "a");
    }
}

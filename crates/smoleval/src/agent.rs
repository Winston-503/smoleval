use serde::{Deserialize, Serialize};

use crate::Result;

/// A tool call made by an agent during its response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    /// The name of the tool that was called.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The arguments passed to the tool.
    pub fn arguments(&self) -> &serde_json::Value {
        &self.arguments
    }
}

/// The response from an agent after processing a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentResponse {
    text: String,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

impl AgentResponse {
    /// Create a new agent response.
    pub fn new(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            text: text.into(),
            tool_calls,
        }
    }

    /// The text content of the agent's response.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The tool calls the agent made during processing.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }
}

/// The outcome of running an agent on a prompt.
///
/// Either a successful [`AgentResponse`] or an error message.
#[derive(Debug, Clone)]
pub enum AgentOutcome {
    /// The agent returned a response successfully.
    Response(AgentResponse),
    /// The agent (or check creation) failed with this error message.
    Error(String),
}

impl AgentOutcome {
    /// Returns `true` if this outcome is an error.
    pub fn is_error(&self) -> bool {
        matches!(self, AgentOutcome::Error(_))
    }

    /// Returns the response if successful, or `None`.
    pub fn response(&self) -> Option<&AgentResponse> {
        match self {
            AgentOutcome::Response(r) => Some(r),
            AgentOutcome::Error(_) => None,
        }
    }

    /// Returns the error message if failed, or `None`.
    pub fn error(&self) -> Option<&str> {
        match self {
            AgentOutcome::Error(e) => Some(e),
            AgentOutcome::Response(_) => None,
        }
    }
}

/// Trait that agent connectors implement.
pub trait Agent: Send + Sync {
    fn run(&self, prompt: &str) -> impl Future<Output = Result<AgentResponse>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_deserialize_with_arguments() {
        let json = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tc.name(), "get_weather");
        assert_eq!(tc.arguments()["city"], "Paris");
    }

    #[test]
    fn tool_call_deserialize_without_arguments() {
        let json = r#"{"name": "noop"}"#;
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tc.name(), "noop");
        assert_eq!(*tc.arguments(), serde_json::Value::Null);
    }

    #[test]
    fn tool_call_serialize_roundtrip() {
        let tc = ToolCall::new("search", serde_json::json!({"query": "rust"}));
        let json = serde_json::to_string(&tc).unwrap();
        let tc2: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(tc2.name(), "search");
        assert_eq!(tc2.arguments()["query"], "rust");
    }

    #[test]
    fn agent_response_deserialize_full() {
        let json = r#"{
            "text": "It's sunny",
            "toolCalls": [{"name": "weather", "arguments": {}}]
        }"#;
        let resp: AgentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text(), "It's sunny");
        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].name(), "weather");
    }

    #[test]
    fn agent_response_deserialize_no_tool_calls() {
        let json = r#"{"text": "hello"}"#;
        let resp: AgentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text(), "hello");
        assert!(resp.tool_calls().is_empty());
    }

    #[test]
    fn agent_response_serialize_roundtrip() {
        let resp = AgentResponse::new(
            "result",
            vec![
                ToolCall::new("a", serde_json::Value::Null),
                ToolCall::new("b", serde_json::json!(42)),
            ],
        );
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: AgentResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp2.text(), "result");
        assert_eq!(resp2.tool_calls().len(), 2);
        assert_eq!(resp2.tool_calls()[0].name(), "a");
        assert_eq!(resp2.tool_calls()[1].name(), "b");
    }

    #[test]
    fn agent_response_missing_text_fails() {
        let json = r#"{"toolCalls": []}"#;
        assert!(serde_json::from_str::<AgentResponse>(json).is_err());
    }
}

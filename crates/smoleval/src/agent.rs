use serde::{Deserialize, Serialize};

use crate::Result;

/// A tool call made by an agent during its response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Name of the tool that was called.
    pub name: String,
    /// Arguments passed to the tool.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// The response from an agent after processing a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentResponse {
    /// The text content of the agent's final response.
    pub text: String,
    /// Tool calls the agent made during processing.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// Trait that agent connectors implement.
///
/// Uses async (RPITIT) because real agent communication is I/O-bound.
/// For sync test agents, just write `async fn` that returns immediately.
pub trait Agent: Send + Sync {
    fn run(&self, prompt: &str) -> impl std::future::Future<Output = Result<AgentResponse>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_deserialize_with_arguments() {
        let json = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.arguments["city"], "Paris");
    }

    #[test]
    fn tool_call_deserialize_without_arguments() {
        let json = r#"{"name": "noop"}"#;
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tc.name, "noop");
        assert_eq!(tc.arguments, serde_json::Value::Null);
    }

    #[test]
    fn tool_call_serialize_roundtrip() {
        let tc = ToolCall {
            name: "search".into(),
            arguments: serde_json::json!({"query": "rust"}),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let tc2: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(tc2.name, "search");
        assert_eq!(tc2.arguments["query"], "rust");
    }

    #[test]
    fn agent_response_deserialize_full() {
        let json = r#"{
            "text": "It's sunny",
            "toolCalls": [{"name": "weather", "arguments": {}}]
        }"#;
        let resp: AgentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "It's sunny");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "weather");
    }

    #[test]
    fn agent_response_deserialize_no_tool_calls() {
        let json = r#"{"text": "hello"}"#;
        let resp: AgentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "hello");
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn agent_response_serialize_roundtrip() {
        let resp = AgentResponse {
            text: "result".into(),
            tool_calls: vec![
                ToolCall {
                    name: "a".into(),
                    arguments: serde_json::Value::Null,
                },
                ToolCall {
                    name: "b".into(),
                    arguments: serde_json::json!(42),
                },
            ],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: AgentResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp2.text, "result");
        assert_eq!(resp2.tool_calls.len(), 2);
        assert_eq!(resp2.tool_calls[0].name, "a");
        assert_eq!(resp2.tool_calls[1].name, "b");
    }

    #[test]
    fn agent_response_missing_text_fails() {
        let json = r#"{"toolCalls": []}"#;
        assert!(serde_json::from_str::<AgentResponse>(json).is_err());
    }
}

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

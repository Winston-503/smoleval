use std::time::Duration;

use serde::Serialize;

use crate::{Agent, AgentResponse, Result, SmolError};

/// Simple HTTP JSON agent connector.
///
/// Sends a POST request with `{"prompt": "..."}` and expects an [`AgentResponse`] JSON back.
pub struct HttpAgent {
    client: reqwest::Client,
    url: String,
}

#[derive(Serialize)]
struct PromptRequest<'a> {
    prompt: &'a str,
}

impl HttpAgent {
    /// Create a new `HttpAgent` pointing at the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.into(),
        }
    }

    /// Create a new `HttpAgent` with a custom timeout.
    pub fn with_timeout(url: impl Into<String>, timeout: Duration) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .expect("failed to build HTTP client"),
            url: url.into(),
        }
    }
}

impl Agent for HttpAgent {
    async fn run(&self, prompt: &str) -> Result<AgentResponse> {
        let resp = self
            .client
            .post(&self.url)
            .json(&PromptRequest { prompt })
            .send()
            .await
            .map_err(|e| SmolError::AgentError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(SmolError::AgentError(format!(
                "HTTP {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_else(|_| "failed to read body".into())
            )));
        }

        resp.json::<AgentResponse>()
            .await
            .map_err(|e| SmolError::AgentError(format!("failed to parse response: {e}")))
    }
}

pub mod agent;
pub mod check;
pub mod dataset;
pub mod error;
pub mod eval;
#[cfg(feature = "http")]
pub mod http_agent;

pub use agent::{Agent, AgentOutcome, AgentResponse, ToolCall};
pub use check::{Check, CheckLabel, CheckRegistry, CheckResult, CheckSpec};
pub use dataset::{EvalDataset, TestCase};
pub use error::{Result, SmolError};
pub use eval::{EvalOptions, EvalReport, TestCaseLabel, TestCaseResult, evaluate, print_on_result};
#[cfg(feature = "http")]
pub use http_agent::{HttpAgent, PromptRequest};

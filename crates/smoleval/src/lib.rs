pub mod agent;
pub mod check;
pub mod dataset;
pub mod error;
pub mod eval;
#[cfg(feature = "http")]
pub mod http_agent;

pub use agent::{Agent, AgentOutcome, AgentResponse, ToolCall};
pub use check::{Check, CheckLabel, CheckRegistry, CheckResult, CheckSpec, ToolStrictness};
pub use dataset::{EvalDataset, TestCase};
pub use error::{Result, SmolError};
pub use eval::{EvalOptions, EvalReport, TestCaseLabel, TestCaseResult, evaluate, evaluate_with_options};
#[cfg(feature = "http")]
pub use http_agent::HttpAgent;

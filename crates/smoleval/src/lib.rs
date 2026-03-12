pub mod agent;
pub mod check;
pub mod dataset;
pub mod error;
pub mod eval;

pub use agent::{Agent, AgentResponse, ToolCall};
pub use check::{Check, CheckLabel, CheckRegistry, CheckResult, CheckSpec, ToolStrictness};
pub use dataset::{EvalDataset, TestCase};
pub use error::{Result, SmolError};
pub use eval::{EvalOptions, EvalReport, TestCaseLabel, TestCaseResult, evaluate, evaluate_with_options};

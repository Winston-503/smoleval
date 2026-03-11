pub mod agent;
pub mod check;
pub mod dataset;
pub mod error;
pub mod eval;

pub use agent::{Agent, AgentResponse, ToolCall};
pub use check::{Check, CheckDef, CheckRegistry, CheckResult, ToolStrictness};
pub use dataset::{EvalDataset, TestCase};
pub use error::{Result, SmolError};
pub use eval::{EvalOptions, EvalReport, TestCaseResult, evaluate, evaluate_with_options};

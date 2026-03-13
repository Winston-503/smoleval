use rig::client::ProviderClient;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use smoleval::check::parse_config;
use smoleval::{
    Agent, AgentResponse, Check, CheckRegistry, CheckResult, EvalDataset, EvalOptions, ToolCall, evaluate_with_options,
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// A mock agent that echoes its input after delay and always calls "echo_tool"
// ---------------------------------------------------------------------------

struct MockAgent {
    delay: Duration,
}

impl MockAgent {
    pub fn new(delay: Duration) -> Self {
        MockAgent { delay }
    }
}

impl Agent for MockAgent {
    async fn run(&self, prompt: &str) -> smoleval::Result<AgentResponse> {
        tokio::time::sleep(self.delay).await;

        Ok(AgentResponse::new(
            prompt,
            vec![ToolCall::new("echo_tool", serde_json::Value::Null)],
        ))
    }
}

// ---------------------------------------------------------------------------
// CustomCheck example
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomCheckConfig {
    custom_value: String,
}

pub struct CustomCheck {
    custom_value: String,
}

impl CustomCheck {
    fn from_config(config: &serde_json::Value) -> smoleval::Result<Box<dyn Check>> {
        let cfg: CustomCheckConfig = parse_config(config)?;
        Ok(Box::new(Self {
            custom_value: cfg.custom_value,
        }))
    }
}

impl Check for CustomCheck {
    fn run(&self, _response: &AgentResponse) -> CheckResult {
        let reason = format!("This check always passes; customValue: {}", self.custom_value);
        CheckResult::pass(reason)
    }
}

// ---------------------------------------------------------------------------
// LLM-as-a-judge check — uses Rig + OpenAI structured outputs
// ---------------------------------------------------------------------------

/// Structured output from the LLM judge.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct JudgeVerdict {
    /// Step-by-step reasoning for the verdict.
    reasoning: String,
    /// Whether the response passes the check.
    pass: bool,
}

impl From<JudgeVerdict> for CheckResult {
    fn from(val: JudgeVerdict) -> Self {
        if val.pass {
            CheckResult::pass(val.reasoning)
        } else {
            CheckResult::fail(val.reasoning)
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmJudgeConfig {
    /// The judging criteria / prompt sent to the LLM.
    criteria: String,
}

struct LlmJudge {
    extractor: rig::extractor::Extractor<openai::responses_api::ResponsesCompletionModel, JudgeVerdict>,
    criteria: String,
}

impl LlmJudge {
    fn from_config(config: &serde_json::Value) -> smoleval::Result<Box<dyn Check>> {
        let cfg: LlmJudgeConfig = parse_config(config)?;

        let openai_client = openai::Client::from_env();

        let extractor = openai_client
            .extractor::<JudgeVerdict>(openai::GPT_4_1_MINI)
            .preamble(
                "You are an evaluation judge. You will receive an agent's response and judging criteria. \
                 Evaluate whether the response meets the criteria. \
                 Provide step-by-step reasoning, then a pass/fail verdict.",
            )
            .build();

        Ok(Box::new(Self {
            extractor,
            criteria: cfg.criteria,
        }))
    }
}

impl Check for LlmJudge {
    fn run(&self, response: &AgentResponse) -> CheckResult {
        let input = format!(
            "## Agent response\n{}\n\n## Criteria\n{}",
            response.text(),
            self.criteria,
        );

        // Bridge async → sync. We're inside a tokio multi-threaded runtime,
        // so block_in_place + block_on is safe here.
        let verdict =
            tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(self.extractor.extract(&input)));

        match verdict {
            Ok(v) => v.into(),
            Err(e) => CheckResult::fail(format!("[LLM judge error] {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mock_eval_dataset.yaml");

    let dataset = EvalDataset::from_file(&path)?;
    let mut registry = CheckRegistry::with_builtins();
    registry.register("customCheck", Box::new(CustomCheck::from_config));

    let llm_judge_enabled = std::env::var("OPENAI_API_KEY").is_ok();
    if llm_judge_enabled {
        registry.register("llmJudge", Box::new(LlmJudge::from_config));
    } else {
        println!("NOTE: llmJudge check is disabled. Set OPENAI_API_KEY to enable LLM-as-a-judge evaluation.\n");
    }

    println!("=== {} ===\n", dataset.name());

    let agent = MockAgent::new(Duration::from_millis(500));
    let options = EvalOptions::new().with_print_on_result();

    let report = evaluate_with_options(&agent, &dataset, &registry, &options).await?;

    println!(
        "Results: {}/{} passed | Mean score: {:.2}",
        report.passed_count(),
        report.total_count(),
        report.mean_score()
    );

    Ok(())
}

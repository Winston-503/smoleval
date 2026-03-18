use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, routing::post};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::completion::ToolDefinition;
use rig::providers::openai;
use rig::tool::{Tool, ToolDyn};
use serde::Deserialize;
use serde_json::json;
use smoleval::{AgentResponse, PromptRequest, ToolCall};

// ---------------------------------------------------------------------------
// Tool-call recorder: each request gets its own instance
// ---------------------------------------------------------------------------
type ToolCallRecorder = Arc<Mutex<Vec<ToolCall>>>;

// ---------------------------------------------------------------------------
// Rig tool: a simple adder that records every invocation
// ---------------------------------------------------------------------------
#[derive(Deserialize)]
struct AddArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

struct Adder {
    recorder: ToolCallRecorder,
}

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = AddArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add two numbers together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.recorder
            .lock()
            .unwrap()
            .push(ToolCall::new("add", json!({"x": args.x, "y": args.y})));
        Ok(args.x + args.y)
    }
}

// ---------------------------------------------------------------------------
// Axum application state
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct AppState {
    openai_client: Arc<openai::Client>,
}

// ---------------------------------------------------------------------------
// Handler: POST / {"prompt": "..."} -> {"text": "...", "toolCalls": [...]}
// ---------------------------------------------------------------------------
async fn handle(State(state): State<AppState>, Json(req): Json<PromptRequest>) -> Json<AgentResponse> {
    // Build a fresh agent (and recorder) per request so tool calls are isolated.
    let recorder: ToolCallRecorder = Arc::new(Mutex::new(Vec::new()));

    let tools: Vec<Box<dyn ToolDyn>> = vec![Box::new(Adder {
        recorder: recorder.clone(),
    })];

    let agent = state
        .openai_client
        .agent(openai::GPT_4_1_MINI)
        .preamble(
            "You are a calculator assistant. \
             Use the `add` tool to perform addition. \
             Always use the tool rather than computing in your head.",
        )
        .tools(tools)
        .max_tokens(1024)
        .build();

    match agent.prompt(req.prompt()).await {
        Ok(text) => {
            let tool_calls = recorder.lock().unwrap().drain(..).collect();
            Json(AgentResponse::new(text, tool_calls))
        }
        Err(e) => Json(AgentResponse::new(format!("Error: {e}"), vec![])),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let openai_client = openai::Client::from_env();

    let state = AppState {
        openai_client: Arc::new(openai_client),
    };

    let app = Router::new().route("/", post(handle)).with_state(state);

    let addr = "0.0.0.0:3825";
    println!("Rig calculator agent listening on {addr}");
    println!();
    println!("Run the eval with:");
    println!("  cargo run -p smoleval-cli -- \\");
    println!("    --dataset crates/smoleval-cli-example/data/eval_dataset.yaml \\");
    println!("    --agent http://localhost:3825");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

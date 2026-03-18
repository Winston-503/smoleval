use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, routing::post};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::completion::ToolDefinition;
use rig::providers::openai;
use rig::providers::openai::responses_api::ResponsesCompletionModel;
use rig::tool::{Tool, ToolDyn};
use serde::Deserialize;
use serde_json::json;
use smoleval::{AgentResponse, PromptRequest, ToolCall};
use tokio::sync::Mutex as TokioMutex;

// ---------------------------------------------------------------------------
// Tool-call recorder: shared between the Rig tool and the HTTP handler
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
    agent: Arc<rig::agent::Agent<ResponsesCompletionModel>>,
    recorder: ToolCallRecorder,
    /// Serialises requests so each clear→prompt→drain cycle is atomic.
    request_lock: Arc<TokioMutex<()>>,
}

// ---------------------------------------------------------------------------
// Handler: POST / {"prompt": "..."} -> {"text": "...", "toolCalls": [...]}
// ---------------------------------------------------------------------------
async fn handle(State(state): State<AppState>, Json(req): Json<PromptRequest>) -> Json<AgentResponse> {
    // Hold an async lock for the entire request so concurrent requests don't interleave their tool-call recordings.
    let _guard = state.request_lock.lock().await;

    state.recorder.lock().unwrap().clear();

    match state.agent.prompt(req.prompt()).await {
        Ok(text) => {
            let tool_calls = state.recorder.lock().unwrap().drain(..).collect();
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

    let recorder: ToolCallRecorder = Arc::new(Mutex::new(Vec::new()));

    let openai_client = openai::Client::from_env();

    let tools: Vec<Box<dyn ToolDyn>> = vec![Box::new(Adder {
        recorder: recorder.clone(),
    })];

    let agent = openai_client
        .agent(openai::GPT_4_1_MINI)
        .preamble(
            "You are a calculator assistant. \
             Use the `add` tool to perform addition. \
             Always use the tool rather than computing in your head.",
        )
        .tools(tools)
        .max_tokens(1024)
        .build();

    let state = AppState {
        agent: Arc::new(agent),
        recorder,
        request_lock: Arc::new(TokioMutex::new(())),
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

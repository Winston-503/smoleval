# smoleval-cli-example

Example HTTP agent server designed to be evaluated with [smoleval-cli](https://crates.io/crates/smoleval-cli).

Builds an [Axum](https://crates.io/crates/axum) web server backed by [Rig](https://crates.io/crates/rig-core) with an OpenAI GPT-4 agent and a custom tool. The server exposes a POST endpoint that accepts prompts and returns `AgentResponse` JSON compatible with smoleval's `HttpAgent`.

## Running

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Start the server
cargo run -p smoleval-cli-example

# In another terminal, run the eval
smoleval --dataset datasets/eval.yaml --agent http://localhost:3825
```

## License

Apache-2.0

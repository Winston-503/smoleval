# smoleval-cli-example

Example HTTP agent servers designed to be evaluated with [smoleval-cli](https://crates.io/crates/smoleval-cli).

Both examples implement the same calculator agent with an `add` tool, demonstrating that smoleval works with any language over HTTP.

## Rust / Rig (port 3825)

Builds an [Axum](https://crates.io/crates/axum) web server backed by [Rig](https://crates.io/crates/rig-core) with an OpenAI GPT-4 agent and a custom tool.

```bash
export OPENAI_API_KEY=sk-...

# Start the server
make example-rig-agent

# In another terminal, run the eval
smoleval --dataset crates/smoleval-cli-example/data/eval_dataset.yaml --agent http://localhost:3825
```

## Python / LangGraph (port 3826)

A single-file [Flask](https://flask.palletsprojects.com) server backed by [LangGraph](https://langchain-ai.github.io/langgraph/). Uses [PEP 723](https://peps.python.org/pep-0723/) inline metadata so `uv run` handles dependencies automatically.

```bash
export OPENAI_API_KEY=sk-...

# Start the server (requires uv)
make example-langchain-agent

# In another terminal, run the eval
smoleval --dataset crates/smoleval-cli-example/data/eval_dataset.yaml --agent http://localhost:3826
```

## License

Apache-2.0

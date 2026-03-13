# smoleval-cli-example

Example HTTP agent servers designed to be evaluated with [smoleval-cli](https://crates.io/crates/smoleval-cli).

Both examples implement the same calculator agent with an `add` tool, demonstrating that smoleval works with any language over HTTP.

## Rust / Rig (port 3825)

[Axum](https://crates.io/crates/axum) web server backed by [Rig](https://crates.io/crates/rig-core).

```bash
export OPENAI_API_KEY=sk-...

# Start the server
make example-rig-agent

# In another terminal, run the eval
smoleval --dataset crates/smoleval-cli-example/data/eval_dataset.yaml --agent http://localhost:3825
```

## Python / LangChain (port 3826)

[Flask](https://flask.palletsprojects.com) server backed by [LangChain](https://docs.langchain.com).

```bash
export OPENAI_API_KEY=sk-...

# Start the server (requires uv)
make example-langchain-agent

# In another terminal, run the eval
smoleval --dataset crates/smoleval-cli-example/data/eval_dataset.yaml --agent http://localhost:3826
```

## CLI Options

Run tests concurrently with `--concurrency`. Pair it with `--quiet` and `--output` so live agent logs stay readable and the eval report goes to a file:

```bash
smoleval \
  --dataset crates/smoleval-cli-example/data/eval_dataset.yaml \
  --agent http://localhost:3825 \
  --concurrency 3 \
  --quiet \
  --format json \
  --output results.json
```

Use `--format json` to get machine-readable output, or `--format junit` for CI integration.

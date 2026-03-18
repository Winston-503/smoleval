# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv==1.2.2",
#     "flask==3.1.3",
#     "langchain==1.2.12",
#     "langchain-openai==1.1.11",
# ]
# ///
"""Minimal LangChain agent server compatible with smoleval's HTTP protocol."""

import contextvars
import threading

import dotenv
from flask import Flask, jsonify, request
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

dotenv.load_dotenv()

# Per-request tool-call log, isolated via threading.local().
_request_log = threading.local()


def _get_log() -> list[dict]:
    """Return the tool-call log for the current request."""
    if not hasattr(_request_log, "calls"):
        _request_log.calls = []
    return _request_log.calls


@tool
def add(x: int, y: int) -> int:
    """Add two numbers together."""
    _get_log().append({"name": "add", "arguments": {"x": x, "y": y}})
    return x + y


model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model,
    tools=[add],
    system_prompt=(
        "You are a calculator assistant. "
        "Use the `add` tool to perform addition. "
        "Always use the tool rather than computing in your head."
    ),
)

app = Flask(__name__)


@app.post("/")
def handle():
    # Reset the per-thread log for this request.
    _request_log.calls = []
    prompt = request.json["prompt"]
    print(f"\n>>> Received prompt: {prompt}")
    result = agent.invoke({"messages": [("user", prompt)]})
    text = result["messages"][-1].content
    log = _get_log()
    if log:
        print(f"    Tools used: {[tc['name'] for tc in log]}")
    print(f"    Response: {text}")
    return jsonify({"text": text, "toolCalls": log})


if __name__ == "__main__":
    addr, port = "0.0.0.0", 3826
    print(f"LangChain calculator agent listening on {addr}:{port}")
    print()
    print("Run the eval with:")
    print("  cargo run -p smoleval-cli -- \\")
    print("    --dataset crates/smoleval-cli-example/data/eval_dataset.yaml \\")
    print(f"    --agent http://localhost:{port}")
    app.run(host=addr, port=port)

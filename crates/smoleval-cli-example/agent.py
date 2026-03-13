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

import dotenv
from flask import Flask, jsonify, request
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

dotenv.load_dotenv()

# Records tool invocations for the current request.
tool_call_log: list[dict] = []


@tool
def add(x: int, y: int) -> int:
    """Add two numbers together."""
    tool_call_log.append({"name": "add", "arguments": {"x": x, "y": y}})
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
    tool_call_log.clear()
    prompt = request.json["prompt"]
    print(f"\n>>> Received prompt: {prompt}")
    result = agent.invoke({"messages": [("user", prompt)]})
    text = result["messages"][-1].content
    if tool_call_log:
        print(f"    Tools used: {[tc['name'] for tc in tool_call_log]}")
    print(f"    Response: {text}")
    return jsonify({"text": text, "toolCalls": tool_call_log})


if __name__ == "__main__":
    addr, port = "0.0.0.0", 3826
    print(f"LangChain calculator agent listening on {addr}:{port}")
    print()
    print("Run the eval with:")
    print("  cargo run -p smoleval-cli -- \\")
    print("    --dataset crates/smoleval-cli-example/data/eval_dataset.yaml \\")
    print(f"    --agent http://localhost:{port}")
    app.run(host=addr, port=port)

<div align="center">

![Dragen Logo](https://github.com/chonkie-inc/dragen/blob/main/assets/dragen.png?raw=true)

# Dragen

### CodeAct-style AI agents that write Python, not JSON.

[![PyPI version](https://img.shields.io/pypi/v/dragen.svg)](https://pypi.org/project/dragen/)
[![Crates.io](https://img.shields.io/crates/v/dragen.svg)](https://crates.io/crates/dragen)
[![License](https://img.shields.io/github/license/chonkie-inc/dragen.svg)](https://github.com/chonkie-inc/dragen/blob/main/LICENSE)
[![CI](https://github.com/chonkie-inc/dragen/actions/workflows/ci.yml/badge.svg)](https://github.com/chonkie-inc/dragen/actions/workflows/ci.yml)

</div>

---

Instead of making your LLM fill in JSON schemas one tool call at a time, Dragen gives it a Python sandbox. The agent writes real code — loops, branches, error handling, multi-step reasoning — all in one shot. That's the [CodeAct](https://arxiv.org/abs/2402.01030) pattern.

Code runs in a [Littrs](https://github.com/chonkie-inc/littrs) sandbox — a Python-to-bytecode compiler and stack VM embedded directly in your process. No containers, no cloud sandboxing services, no `exec()`. Zero ambient capabilities: no filesystem, no network, no env vars. The only way sandboxed code reaches the outside world is through tools you explicitly provide.

## Installation

```bash
pip install dragen
```

## Quick Start

```python
import dragen

agent = dragen.Agent("gpt-4o")

@agent.tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

result = agent.run("Search for recent AI agent frameworks")
print(result)
```

## Examples

### Structured output with self-correction

Pass a schema and the agent retries until the output validates:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    sentiment: str  # positive, negative, neutral
    confidence: float

agent = dragen.Agent("gpt-4o")
result = agent.run(
    "Analyze the sentiment of: 'This product is amazing!'",
    schema=Analysis.model_json_schema()
)
analysis = Analysis(**result)
```

### Multi-agent pipeline with shared context

Agents pass typed data to each other through a shared `Context`:

```python
from dragen import Agent, Context

ctx = Context()

# Planner researches and writes a plan
planner = Agent("gpt-4o").to_context(ctx, "plan")
planner.run("Create a research plan for: quantum computing trends")

# Writer reads the plan and produces content
writer = Agent("gpt-4o").from_context(ctx, "plan")
result = writer.run("Write a report based on the research plan")
```

### Sandbox with tools, limits, and file access

```python
sandbox = dragen.Sandbox(builtins=True)
sandbox.limit(max_instructions=50_000, max_recursion_depth=30)
sandbox.mount("data.csv", "./input/data.csv")

@sandbox.tool
def summarize(text: str) -> str:
    """Summarize text using an external API."""
    return call_summary_api(text)

agent = dragen.Agent("gpt-4o", sandbox=sandbox)
result = agent.run("Read data.csv and summarize its contents")
```

### Recursive Language Model ([RLM](https://arxiv.org/abs/2512.24601))

Process inputs far beyond the context window — the long input lives in the sandbox as a variable and the agent writes code to slice and summarize it across iterations:

```python
sandbox = dragen.Sandbox(builtins=True)
sandbox["document"] = very_long_text  # e.g. 500K tokens

agent = dragen.Agent("gpt-4o", max_iterations=20, sandbox=sandbox)
result = agent.run("""
The variable `document` contains a very long research paper.
Extract all key findings, then synthesize them into a structured summary.
You can slice `document` with Python string indexing to read it in parts.
""")
```

## Configuration

```python
agent = dragen.Agent(
    "gpt-4o",
    max_iterations=10,
    temperature=0.7,
    max_tokens=4096,
    system="You are a helpful assistant"
)
```

## Event Callbacks

```python
agent = dragen.Agent("gpt-4o")

@agent.on_code
def on_code(code):
    print(f"Executing:\n{code}")

@agent.on_output
def on_output(output):
    print(f"Output: {output}")

@agent.on_finish
def on_finish(result):
    print(f"Done: {result}")
```

For the full feature reference, see **[DOCS.md](https://github.com/chonkie-inc/dragen/blob/main/DOCS.md)**. More examples in **[examples/](https://github.com/chonkie-inc/dragen/tree/main/examples)**.

## License

Apache-2.0

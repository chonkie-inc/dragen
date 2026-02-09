<div align="center">

![Dragen Logo](https://github.com/chonkie-inc/dragen/blob/main/assets/dragen.png?raw=true)

# Dragen

### CodeAct-style AI agents that write Python, not JSON.

[![Crates.io](https://img.shields.io/crates/v/dragen.svg)](https://crates.io/crates/dragen)
[![PyPI version](https://img.shields.io/pypi/v/dragen.svg)](https://pypi.org/project/dragen/)
[![License](https://img.shields.io/github/license/chonkie-inc/dragen.svg)](https://github.com/chonkie-inc/dragen/blob/main/LICENSE)
[![CI](https://github.com/chonkie-inc/dragen/actions/workflows/ci.yml/badge.svg)](https://github.com/chonkie-inc/dragen/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/dragen.svg)](https://github.com/chonkie-inc/dragen/stargazers)

</div>

---

When you ask an LLM to call tools via JSON schemas, you're asking it to work in a format it wasn't trained on. It can't loop over results, can't branch on conditions, can't compose tool outputs — it fills in one schema at a time and waits. But give it a Python sandbox and it writes code: loops, branches, error handling, multi-step reasoning — all in one shot.

That's the [CodeAct](https://arxiv.org/abs/2402.01030) pattern, and Dragen is a framework built around it. You register tools as Python functions, hand the agent a task, and it writes code to solve it. The hard part is sandboxing — most CodeAct frameworks rely on restricted Python interpreters that block dangerous imports, but these have a large attack surface and have led to sandbox escapes in practice ([CVE-2025-5120](https://nvd.nist.gov/vuln/detail/CVE-2025-5120), [CVE-2025-9959](https://nvd.nist.gov/vuln/detail/CVE-2025-9959)). The alternative — Docker, E2B, or Modal — adds infrastructure and latency. Dragen sidesteps this with [Littrs](https://github.com/chonkie-inc/littrs), a Python-to-bytecode compiler and stack VM with **zero ambient capabilities**: no filesystem, no network, no env vars, no dangerous imports. Resource limits are enforced at the VM level and cannot be caught by `try`/`except`. The only way sandboxed code can reach the outside world is through the tools you explicitly provide. All of this runs in-process — `cargo add dragen` or `pip install dragen` and you're done.

### What you get

* **Secure sandbox** — [Littrs](https://github.com/chonkie-inc/littrs) with resource limits, file mounting, and custom modules. [Details](DOCS.md#sandbox-configuration)
* **Structured output** — JSON Schema validation with self-correction. Works with Pydantic. [Details](DOCS.md#structured-output)
* **Multi-agent pipelines** — shared `Context` for typed data passing between agents. [Details](DOCS.md#shared-context)
* **Parallel execution** — `agent.map(tasks)` runs concurrent tasks on cloned agents. [Details](DOCS.md#parallel-execution)
* **Any LLM** — OpenAI, Anthropic, Groq, Ollama, or any compatible API via [Tanukie](https://github.com/chonkie-inc/tanukie)
* **Observable** — event callbacks for every step of the agent loop. [Details](DOCS.md#event-callbacks)

## Installation

### Rust

```bash
cargo add dragen
```

### Python

```bash
pip install dragen
```

## Quick Start

### Rust

```rust
use dragen::{Agent, AgentConfig};
use littrs::tool;

/// Search the web for information.
///
/// Args:
///     query: The search query
#[tool]
fn search(query: String) -> String {
    format!("Results for: {}", query)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = Agent::new(AgentConfig::new("moonshotai/kimi-k2.5"));
    agent.register(search::Tool);

    let result: String = agent.run("Search for recent AI agent frameworks").await?;
    println!("{}", result);
    Ok(())
}
```

### Python

```python
import dragen

agent = dragen.Agent("moonshotai/kimi-k2.5")

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

agent = dragen.Agent("moonshotai/kimi-k2.5")
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
planner = Agent("moonshotai/kimi-k2.5").to_context(ctx, "plan")
planner.run("Create a research plan for: quantum computing trends")

# Writer reads the plan and produces content
writer = Agent("moonshotai/kimi-k2.5").from_context(ctx, "plan")
result = writer.run("Write a report based on the research plan")
```

### Recursive Language Model ([RLM](https://arxiv.org/abs/2512.24601))

RLMs let an LLM recursively call itself to process inputs far beyond its context window. The long input lives in the sandbox as a variable — the agent writes code to slice, examine, and summarize chunks, accumulating results across iterations:

```python
sandbox = dragen.Sandbox(builtins=True)
sandbox["document"] = very_long_text  # e.g. 500K tokens

agent = dragen.Agent("moonshotai/kimi-k2.5", max_iterations=20, sandbox=sandbox)
result = agent.run("""
The variable `document` contains a very long research paper.
Extract all key findings, then synthesize them into a structured summary.
You can slice `document` with Python string indexing to read it in parts.
""")
```

The agent writes code like `chunk = document[0:5000]`, processes it, then `chunk = document[5000:10000]`, accumulating findings in a list variable across iterations — recursively decomposing the input without ever exceeding its context window.

### Custom sandbox with limits and file access

Pre-configure a sandbox with resource limits and mounted files:

```python
sandbox = dragen.Sandbox(builtins=True)
sandbox.limit(max_instructions=50_000, max_recursion_depth=30)
sandbox.mount("data.csv", "./input/data.csv")
sandbox.mount("report.md", "./output/report.md", writable=True)

agent = dragen.Agent("moonshotai/kimi-k2.5", sandbox=sandbox)
result = agent.run("Read data.csv and write a summary report to report.md")
```

For the full feature reference, see **[DOCS.md](DOCS.md)**. More examples in **[examples/](examples/)**.

## Citation

If you use Dragen in your research, please cite it as:

```bibtex
@software{dragen,
  title = {Dragen: CodeAct-style AI Agent Framework},
  author = {Chonkie Inc.},
  url = {https://github.com/chonkie-inc/dragen},
  license = {Apache-2.0},
  year = {2025-2026}
}
```

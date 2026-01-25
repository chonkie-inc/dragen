# dragen

CodeAct-style AI agent framework for Python.

## Installation

```bash
pip install dragen
```

## Quick Start

```python
from dragen import Agent

# Create an agent
agent = Agent("gpt-4o")

# Register a tool
def search(args):
    query = args[0]
    return {"results": [f"Found: {query}"]}

agent.register_function("search", search)

# Run a task
result = agent.run("Search for Python tutorials")
print(result)
```

## Features

- **CodeAct pattern**: LLM writes Python code to accomplish tasks
- **Secure sandbox**: Code runs in a sandboxed environment
- **Tool registration**: Expose Python functions as tools for the agent
- **Multi-agent support**: Share data between agents via Context

## Configuration

```python
from dragen import Agent

agent = Agent(
    "gpt-4o",
    max_iterations=10,
    temperature=0.7,
    max_tokens=4096,
    system="You are a helpful assistant"
)
```

## Tool Registration

### Simple function

```python
def add(args):
    return args[0] + args[1]

agent.register_function("add", add)
```

### With type information

```python
from dragen import ToolInfo

info = ToolInfo("search", "Search the web") \
    .arg_required("query", "str", "The search query") \
    .returns("dict")

def search(args):
    query = args[0]
    return {"results": ["result1", "result2"]}

agent.register_tool(info, search)
```

## Multi-Agent with Context

```python
from dragen import Agent, Context

ctx = Context()

# Planner writes output to context
planner = Agent("gpt-4o")
planner.to_context(ctx, "plan")
planner.run("Create a research plan about AI")

# Executor reads from context
executor = Agent("gpt-4o")
executor.from_context(ctx, "plan")
executor.run("Execute the research plan")
```

## Environment Variables

Set your API key based on the model provider:

- `OPENAI_API_KEY` for OpenAI models (gpt-4o, etc.)
- `ANTHROPIC_API_KEY` for Anthropic models (claude-*, etc.)
- `GROQ_API_KEY` for Groq models

## License

Apache-2.0

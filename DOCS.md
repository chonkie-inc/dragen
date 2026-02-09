# Dragen Documentation

Full reference for all dragen features. For a quick overview, see the [README](README.md).

## Table of Contents

- [How It Works](#how-it-works)
- [Sandbox Configuration](#sandbox-configuration)
- [Tool Registration](#tool-registration)
- [Structured Output](#structured-output)
- [Parallel Execution](#parallel-execution)
- [Shared Context](#shared-context)
- [Event Callbacks](#event-callbacks)
- [Configuration](#configuration)
- [Python API](#python-api)

## How It Works

Dragen implements the [CodeAct](https://arxiv.org/abs/2402.01030) pattern. Instead of JSON function calls, the LLM writes Python code to solve tasks:

1. You register tools as Python functions (`#[tool]` in Rust, `@agent.tool` in Python)
2. The agent builds a system prompt with auto-generated tool signatures and docstrings
3. The LLM receives your task and writes Python code to solve it
4. Code executes in the [Littrs](https://github.com/chonkie-inc/littrs) sandbox — tools are the only way to reach the outside world
5. Execution output (return values + print output) is sent back to the LLM
6. The LLM calls `finish(answer)` when done, returning a typed result

```
User Task → LLM → Python Code → Sandbox → Result → LLM → ... → finish()
```

The agent extracts code from both ` ```python ` fenced blocks and `<code>...</code>` XML tags. For structured output, the LLM can also use `<finish>{json}</finish>` blocks to return JSON directly without code execution.

## Sandbox Configuration

By default, `Agent::new()` creates a sandbox with builtins enabled (`json`, `math`, `typing`). For full control, configure a sandbox and pass it in with `Agent::with_sandbox()`.

### Rust

```rust
use littrs::{Sandbox, Limits};
use dragen::{Agent, AgentConfig};

let mut sandbox = Sandbox::with_builtins();

// Resource limits (uncatchable by try/except in sandbox code)
sandbox.limit(Limits {
    max_instructions: Some(100_000),
    max_recursion_depth: Some(50),
});

// Mount host files into the sandbox
sandbox.mount("input.json", "./data/input.json", false);    // read-only
sandbox.mount("output.txt", "./results/output.txt", true);  // read-write

// Register a custom module accessible via `import config`
sandbox.module("config", |m| {
    m.constant("api_version", littrs::PyValue::Str("v2".into()));
    m.function("get_flag", |_args| littrs::PyValue::Bool(true));
});

let agent = Agent::with_sandbox(sandbox, AgentConfig::new("moonshotai/kimi-k2.5"));
```

### Python

```python
import dragen

sandbox = dragen.Sandbox(builtins=True)
sandbox.limit(max_instructions=100_000, max_recursion_depth=50)
sandbox.mount("input.json", "./data/input.json")
sandbox.mount("output.txt", "./results/output.txt", writable=True)

agent = dragen.Agent("moonshotai/kimi-k2.5", sandbox=sandbox)
```

### Sandbox capabilities

| Feature | Description |
|---------|-------------|
| **Resource limits** | Cap bytecode instructions and recursion depth per execution. Enforced at the VM level — `try`/`except` cannot suppress them |
| **File mounting** | Mount host files into the sandbox. Read-only by default, optionally read-write. Unmounted paths raise `FileNotFoundError` |
| **Custom modules** | Register modules with constants and functions, accessible via `import` |
| **Built-in modules** | `json`, `math`, `typing` available with `Sandbox::with_builtins()` / `Sandbox(builtins=True)` |
| **Variable injection** | Set sandbox variables from host code with `sandbox.set(name, value)` |
| **Zero ambient capabilities** | No filesystem, no network, no env vars, no dangerous imports. Tools are the only gateway |

## Tool Registration

### The `#[tool]` macro (Rust)

The easiest way to register tools. Write a normal function with doc comments, and the macro generates everything:

```rust
use littrs::tool;

/// Get current weather for a city.
///
/// Args:
///     city: The city name
///     units: Temperature units (C or F)
#[tool]
fn get_weather(city: String, units: Option<String>) -> String {
    format!("{}: 22°{}", city, units.unwrap_or("C".into()))
}

let mut agent = Agent::new(config);
agent.register(get_weather::Tool);
```

The macro handles `PyValue` ↔ Rust type conversion automatically. `Option<T>` parameters become optional arguments.

### The `@agent.tool` decorator (Python)

```python
@agent.tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp": 22, "units": units}
```

Type hints (`str`, `int`, `float`, `bool`, `list`, `dict`, `any`) are extracted and shown in the tool documentation the LLM sees.

### Low-level registration with `ToolInfo`

For cases where the macro/decorator isn't suitable:

```rust
use littrs::{ToolInfo, PyValue};

let info = ToolInfo::new("search", "Search the web")
    .arg("query", "str", "The search query")
    .arg_opt("limit", "int", "Max results")
    .returns("list");

agent.register_tool(info, |args| {
    let query = args[0].as_str().unwrap_or("");
    PyValue::List(vec![PyValue::Str(format!("Result for: {}", query))])
});
```

### Custom finish tools

Override the default `finish()` to add custom logic:

```rust
let finish_info = ToolInfo::new("finish", "Return the final report")
    .arg("report", "dict", "The structured report")
    .returns("dict");

agent.register_finish(finish_info, |args| {
    // Custom processing before returning
    args.first().cloned().unwrap_or(PyValue::None)
});
```

### Auto-generated documentation

`sandbox.describe()` produces Python-style signatures from registered tools, embedded in the system prompt:

```
def get_weather(city: str, units?: str) -> str:
    """Get current weather for a city."""

def search(query: str, limit?: int) -> list:
    """Search the web."""
```

## Structured Output

### JSON Schema validation

Pass a JSON Schema and the agent self-corrects until the output validates:

```rust
let schema = serde_json::json!({
    "type": "object",
    "required": ["summary", "sentiment", "confidence"],
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    }
});

let mut agent = Agent::with_model("moonshotai/kimi-k2.5").schema(schema);
let result: Analysis = agent.run("Analyze: 'This product is amazing!'").await?;
```

If the LLM's output fails validation, the error message is sent back with the schema details, and the LLM gets another iteration to fix it.

### Pydantic integration (Python)

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float

result = agent.run(
    "Analyze: 'This product is amazing!'",
    schema=Analysis.model_json_schema()
)
analysis = Analysis(**result)
```

### How finish works

The LLM can return results in two ways:

1. **`finish()` function call** — the agent registers a `finish` tool that captures the value and signals completion
2. **`<finish>` XML block** — the LLM writes `<finish>{"key": "value"}</finish>` to return JSON directly without code execution

Both paths go through schema validation if a schema is set.

## Parallel Execution

Run multiple tasks concurrently with `agent.map()`:

```rust
let agent = Agent::new(config);
agent.register(search::Tool);

let results: Vec<Result<String>> = agent.map(vec![
    "Research topic A".into(),
    "Research topic B".into(),
    "Research topic C".into(),
]).await;

for (i, result) in results.iter().enumerate() {
    match result {
        Ok(value) => println!("Task {} succeeded: {}", i, value),
        Err(e) => println!("Task {} failed: {}", i, e),
    }
}
```

Each task runs on a **fresh clone** of the agent with:
- Same configuration, tools, and schema
- Independent sandbox state and message history
- Shared `Context` access (if configured)

Individual task failures don't affect other tasks. Results are returned in order.

### Python

```python
results = agent.map([
    "Research topic A",
    "Research topic B",
    "Research topic C",
])
```

## Shared Context

`Context` is a thread-safe key-value store for passing data between agents without manual `Arc<Mutex<>>` management.

### Basic usage

```rust
use dragen::{Agent, AgentConfig, Context};

let ctx = Context::new();

// Planner writes output to context
let mut planner = Agent::new(AgentConfig::new("moonshotai/kimi-k2.5"))
    .to_context(&ctx, "plan");
planner.run::<PlanOutput>(&task).await?;

// Executor reads from context (auto-injected into prompt)
let mut executor = Agent::new(AgentConfig::new("moonshotai/kimi-k2.5"))
    .from_context(&ctx, "plan");
executor.run::<String>("Execute the plan").await?;
```

### How it works

- `.to_context(&ctx, "key")` — after `run()` completes, the result is serialized to JSON and stored under the given key
- `.from_context(&ctx, "key")` — before calling the LLM, the stored value is injected into the task prompt as a `<context>` block
- Cloning a `Context` is cheap (Arc-based) — all agents share the same underlying store
- Multiple agents can read from and write to the same context concurrently

### Context API

```rust
let ctx = Context::new();
ctx.set("key", &value);                    // Store any serializable value
let val: Option<T> = ctx.get("key");        // Retrieve and deserialize
ctx.contains("key");                        // Check existence
ctx.remove("key");                          // Remove a key
ctx.keys();                                 // List all keys
ctx.clear();                                // Clear everything
```

## Event Callbacks

Hook into every step of the agent loop for logging, debugging, or custom control.

### Available events

| Event | Fields | Description |
|-------|--------|-------------|
| `IterationStart` | `iteration`, `max_iterations` | Beginning of a new iteration |
| `LLMRequest` | `message_count` | About to call the LLM |
| `LLMResponse` | `content`, `tokens_used` | LLM responded |
| `Thinking` | `content` | Extracted from custom thinking tags |
| `CodeGenerated` | `code` | Code block extracted from response |
| `CodeExecuted` | `code`, `output`, `success` | Code ran in sandbox |
| `ToolCall` | `name`, `args` | A tool was invoked |
| `ToolResult` | `name`, `result` | A tool returned |
| `Finish` | `value` | Agent called `finish()` |
| `Error` | `message` | An error occurred |

### Rust callbacks

```rust
let agent = Agent::with_model("moonshotai/kimi-k2.5")
    .on_code_generated(|e| {
        if let AgentEvent::CodeGenerated { code } = e {
            println!("Code: {}", code);
        }
    })
    .on_code_executed(|e| {
        if let AgentEvent::CodeExecuted { output, success, .. } = e {
            println!("[{}] {}", if *success { "OK" } else { "ERR" }, output);
        }
    })
    .on_finish(|e| {
        if let AgentEvent::Finish { value } = e {
            println!("Done: {:?}", value);
        }
    });
```

### Verbose mode

Built-in logging to stderr:

```rust
let agent = Agent::with_model("moonshotai/kimi-k2.5").verbose(true);
```

### Python callbacks

```python
@agent.on("code_executed")
def on_code(event):
    print(f"[{'OK' if event['success'] else 'ERR'}] {event['output']}")

@agent.on("finish")
def on_finish(event):
    print(f"Done: {event['value']}")
```

### Thinking tags

Extract structured thinking from custom XML tags:

```rust
let config = AgentConfig::new("moonshotai/kimi-k2.5")
    .thinking_tag("intent");

// The agent will extract content from <intent>...</intent> blocks
// and emit Thinking events
```

## Configuration

`AgentConfig` uses a builder pattern:

```rust
let config = AgentConfig::new("moonshotai/kimi-k2.5")
    .max_iterations(10)        // Max code execution iterations (default: 10)
    .temperature(0.7)          // LLM sampling temperature
    .max_tokens(4096)          // Max tokens per LLM response
    .no_max_tokens()           // Remove token limit (use model default)
    .system("You are a helpful research assistant.")  // Custom system description
    .thinking_tag("intent");   // Extract <intent>...</intent> blocks
```

### Python

```python
agent = dragen.Agent(
    "moonshotai/kimi-k2.5",
    max_iterations=10,
    temperature=0.7,
    max_tokens=4096,
    system="You are a helpful research assistant.",
    verbose=True,
    sandbox=sandbox,           # Optional pre-configured sandbox
)
```

## Python API

The Python package provides the same capabilities as Rust with a Pythonic API.

### Sandbox class

```python
sandbox = dragen.Sandbox(builtins=True)  # or builtins=False for minimal sandbox
sandbox.limit(max_instructions=100_000, max_recursion_depth=50)
sandbox.mount("data.json", "./data.json")
sandbox.mount("out.txt", "./out.txt", writable=True)
sandbox.set("x", 42)
sandbox.run("x + 1")                      # => 43
sandbox.files()                            # {"out.txt": "..."}
```

### Tool decorator

```python
@agent.tool
def search(query: str, limit: int = 5) -> list:
    """Search the web for information.

    Args:
        query: The search query
        limit: Maximum number of results
    """
    return [{"title": f"Result for {query}"}]
```

Type hints are extracted from the function signature. Docstrings become tool descriptions. Default values make parameters optional.

### Finish tools

```python
@agent.tool(finish=True)
def submit_report(title: str, content: str, sources: list) -> dict:
    """Submit the final research report."""
    return {"title": title, "content": content, "sources": sources}
```

### Event handling

```python
@agent.on("code_executed")
def handle(event):
    print(event)  # {"code": "...", "output": "...", "success": True}
```

### Parallel execution

```python
results = agent.map(["Task A", "Task B", "Task C"])
```

### Structured output with Pydantic

```python
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    sections: list[str]
    sources: list[str]

result = agent.run("Write a report on AI agents", schema=Report.model_json_schema())
report = Report(**result)
```

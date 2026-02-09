//! Python bindings for dragen - a CodeAct-style AI agent framework.
//!
//! This module provides Python access to dragen agents, allowing
//! LLM-powered code execution with tool registration.

use ::dragen::{
    Agent as RustAgent, AgentConfig as RustAgentConfig, AgentEvent, Context as RustContext,
};
use ::littrs::{Limits, PyValue, Sandbox as RustSandbox, ToolInfo as RustToolInfo};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple};
use pyo3::IntoPy;
use tokio::runtime::Runtime;

// ============================================================================
// PyValue conversion (similar to littrs-python)
// ============================================================================

/// Convert a littrs::PyValue to a Python object.
fn pyvalue_to_py(py: Python<'_>, value: &PyValue) -> PyObject {
    match value {
        PyValue::None => py.None(),
        PyValue::Bool(b) => b.into_py(py),
        PyValue::Int(i) => i.into_py(py),
        PyValue::Float(f) => f.into_py(py),
        PyValue::Str(s) => s.into_py(py),
        PyValue::List(items) => {
            let list: Vec<PyObject> = items.iter().map(|v| pyvalue_to_py(py, v)).collect();
            list.into_py(py)
        }
        PyValue::Tuple(items) => {
            let elements: Vec<PyObject> = items.iter().map(|v| pyvalue_to_py(py, v)).collect();
            PyTuple::new(py, &elements).unwrap().into_py(py)
        }
        PyValue::Dict(pairs) => {
            let dict = PyDict::new(py);
            for (k, v) in pairs {
                dict.set_item(pyvalue_to_py(py, k), pyvalue_to_py(py, v)).unwrap();
            }
            dict.into_py(py)
        }
        PyValue::Set(items) => {
            let set = PySet::new(py, &items.iter().map(|v| pyvalue_to_py(py, v)).collect::<Vec<_>>()).unwrap();
            set.into_py(py)
        }
        _ => py.None(),
    }
}

/// Convert a Python object to a littrs::PyValue.
fn py_to_pyvalue(obj: &Bound<'_, PyAny>) -> PyResult<PyValue> {
    if obj.is_none() {
        Ok(PyValue::None)
    } else if let Ok(b) = obj.downcast::<PyBool>() {
        Ok(PyValue::Bool(b.is_true()))
    } else if let Ok(i) = obj.downcast::<PyInt>() {
        Ok(PyValue::Int(i.extract()?))
    } else if let Ok(f) = obj.downcast::<PyFloat>() {
        Ok(PyValue::Float(f.extract()?))
    } else if let Ok(s) = obj.downcast::<PyString>() {
        Ok(PyValue::Str(s.to_string()))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: PyResult<Vec<_>> = list.iter().map(|item| py_to_pyvalue(&item)).collect();
        Ok(PyValue::List(items?))
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let items: PyResult<Vec<_>> = tuple.iter().map(|item| py_to_pyvalue(&item)).collect();
        Ok(PyValue::Tuple(items?))
    } else if let Ok(set) = obj.downcast::<PySet>() {
        let items: PyResult<Vec<_>> = set.iter().map(|item| py_to_pyvalue(&item)).collect();
        Ok(PyValue::Set(items?))
    } else if let Ok(set) = obj.downcast::<PyFrozenSet>() {
        let items: PyResult<Vec<_>> = set.iter().map(|item| py_to_pyvalue(&item)).collect();
        Ok(PyValue::Set(items?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut pairs = Vec::new();
        for (k, v) in dict.iter() {
            pairs.push((py_to_pyvalue(&k)?, py_to_pyvalue(&v)?));
        }
        Ok(PyValue::Dict(pairs))
    } else {
        Err(PyTypeError::new_err(format!(
            "Cannot convert {} to sandbox value",
            obj.get_type().name()?
        )))
    }
}

/// Convert a serde_json::Value to a Python object.
fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyObject {
    match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_py(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.into_py(py),
        serde_json::Value::Array(arr) => {
            let list: Vec<PyObject> = arr.iter().map(|v| json_to_py(py, v)).collect();
            list.into_py(py)
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)).unwrap();
            }
            dict.into_py(py)
        }
    }
}

/// Convert a Python object to serde_json::Value.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.downcast::<PyBool>() {
        Ok(serde_json::Value::Bool(b.is_true()))
    } else if let Ok(i) = obj.downcast::<PyInt>() {
        let val: i64 = i.extract()?;
        Ok(serde_json::Value::Number(val.into()))
    } else if let Ok(f) = obj.downcast::<PyFloat>() {
        let val: f64 = f.extract()?;
        Ok(serde_json::json!(val))
    } else if let Ok(s) = obj.downcast::<PyString>() {
        Ok(serde_json::Value::String(s.to_string()))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: PyResult<Vec<_>> = list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(items?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        // Try to convert to string as fallback
        let s = obj.str()?.to_string();
        Ok(serde_json::Value::String(s))
    }
}

/// Helper to create a PyValue error dict.
fn pyvalue_error_dict(message: String) -> PyValue {
    PyValue::Dict(vec![(
        PyValue::Str("error".to_string()),
        PyValue::Str(message),
    )])
}

// ============================================================================
// ToolInfo wrapper
// ============================================================================

/// Tool information for registration with type validation.
///
/// Example:
///     >>> info = ToolInfo("search", "Search the web")
///     >>> info = info.arg("query", "str", "The search query")
///     >>> info = info.returns("list")
#[pyclass]
#[derive(Clone)]
struct ToolInfo {
    inner: RustToolInfo,
}

#[pymethods]
impl ToolInfo {
    /// Create a new tool info.
    ///
    /// Args:
    ///     name: The tool name (will be the function name in Python sandbox)
    ///     description: Description of what the tool does
    #[new]
    fn new(name: &str, description: &str) -> Self {
        Self {
            inner: RustToolInfo::new(name, description),
        }
    }

    /// Add a required argument.
    ///
    /// Args:
    ///     name: Argument name
    ///     python_type: Type hint (str, int, float, bool, list, dict, any)
    ///     description: Description of the argument
    fn arg_required(&self, name: &str, python_type: &str, description: &str) -> Self {
        Self {
            inner: self.inner.clone().arg(name, python_type, description),
        }
    }

    /// Add an optional argument.
    ///
    /// Args:
    ///     name: Argument name
    ///     python_type: Type hint (str, int, float, bool, list, dict, any)
    ///     description: Description of the argument
    fn arg_optional(&self, name: &str, python_type: &str, description: &str) -> Self {
        Self {
            inner: self.inner.clone().arg_opt(name, python_type, description),
        }
    }

    /// Set the return type.
    fn returns(&self, python_type: &str) -> Self {
        Self {
            inner: self.inner.clone().returns(python_type),
        }
    }
}

// ============================================================================
// AgentConfig wrapper
// ============================================================================

/// Configuration for a CodeAct agent.
///
/// Example:
///     >>> config = AgentConfig("gpt-4o")
///     >>> config = config.max_iterations(10).temperature(0.7)
#[pyclass]
#[derive(Clone)]
struct AgentConfig {
    inner: RustAgentConfig,
}

#[pymethods]
impl AgentConfig {
    /// Create a new configuration with the specified model.
    ///
    /// Args:
    ///     model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
    #[new]
    fn new(model: &str) -> Self {
        Self {
            inner: RustAgentConfig::new(model),
        }
    }

    /// Set the maximum number of iterations (code executions).
    fn max_iterations(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().max_iterations(n),
        }
    }

    /// Set the temperature for LLM sampling.
    fn temperature(&self, t: f32) -> Self {
        Self {
            inner: self.inner.clone().temperature(t),
        }
    }

    /// Set the maximum tokens for LLM response.
    fn max_tokens(&self, n: u32) -> Self {
        Self {
            inner: self.inner.clone().max_tokens(n),
        }
    }

    /// Remove the max tokens limit.
    fn no_max_tokens(&self) -> Self {
        Self {
            inner: self.inner.clone().no_max_tokens(),
        }
    }

    /// Set a custom system description.
    fn system(&self, system: &str) -> Self {
        Self {
            inner: self.inner.clone().system(system),
        }
    }
}

// ============================================================================
// Sandbox wrapper
// ============================================================================

/// A configurable Python sandbox for executing code securely.
///
/// Create and configure a sandbox, then pass it to an Agent.
///
/// Example:
///     >>> sandbox = Sandbox(builtins=True)
///     >>> sandbox.limit(max_instructions=100000, max_recursion_depth=50)
///     >>> sandbox.mount("data.json", "./data/input.json")
///     >>> sandbox.set("items", [1, 2, 3])
///     >>>
///     >>> agent = Agent("gpt-4o", sandbox=sandbox)
#[pyclass]
#[derive(Clone)]
struct Sandbox {
    inner: RustSandbox,
}

#[pymethods]
impl Sandbox {
    /// Create a new sandbox.
    ///
    /// Args:
    ///     builtins: If True, enable built-in modules (json, math, typing). Default: True.
    #[new]
    #[pyo3(signature = (builtins=true))]
    fn new(builtins: bool) -> Self {
        Self {
            inner: if builtins {
                RustSandbox::with_builtins()
            } else {
                RustSandbox::new()
            },
        }
    }

    /// Set resource limits for sandbox execution.
    ///
    /// Args:
    ///     max_instructions: Maximum bytecode instructions per execution (None for unlimited)
    ///     max_recursion_depth: Maximum call stack depth (None for unlimited)
    #[pyo3(signature = (max_instructions=None, max_recursion_depth=None))]
    fn limit(&mut self, max_instructions: Option<u64>, max_recursion_depth: Option<usize>) {
        self.inner.limit(Limits {
            max_instructions,
            max_recursion_depth,
        });
    }

    /// Mount a file into the sandbox's virtual filesystem.
    ///
    /// Args:
    ///     virtual_path: The path visible to sandbox code
    ///     host_path: The actual file path on the host
    ///     writable: If True, sandbox code can write to this file. Default: False.
    #[pyo3(signature = (virtual_path, host_path, writable=false))]
    fn mount(&mut self, virtual_path: &str, host_path: &str, writable: bool) {
        self.inner.mount(virtual_path, host_path, writable);
    }

    /// Set a variable in the sandbox's global scope.
    ///
    /// Args:
    ///     name: Variable name
    ///     value: The value to set
    fn set(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let pyvalue = py_to_pyvalue(value)?;
        self.inner.set(name, pyvalue);
        Ok(())
    }

    /// Get the contents of all writable mounted files.
    ///
    /// Returns:
    ///     Dict mapping virtual paths to file contents
    fn files(&self, py: Python<'_>) -> PyObject {
        let files = self.inner.files();
        let dict = PyDict::new(py);
        for (path, content) in files {
            dict.set_item(path, content).unwrap();
        }
        dict.into_py(py)
    }

    /// Run Python code directly in the sandbox.
    ///
    /// Returns:
    ///     The result of the last expression
    fn run(&mut self, py: Python<'_>, code: &str) -> PyResult<PyObject> {
        match self.inner.run(code) {
            Ok(value) => Ok(pyvalue_to_py(py, &value)),
            Err(e) => Err(PyRuntimeError::new_err(format!("{}", e))),
        }
    }
}

// ============================================================================
// Context wrapper
// ============================================================================

/// Shared context for passing data between agents.
///
/// Context is a thread-safe key-value store that allows agents to share data.
///
/// Example:
///     >>> ctx = Context()
///     >>> ctx.set("plan", {"title": "My Plan", "steps": [1, 2, 3]})
///     >>> ctx.get("plan")
///     {'title': 'My Plan', 'steps': [1, 2, 3]}
#[pyclass]
#[derive(Clone)]
struct Context {
    inner: RustContext,
}

#[pymethods]
impl Context {
    /// Create a new empty context.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustContext::new(),
        }
    }

    /// Store a value in the context.
    ///
    /// Args:
    ///     key: The key to store under
    ///     value: The value to store (any JSON-serializable Python object)
    fn set(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let json_value = py_to_json(value)?;
        self.inner.set(key, &json_value);
        Ok(())
    }

    /// Retrieve a value from the context.
    ///
    /// Args:
    ///     key: The key to retrieve
    ///
    /// Returns:
    ///     The stored value, or None if key doesn't exist
    fn get(&self, py: Python<'_>, key: &str) -> PyObject {
        match self.inner.get_raw(key) {
            Some(value) => json_to_py(py, &value),
            None => py.None(),
        }
    }

    /// Check if a key exists in the context.
    fn contains(&self, key: &str) -> bool {
        self.inner.contains(key)
    }

    /// Remove a value from the context.
    fn remove(&self, py: Python<'_>, key: &str) -> PyObject {
        match self.inner.remove(key) {
            Some(value) => json_to_py(py, &value),
            None => py.None(),
        }
    }

    /// Get all keys in the context.
    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    /// Clear all data from the context.
    fn clear(&self) {
        self.inner.clear();
    }
}

// ============================================================================
// Agent wrapper
// ============================================================================

/// A CodeAct-style agent that executes Python code in a sandbox.
///
/// The agent uses an LLM to generate Python code which is executed in a
/// secure sandbox. Tools are exposed as Python functions.
///
/// Example:
///     >>> agent = Agent("gpt-4o", max_iterations=10, temperature=0.7)
///     >>>
///     >>> def search(args):
///     ...     query = args[0]
///     ...     return {"results": [f"Found: {query}"]}
///     >>>
///     >>> agent.register_function("search", search)
///     >>> result = agent.run("Search for Python tutorials")
///
/// Helper to get event type string from AgentEvent
fn event_type_for_event(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::IterationStart { .. } => "iteration_start",
        AgentEvent::LLMRequest { .. } => "llm_request",
        AgentEvent::LLMResponse { .. } => "llm_response",
        AgentEvent::Thinking { .. } => "thinking",
        AgentEvent::CodeGenerated { .. } => "code_generated",
        AgentEvent::CodeExecuted { .. } => "code_executed",
        AgentEvent::ToolCall { .. } => "tool_call",
        AgentEvent::ToolResult { .. } => "tool_result",
        AgentEvent::Finish { .. } => "finish",
        AgentEvent::Error { .. } => "error",
    }
}

/// Note: Agent is not thread-safe and must be used from a single thread.
#[pyclass(unsendable)]
struct Agent {
    inner: RustAgent,
    runtime: Runtime,
    context: Option<Context>,
    context_reads: Vec<String>,
    context_write: Option<String>,
}

#[pymethods]
impl Agent {
    /// Create a new agent.
    ///
    /// Args:
    ///     model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
    ///     max_iterations: Maximum number of code execution iterations (default: 10)
    ///     temperature: Temperature for LLM sampling (default: 0.7)
    ///     max_tokens: Maximum tokens for LLM response (default: 4096, None for unlimited)
    ///     system: Custom system description
    ///     sandbox: Pre-configured Sandbox instance. If not provided, creates one with builtins.
    #[new]
    #[pyo3(signature = (model, max_iterations=None, temperature=None, max_tokens=None, system=None, verbose=None, sandbox=None))]
    fn new(
        model: &str,
        max_iterations: Option<usize>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        system: Option<&str>,
        verbose: Option<bool>,
        sandbox: Option<Sandbox>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let mut config = RustAgentConfig::new(model);
        if let Some(n) = max_iterations {
            config = config.max_iterations(n);
        }
        if let Some(t) = temperature {
            config = config.temperature(t);
        }
        if let Some(n) = max_tokens {
            config = config.max_tokens(n);
        }
        if let Some(s) = system {
            config = config.system(s);
        }

        let mut agent = match sandbox {
            Some(sb) => RustAgent::with_sandbox(sb.inner, config),
            None => RustAgent::new(config),
        };
        if verbose.unwrap_or(false) {
            agent = agent.verbose(true);
        }

        Ok(Self {
            inner: agent,
            runtime,
            context: None,
            context_reads: Vec::new(),
            context_write: None,
        })
    }

    /// Register a callback for agent events.
    ///
    /// Use as a decorator to handle specific events during agent execution.
    /// Callbacks fire in real-time as events occur during run().
    ///
    /// Event types: iteration_start, llm_request, llm_response, code_generated,
    ///              code_executed, tool_call, tool_result, finish, error
    ///
    /// Example:
    ///     >>> agent = Agent("opus")
    ///     >>>
    ///     >>> @agent.on("tool_call")
    ///     ... def log_tool(event):
    ///     ...     print(f"Tool: {event['name']}({event['args']})")
    ///     >>>
    ///     >>> @agent.on("code_executed")
    ///     ... def log_code(event):
    ///     ...     print(f"Code: {event['code'][:50]}...")
    ///     >>>
    ///     >>> result = agent.run("Search for Python tutorials")
    #[pyo3(signature = (event_type))]
    fn on(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        event_type: String,
    ) -> PyResult<PyObject> {
        let event_type_clone = event_type.clone();

        // Return a decorator function
        let decorator = PyModule::from_code(
            py,
            c"
def make_decorator(agent, event_type):
    def decorator(func):
        agent._register_callback(event_type, func)
        return func
    return decorator
",
            c"_dragen_on_decorator",
            c"_dragen_on_decorator",
        )?;

        let make_decorator = decorator.getattr("make_decorator")?;
        let result = make_decorator.call1((slf.into_pyobject(py)?, event_type_clone))?;
        Ok(result.into_py(py))
    }

    /// Internal method to register a callback (called by decorator).
    /// This registers a Rust callback that calls Python via with_gil().
    #[pyo3(name = "_register_callback")]
    fn register_callback(
        &mut self,
        py: Python<'_>,
        event_type: String,
        func: PyObject,
    ) {
        let func = func.clone_ref(py);

        // Create a Rust callback that calls Python via with_gil
        let rust_callback = move |event: &AgentEvent| {
            Python::with_gil(|py| {
                // Build event dict
                let dict = PyDict::new(py);
                let _ = dict.set_item("type", event_type_for_event(event));

                match event {
                    AgentEvent::IterationStart { iteration, max_iterations } => {
                        let _ = dict.set_item("iteration", *iteration);
                        let _ = dict.set_item("max_iterations", *max_iterations);
                    }
                    AgentEvent::LLMRequest { message_count } => {
                        let _ = dict.set_item("message_count", *message_count);
                    }
                    AgentEvent::LLMResponse { content, tokens_used } => {
                        let _ = dict.set_item("content", content.clone());
                        let _ = dict.set_item("tokens_used", *tokens_used);
                    }
                    AgentEvent::Thinking { content } => {
                        let _ = dict.set_item("content", content.clone());
                    }
                    AgentEvent::CodeGenerated { code } => {
                        let _ = dict.set_item("code", code.clone());
                    }
                    AgentEvent::CodeExecuted { code, output, success } => {
                        let _ = dict.set_item("code", code.clone());
                        let _ = dict.set_item("output", output.clone());
                        let _ = dict.set_item("success", *success);
                    }
                    AgentEvent::ToolCall { name, args } => {
                        let _ = dict.set_item("name", name.clone());
                        let py_args: Vec<PyObject> = args.iter().map(|v| pyvalue_to_py(py, v)).collect();
                        let _ = dict.set_item("args", py_args);
                    }
                    AgentEvent::ToolResult { name, result } => {
                        let _ = dict.set_item("name", name.clone());
                        let _ = dict.set_item("result", pyvalue_to_py(py, result));
                    }
                    AgentEvent::Finish { value } => {
                        let _ = dict.set_item("value", pyvalue_to_py(py, value));
                    }
                    AgentEvent::Error { message } => {
                        let _ = dict.set_item("message", message.clone());
                    }
                }

                // Call the Python callback, ignoring errors
                let _ = func.call1(py, (dict,));
            });
        };

        // Register the Rust callback with the appropriate event type
        // We need to swap out the inner agent since the builder methods consume self
        let inner = std::mem::replace(&mut self.inner, RustAgent::with_model("temp"));
        self.inner = match event_type.as_str() {
            "iteration_start" => inner.on_iteration_start(rust_callback),
            "llm_request" => inner.on_llm_request(rust_callback),
            "llm_response" => inner.on_llm_response(rust_callback),
            "thinking" => inner.on_thinking(rust_callback),
            "code_generated" => inner.on_code_generated(rust_callback),
            "code_executed" => inner.on_code_executed(rust_callback),
            "tool_call" => inner.on_tool_call(rust_callback),
            "tool_result" => inner.on_tool_result(rust_callback),
            "finish" => inner.on_finish(rust_callback),
            "error" => inner.on_error(rust_callback),
            _ => inner.on_event(rust_callback), // Unknown event type -> catch-all
        };
    }

    /// Register a Python callable as a tool in the agent's sandbox.
    ///
    /// The callable receives a list of arguments and should return a value.
    ///
    /// Args:
    ///     name: The function name in the sandbox
    ///     func: A Python callable that takes a list of arguments
    ///
    /// Example:
    ///     >>> def add(args):
    ///     ...     return args[0] + args[1]
    ///     >>> agent.register_function("add", add)
    fn register_function(&mut self, py: Python<'_>, name: &str, func: PyObject) -> PyResult<()> {
        if !func.bind(py).is_callable() {
            return Err(PyTypeError::new_err("func must be callable"));
        }

        let func = func.clone_ref(py);
        self.inner.sandbox_mut().register_fn(name, move |args: Vec<PyValue>| {
            Python::with_gil(|py| {
                let py_args: Vec<PyObject> = args.iter().map(|v| pyvalue_to_py(py, v)).collect();
                match func.call1(py, (py_args,)) {
                    Ok(result) => py_to_pyvalue(result.bind(py)).unwrap_or(PyValue::None),
                    Err(e) => pyvalue_error_dict(format!("{}", e)),
                }
            })
        });

        Ok(())
    }

    /// Register a tool with explicit type information.
    ///
    /// Args:
    ///     info: Tool information (name, description, arguments)
    ///     func: A Python callable that takes a list of arguments
    fn register_tool(&mut self, py: Python<'_>, info: &ToolInfo, func: PyObject) -> PyResult<()> {
        if !func.bind(py).is_callable() {
            return Err(PyTypeError::new_err("func must be callable"));
        }

        let func = func.clone_ref(py);
        self.inner.register_tool(info.inner.clone(), move |args: Vec<PyValue>| {
            Python::with_gil(|py| {
                let py_args: Vec<PyObject> = args.iter().map(|v| pyvalue_to_py(py, v)).collect();
                match func.call1(py, (py_args,)) {
                    Ok(result) => py_to_pyvalue(result.bind(py)).unwrap_or(PyValue::None),
                    Err(e) => pyvalue_error_dict(format!("{}", e)),
                }
            })
        });

        Ok(())
    }

    /// Register a custom finish tool with specified arguments.
    ///
    /// Args:
    ///     info: Tool information for the finish tool
    ///     func: A Python callable that processes finish arguments
    fn register_finish(&mut self, py: Python<'_>, info: &ToolInfo, func: PyObject) -> PyResult<()> {
        if !func.bind(py).is_callable() {
            return Err(PyTypeError::new_err("func must be callable"));
        }

        let func = func.clone_ref(py);
        self.inner.register_finish(info.inner.clone(), move |args: Vec<PyValue>| {
            Python::with_gil(|py| {
                let py_args: Vec<PyObject> = args.iter().map(|v| pyvalue_to_py(py, v)).collect();
                match func.call1(py, (py_args,)) {
                    Ok(result) => py_to_pyvalue(result.bind(py)).unwrap_or(PyValue::None),
                    Err(e) => pyvalue_error_dict(format!("{}", e)),
                }
            })
        });

        Ok(())
    }

    /// Read data from a shared context and inject it into the agent's prompt.
    ///
    /// Args:
    ///     ctx: The shared context
    ///     key: The key to read from
    fn from_context(&mut self, ctx: &Context, key: &str) {
        self.inner = std::mem::replace(&mut self.inner, RustAgent::with_model("temp"))
            .from_context(&ctx.inner, key);
        if self.context.is_none() {
            self.context = Some(ctx.clone());
        }
        self.context_reads.push(key.to_string());
    }

    /// Save the agent's output to a shared context.
    ///
    /// Args:
    ///     ctx: The shared context
    ///     key: The key to write to
    fn to_context(&mut self, ctx: &Context, key: &str) {
        self.inner = std::mem::replace(&mut self.inner, RustAgent::with_model("temp"))
            .to_context(&ctx.inner, key);
        if self.context.is_none() {
            self.context = Some(ctx.clone());
        }
        self.context_write = Some(key.to_string());
    }

    /// Run the agent on a task.
    ///
    /// This method blocks until the agent completes or reaches max iterations.
    /// Callbacks registered via @agent.on() fire in real-time during execution.
    ///
    /// Args:
    ///     task: The task description for the agent
    ///     schema: Optional JSON Schema dict for validating finish() output.
    ///             If validation fails, error is sent to LLM for self-correction.
    ///             Works with Pydantic: `schema=MyModel.model_json_schema()`
    ///
    /// Returns:
    ///     The result as a Python object (dict, list, string, etc.)
    ///
    /// Raises:
    ///     RuntimeError: If the agent fails (LLM error, sandbox error, etc.)
    ///     ValueError: If the result cannot be deserialized or schema validation fails
    ///
    /// Example:
    ///     >>> from pydantic import BaseModel
    ///     >>> class Output(BaseModel):
    ///     ...     content: str
    ///     ...     sources: list[str]
    ///     >>> result = agent.run(task, schema=Output.model_json_schema())
    #[pyo3(signature = (task, schema=None))]
    fn run(&mut self, py: Python<'_>, task: &str, schema: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        // Set schema if provided
        if let Some(schema_obj) = schema {
            let schema_json = py_to_json(schema_obj)?;
            self.inner.set_schema(schema_json);
        } else {
            self.inner.clear_schema();
        }

        // Release GIL while running async code
        // Callbacks registered via @agent.on() use Python::with_gil() to re-acquire
        // the GIL and call Python functions during execution
        let result = py.allow_threads(|| {
            self.runtime.block_on(async {
                self.inner.run::<serde_json::Value>(task).await
            })
        });

        match result {
            Ok(value) => Ok(json_to_py(py, &value)),
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("MaxIterations") {
                    Err(PyRuntimeError::new_err(format!("Agent reached maximum iterations: {}", e)))
                } else if err_str.contains("Deserialization") || err_str.contains("Schema validation") {
                    Err(PyValueError::new_err(format!("Failed to parse result: {}", e)))
                } else {
                    Err(PyRuntimeError::new_err(format!("Agent error: {}", e)))
                }
            }
        }
    }

    /// Get the conversation history.
    ///
    /// Returns:
    ///     List of message dicts with 'role' and 'content' keys
    fn messages(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let messages = self.inner.messages();
        let mut result = Vec::new();

        for msg in messages {
            let dict = PyDict::new(py);
            dict.set_item("role", format!("{:?}", msg.role).to_lowercase())?;
            dict.set_item("content", &msg.content)?;
            result.push(dict.into_pyobject(py).unwrap().into_any().unbind());
        }

        Ok(result)
    }

    /// Get the structured finish value (if finish() was called).
    ///
    /// Returns:
    ///     The finish value as a Python object, or None
    fn finish_value(&self, py: Python<'_>) -> PyObject {
        match self.inner.finish_value() {
            Some(value) => pyvalue_to_py(py, &value),
            None => py.None(),
        }
    }

    /// Get tool documentation for the registered tools.
    fn describe_tools(&self) -> String {
        self.inner.sandbox().describe()
    }

    /// Decorator to register a function as a tool.
    ///
    /// Automatically extracts tool info from type hints and docstring.
    ///
    /// Example:
    ///     >>> @agent.tool
    ///     ... def search(query: str, limit: int = 5) -> list:
    ///     ...     """Search the web for information."""
    ///     ...     return [{"result": query}]
    ///
    /// Or with explicit name:
    ///     >>> @agent.tool(name="web_search")
    ///     ... def search(query: str) -> list:
    ///     ...     """Search the web."""
    ///     ...     return []
    ///
    /// For custom finish tools, use finish=True:
    ///     >>> @agent.tool(finish=True)
    ///     ... def finish(result: dict) -> dict:
    ///     ...     """Custom finish with validation."""
    ///     ...     if not result.get("sources"):
    ///     ...         raise ValueError("No sources collected!")
    ///     ...     return result
    #[pyo3(signature = (func=None, *, name=None, finish=false))]
    fn tool(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        func: Option<PyObject>,
        name: Option<String>,
        finish: bool,
    ) -> PyResult<PyObject> {
        // If func is provided, we're being called as @agent.tool (no parens)
        // If func is None, we're being called as @agent.tool() or @agent.tool(name="...")
        match func {
            Some(f) => {
                // Direct decoration: @agent.tool
                if finish {
                    register_finish_from_function(&mut slf, py, f.clone_ref(py), name)?;
                } else {
                    register_tool_from_function(&mut slf, py, f.clone_ref(py), name)?;
                }
                Ok(f)
            }
            None => {
                // Parameterized decoration: @agent.tool() or @agent.tool(name="...", finish=True)
                // Return a decorator function
                let name_clone = name.clone();

                // We need to create a Python function that will be called with the actual function
                // Use a closure captured in a Python lambda
                let decorator = PyModule::from_code(
                    py,
                    c"
def make_decorator(agent, name, is_finish):
    def decorator(func):
        if is_finish:
            agent._register_finish_from_func(func, name)
        else:
            agent._register_tool_from_func(func, name)
        return func
    return decorator
",
                    c"_dragen_decorator",
                    c"_dragen_decorator",
                )?;

                let make_decorator = decorator.getattr("make_decorator")?;
                let result = make_decorator.call1((slf.into_pyobject(py)?, name_clone, finish))?;
                Ok(result.into_py(py))
            }
        }
    }

    /// Internal method to register a tool from a function (called by decorator).
    #[pyo3(name = "_register_tool_from_func", signature = (func, name=None))]
    fn register_tool_from_func(
        &mut self,
        py: Python<'_>,
        func: PyObject,
        name: Option<String>,
    ) -> PyResult<()> {
        register_tool_from_function(self, py, func, name)
    }

    /// Internal method to register a finish tool from a function (called by decorator).
    #[pyo3(name = "_register_finish_from_func", signature = (func, name=None))]
    fn register_finish_from_func(
        &mut self,
        py: Python<'_>,
        func: PyObject,
        name: Option<String>,
    ) -> PyResult<()> {
        register_finish_from_function(self, py, func, name)
    }

    /// Run multiple tasks in parallel, each with a fresh agent clone.
    ///
    /// Creates independent agent instances with the same configuration and tools,
    /// runs each task in parallel, and returns results in order. Individual task
    /// failures do not affect other tasks.
    ///
    /// Args:
    ///     tasks: List of task strings to run
    ///     schema: Optional JSON Schema dict for validating finish() output (applied to all tasks)
    ///
    /// Returns:
    ///     List of results in the same order as input tasks. Failed tasks return
    ///     a dict with an "error" key containing the error message.
    ///
    /// Example:
    ///     >>> agent = Agent("gpt-4o")
    ///     >>> @agent.tool
    ///     ... def search(query: str) -> list:
    ///     ...     return [{"result": query}]
    ///     >>>
    ///     >>> results = agent.map([
    ///     ...     "Write about topic A",
    ///     ...     "Write about topic B",
    ///     ...     "Write about topic C",
    ///     ... ])
    ///     >>> for i, result in enumerate(results):
    ///     ...     if "error" in result:
    ///     ...         print(f"Task {i} failed: {result['error']}")
    ///     ...     else:
    ///     ...         print(f"Task {i} succeeded")
    #[pyo3(signature = (tasks, schema=None))]
    fn map(&mut self, py: Python<'_>, tasks: Vec<String>, schema: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        // Set schema if provided (will be inherited by cloned agents)
        if let Some(schema_obj) = schema {
            let schema_json = py_to_json(schema_obj)?;
            self.inner.set_schema(schema_json);
        } else {
            self.inner.clear_schema();
        }

        // Use Rust-side parallel execution - returns Vec<Result<T>>
        let results = py.allow_threads(|| {
            self.runtime.block_on(async {
                self.inner.map::<serde_json::Value>(tasks).await
            })
        });

        // Convert each result - Ok becomes the value, Err becomes {"error": "..."}
        let py_results: Vec<PyObject> = results
            .into_iter()
            .map(|r| match r {
                Ok(value) => json_to_py(py, &value),
                Err(e) => {
                    let dict = PyDict::new(py);
                    dict.set_item("error", format!("{}", e)).unwrap();
                    dict.into_py(py)
                }
            })
            .collect();

        Ok(py_results.into_py(py))
    }

    /// Set a JSON Schema for validating finish() output.
    ///
    /// This is useful when you want to set the schema once and run multiple tasks,
    /// or when using map() for parallel execution.
    ///
    /// Args:
    ///     schema: JSON Schema dict (e.g., from Pydantic's model_json_schema())
    fn set_schema(&mut self, schema: &Bound<'_, PyAny>) -> PyResult<()> {
        let schema_json = py_to_json(schema)?;
        self.inner.set_schema(schema_json);
        Ok(())
    }

    /// Clear the schema validation.
    fn clear_schema(&mut self) {
        self.inner.clear_schema();
    }
}

// ============================================================================
// Tool registration helper
// ============================================================================

/// Register a tool by introspecting a Python function's signature and docstring.
fn register_tool_from_function(
    agent: &mut Agent,
    py: Python<'_>,
    func: PyObject,
    override_name: Option<String>,
) -> PyResult<()> {
    let func_bound = func.bind(py);

    if !func_bound.is_callable() {
        return Err(PyTypeError::new_err("func must be callable"));
    }

    // Get function name
    let func_name: String = match override_name {
        Some(n) => n,
        None => func_bound
            .getattr("__name__")
            .map(|n| n.extract::<String>().unwrap_or_else(|_| "tool".to_string()))
            .unwrap_or_else(|_| "tool".to_string()),
    };

    // Get docstring for description
    let description: String = func_bound
        .getattr("__doc__")
        .ok()
        .and_then(|doc| {
            if doc.is_none() {
                None
            } else {
                doc.extract::<String>().ok()
            }
        })
        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
        .unwrap_or_else(|| format!("{} tool", func_name));

    // Use inspect module to get signature
    let inspect = py.import("inspect")?;
    let signature = inspect.call_method1("signature", (&func,))?;
    let parameters = signature.getattr("parameters")?;

    // Build ToolInfo
    let mut tool_info = RustToolInfo::new(&func_name, &description);

    // Track parameter names for wrapper
    let mut param_names: Vec<String> = Vec::new();
    let mut param_defaults: Vec<Option<PyObject>> = Vec::new();

    // Iterate over parameters
    let items = parameters.call_method0("items")?;
    for item in items.iter()? {
        let item = item?;
        let tuple = item.downcast::<pyo3::types::PyTuple>()?;
        let param_name: String = tuple.get_item(0)?.extract()?;
        let param = tuple.get_item(1)?;

        // Skip *args and **kwargs
        let kind: i32 = param.getattr("kind")?.extract()?;
        if kind == 2 || kind == 4 {
            // VAR_POSITIONAL or VAR_KEYWORD
            continue;
        }

        param_names.push(param_name.clone());

        // Get type annotation
        let annotation = param.getattr("annotation")?;
        let inspect_empty = inspect.getattr("Parameter")?.getattr("empty")?;
        let type_str = if annotation.is(&inspect_empty) {
            "any".to_string()
        } else {
            // Try to get __name__ or use str()
            annotation
                .getattr("__name__")
                .map(|n| n.extract::<String>().unwrap_or_else(|_| "any".to_string()))
                .unwrap_or_else(|_| {
                    annotation
                        .str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| "any".to_string())
                })
        };

        // Check if has default
        let default = param.getattr("default")?;
        let has_default = !default.is(&inspect_empty);

        if has_default {
            tool_info = tool_info.arg_opt(&param_name, &type_str, "");
            param_defaults.push(Some(default.into_py(py)));
        } else {
            tool_info = tool_info.arg(&param_name, &type_str, "");
            param_defaults.push(None);
        }
    }

    // Get return type
    let return_annotation = signature.getattr("return_annotation")?;
    let inspect_empty = inspect.getattr("Signature")?.getattr("empty")?;
    if !return_annotation.is(&inspect_empty) {
        let return_type = return_annotation
            .getattr("__name__")
            .map(|n| n.extract::<String>().unwrap_or_else(|_| "any".to_string()))
            .unwrap_or_else(|_| "any".to_string());
        tool_info = tool_info.returns(&return_type);
    }

    // Create wrapper that converts positional args to kwargs
    let func_clone = func.clone_ref(py);
    let param_names_clone = param_names.clone();
    let param_defaults_clone: Vec<Option<serde_json::Value>> = param_defaults
        .iter()
        .map(|d| d.as_ref().and_then(|obj| py_to_json(obj.bind(py)).ok()))
        .collect();

    agent.inner.register_tool(tool_info, move |args: Vec<PyValue>| {
        Python::with_gil(|py| {
            // Build kwargs dict
            let kwargs = PyDict::new(py);

            for (i, name) in param_names_clone.iter().enumerate() {
                if i < args.len() {
                    // Use provided argument
                    kwargs.set_item(name, pyvalue_to_py(py, &args[i])).unwrap();
                } else if let Some(Some(default)) = param_defaults_clone.get(i) {
                    // Use default value
                    kwargs.set_item(name, json_to_py(py, default)).unwrap();
                }
            }

            // Call with kwargs
            match func_clone.call(py, (), Some(&kwargs)) {
                Ok(result) => py_to_pyvalue(result.bind(py)).unwrap_or(PyValue::None),
                Err(e) => pyvalue_error_dict(format!("{}", e)),
            }
        })
    });

    Ok(())
}

/// Register a finish tool by introspecting a Python function's signature and docstring.
/// Similar to register_tool_from_function but registers as the finish tool.
fn register_finish_from_function(
    agent: &mut Agent,
    py: Python<'_>,
    func: PyObject,
    override_name: Option<String>,
) -> PyResult<()> {
    let func_bound = func.bind(py);

    if !func_bound.is_callable() {
        return Err(PyTypeError::new_err("func must be callable"));
    }

    // Get function name (default to "finish")
    let func_name: String = match override_name {
        Some(n) => n,
        None => func_bound
            .getattr("__name__")
            .map(|n| n.extract::<String>().unwrap_or_else(|_| "finish".to_string()))
            .unwrap_or_else(|_| "finish".to_string()),
    };

    // Get docstring for description
    let description: String = func_bound
        .getattr("__doc__")
        .ok()
        .and_then(|doc| {
            if doc.is_none() {
                None
            } else {
                doc.extract::<String>().ok()
            }
        })
        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
        .unwrap_or_else(|| "Complete the task with the final result".to_string());

    // Use inspect module to get signature
    let inspect = py.import("inspect")?;
    let signature = inspect.call_method1("signature", (&func,))?;
    let parameters = signature.getattr("parameters")?;

    // Build ToolInfo
    let mut tool_info = RustToolInfo::new(&func_name, &description);

    // Track parameter names for wrapper
    let mut param_names: Vec<String> = Vec::new();
    let mut param_defaults: Vec<Option<PyObject>> = Vec::new();

    // Iterate over parameters
    let items = parameters.call_method0("items")?;
    for item in items.iter()? {
        let item = item?;
        let tuple = item.downcast::<pyo3::types::PyTuple>()?;
        let param_name: String = tuple.get_item(0)?.extract()?;
        let param = tuple.get_item(1)?;

        // Skip *args and **kwargs
        let kind: i32 = param.getattr("kind")?.extract()?;
        if kind == 2 || kind == 4 {
            continue;
        }

        param_names.push(param_name.clone());

        // Get type annotation
        let annotation = param.getattr("annotation")?;
        let inspect_empty = inspect.getattr("Parameter")?.getattr("empty")?;
        let type_str = if annotation.is(&inspect_empty) {
            "any".to_string()
        } else {
            annotation
                .getattr("__name__")
                .map(|n| n.extract::<String>().unwrap_or_else(|_| "any".to_string()))
                .unwrap_or_else(|_| {
                    annotation
                        .str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| "any".to_string())
                })
        };

        // Check if has default
        let default = param.getattr("default")?;
        let has_default = !default.is(&inspect_empty);

        if has_default {
            tool_info = tool_info.arg_opt(&param_name, &type_str, "");
            param_defaults.push(Some(default.into_py(py)));
        } else {
            tool_info = tool_info.arg(&param_name, &type_str, "");
            param_defaults.push(None);
        }
    }

    // Create wrapper that converts positional args to kwargs
    let func_clone = func.clone_ref(py);
    let param_names_clone = param_names.clone();
    let param_defaults_clone: Vec<Option<serde_json::Value>> = param_defaults
        .iter()
        .map(|d| d.as_ref().and_then(|obj| py_to_json(obj.bind(py)).ok()))
        .collect();

    // Register as finish tool instead of regular tool
    agent.inner.register_finish(tool_info, move |args: Vec<PyValue>| {
        Python::with_gil(|py| {
            // Build kwargs dict
            let kwargs = PyDict::new(py);

            for (i, name) in param_names_clone.iter().enumerate() {
                if i < args.len() {
                    kwargs.set_item(name, pyvalue_to_py(py, &args[i])).unwrap();
                } else if let Some(Some(default)) = param_defaults_clone.get(i) {
                    kwargs.set_item(name, json_to_py(py, default)).unwrap();
                }
            }

            // Call with kwargs - if it raises an exception, return error dict
            match func_clone.call(py, (), Some(&kwargs)) {
                Ok(result) => py_to_pyvalue(result.bind(py)).unwrap_or(PyValue::None),
                Err(e) => pyvalue_error_dict(format!("{}", e)),
            }
        })
    });

    Ok(())
}

// ============================================================================
// Module definition
// ============================================================================

/// dragen - CodeAct-style AI agent framework for Python.
///
/// This module provides tools for building AI agents that execute Python code
/// in a secure sandbox.
///
/// Example:
///     >>> from dragen import Agent, AgentConfig, Context, Sandbox, ToolInfo
///     >>>
///     >>> # Create a configured sandbox
///     >>> sandbox = Sandbox(builtins=True)
///     >>> sandbox.limit(max_instructions=100000)
///     >>>
///     >>> # Create an agent with the sandbox
///     >>> agent = Agent("gpt-4o", sandbox=sandbox)
///     >>>
///     >>> # Register a tool
///     >>> @agent.tool
///     ... def search(query: str) -> list:
///     ...     """Search the web."""
///     ...     return [{"result": query}]
///     >>>
///     >>> # Run a task
///     >>> result = agent.run("Search for Python tutorials")
///     >>> print(result)
///
/// Multi-agent with shared context:
///     >>> ctx = Context()
///     >>>
///     >>> # Planner writes to context
///     >>> planner = Agent("gpt-4o")
///     >>> planner = planner.to_context(ctx, "plan")
///     >>> planner.run("Create a research plan")
///     >>>
///     >>> # Executor reads from context
///     >>> executor = Agent("gpt-4o")
///     >>> executor = executor.from_context(ctx, "plan")
///     >>> executor.run("Execute the plan")
#[pymodule]
fn dragen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_class::<Context>()?;
    m.add_class::<Sandbox>()?;
    m.add_class::<ToolInfo>()?;
    Ok(())
}

//! Python bindings for dragen - a CodeAct-style AI agent framework.
//!
//! This module provides Python access to dragen agents, allowing
//! LLM-powered code execution with tool registration.

use ::dragen::{
    Agent as RustAgent, AgentConfig as RustAgentConfig, Context as RustContext,
};
use ::littrs::{PyValue, ToolInfo as RustToolInfo};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
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
        PyValue::Dict(pairs) => {
            let dict = PyDict::new(py);
            for (k, v) in pairs {
                dict.set_item(k, pyvalue_to_py(py, v)).unwrap();
            }
            dict.into_py(py)
        }
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
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut pairs = Vec::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            pairs.push((key, py_to_pyvalue(&v)?));
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

// ============================================================================
// ToolInfo wrapper
// ============================================================================

/// Tool information for registration with type validation.
///
/// Example:
///     >>> info = ToolInfo("search", "Search the web")
///     >>> info = info.arg_required("query", "str", "The search query")
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
            inner: self.inner.clone().arg_required(name, python_type, description),
        }
    }

    /// Add an optional argument.
    fn arg_optional(&self, name: &str, python_type: &str, description: &str) -> Self {
        Self {
            inner: self.inner.clone().arg_optional(name, python_type, description),
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
    #[new]
    #[pyo3(signature = (model, max_iterations=None, temperature=None, max_tokens=None, system=None))]
    fn new(
        model: &str,
        max_iterations: Option<usize>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        system: Option<&str>,
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

        Ok(Self {
            inner: RustAgent::new(config),
            runtime,
            context: None,
            context_reads: Vec::new(),
            context_write: None,
        })
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
                    Err(e) => PyValue::Dict(vec![(
                        "error".to_string(),
                        PyValue::Str(format!("{}", e)),
                    )]),
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
                    Err(e) => PyValue::Dict(vec![(
                        "error".to_string(),
                        PyValue::Str(format!("{}", e)),
                    )]),
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
                    Err(e) => PyValue::Dict(vec![(
                        "error".to_string(),
                        PyValue::Str(format!("{}", e)),
                    )]),
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
    ///
    /// Args:
    ///     task: The task description for the agent
    ///
    /// Returns:
    ///     The result as a Python object (dict, list, string, etc.)
    ///
    /// Raises:
    ///     RuntimeError: If the agent fails (LLM error, sandbox error, etc.)
    ///     ValueError: If the result cannot be deserialized
    fn run(&mut self, py: Python<'_>, task: &str) -> PyResult<PyObject> {
        // Release GIL while running async code
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
                } else if err_str.contains("Deserialization") {
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
        self.inner.sandbox().describe_tools()
    }
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
///     >>> from dragen import Agent, AgentConfig, Context, ToolInfo
///     >>>
///     >>> # Create an agent
///     >>> agent = Agent(AgentConfig("gpt-4o").max_iterations(10))
///     >>>
///     >>> # Register a tool
///     >>> def search(args):
///     ...     query = args[0]
///     ...     return {"results": [f"Found: {query}"]}
///     >>>
///     >>> agent.register_function("search", search)
///     >>>
///     >>> # Run a task
///     >>> result = agent.run("Search for Python tutorials")
///     >>> print(result)
///
/// Multi-agent with shared context:
///     >>> ctx = Context()
///     >>>
///     >>> # Planner writes to context
///     >>> planner = Agent(AgentConfig("gpt-4o"))
///     >>> planner = planner.to_context(ctx, "plan")
///     >>> planner.run("Create a research plan")
///     >>>
///     >>> # Executor reads from context
///     >>> executor = Agent(AgentConfig("gpt-4o"))
///     >>> executor = executor.from_context(ctx, "plan")
///     >>> executor.run("Execute the plan")
#[pymodule]
fn dragen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_class::<Context>()?;
    m.add_class::<ToolInfo>()?;
    Ok(())
}

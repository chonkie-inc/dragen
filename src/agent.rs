//! CodeAct-style agent implementation.
//!
//! The agent uses an LLM to generate Python code which is executed in a
//! secure Litter sandbox. Tools are exposed as Python functions.

use crate::error::{Error, Result};
use litter::{PyValue, Sandbox, ToolInfo};
use regex::Regex;
use serde::de::DeserializeOwned;
use std::sync::{Arc, Mutex};
use tanukie::{Client, Message, Role};

/// Marker prefix for finish tool output
const FINISH_MARKER: &str = "___FINISH___:";

/// System prompt template for the CodeAct agent.
const SYSTEM_PROMPT_TEMPLATE: &str = r#"{system}

<functions>
{tools}
</functions>

<format>
To execute code, write it in a <code> block or ```python block:

<code>
result = some_function(arg1, arg2)
print(result)
</code>

To return structured data directly, use a <finish> block:

<finish>
{"key": "value", "items": [1, 2, 3]}
</finish>
</format>

<rules>
- Write ONE code block per response, then STOP and wait for results
- Do NOT assume or predict output - you will see actual results after execution
- Only use functions listed above - no imports allowed
- Use print() to see values
- Variables persist between executions
- When done, either call finish(answer) in code OR use a <finish>JSON</finish> block for structured output
</rules>
"#;

const DEFAULT_SYSTEM: &str = "You are an AI assistant that solves tasks by writing and executing Python code.";

/// Configuration for the CodeAct agent.
#[derive(Clone)]
pub struct AgentConfig {
    /// The model to use (e.g., "gpt-4o", "llama-3.3-70b-versatile")
    pub model: String,
    /// Maximum number of iterations (code executions)
    pub max_iterations: usize,
    /// Temperature for LLM sampling
    pub temperature: Option<f32>,
    /// Maximum tokens for LLM response
    pub max_tokens: Option<u32>,
    /// Custom system description (embedded in the full prompt template)
    pub system: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "llama-3.3-70b-versatile".to_string(),
            max_iterations: 10,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            system: None,
        }
    }
}

impl AgentConfig {
    /// Create a new config with the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set the maximum number of iterations.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the max tokens.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set a custom system description (embedded in the full prompt template).
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
}

/// A CodeAct-style agent that executes Python code in a sandbox.
pub struct Agent {
    client: Client,
    sandbox: Sandbox,
    config: AgentConfig,
    messages: Vec<Message>,
    code_regex: Regex,
    /// Regex to match <finish>...</finish> blocks (direct structured output)
    finish_regex: Regex,
    /// Holds the final answer when finish() is called (as structured PyValue)
    finish_answer: Arc<Mutex<Option<PyValue>>>,
}

impl Agent {
    /// Create a new agent with the given configuration.
    pub fn new(config: AgentConfig) -> Self {
        Self {
            client: Client::new(),
            sandbox: Sandbox::new(),
            config,
            messages: Vec::new(),
            // Match either <code>...</code> or ```python...``` blocks
            code_regex: Regex::new(r"(?:<code>\s*([\s\S]*?)</code>|```(?:python|py)?\s*\n([\s\S]*?)```)").unwrap(),
            // Match <finish>...</finish> blocks for direct structured output
            finish_regex: Regex::new(r"<finish>\s*([\s\S]*?)</finish>").unwrap(),
            finish_answer: Arc::new(Mutex::new(None)),
        }
    }

    /// Register the default finish tool if no custom one exists.
    fn ensure_finish_tool(&mut self) {
        // Check if a "finish" tool is already registered
        let has_finish = self.sandbox.tool_infos().iter().any(|t| t.name == "finish");
        if has_finish {
            return;
        }

        // Register the built-in finish tool
        let finish_info = ToolInfo::new("finish", "Complete the task and return the final answer")
            .arg_required("answer", "any", "The final answer to return")
            .returns("any");

        let finish_answer_clone = self.finish_answer.clone();
        self.sandbox.register_tool(finish_info, move |args| {
            let value = args.get(0).cloned().unwrap_or(PyValue::None);
            let answer_str = pyvalue_to_string(&value);

            if let Ok(mut fa) = finish_answer_clone.lock() {
                *fa = Some(value);
            }

            // Return a marker that the run loop can detect
            PyValue::Str(format!("{}{}", FINISH_MARKER, answer_str))
        });
    }

    /// Create a new agent with default configuration.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self::new(AgentConfig::new(model))
    }

    /// Register a tool with the agent's sandbox.
    ///
    /// Tools created with `#[tool]` macro can be registered like:
    /// ```ignore
    /// agent.register(my_tool::Tool);
    /// ```
    pub fn register<T: litter::Tool + 'static>(&mut self, tool: T) {
        self.sandbox.register(tool);
    }

    /// Register a tool with explicit info and callback.
    pub fn register_tool<F>(&mut self, info: ToolInfo, f: F)
    where
        F: Fn(Vec<PyValue>) -> PyValue + Send + Sync + 'static,
    {
        self.sandbox.register_tool(info, f);
    }

    /// Register a custom finish tool with specified arguments.
    ///
    /// The callback receives the arguments and should return a PyValue.
    /// The agent will automatically handle the finish marker.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Planner agent that returns structured sections
    /// agent.register_finish(
    ///     ToolInfo::new("finish", "Return research sections")
    ///         .arg_required("sections", "dict", "Section titles mapped to descriptions"),
    ///     |args| {
    ///         // Process and return the sections
    ///         args.get(0).cloned().unwrap_or(PyValue::None)
    ///     }
    /// );
    /// ```
    pub fn register_finish<F>(&mut self, info: ToolInfo, f: F)
    where
        F: Fn(Vec<PyValue>) -> PyValue + Send + Sync + 'static,
    {
        let finish_answer_clone = self.finish_answer.clone();

        // Wrap the callback to handle finish marker
        self.sandbox.register_tool(info, move |args| {
            let result = f(args);
            let answer_str = pyvalue_to_string(&result);

            if let Ok(mut fa) = finish_answer_clone.lock() {
                *fa = Some(result);
            }

            // Return marker so run loop detects completion
            PyValue::Str(format!("{}{}", FINISH_MARKER, answer_str))
        });
    }

    /// Get the tool documentation for the system prompt.
    fn tool_docs(&self) -> String {
        let docs = self.sandbox.describe_tools();
        if docs.is_empty() {
            "No tools available.".to_string()
        } else {
            docs
        }
    }

    /// Build the system prompt with tool documentation.
    fn system_prompt(&self) -> String {
        let system = self
            .config
            .system
            .as_deref()
            .unwrap_or(DEFAULT_SYSTEM);
        SYSTEM_PROMPT_TEMPLATE
            .replace("{system}", system)
            .replace("{tools}", &self.tool_docs())
    }

    /// Extract Python code from a response.
    /// Supports both <code>...</code> and ```python...``` blocks.
    fn extract_code(&self, text: &str) -> Option<String> {
        self.code_regex.captures(text).map(|cap| {
            // Group 1 is <code> content, Group 2 is ```python content
            cap.get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default()
        })
    }

    /// Extract a direct finish block from a response.
    /// Supports <finish>JSON</finish> blocks for direct structured output.
    /// This allows LLMs to return structured data without executing code.
    fn extract_finish(&self, text: &str) -> Option<String> {
        self.finish_regex.captures(text).map(|cap| {
            cap.get(1)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default()
        })
    }

    /// Execute code in the sandbox and format the result.
    fn execute_code(&mut self, code: &str) -> String {
        match self.sandbox.execute_with_output(code) {
            Ok(output) => {
                let mut parts = Vec::new();

                // Add print output if any
                if output.has_output() {
                    parts.push(output.print_output());
                }

                // Add result if not None
                let result_str = format_pyvalue(&output.result);
                if result_str != "None" {
                    parts.push(format!("=> {}", result_str));
                }

                if parts.is_empty() {
                    "Code executed successfully (no output).".to_string()
                } else {
                    parts.join("\n")
                }
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    /// Run the agent on a task and return the result as the specified type.
    ///
    /// The return type `T` can be:
    /// - `String`: Returns the string representation of the finish value
    /// - Any struct implementing `DeserializeOwned`: Deserializes the finish value
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get result as string
    /// let answer: String = agent.run("What is 2+2?").await?;
    ///
    /// // Get result as typed struct
    /// #[derive(Deserialize)]
    /// struct PlanOutput {
    ///     title: String,
    ///     sections: Vec<String>,
    /// }
    /// let plan: PlanOutput = agent.run("Create a plan").await?;
    /// ```
    pub async fn run<T>(&mut self, task: &str) -> Result<T>
    where
        T: DeserializeOwned,
    {
        // Ensure finish tool exists (registers default if no custom one)
        self.ensure_finish_tool();

        // Clear any previous finish answer
        if let Ok(mut fa) = self.finish_answer.lock() {
            *fa = None;
        }

        // Initialize conversation with system prompt and user task
        self.messages.clear();
        self.messages.push(Message {
            role: Role::System,
            content: self.system_prompt(),
            name: None,
            tool_call_id: None,
        });
        self.messages.push(Message {
            role: Role::User,
            content: task.to_string(),
            name: None,
            tool_call_id: None,
        });

        let mut iterations = 0;

        loop {
            if iterations >= self.config.max_iterations {
                return Err(Error::MaxIterations(self.config.max_iterations));
            }

            // Call LLM
            let response = self.call_llm().await?;
            let text = response.text.clone();

            // Add assistant response to history
            self.messages.push(Message {
                role: Role::Assistant,
                content: text.clone(),
                name: None,
                tool_call_id: None,
            });

            // Check for direct <finish>JSON</finish> block first
            // This allows LLMs to return structured data directly without code execution
            if let Some(finish_content) = self.extract_finish(&text) {
                // Try to parse the content as JSON
                match serde_json::from_str::<T>(&finish_content) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // Give the LLM feedback with the exact error and its output
                        iterations += 1;
                        if iterations >= self.config.max_iterations {
                            return Err(Error::Deserialization(format!(
                                "Invalid JSON in <finish> block: {}", e
                            )));
                        }

                        self.messages.push(Message {
                            role: Role::User,
                            content: format!(
                                "Error parsing your <finish> block:\n\n{}\n\nYour output:\n```\n{}\n```\n\nPlease fix and try again.",
                                e, finish_content
                            ),
                            name: None,
                            tool_call_id: None,
                        });
                        continue;
                    }
                }
            }

            // Check for code block
            if let Some(code) = self.extract_code(&text) {
                iterations += 1;

                // Execute the code
                let output = self.execute_code(&code);

                // Check if finish() was called
                if output.contains(FINISH_MARKER) {
                    // Get the answer from the mutex and deserialize
                    if let Ok(fa) = self.finish_answer.lock() {
                        if let Some(value) = fa.as_ref() {
                            let json = pyvalue_to_json(value);
                            match serde_json::from_value::<T>(json.clone()) {
                                Ok(result) => return Ok(result),
                                Err(e) => {
                                    // Give the LLM feedback with the exact error and its output
                                    if iterations >= self.config.max_iterations {
                                        return Err(Error::Deserialization(format!(
                                            "Invalid finish() output: {}", e
                                        )));
                                    }

                                    // Clear the finish answer so we can try again
                                    drop(fa);
                                    if let Ok(mut fa) = self.finish_answer.lock() {
                                        *fa = None;
                                    }

                                    self.messages.push(Message {
                                        role: Role::User,
                                        content: format!(
                                            "Error parsing your finish() output:\n\n{}\n\nYour output:\n```\n{}\n```\n\nPlease fix and try again.",
                                            e, json
                                        ),
                                        name: None,
                                        tool_call_id: None,
                                    });
                                    continue;
                                }
                            }
                        }
                    }
                    // Fallback: no finish value captured
                    return Err(Error::Deserialization("No finish value captured".to_string()));
                }

                // Add execution result as user message
                self.messages.push(Message {
                    role: Role::User,
                    content: format!("Execution output:\n```\n{}\n```", output),
                    name: None,
                    tool_call_id: None,
                });
            } else {
                // No code block or finish block - agent is done (fallback behavior)
                // Try to deserialize the text as JSON, or wrap as string
                return serde_json::from_str(&text)
                    .or_else(|_| {
                        // If text isn't valid JSON, wrap it as a JSON string for String return type
                        serde_json::from_value(serde_json::Value::String(text))
                    })
                    .map_err(|e| Error::Deserialization(e.to_string()));
            }
        }
    }

    /// Call the LLM with current messages.
    async fn call_llm(&self) -> Result<tanukie::Response> {
        let mut options = tanukie::GenerateOptions::default();
        options.temperature = self.config.temperature;
        options.max_tokens = self.config.max_tokens;

        let response = self
            .client
            .agenerate_with(&self.config.model, self.messages.clone(), options)
            .await?;

        Ok(response)
    }

    /// Get the conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get access to the sandbox for inspection.
    pub fn sandbox(&self) -> &Sandbox {
        &self.sandbox
    }

    /// Get mutable access to the sandbox.
    pub fn sandbox_mut(&mut self) -> &mut Sandbox {
        &mut self.sandbox
    }

    /// Get the structured finish value (if finish() was called with structured data).
    ///
    /// This allows accessing the raw PyValue returned by custom finish tools,
    /// avoiding the need to parse the string representation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After agent.run() completes:
    /// if let Some(value) = agent.finish_value() {
    ///     if let PyValue::Dict(pairs) = value {
    ///         // Access structured data directly
    ///         for (key, val) in pairs {
    ///             println!("{}: {:?}", key, val);
    ///         }
    ///     }
    /// }
    /// ```
    pub fn finish_value(&self) -> Option<PyValue> {
        self.finish_answer.lock().ok().and_then(|fa| fa.clone())
    }
}

/// Format a PyValue for display.
fn format_pyvalue(value: &PyValue) -> String {
    match value {
        PyValue::None => "None".to_string(),
        PyValue::Bool(b) => b.to_string(),
        PyValue::Int(i) => i.to_string(),
        PyValue::Float(f) => f.to_string(),
        PyValue::Str(s) => format!("\"{}\"", s),
        PyValue::List(items) => {
            let formatted: Vec<String> = items.iter().map(format_pyvalue).collect();
            format!("[{}]", formatted.join(", "))
        }
        PyValue::Dict(pairs) => {
            let formatted: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("\"{}\": {}", k, format_pyvalue(v)))
                .collect();
            format!("{{{}}}", formatted.join(", "))
        }
    }
}

/// Convert a PyValue to a plain string (without quotes for strings).
/// Used by finish() to get the final answer as a string.
fn pyvalue_to_string(value: &PyValue) -> String {
    match value {
        PyValue::None => "None".to_string(),
        PyValue::Bool(b) => b.to_string(),
        PyValue::Int(i) => i.to_string(),
        PyValue::Float(f) => f.to_string(),
        PyValue::Str(s) => s.clone(),
        PyValue::List(items) => {
            let formatted: Vec<String> = items.iter().map(pyvalue_to_string).collect();
            format!("[{}]", formatted.join(", "))
        }
        PyValue::Dict(pairs) => {
            let formatted: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, pyvalue_to_string(v)))
                .collect();
            formatted.join("\n")
        }
    }
}

/// Convert a PyValue to a serde_json::Value for typed deserialization.
pub fn pyvalue_to_json(value: &PyValue) -> serde_json::Value {
    match value {
        PyValue::None => serde_json::Value::Null,
        PyValue::Bool(b) => serde_json::Value::Bool(*b),
        PyValue::Int(i) => serde_json::Value::Number((*i).into()),
        PyValue::Float(f) => {
            serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        PyValue::Str(s) => serde_json::Value::String(s.clone()),
        PyValue::List(items) => {
            serde_json::Value::Array(items.iter().map(pyvalue_to_json).collect())
        }
        PyValue::Dict(pairs) => {
            let map: serde_json::Map<String, serde_json::Value> = pairs
                .iter()
                .map(|(k, v)| (k.clone(), pyvalue_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code() {
        let agent = Agent::with_model("test");

        let text = r#"Let me calculate that:

```python
x = 1 + 2
print(x)
```

This will output 3."#;

        let code = agent.extract_code(text);
        assert_eq!(code, Some("x = 1 + 2\nprint(x)".to_string()));
    }

    #[test]
    fn test_extract_code_no_language() {
        let agent = Agent::with_model("test");

        let text = r#"```
x = 1
```"#;

        let code = agent.extract_code(text);
        assert_eq!(code, Some("x = 1".to_string()));
    }

    #[test]
    fn test_extract_code_xml_tag() {
        let agent = Agent::with_model("test");

        let text = r#"Let me calculate:

<code>
x = add(1, 2)
print(x)
</code>

This will give us the result."#;

        let code = agent.extract_code(text);
        assert_eq!(code, Some("x = add(1, 2)\nprint(x)".to_string()));
    }

    #[test]
    fn test_no_code_block() {
        let agent = Agent::with_model("test");

        let text = "The answer is 42.";
        let code = agent.extract_code(text);
        assert_eq!(code, None);
    }

    #[test]
    fn test_extract_finish_block() {
        let agent = Agent::with_model("test");

        let text = r#"Based on my research, here is the result:

<finish>
{"content": "The market is growing", "sources": ["https://example.com"]}
</finish>

I hope this helps!"#;

        let finish = agent.extract_finish(text);
        assert!(finish.is_some());
        let content = finish.unwrap();
        assert!(content.contains("\"content\""));
        assert!(content.contains("\"sources\""));
    }

    #[test]
    fn test_no_finish_block() {
        let agent = Agent::with_model("test");

        let text = "The answer is 42.";
        let finish = agent.extract_finish(text);
        assert_eq!(finish, None);
    }

    #[test]
    fn test_execute_code() {
        let mut agent = Agent::with_model("test");

        let output = agent.execute_code("1 + 2");
        assert_eq!(output, "=> 3");
    }

    #[test]
    fn test_execute_code_with_print() {
        let mut agent = Agent::with_model("test");

        let output = agent.execute_code("print('hello')");
        assert_eq!(output, "hello");
    }

    #[test]
    fn test_execute_code_with_print_and_result() {
        let mut agent = Agent::with_model("test");

        let output = agent.execute_code("print('calculating')\n1 + 2");
        assert_eq!(output, "calculating\n=> 3");
    }

    #[test]
    fn test_execute_code_error() {
        let mut agent = Agent::with_model("test");

        let output = agent.execute_code("undefined_var");
        assert!(output.starts_with("Error:"));
    }
}

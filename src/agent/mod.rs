//! CodeAct-style agent implementation.
//!
//! The agent uses an LLM to generate Python code which is executed in a
//! secure Littrs sandbox. Tools are exposed as Python functions.

mod config;
mod convert;
mod events;
mod prompt;

pub use config::AgentConfig;
pub use convert::pyvalue_to_json;
pub use events::{AgentCallbacks, AgentEvent};

use crate::context::Context;
use crate::error::{Error, Result};
use convert::{format_pyvalue, json_to_pyvalue, pyvalue_to_string};
use events::verbose_callbacks;
use littrs::{PyValue, Sandbox, ToolInfo};
use prompt::{DEFAULT_SYSTEM, FINISH_MARKER, SYSTEM_PROMPT_TEMPLATE};
use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::{Arc, Mutex};
use tanukie::{Client, Message, Role};

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
    /// Shared context for data passing between agents
    context: Option<Context>,
    /// Keys to read from context and inject into prompt
    context_reads: Vec<String>,
    /// Key to write output to in context
    context_write: Option<String>,
    /// Callbacks for observability
    callbacks: AgentCallbacks,
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
            code_regex: Regex::new(
                r"(?:<code>\s*([\s\S]*?)</code>|```(?:python|py)?\s*\n([\s\S]*?)```)",
            )
            .unwrap(),
            // Match <finish>...</finish> blocks for direct structured output
            finish_regex: Regex::new(r"<finish>\s*([\s\S]*?)</finish>").unwrap(),
            finish_answer: Arc::new(Mutex::new(None)),
            context: None,
            context_reads: Vec::new(),
            context_write: None,
            callbacks: AgentCallbacks::default(),
        }
    }

    /// Create a new agent with default configuration.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self::new(AgentConfig::new(model))
    }

    // =========================================================================
    // Builder methods for callbacks
    // =========================================================================

    /// Enable verbose logging to stderr.
    ///
    /// This prints iteration progress, LLM responses, code execution, and tool calls.
    pub fn verbose(mut self, enabled: bool) -> Self {
        if enabled {
            self.callbacks = verbose_callbacks();
        }
        self
    }

    /// Set a callback for iteration start events.
    pub fn on_iteration_start<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_iteration_start = Some(Arc::new(f));
        self
    }

    /// Set a callback for LLM request events.
    pub fn on_llm_request<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_llm_request = Some(Arc::new(f));
        self
    }

    /// Set a callback for LLM response events.
    pub fn on_llm_response<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_llm_response = Some(Arc::new(f));
        self
    }

    /// Set a callback for code generation events.
    pub fn on_code_generated<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_code_generated = Some(Arc::new(f));
        self
    }

    /// Set a callback for code execution events.
    pub fn on_code_executed<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_code_executed = Some(Arc::new(f));
        self
    }

    /// Set a callback for tool call events.
    pub fn on_tool_call<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_tool_call = Some(Arc::new(f));
        self
    }

    /// Set a callback for tool result events.
    pub fn on_tool_result<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_tool_result = Some(Arc::new(f));
        self
    }

    /// Set a callback for finish events.
    pub fn on_finish<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_finish = Some(Arc::new(f));
        self
    }

    /// Set a callback for error events.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_error = Some(Arc::new(f));
        self
    }

    /// Set a catch-all callback for any event.
    pub fn on_event<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_event = Some(Arc::new(f));
        self
    }

    /// Enable event capture (used internally by Python bindings).
    #[doc(hidden)]
    pub fn capture_events(mut self, enabled: bool) -> Self {
        if enabled {
            self.callbacks.captured_events = Some(Arc::new(Mutex::new(Vec::new())));
        } else {
            self.callbacks.captured_events = None;
        }
        self
    }

    /// Take captured events (used internally by Python bindings).
    #[doc(hidden)]
    pub fn take_events(&mut self) -> Vec<AgentEvent> {
        if let Some(ref events) = self.callbacks.captured_events {
            if let Ok(mut events) = events.lock() {
                return std::mem::take(&mut *events);
            }
        }
        Vec::new()
    }

    // =========================================================================
    // Context methods
    // =========================================================================

    /// Read data from a shared context and inject it into the agent's prompt.
    pub fn from_context(mut self, ctx: &Context, key: &str) -> Self {
        if self.context.is_none() {
            self.context = Some(ctx.clone());
        }
        self.context_reads.push(key.to_string());
        self
    }

    /// Save the agent's output to a shared context.
    pub fn to_context(mut self, ctx: &Context, key: &str) -> Self {
        if self.context.is_none() {
            self.context = Some(ctx.clone());
        }
        self.context_write = Some(key.to_string());
        self
    }

    // =========================================================================
    // Tool registration
    // =========================================================================

    /// Register a tool with the agent's sandbox.
    pub fn register<T: littrs::Tool + 'static>(&mut self, tool: T) {
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
    pub fn register_finish<F>(&mut self, info: ToolInfo, f: F)
    where
        F: Fn(Vec<PyValue>) -> PyValue + Send + Sync + 'static,
    {
        let finish_answer_clone = self.finish_answer.clone();

        self.sandbox.register_tool(info, move |args| {
            let result = f(args);
            let answer_str = pyvalue_to_string(&result);

            if let Ok(mut fa) = finish_answer_clone.lock() {
                *fa = Some(result);
            }

            PyValue::Str(format!("{}{}", FINISH_MARKER, answer_str))
        });
    }

    /// Register the default finish tool if no custom one exists.
    fn ensure_finish_tool(&mut self) {
        let has_finish = self.sandbox.tool_infos().iter().any(|t| t.name == "finish");
        if has_finish {
            return;
        }

        let finish_info = ToolInfo::new("finish", "Complete the task and return the final answer")
            .arg_required("answer", "any", "The final answer to return")
            .returns("any");

        let finish_answer_clone = self.finish_answer.clone();
        self.sandbox.register_tool(finish_info, move |args| {
            let value = args.first().cloned().unwrap_or(PyValue::None);
            let answer_str = pyvalue_to_string(&value);

            if let Ok(mut fa) = finish_answer_clone.lock() {
                *fa = Some(value);
            }

            PyValue::Str(format!("{}{}", FINISH_MARKER, answer_str))
        });
    }

    // =========================================================================
    // Accessors
    // =========================================================================

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
    pub fn finish_value(&self) -> Option<PyValue> {
        self.finish_answer.lock().ok().and_then(|fa| fa.clone())
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Emit an event to registered callbacks.
    fn emit(&self, event: AgentEvent) {
        self.callbacks.emit(&event);
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

    /// Inject context data into the task prompt.
    fn inject_context_into_task(&self, task: &str) -> String {
        let Some(ctx) = &self.context else {
            return task.to_string();
        };

        if self.context_reads.is_empty() {
            return task.to_string();
        }

        let mut injections = Vec::new();
        for key in &self.context_reads {
            if let Some(value) = ctx.get_raw(key) {
                let formatted = serde_json::to_string_pretty(&value).unwrap_or_default();
                injections.push(format!("=== {} ===\n{}", key.to_uppercase(), formatted));
            }
        }

        if injections.is_empty() {
            task.to_string()
        } else {
            format!(
                "<context>\n{}\n</context>\n\n{}",
                injections.join("\n\n"),
                task
            )
        }
    }

    /// Save the result to context if configured.
    fn save_to_context<T: Serialize>(&self, result: &T) {
        if let (Some(ctx), Some(key)) = (&self.context, &self.context_write) {
            ctx.set(key, result);
        }
    }

    /// Build the system prompt with tool documentation.
    fn system_prompt(&self) -> String {
        let system = self.config.system.as_deref().unwrap_or(DEFAULT_SYSTEM);
        SYSTEM_PROMPT_TEMPLATE
            .replace("{system}", system)
            .replace("{tools}", &self.tool_docs())
    }

    /// Extract Python code from a response.
    fn extract_code(&self, text: &str) -> Option<String> {
        self.code_regex.captures(text).map(|cap| {
            cap.get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default()
        })
    }

    /// Extract a direct finish block from a response.
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

                if output.has_output() {
                    parts.push(output.print_output());
                }

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

    // =========================================================================
    // Main run loop
    // =========================================================================

    /// Run the agent on a task and return the result as the specified type.
    pub async fn run<T>(&mut self, task: &str) -> Result<T>
    where
        T: DeserializeOwned + Serialize,
    {
        self.ensure_finish_tool();

        // Clear any previous finish answer
        if let Ok(mut fa) = self.finish_answer.lock() {
            *fa = None;
        }

        // Inject context data into the task
        let task_with_context = self.inject_context_into_task(task);

        // Initialize conversation
        self.messages.clear();
        self.messages.push(Message {
            role: Role::System,
            content: self.system_prompt(),
            name: None,
            tool_call_id: None,
        });
        self.messages.push(Message {
            role: Role::User,
            content: task_with_context,
            name: None,
            tool_call_id: None,
        });

        let mut iterations = 0;

        loop {
            iterations += 1;

            if iterations > self.config.max_iterations {
                self.emit(AgentEvent::Error {
                    message: format!("Max iterations ({}) reached", self.config.max_iterations),
                });
                return Err(Error::MaxIterations(self.config.max_iterations));
            }

            self.emit(AgentEvent::IterationStart {
                iteration: iterations,
                max_iterations: self.config.max_iterations,
            });

            self.emit(AgentEvent::LLMRequest {
                message_count: self.messages.len(),
            });

            let response = self.call_llm().await?;
            let text = response.text.clone();

            self.emit(AgentEvent::LLMResponse {
                content: text.clone(),
                tokens_used: None,
            });

            self.messages.push(Message {
                role: Role::Assistant,
                content: text.clone(),
                name: None,
                tool_call_id: None,
            });

            // Check for direct <finish>JSON</finish> block first
            if let Some(finish_content) = self.extract_finish(&text) {
                match serde_json::from_str::<T>(&finish_content) {
                    Ok(result) => {
                        if let Ok(json) = serde_json::to_value(&result) {
                            self.emit(AgentEvent::Finish {
                                value: json_to_pyvalue(&json),
                            });
                        }
                        self.save_to_context(&result);
                        return Ok(result);
                    }
                    Err(e) => {
                        self.emit(AgentEvent::Error {
                            message: format!("Invalid JSON in <finish> block: {}", e),
                        });

                        if iterations >= self.config.max_iterations {
                            return Err(Error::Deserialization(format!(
                                "Invalid JSON in <finish> block: {}",
                                e
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
                self.emit(AgentEvent::CodeGenerated { code: code.clone() });

                let output = self.execute_code(&code);
                let success = !output.starts_with("Error:");

                self.emit(AgentEvent::CodeExecuted {
                    code: code.clone(),
                    output: output.clone(),
                    success,
                });

                // Check if finish() was called
                if output.contains(FINISH_MARKER) {
                    if let Ok(fa) = self.finish_answer.lock() {
                        if let Some(value) = fa.as_ref() {
                            self.emit(AgentEvent::Finish {
                                value: value.clone(),
                            });

                            let json = pyvalue_to_json(value);
                            match serde_json::from_value::<T>(json.clone()) {
                                Ok(result) => {
                                    self.save_to_context(&result);
                                    return Ok(result);
                                }
                                Err(e) => {
                                    if iterations >= self.config.max_iterations {
                                        return Err(Error::Deserialization(format!(
                                            "Invalid finish() output: {}",
                                            e
                                        )));
                                    }

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
                    return Err(Error::Deserialization("No finish value captured".to_string()));
                }

                self.messages.push(Message {
                    role: Role::User,
                    content: format!("Execution output:\n```\n{}\n```", output),
                    name: None,
                    tool_call_id: None,
                });
            } else {
                // No code block or finish block - fallback behavior
                let result: T = serde_json::from_str(&text)
                    .or_else(|_| serde_json::from_value(serde_json::Value::String(text)))
                    .map_err(|e| Error::Deserialization(e.to_string()))?;
                self.save_to_context(&result);
                return Ok(result);
            }
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

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
use jsonschema::Validator;
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
    /// Regex to match custom thinking tags (e.g., <intent>...</intent>)
    think_regex: Option<Regex>,
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
    /// Optional JSON Schema for validating finish() output
    schema: Option<serde_json::Value>,
    /// Compiled JSON Schema validator (for performance)
    schema_validator: Option<Arc<Validator>>,
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        // Rebuild thinking regex from config
        let think_regex = self.config.thinking_tag.as_ref().map(|tag| {
            Regex::new(&format!(r"<{}>\s*([\s\S]*?)</{}>", tag, tag)).unwrap()
        });

        Self {
            client: self.client.clone(),
            sandbox: self.sandbox.clone(),
            config: self.config.clone(),
            messages: Vec::new(), // Fresh message history
            code_regex: self.code_regex.clone(),
            think_regex,
            finish_regex: self.finish_regex.clone(),
            finish_answer: Arc::new(Mutex::new(None)), // Fresh finish state
            context: self.context.clone(), // Shared context (intentional)
            context_reads: self.context_reads.clone(),
            context_write: self.context_write.clone(),
            callbacks: self.callbacks.clone(),
            schema: self.schema.clone(),
            schema_validator: self.schema_validator.clone(),
        }
    }
}

impl Agent {
    /// Create a new agent with the given configuration.
    ///
    /// Uses `Sandbox::with_builtins()` by default, which enables
    /// `import json`, `import math`, and `import typing`.
    pub fn new(config: AgentConfig) -> Self {
        Self::with_sandbox(Sandbox::with_builtins(), config)
    }

    /// Create a new agent with a pre-configured sandbox.
    ///
    /// This allows full control over the sandbox configuration:
    /// resource limits, mounted files, custom modules, and builtins.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use littrs::{Sandbox, Limits};
    /// use dragen::{Agent, AgentConfig};
    ///
    /// let mut sandbox = Sandbox::with_builtins();
    /// sandbox.limit(Limits {
    ///     max_instructions: Some(100_000),
    ///     max_recursion_depth: Some(50),
    /// });
    /// sandbox.mount("input.json", "./data/input.json", false);
    ///
    /// let agent = Agent::with_sandbox(sandbox, AgentConfig::new("gpt-4o"));
    /// ```
    pub fn with_sandbox(sandbox: Sandbox, config: AgentConfig) -> Self {
        // Build thinking tag regex if configured
        let think_regex = config.thinking_tag.as_ref().map(|tag| {
            Regex::new(&format!(r"<{}>\s*([\s\S]*?)</{}>", tag, tag)).unwrap()
        });

        Self {
            client: Client::new(),
            sandbox,
            config,
            messages: Vec::new(),
            // Match either <code>...</code> or ```python...``` blocks
            code_regex: Regex::new(
                r"(?:<code>\s*([\s\S]*?)</code>|```(?:python|py)?\s*\n([\s\S]*?)```)",
            )
            .unwrap(),
            think_regex,
            // Match <finish>...</finish> blocks for direct structured output
            finish_regex: Regex::new(r"<finish>\s*([\s\S]*?)</finish>").unwrap(),
            finish_answer: Arc::new(Mutex::new(None)),
            context: None,
            context_reads: Vec::new(),
            context_write: None,
            callbacks: AgentCallbacks::default(),
            schema: None,
            schema_validator: None,
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

    /// Set a callback for thinking events (extracted from <think> tags).
    pub fn on_thinking<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_thinking = Some(Arc::new(f));
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
        if let Some(ref events) = self.callbacks.captured_events
            && let Ok(mut events) = events.lock()
        {
            return std::mem::take(&mut *events);
        }
        Vec::new()
    }

    // =========================================================================
    // Schema validation
    // =========================================================================

    /// Set a JSON Schema for validating the finish() output.
    ///
    /// When set, the agent will validate the result against this schema after
    /// finish() is called. If validation fails, an error message is sent back
    /// to the LLM for self-correction.
    ///
    /// # Example (Rust)
    ///
    /// ```ignore
    /// let schema = serde_json::json!({
    ///     "type": "object",
    ///     "required": ["content", "sources"],
    ///     "properties": {
    ///         "content": {"type": "string"},
    ///         "sources": {"type": "array", "items": {"type": "string"}}
    ///     }
    /// });
    /// let agent = Agent::with_model("gpt-4o").schema(schema);
    /// ```
    ///
    /// # Example (Python with Pydantic)
    ///
    /// ```python
    /// from pydantic import BaseModel
    ///
    /// class Output(BaseModel):
    ///     content: str
    ///     sources: list[str]
    ///
    /// result = agent.run(task, schema=Output.model_json_schema())
    /// ```
    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        // Compile the schema for validation
        match Validator::new(&schema) {
            Ok(validator) => {
                self.schema = Some(schema);
                self.schema_validator = Some(Arc::new(validator));
            }
            Err(e) => {
                eprintln!("Warning: Invalid JSON Schema, validation disabled: {}", e);
                self.schema = None;
                self.schema_validator = None;
            }
        }
        self
    }

    /// Set a JSON Schema from a raw JSON Value (for Python bindings).
    #[doc(hidden)]
    pub fn set_schema(&mut self, schema: serde_json::Value) {
        match Validator::new(&schema) {
            Ok(validator) => {
                self.schema = Some(schema);
                self.schema_validator = Some(Arc::new(validator));
            }
            Err(e) => {
                eprintln!("Warning: Invalid JSON Schema, validation disabled: {}", e);
                self.schema = None;
                self.schema_validator = None;
            }
        }
    }

    /// Clear the schema validation.
    pub fn clear_schema(&mut self) {
        self.schema = None;
        self.schema_validator = None;
    }

    /// Validate a value against the schema, returning a formatted error message if invalid.
    fn validate_against_schema(&self, value: &serde_json::Value) -> std::result::Result<(), String> {
        let Some(validator) = &self.schema_validator else {
            return Ok(());
        };

        let result = validator.validate(value);
        if result.is_ok() {
            return Ok(());
        }

        // Collect validation errors into a readable message
        let errors: Vec<String> = validator
            .iter_errors(value)
            .map(|e| format!("- {}: {}", e.instance_path, e))
            .collect();

        let schema_hint = if let Some(schema) = &self.schema {
            if let Some(required) = schema.get("required") {
                format!("\n\nExpected keys: {}", required)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        Err(format!(
            "Schema validation failed:\n{}{}\n\nYour output:\n{}",
            errors.join("\n"),
            schema_hint,
            serde_json::to_string_pretty(value).unwrap_or_default()
        ))
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
        self.sandbox.add(tool);
    }

    /// Register a tool with explicit info and callback.
    pub fn register_tool<F>(&mut self, info: ToolInfo, f: F)
    where
        F: Fn(Vec<PyValue>) -> PyValue + Send + Sync + 'static,
    {
        self.sandbox.register_tool(info, f);
    }

    /// Set a variable in the agent's sandbox.
    ///
    /// This allows you to inject initial state that the agent's code can access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// agent.set_variable("collected_items", PyValue::List(vec![]));
    /// // Now the agent's Python code can use `collected_items`
    /// ```
    pub fn set_variable(&mut self, name: impl Into<String>, value: impl Into<PyValue>) {
        self.sandbox.set(name, value);
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
        let has_finish = self.sandbox.tools().iter().any(|t| t.name == "finish");
        if has_finish {
            return;
        }

        let finish_info = ToolInfo::new("finish", "Complete the task and return the final answer")
            .arg("answer", "any", "The final answer to return")
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
        let docs = self.sandbox.describe();
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

    /// Extract thinking from custom tags (e.g., <intent>...</intent>) in a response.
    fn extract_thinking(&self, text: &str) -> Option<String> {
        self.think_regex.as_ref().and_then(|regex| {
            regex.captures(text).map(|cap| {
                cap.get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default()
            })
        })
    }

    /// Execute code in the sandbox and format the result.
    fn execute_code(&mut self, code: &str) -> String {
        match self.sandbox.capture(code) {
            Ok(output) => {
                let mut parts = Vec::new();

                if !output.output.is_empty() {
                    parts.push(output.output.join("\n"));
                }

                let result_str = format_pyvalue(&output.value);
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
        let options = tanukie::GenerateOptions {
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            ..Default::default()
        };

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

            // Extract and emit thinking if present
            if let Some(thinking) = self.extract_thinking(&text) {
                self.emit(AgentEvent::Thinking {
                    content: thinking,
                });
            }

            self.messages.push(Message {
                role: Role::Assistant,
                content: text.clone(),
                name: None,
                tool_call_id: None,
            });

            // Check for direct <finish>JSON</finish> block first
            if let Some(finish_content) = self.extract_finish(&text) {
                // First parse as generic JSON for validation
                match serde_json::from_str::<serde_json::Value>(&finish_content) {
                    Ok(json_value) => {
                        // Validate against schema if set
                        if let Err(validation_error) = self.validate_against_schema(&json_value) {
                            self.emit(AgentEvent::Error {
                                message: format!("Schema validation failed: {}", validation_error),
                            });

                            if iterations >= self.config.max_iterations {
                                return Err(Error::Deserialization(format!(
                                    "Schema validation failed: {}",
                                    validation_error
                                )));
                            }

                            self.messages.push(Message {
                                role: Role::User,
                                content: format!(
                                    "Your output did not match the expected schema.\n\n{}\n\nPlease fix and try again.",
                                    validation_error
                                ),
                                name: None,
                                tool_call_id: None,
                            });
                            continue;
                        }

                        // Now deserialize to the target type
                        match serde_json::from_value::<T>(json_value.clone()) {
                            Ok(result) => {
                                self.emit(AgentEvent::Finish {
                                    value: json_to_pyvalue(&json_value),
                                });
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
                    if let Ok(fa) = self.finish_answer.lock()
                        && let Some(value) = fa.as_ref()
                    {
                        let json = pyvalue_to_json(value);

                        // Validate against schema if set
                        if let Err(validation_error) = self.validate_against_schema(&json) {
                            self.emit(AgentEvent::Error {
                                message: format!("Schema validation failed: {}", validation_error),
                            });

                            if iterations >= self.config.max_iterations {
                                return Err(Error::Deserialization(format!(
                                    "Schema validation failed: {}",
                                    validation_error
                                )));
                            }

                            drop(fa);
                            if let Ok(mut fa) = self.finish_answer.lock() {
                                *fa = None;
                            }

                            self.messages.push(Message {
                                role: Role::User,
                                content: format!(
                                    "Your output did not match the expected schema.\n\n{}\n\nPlease fix and try again.",
                                    validation_error
                                ),
                                name: None,
                                tool_call_id: None,
                            });
                            continue;
                        }

                        self.emit(AgentEvent::Finish {
                            value: value.clone(),
                        });

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

    /// Chat with the agent, preserving conversation history across calls.
    ///
    /// Unlike `run()`, which clears history and returns a typed `T`, `chat()`
    /// preserves message history for multi-turn conversation and returns a `String`.
    ///
    /// Termination conditions per call:
    /// - **No code block and no finish block** → conversational response → return text
    /// - **Code block** → execute in sandbox, append output, continue looping
    /// - **`finish()` called** → return the finish value as a string
    /// - **`<finish>` block** → return the content as a string
    pub async fn chat(&mut self, message: &str) -> Result<String> {
        // Lazy-initialize on first call
        if self.messages.is_empty() {
            self.ensure_finish_tool();
            self.messages.push(Message {
                role: Role::System,
                content: self.system_prompt(),
                name: None,
                tool_call_id: None,
            });
        }

        // Clear any previous finish answer
        if let Ok(mut fa) = self.finish_answer.lock() {
            *fa = None;
        }

        // Append user message (never clear history)
        self.messages.push(Message {
            role: Role::User,
            content: message.to_string(),
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

            // Extract and emit thinking if present
            if let Some(thinking) = self.extract_thinking(&text) {
                self.emit(AgentEvent::Thinking {
                    content: thinking,
                });
            }

            self.messages.push(Message {
                role: Role::Assistant,
                content: text.clone(),
                name: None,
                tool_call_id: None,
            });

            // Check for direct <finish>JSON</finish> block
            if let Some(finish_content) = self.extract_finish(&text) {
                self.emit(AgentEvent::Finish {
                    value: PyValue::Str(finish_content.clone()),
                });
                return Ok(finish_content);
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
                    if let Ok(fa) = self.finish_answer.lock()
                        && let Some(value) = fa.as_ref()
                    {
                        let result_str = pyvalue_to_string(value);
                        self.emit(AgentEvent::Finish {
                            value: value.clone(),
                        });
                        return Ok(result_str);
                    }
                    return Ok(String::new());
                }

                self.messages.push(Message {
                    role: Role::User,
                    content: format!("Execution output:\n```\n{}\n```", output),
                    name: None,
                    tool_call_id: None,
                });
            } else {
                // No code block or finish block — conversational response
                return Ok(text);
            }
        }
    }

    /// Clear the conversation history.
    ///
    /// The next `chat()` call will re-initialize the system prompt.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Run multiple tasks in parallel using cloned agents.
    ///
    /// Each task is run on a fresh clone of this agent, allowing parallel execution.
    /// Results are returned in the same order as the input tasks. Each result
    /// indicates whether that specific task succeeded or failed.
    ///
    /// # Arguments
    ///
    /// * `tasks` - A list of task strings to run
    ///
    /// # Example
    ///
    /// ```ignore
    /// let agent = Agent::new(config);
    /// agent.register(search_tool::Tool);
    ///
    /// let results: Vec<Result<String>> = agent.map(vec![
    ///     "Write about topic A".to_string(),
    ///     "Write about topic B".to_string(),
    ///     "Write about topic C".to_string(),
    /// ]).await;
    ///
    /// for (i, result) in results.iter().enumerate() {
    ///     match result {
    ///         Ok(value) => println!("Task {} succeeded", i),
    ///         Err(e) => println!("Task {} failed: {}", i, e),
    ///     }
    /// }
    /// ```
    pub async fn map<T>(&self, tasks: Vec<String>) -> Vec<Result<T>>
    where
        T: DeserializeOwned + Serialize + Send + 'static,
    {
        use futures::future::join_all;

        let futures: Vec<_> = tasks
            .into_iter()
            .map(|task| {
                let mut agent = self.clone();
                async move { agent.run::<T>(&task).await }
            })
            .collect();

        join_all(futures).await
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

    #[test]
    fn test_chat_initializes_system_prompt() {
        let mut agent = Agent::with_model("test");
        assert!(agent.messages().is_empty());

        // Simulate what chat() does on first call: lazy-init
        agent.ensure_finish_tool();
        agent.messages.push(Message {
            role: Role::System,
            content: agent.system_prompt(),
            name: None,
            tool_call_id: None,
        });
        agent.messages.push(Message {
            role: Role::User,
            content: "hello".to_string(),
            name: None,
            tool_call_id: None,
        });

        assert_eq!(agent.messages().len(), 2);
        assert_eq!(agent.messages()[0].role, Role::System);
        assert_eq!(agent.messages()[1].role, Role::User);
        assert_eq!(agent.messages()[1].content, "hello");
    }

    #[test]
    fn test_chat_preserves_history() {
        let mut agent = Agent::with_model("test");

        // Simulate two chat turns by manually building messages
        agent.messages.push(Message {
            role: Role::System,
            content: agent.system_prompt(),
            name: None,
            tool_call_id: None,
        });
        agent.messages.push(Message {
            role: Role::User,
            content: "first".to_string(),
            name: None,
            tool_call_id: None,
        });
        agent.messages.push(Message {
            role: Role::Assistant,
            content: "response 1".to_string(),
            name: None,
            tool_call_id: None,
        });
        let len_after_first = agent.messages().len();

        // Second turn appends, doesn't clear
        agent.messages.push(Message {
            role: Role::User,
            content: "second".to_string(),
            name: None,
            tool_call_id: None,
        });
        agent.messages.push(Message {
            role: Role::Assistant,
            content: "response 2".to_string(),
            name: None,
            tool_call_id: None,
        });

        assert!(agent.messages().len() > len_after_first);
        assert_eq!(agent.messages().len(), 5);
    }

    #[test]
    fn test_clear() {
        let mut agent = Agent::with_model("test");

        // Add some messages
        agent.messages.push(Message {
            role: Role::System,
            content: "system".to_string(),
            name: None,
            tool_call_id: None,
        });
        agent.messages.push(Message {
            role: Role::User,
            content: "hello".to_string(),
            name: None,
            tool_call_id: None,
        });
        assert_eq!(agent.messages().len(), 2);

        agent.clear();
        assert!(agent.messages().is_empty());
    }
}

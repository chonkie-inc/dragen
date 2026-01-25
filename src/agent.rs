//! CodeAct-style agent implementation.
//!
//! The agent uses an LLM to generate Python code which is executed in a
//! secure Littrs sandbox. Tools are exposed as Python functions.

use crate::context::Context;
use crate::error::{Error, Result};
use littrs::{PyValue, Sandbox, ToolInfo};
use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::{Arc, Mutex};
use tanukie::{Client, Message, Role};

// ============================================================================
// Agent Events & Callbacks
// ============================================================================

/// Events emitted during agent execution for observability.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Starting a new iteration
    IterationStart {
        iteration: usize,
        max_iterations: usize,
    },
    /// About to call the LLM
    LLMRequest {
        message_count: usize,
    },
    /// LLM responded
    LLMResponse {
        content: String,
        #[allow(dead_code)]
        tokens_used: Option<usize>,
    },
    /// Agent generated code to execute
    CodeGenerated {
        code: String,
    },
    /// Code was executed in sandbox
    CodeExecuted {
        code: String,
        output: String,
        success: bool,
    },
    /// A tool was called
    ToolCall {
        name: String,
        args: Vec<PyValue>,
    },
    /// A tool returned a result
    ToolResult {
        name: String,
        result: PyValue,
    },
    /// Agent called finish()
    Finish {
        value: PyValue,
    },
    /// An error occurred
    Error {
        message: String,
    },
}

/// Type alias for event callbacks
pub type EventCallback = Arc<dyn Fn(&AgentEvent) + Send + Sync>;

/// Storage for agent callbacks
#[derive(Default, Clone)]
pub struct AgentCallbacks {
    pub on_iteration_start: Option<EventCallback>,
    pub on_llm_request: Option<EventCallback>,
    pub on_llm_response: Option<EventCallback>,
    pub on_code_generated: Option<EventCallback>,
    pub on_code_executed: Option<EventCallback>,
    pub on_tool_call: Option<EventCallback>,
    pub on_tool_result: Option<EventCallback>,
    pub on_finish: Option<EventCallback>,
    pub on_error: Option<EventCallback>,
    /// Catch-all callback for any event
    pub on_event: Option<EventCallback>,
    /// Captured events (used internally by Python bindings)
    captured_events: Option<Arc<Mutex<Vec<AgentEvent>>>>,
}

impl AgentCallbacks {
    /// Emit an event to the appropriate callback(s)
    fn emit(&self, event: &AgentEvent) {
        // Capture event if enabled (used by Python bindings)
        if let Some(ref events) = self.captured_events {
            if let Ok(mut events) = events.lock() {
                events.push(event.clone());
            }
        }

        // Call specific callback
        let specific = match event {
            AgentEvent::IterationStart { .. } => &self.on_iteration_start,
            AgentEvent::LLMRequest { .. } => &self.on_llm_request,
            AgentEvent::LLMResponse { .. } => &self.on_llm_response,
            AgentEvent::CodeGenerated { .. } => &self.on_code_generated,
            AgentEvent::CodeExecuted { .. } => &self.on_code_executed,
            AgentEvent::ToolCall { .. } => &self.on_tool_call,
            AgentEvent::ToolResult { .. } => &self.on_tool_result,
            AgentEvent::Finish { .. } => &self.on_finish,
            AgentEvent::Error { .. } => &self.on_error,
        };

        if let Some(cb) = specific {
            cb(event);
        }

        // Call catch-all callback
        if let Some(cb) = &self.on_event {
            cb(event);
        }
    }
}

/// Create verbose logging callbacks
fn verbose_callbacks() -> AgentCallbacks {
    AgentCallbacks {
        on_iteration_start: Some(Arc::new(|e| {
            if let AgentEvent::IterationStart { iteration, max_iterations } = e {
                eprintln!("[dragen] Iteration {}/{}", iteration, max_iterations);
            }
        })),
        on_llm_response: Some(Arc::new(|e| {
            if let AgentEvent::LLMResponse { content, .. } = e {
                let preview: String = content.chars().take(100).collect();
                let suffix = if content.len() > 100 { "..." } else { "" };
                eprintln!("[dragen] LLM: {}{}", preview.replace('\n', "\\n"), suffix);
            }
        })),
        on_code_generated: Some(Arc::new(|e| {
            if let AgentEvent::CodeGenerated { code } = e {
                let lines: Vec<&str> = code.lines().take(3).collect();
                let preview = lines.join("\\n");
                let suffix = if code.lines().count() > 3 { "..." } else { "" };
                eprintln!("[dragen] Code: {}{}", preview, suffix);
            }
        })),
        on_code_executed: Some(Arc::new(|e| {
            if let AgentEvent::CodeExecuted { output, success, .. } = e {
                let status = if *success { "✓" } else { "✗" };
                let preview: String = output.chars().take(80).collect();
                let suffix = if output.len() > 80 { "..." } else { "" };
                eprintln!("[dragen] {} {}{}", status, preview.replace('\n', "\\n"), suffix);
            }
        })),
        on_tool_call: Some(Arc::new(|e| {
            if let AgentEvent::ToolCall { name, args } = e {
                let args_preview: Vec<String> = args.iter().take(3).map(|a| format!("{:?}", a)).collect();
                let suffix = if args.len() > 3 { ", ..." } else { "" };
                eprintln!("[dragen] Tool: {}({}{})", name, args_preview.join(", "), suffix);
            }
        })),
        on_finish: Some(Arc::new(|e| {
            if let AgentEvent::Finish { value } = e {
                eprintln!("[dragen] Finish: {:?}", value);
            }
        })),
        on_error: Some(Arc::new(|e| {
            if let AgentEvent::Error { message } = e {
                eprintln!("[dragen] Error: {}", message);
            }
        })),
        ..Default::default()
    }
}

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

    /// Remove the max tokens limit (let the model use its default).
    pub fn no_max_tokens(mut self) -> Self {
        self.max_tokens = None;
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
            code_regex: Regex::new(r"(?:<code>\s*([\s\S]*?)</code>|```(?:python|py)?\s*\n([\s\S]*?)```)").unwrap(),
            // Match <finish>...</finish> blocks for direct structured output
            finish_regex: Regex::new(r"<finish>\s*([\s\S]*?)</finish>").unwrap(),
            finish_answer: Arc::new(Mutex::new(None)),
            context: None,
            context_reads: Vec::new(),
            context_write: None,
            callbacks: AgentCallbacks::default(),
        }
    }

    /// Enable verbose logging to stderr.
    ///
    /// This prints iteration progress, LLM responses, code execution, and tool calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let agent = Agent::new(config).verbose(true);
    /// ```
    pub fn verbose(mut self, enabled: bool) -> Self {
        if enabled {
            self.callbacks = verbose_callbacks();
        }
        self
    }

    /// Set a callback for iteration start events.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let agent = Agent::new(config)
    ///     .on_iteration_start(|e| {
    ///         if let AgentEvent::IterationStart { iteration, max_iterations } = e {
    ///             println!("[{}/{}]", iteration, max_iterations);
    ///         }
    ///     });
    /// ```
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
    ///
    /// This is called for every event in addition to specific callbacks.
    pub fn on_event<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        self.callbacks.on_event = Some(Arc::new(f));
        self
    }

    /// Emit an event to registered callbacks.
    fn emit(&self, event: AgentEvent) {
        self.callbacks.emit(&event);
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

    /// Read data from a shared context and inject it into the agent's prompt.
    ///
    /// The context value is automatically formatted and included in the task prompt,
    /// so the agent can see the data without making any tool calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = Context::new();
    /// ctx.set("plan", &planner_output);
    ///
    /// let mut executor = Agent::new(config)
    ///     .from_context(&ctx, "plan")
    ///     .from_context(&ctx, "search_log");
    /// ```
    pub fn from_context(mut self, ctx: &Context, key: &str) -> Self {
        if self.context.is_none() {
            self.context = Some(ctx.clone());
        }
        self.context_reads.push(key.to_string());
        self
    }

    /// Save the agent's output to a shared context.
    ///
    /// When the agent finishes (via `finish()` or `<finish>` block), the output
    /// is automatically stored in the context under the specified key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = Context::new();
    ///
    /// let mut planner = Agent::new(config)
    ///     .to_context(&ctx, "plan");
    /// planner.run::<PlannerOutput>(&query).await?;
    ///
    /// // ctx now contains "plan" -> PlannerOutput
    /// ```
    pub fn to_context(mut self, ctx: &Context, key: &str) -> Self {
        if self.context.is_none() {
            self.context = Some(ctx.clone());
        }
        self.context_write = Some(key.to_string());
        self
    }

    /// Register a tool with the agent's sandbox.
    ///
    /// Tools created with `#[tool]` macro can be registered like:
    /// ```ignore
    /// agent.register(my_tool::Tool);
    /// ```
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
    /// If the agent is configured with `to_context()`, the result is automatically
    /// saved to the shared context.
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
        T: DeserializeOwned + Serialize,
    {
        // Ensure finish tool exists (registers default if no custom one)
        self.ensure_finish_tool();

        // Clear any previous finish answer
        if let Ok(mut fa) = self.finish_answer.lock() {
            *fa = None;
        }

        // Inject context data into the task
        let task_with_context = self.inject_context_into_task(task);

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

            // Emit iteration start
            self.emit(AgentEvent::IterationStart {
                iteration: iterations,
                max_iterations: self.config.max_iterations,
            });

            // Emit LLM request
            self.emit(AgentEvent::LLMRequest {
                message_count: self.messages.len(),
            });

            // Call LLM
            let response = self.call_llm().await?;
            let text = response.text.clone();

            // Emit LLM response
            self.emit(AgentEvent::LLMResponse {
                content: text.clone(),
                tokens_used: None, // TODO: get from response if available
            });

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
                    Ok(result) => {
                        // Emit finish event
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

                        // Give the LLM feedback with the exact error and its output
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
                // Emit code generated
                self.emit(AgentEvent::CodeGenerated {
                    code: code.clone(),
                });

                // Execute the code
                let output = self.execute_code(&code);
                let success = !output.starts_with("Error:");

                // Emit code executed
                self.emit(AgentEvent::CodeExecuted {
                    code: code.clone(),
                    output: output.clone(),
                    success,
                });

                // Check if finish() was called
                if output.contains(FINISH_MARKER) {
                    // Get the answer from the mutex and deserialize
                    if let Ok(fa) = self.finish_answer.lock() {
                        if let Some(value) = fa.as_ref() {
                            // Emit finish event
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
                let result: T = serde_json::from_str(&text)
                    .or_else(|_| {
                        // If text isn't valid JSON, wrap it as a JSON string for String return type
                        serde_json::from_value(serde_json::Value::String(text))
                    })
                    .map_err(|e| Error::Deserialization(e.to_string()))?;
                self.save_to_context(&result);
                return Ok(result);
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

/// Convert a serde_json::Value to a PyValue.
fn json_to_pyvalue(value: &serde_json::Value) -> PyValue {
    match value {
        serde_json::Value::Null => PyValue::None,
        serde_json::Value::Bool(b) => PyValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PyValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                PyValue::Float(f)
            } else {
                PyValue::None
            }
        }
        serde_json::Value::String(s) => PyValue::Str(s.clone()),
        serde_json::Value::Array(arr) => {
            PyValue::List(arr.iter().map(json_to_pyvalue).collect())
        }
        serde_json::Value::Object(map) => {
            PyValue::Dict(
                map.iter()
                    .map(|(k, v)| (k.clone(), json_to_pyvalue(v)))
                    .collect(),
            )
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

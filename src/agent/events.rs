//! Agent events and callbacks for observability.

use littrs::PyValue;
use std::sync::{Arc, Mutex};

/// Events emitted during agent execution for observability.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Starting a new iteration
    IterationStart {
        iteration: usize,
        max_iterations: usize,
    },
    /// About to call the LLM
    LLMRequest { message_count: usize },
    /// LLM responded
    LLMResponse {
        content: String,
        #[allow(dead_code)]
        tokens_used: Option<usize>,
    },
    /// Agent is thinking (extracted from <think> tags)
    Thinking { content: String },
    /// Agent generated code to execute
    CodeGenerated { code: String },
    /// Code was executed in sandbox
    CodeExecuted {
        code: String,
        output: String,
        success: bool,
    },
    /// A tool was called
    ToolCall { name: String, args: Vec<PyValue> },
    /// A tool returned a result
    ToolResult { name: String, result: PyValue },
    /// Agent called finish()
    Finish { value: PyValue },
    /// An error occurred
    Error { message: String },
}

/// Type alias for event callbacks
pub type EventCallback = Arc<dyn Fn(&AgentEvent) + Send + Sync>;

/// Storage for agent callbacks
#[derive(Default, Clone)]
pub struct AgentCallbacks {
    pub on_iteration_start: Option<EventCallback>,
    pub on_llm_request: Option<EventCallback>,
    pub on_llm_response: Option<EventCallback>,
    pub on_thinking: Option<EventCallback>,
    pub on_code_generated: Option<EventCallback>,
    pub on_code_executed: Option<EventCallback>,
    pub on_tool_call: Option<EventCallback>,
    pub on_tool_result: Option<EventCallback>,
    pub on_finish: Option<EventCallback>,
    pub on_error: Option<EventCallback>,
    /// Catch-all callback for any event
    pub on_event: Option<EventCallback>,
    /// Captured events (used internally by Python bindings)
    pub(crate) captured_events: Option<Arc<Mutex<Vec<AgentEvent>>>>,
}

impl AgentCallbacks {
    /// Emit an event to the appropriate callback(s)
    pub fn emit(&self, event: &AgentEvent) {
        // Capture event if enabled (used by Python bindings)
        if let Some(ref events) = self.captured_events
            && let Ok(mut events) = events.lock()
        {
            events.push(event.clone());
        }

        // Call specific callback
        let specific = match event {
            AgentEvent::IterationStart { .. } => &self.on_iteration_start,
            AgentEvent::LLMRequest { .. } => &self.on_llm_request,
            AgentEvent::LLMResponse { .. } => &self.on_llm_response,
            AgentEvent::Thinking { .. } => &self.on_thinking,
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
pub fn verbose_callbacks() -> AgentCallbacks {
    AgentCallbacks {
        on_iteration_start: Some(Arc::new(|e| {
            if let AgentEvent::IterationStart {
                iteration,
                max_iterations,
            } = e
            {
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
                eprintln!(
                    "[dragen] {} {}{}",
                    status,
                    preview.replace('\n', "\\n"),
                    suffix
                );
            }
        })),
        on_tool_call: Some(Arc::new(|e| {
            if let AgentEvent::ToolCall { name, args } = e {
                let args_preview: Vec<String> =
                    args.iter().take(3).map(|a| format!("{:?}", a)).collect();
                let suffix = if args.len() > 3 { ", ..." } else { "" };
                eprintln!(
                    "[dragen] Tool: {}({}{})",
                    name,
                    args_preview.join(", "),
                    suffix
                );
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

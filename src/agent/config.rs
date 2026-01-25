//! Agent configuration.

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

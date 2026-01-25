//! Error types for Dragen agent.

use thiserror::Error;

/// Errors that can occur during agent execution.
#[derive(Error, Debug)]
pub enum Error {
    /// LLM client error
    #[error("LLM error: {0}")]
    Llm(#[from] tanukie::TanukieError),

    /// Sandbox execution error
    #[error("Sandbox error: {0}")]
    Sandbox(#[from] littrs::Error),

    /// No code found in LLM response
    #[error("No code block found in response")]
    NoCodeFound,

    /// Maximum iterations reached
    #[error("Maximum iterations ({0}) reached")]
    MaxIterations(usize),

    /// Agent was stopped by the LLM
    #[error("Agent completed: {0}")]
    Completed(String),

    /// Deserialization error when converting finish value to typed output
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}

/// Result type for Dragen operations.
pub type Result<T> = std::result::Result<T, Error>;

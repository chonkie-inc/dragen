//! Dragen - CodeAct-style agent framework
//!
//! Dragen provides a simple framework for building AI agents that execute
//! Python code in a secure sandbox. It uses the CodeAct pattern where the
//! LLM writes Python code to accomplish tasks, with tools exposed as
//! Python functions.
//!
//! # Quick Start
//!
//! ```ignore
//! use dragen::{Agent, AgentConfig};
//! use litter::tool;
//!
//! /// Get the current weather for a city.
//! ///
//! /// Args:
//! ///     city: The city name
//! #[tool]
//! fn get_weather(city: String) -> String {
//!     format!("Weather in {}: Sunny, 22°C", city)
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut agent = Agent::new(AgentConfig::new("gpt-4o"));
//!     agent.register(get_weather::Tool);
//!
//!     let result = agent.run("What's the weather in Paris?").await.unwrap();
//!     println!("{}", result);
//! }
//! ```
//!
//! # Sharing Data Between Agents
//!
//! Use `Context` to pass data between agents without manual Arc/Mutex management:
//!
//! ```ignore
//! use dragen::{Agent, AgentConfig, Context};
//!
//! let ctx = Context::new();
//!
//! // Planner writes output to context
//! let mut planner = Agent::new(AgentConfig::new("gpt-4o"))
//!     .to_context(&ctx, "plan");
//! planner.run::<PlanOutput>(&query).await?;
//!
//! // Executor reads from context (injected into prompt)
//! let mut executor = Agent::new(AgentConfig::new("gpt-4o"))
//!     .from_context(&ctx, "plan");
//! executor.run::<String>("Execute the plan").await?;
//! ```

mod agent;
mod context;
mod error;

pub use agent::{pyvalue_to_json, Agent, AgentConfig};
pub use context::Context;
pub use error::{Error, Result};

// Re-export litter for convenience
pub use litter;

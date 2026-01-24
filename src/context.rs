//! Shared context for passing data between agents.
//!
//! Context provides a simple key-value store that agents can read from and write to,
//! enabling data sharing without manual Arc<Mutex<>> management.
//!
//! # Example
//!
//! ```ignore
//! use dragen::{Agent, AgentConfig, Context};
//!
//! let ctx = Context::new();
//!
//! // Planner writes output to context
//! let mut planner = Agent::new(AgentConfig::new("gpt-4o"))
//!     .to_context(&ctx, "plan");
//! planner.run::<PlannerOutput>(&query).await?;
//!
//! // Executor reads from context (injected into prompt)
//! let mut executor = Agent::new(AgentConfig::new("gpt-4o"))
//!     .from_context(&ctx, "plan");
//! executor.run::<ExecutorOutput>("Write the section").await?;
//! ```

use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shared context for passing data between agents.
///
/// Context is a thread-safe key-value store that allows agents to share data.
/// Cloning is cheap (Arc-based), so you can pass the same context to multiple agents.
///
/// Data is stored as JSON internally, allowing any serializable type to be stored
/// and retrieved.
#[derive(Clone, Default)]
pub struct Context {
    data: Arc<Mutex<HashMap<String, serde_json::Value>>>,
}

impl Context {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a value in the context.
    ///
    /// The value is serialized to JSON internally.
    pub fn set<T: Serialize>(&self, key: &str, value: &T) {
        let json = serde_json::to_value(value).unwrap_or(serde_json::Value::Null);
        self.data.lock().unwrap().insert(key.to_string(), json);
    }

    /// Retrieve a value from the context.
    ///
    /// Returns `None` if the key doesn't exist or if deserialization fails.
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data
            .lock()
            .unwrap()
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get the raw JSON value for a key.
    ///
    /// Used internally for prompt injection.
    pub fn get_raw(&self, key: &str) -> Option<serde_json::Value> {
        self.data.lock().unwrap().get(key).cloned()
    }

    /// Check if a key exists in the context.
    pub fn contains(&self, key: &str) -> bool {
        self.data.lock().unwrap().contains_key(key)
    }

    /// Remove a value from the context.
    pub fn remove(&self, key: &str) -> Option<serde_json::Value> {
        self.data.lock().unwrap().remove(key)
    }

    /// Get all keys in the context.
    pub fn keys(&self) -> Vec<String> {
        self.data.lock().unwrap().keys().cloned().collect()
    }

    /// Clear all data from the context.
    pub fn clear(&self) {
        self.data.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let ctx = Context::new();
        ctx.set("name", &"Alice".to_string());

        let name: Option<String> = ctx.get("name");
        assert_eq!(name, Some("Alice".to_string()));
    }

    #[test]
    fn test_get_nonexistent() {
        let ctx = Context::new();
        let value: Option<String> = ctx.get("missing");
        assert_eq!(value, None);
    }

    #[test]
    fn test_complex_types() {
        #[derive(Serialize, serde::Deserialize, Debug, PartialEq)]
        struct Plan {
            title: String,
            sections: Vec<String>,
        }

        let ctx = Context::new();
        let plan = Plan {
            title: "My Plan".to_string(),
            sections: vec!["Intro".to_string(), "Body".to_string()],
        };

        ctx.set("plan", &plan);

        let retrieved: Option<Plan> = ctx.get("plan");
        assert_eq!(retrieved, Some(plan));
    }

    #[test]
    fn test_clone_shares_data() {
        let ctx1 = Context::new();
        let ctx2 = ctx1.clone();

        ctx1.set("key", &"value".to_string());

        let value: Option<String> = ctx2.get("key");
        assert_eq!(value, Some("value".to_string()));
    }

    #[test]
    fn test_contains() {
        let ctx = Context::new();
        assert!(!ctx.contains("key"));

        ctx.set("key", &42);
        assert!(ctx.contains("key"));
    }

    #[test]
    fn test_remove() {
        let ctx = Context::new();
        ctx.set("key", &42);

        let removed = ctx.remove("key");
        assert!(removed.is_some());
        assert!(!ctx.contains("key"));
    }

    #[test]
    fn test_keys() {
        let ctx = Context::new();
        ctx.set("a", &1);
        ctx.set("b", &2);

        let mut keys = ctx.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }
}

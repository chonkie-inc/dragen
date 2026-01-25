//! Simple agent example demonstrating CodeAct pattern.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run --example simple_agent
//!
//! Or with OpenAI:
//!   OPENAI_API_KEY=your_key cargo run --example simple_agent -- openai

use dragen::{Agent, AgentConfig};
use littrs::{tool, PyValue};

/// Add two numbers together.
///
/// Args:
///     a: First number
///     b: Second number
#[tool]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Multiply two numbers together.
///
/// Args:
///     a: First number
///     b: Second number
#[tool]
fn multiply(a: i64, b: i64) -> i64 {
    a * b
}

/// Get information about a person.
///
/// Args:
///     name: The person's name
#[tool]
fn get_person_info(name: String) -> PyValue {
    // Simulated database lookup
    let info = match name.to_lowercase().as_str() {
        "alice" => vec![
            ("name".to_string(), PyValue::Str("Alice".to_string())),
            ("age".to_string(), PyValue::Int(30)),
            ("city".to_string(), PyValue::Str("New York".to_string())),
        ],
        "bob" => vec![
            ("name".to_string(), PyValue::Str("Bob".to_string())),
            ("age".to_string(), PyValue::Int(25)),
            ("city".to_string(), PyValue::Str("San Francisco".to_string())),
        ],
        _ => vec![
            ("error".to_string(), PyValue::Str(format!("Person '{}' not found", name))),
        ],
    };
    PyValue::Dict(info)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for provider argument
    let args: Vec<String> = std::env::args().collect();
    let model = if args.len() > 1 && args[1] == "openai" {
        println!("Using OpenAI gpt-4o-mini");
        "gpt-4o-mini".to_string()
    } else {
        println!("Using Groq llama-3.3-70b-versatile");
        "llama-3.3-70b-versatile".to_string()
    };

    // Create agent with configuration
    let config = AgentConfig::new(&model)
        .max_iterations(5)
        .temperature(0.7);

    let mut agent = Agent::new(config);

    // Register tools
    agent.register(add::Tool);
    agent.register(multiply::Tool);
    agent.register(get_person_info::Tool);

    println!("\n--- Agent Tools ---");
    println!("{}", agent.sandbox().describe_tools());

    // Run a task
    let task = "What is (15 + 27) * 3? Also, look up information about Alice and tell me her age.";
    println!("\n--- Task ---");
    println!("{}", task);

    println!("\n--- Running Agent ---\n");

    match agent.run::<String>(task).await {
        Ok(result) => {
            println!("\n--- Final Answer ---");
            println!("{}", result);
        }
        Err(e) => {
            eprintln!("Agent error: {}", e);
        }
    }

    // Show conversation history
    println!("\n--- Conversation History ({} messages) ---", agent.messages().len());
    for (i, msg) in agent.messages().iter().enumerate() {
        let role = format!("{:?}", msg.role);
        let content_preview = if msg.content.len() > 100 {
            format!("{}...", &msg.content[..100])
        } else {
            msg.content.clone()
        };
        println!("[{}] {}: {}", i, role, content_preview);
    }

    Ok(())
}

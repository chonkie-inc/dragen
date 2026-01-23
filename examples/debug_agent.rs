//! Debug example showing exact prompts and responses.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run --example debug_agent

use dragen::{Agent, AgentConfig};
use litter::{tool, PyValue};

/// Add two numbers together.
///
/// Args:
///     a: First number
///     b: Second number
#[tool]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Get information about a person.
///
/// Args:
///     name: The person's name
#[tool]
fn get_person_info(name: String) -> PyValue {
    PyValue::Dict(vec![
        ("name".to_string(), PyValue::Str(name)),
        ("age".to_string(), PyValue::Int(30)),
    ])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AgentConfig::new("llama-3.3-70b-versatile")
        .max_iterations(3);

    let mut agent = Agent::new(config);
    agent.register(add::Tool);
    agent.register(get_person_info::Tool);

    let task = "Calculate 5 + 3 and look up Alice's age.";

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    SYSTEM PROMPT                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // The system prompt is built automatically from the template
    // For this debug example, we'll show what the first message looks like after agent.run() starts
    println!("(System prompt will be shown in conversation log below)");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    USER TASK                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("{}\n", task);

    // Run the agent
    match agent.run::<String>(task).await {
        Ok(result) => {
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                    CONVERSATION LOG                          ║");
            println!("╚══════════════════════════════════════════════════════════════╝\n");

            for (i, msg) in agent.messages().iter().enumerate() {
                let role = format!("{:?}", msg.role).to_uppercase();
                println!("--- [{}] {} ---", i, role);
                println!("{}\n", msg.content);
            }

            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║                    FINAL ANSWER                              ║");
            println!("╚══════════════════════════════════════════════════════════════╝\n");
            println!("{}", result);
        }
        Err(e) => {
            eprintln!("Agent error: {}", e);
        }
    }

    Ok(())
}

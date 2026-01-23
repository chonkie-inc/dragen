//! Exa search agent using the #[tool] macro.
//!
//! This example shows how to use the `#[tool]` macro for cleaner tool definitions.
//!
//! Run with:
//!   EXA_API_KEY=your_key GROQ_API_KEY=your_key cargo run --example exa_search_macro

use dragen::{Agent, AgentConfig};
use litter::{tool, PyValue};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExaSearchRequest {
    query: String,
    num_results: u32,
    #[serde(rename = "type")]
    search_type: String,
    contents: ExaContents,
}

#[derive(Serialize)]
struct ExaContents {
    text: bool,
}

#[derive(Deserialize)]
struct ExaSearchResponse {
    results: Vec<ExaResult>,
}

#[derive(Deserialize)]
struct ExaResult {
    title: String,
    url: String,
    text: Option<String>,
}

/// Search the web using Exa AI.
///
/// Args:
///     query: The search query string
///     num_results: Number of results to return (1-10)
#[tool]
fn search(query: String, num_results: Option<i64>) -> PyValue {
    let api_key = match env::var("EXA_API_KEY") {
        Ok(key) => key,
        Err(_) => return PyValue::Str("Error: EXA_API_KEY not set".to_string()),
    };

    let n = num_results.unwrap_or(3).max(1).min(10) as u32;

    let request = ExaSearchRequest {
        query,
        num_results: n,
        search_type: "auto".to_string(),
        contents: ExaContents { text: true },
    };

    let response = ureq::post("https://api.exa.ai/search")
        .header("x-api-key", &api_key)
        .header("Content-Type", "application/json")
        .send_json(&request);

    match response {
        Ok(mut resp) => match resp.body_mut().read_json::<ExaSearchResponse>() {
            Ok(data) => {
                let results: Vec<PyValue> = data
                    .results
                    .into_iter()
                    .map(|r| {
                        PyValue::Dict(vec![
                            ("title".to_string(), PyValue::Str(r.title)),
                            ("url".to_string(), PyValue::Str(r.url)),
                            (
                                "text".to_string(),
                                PyValue::Str(r.text.unwrap_or_default().chars().take(500).collect()),
                            ),
                        ])
                    })
                    .collect();
                PyValue::List(results)
            }
            Err(e) => PyValue::Str(format!("Error parsing response: {}", e)),
        },
        Err(ureq::Error::StatusCode(code)) => PyValue::Str(format!("HTTP error {}", code)),
        Err(e) => PyValue::Str(format!("Request error: {:?}", e)),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("EXA_API_KEY").is_err() {
        eprintln!("Error: EXA_API_KEY environment variable not set");
        std::process::exit(1);
    }

    let config = AgentConfig::new("llama-3.3-70b-versatile")
        .max_iterations(5)
        .system("You are a research assistant that searches the web to answer questions.");

    let mut agent = Agent::new(config);

    // Using the #[tool] macro - much cleaner!
    agent.register(search::Tool);

    let task = "What are the key features of Rust 1.85?";

    println!("Task: {}\n", task);
    println!("Running agent...\n");

    match agent.run::<String>(task).await {
        Ok(result) => {
            println!("═══════════════════════════════════════════════════════════════");
            println!("                         FINAL ANSWER                          ");
            println!("═══════════════════════════════════════════════════════════════\n");
            println!("{}", result);
        }
        Err(e) => {
            eprintln!("Agent error: {}", e);
        }
    }

    Ok(())
}

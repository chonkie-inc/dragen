//! Exa search agent example.
//!
//! Demonstrates using a real web search tool with the agent.
//!
//! Run with:
//!   EXA_API_KEY=your_key GROQ_API_KEY=your_key cargo run --example exa_search_agent

use dragen::{Agent, AgentConfig};
use littrs::{PyValue, ToolInfo};
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

fn search_web(query: String, num_results: i64) -> PyValue {
    let api_key = match env::var("EXA_API_KEY") {
        Ok(key) => key,
        Err(_) => return PyValue::Str("Error: EXA_API_KEY not set".to_string()),
    };

    let request = ExaSearchRequest {
        query,
        num_results: num_results.max(1).min(10) as u32,
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
                            (PyValue::Str("title".to_string()), PyValue::Str(r.title)),
                            (PyValue::Str("url".to_string()), PyValue::Str(r.url)),
                            (
                                PyValue::Str("text".to_string()),
                                PyValue::Str(
                                    r.text.unwrap_or_default().chars().take(500).collect(),
                                ),
                            ),
                        ])
                    })
                    .collect();
                PyValue::List(results)
            }
            Err(e) => PyValue::Str(format!("Error parsing response: {}", e)),
        },
        Err(ureq::Error::StatusCode(code)) => {
            PyValue::Str(format!("HTTP error {}", code))
        }
        Err(e) => {
            // Try to get more error details
            PyValue::Str(format!("Request error: {:?}", e))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for API keys
    if env::var("EXA_API_KEY").is_err() {
        eprintln!("Error: EXA_API_KEY environment variable not set");
        std::process::exit(1);
    }

    let config = AgentConfig::new("llama-3.3-70b-versatile")
        .max_iterations(5)
        .system("You are a research assistant that searches the web to answer questions. Use the search tool to find relevant information, then synthesize a clear answer.");

    let mut agent = Agent::new(config);

    // Register the search tool with metadata
    let search_info = ToolInfo::new("search", "Search the web using Exa")
        .arg("query", "str", "The search query")
        .arg_opt("num_results", "int", "Number of results (1-10, default 3)")
        .returns("list");

    agent.register_tool(search_info, |args| {
        let query = args
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let num_results = args.get(1).and_then(|v| v.as_int()).unwrap_or(3);
        search_web(query, num_results)
    });

    let task = "What are the latest developments in Rust programming language in 2024?";

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

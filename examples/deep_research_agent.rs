//! Deep research agent example.
//!
//! Demonstrates multi-step research where the agent performs multiple searches,
//! analyzes results, and synthesizes comprehensive answers.
//!
//! Run with:
//!   EXA_API_KEY=your_key GROQ_API_KEY=your_key cargo run --example deep_research_agent

use dragen::{Agent, AgentConfig};
use litter::{PyValue, ToolInfo};
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
#[serde(rename_all = "camelCase")]
struct ExaContents {
    text: ExaTextConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExaTextConfig {
    max_characters: u32,
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
        query: query.clone(),
        num_results: num_results.max(1).min(10) as u32,
        search_type: "auto".to_string(),
        contents: ExaContents {
            text: ExaTextConfig { max_characters: 1500 },
        },
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
                                "snippet".to_string(),
                                PyValue::Str(r.text.unwrap_or_default()),
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

use tanukie::Message;

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

fn extract_code_block(text: &str) -> Option<String> {
    // Try to extract code from ```python or <code> blocks
    if let Some(start) = text.find("```") {
        let after_start = &text[start + 3..];
        // Skip the language identifier line
        if let Some(newline) = after_start.find('\n') {
            let code_start = &after_start[newline + 1..];
            if let Some(end) = code_start.find("```") {
                return Some(code_start[..end].trim().to_string());
            }
        }
    }
    if let Some(start) = text.find("<code>") {
        let after_start = &text[start + 6..];
        if let Some(end) = after_start.find("</code>") {
            return Some(after_start[..end].trim().to_string());
        }
    }
    None
}

fn print_research_steps(messages: &[Message]) {
    let mut step = 0;

    // Skip system message
    for msg in messages.iter().skip(1) {
        let role = format!("{:?}", msg.role);

        match role.as_str() {
            "User" => {
                if msg.content.starts_with("Execution output:") {
                    // This is a tool result
                    let output = msg.content.replace("Execution output:\n```\n", "")
                        .replace("\n```", "");

                    // Check if it's a search result (contains title patterns)
                    let result_count = output.matches("'title'").count()
                        + output.matches("\"title\"").count();

                    if result_count > 0 {
                        println!("   📊 Received {} search results\n", result_count);
                    } else if output.contains("Note saved") {
                        println!("   ✓ {}\n", output.trim());
                    } else if output.trim().starts_with('[') || output.trim().starts_with("=> [") {
                        println!("   📊 Results received\n");
                    } else {
                        println!("   → {}\n", truncate(&output, 100));
                    }
                } else {
                    // Initial task
                    println!("📋 TASK: {}\n", truncate(&msg.content, 200));
                }
            }
            "Assistant" => {
                if let Some(code) = extract_code_block(&msg.content) {
                    step += 1;
                    println!("────────────────────────────────────────────────────────────────");
                    println!("STEP {}", step);
                    println!("────────────────────────────────────────────────────────────────");

                    // Show any thinking/reasoning before the code
                    let thinking = if let Some(code_start) = msg.content.find("```") {
                        msg.content[..code_start].trim()
                    } else if let Some(code_start) = msg.content.find("<code>") {
                        msg.content[..code_start].trim()
                    } else {
                        ""
                    };

                    if !thinking.is_empty() {
                        println!("💭 {}\n", truncate(thinking, 200));
                    }

                    // Parse and show the tool calls - there might be multiple in one block
                    let mut shown_action = false;

                    if code.contains("search(") {
                        // Extract the search query
                        if let Some(query_start) = code.find('"') {
                            let after_quote = &code[query_start + 1..];
                            if let Some(query_end) = after_quote.find('"') {
                                let query = &after_quote[..query_end];
                                println!("🔍 SEARCH: \"{}\"", query);
                                shown_action = true;
                            }
                        } else if let Some(query_start) = code.find('\'') {
                            let after_quote = &code[query_start + 1..];
                            if let Some(query_end) = after_quote.find('\'') {
                                let query = &after_quote[..query_end];
                                println!("🔍 SEARCH: \"{}\"", query);
                                shown_action = true;
                            }
                        }
                    }

                    if code.contains("note(") {
                        // Extract note content
                        let note_pattern = if code.contains("note(\"") { "note(\"" } else { "note('" };
                        if let Some(start) = code.find(note_pattern) {
                            let after_start = &code[start + note_pattern.len()..];
                            let end_char = if note_pattern.contains('"') { '"' } else { '\'' };
                            if let Some(end) = after_start.find(end_char) {
                                let note_content = &after_start[..end];
                                println!("📝 NOTE: \"{}\"", truncate(note_content, 60));
                                shown_action = true;
                            }
                        }
                        if !shown_action {
                            println!("📝 SAVING NOTE");
                            shown_action = true;
                        }
                    }

                    if code.contains("get_notes(") {
                        println!("📋 RETRIEVING NOTES");
                        shown_action = true;
                    }

                    if !shown_action {
                        println!("⚡ CODE: {}", truncate(&code, 80));
                    }
                }
            }
            _ => {}
        }
    }
    println!();
}

const DEEP_RESEARCH_SYSTEM: &str = r#"You are an expert research assistant that conducts thorough, multi-step investigations.

IMPORTANT: Perform 4-5 searches maximum to gather information, then call finish() with your synthesis. Do not exceed 5 searches.

CRITICAL FORMAT REQUIREMENT:
- You MUST write Python code in a ```python code block to call functions
- Do NOT use XML tags like <search> or <note>
- Example of correct format:

```python
results = search("your query here", 5)
print(results)
```

Then wait for the execution output before continuing.

Your research process (follow these steps in order):

SEARCH 1 - BROAD OVERVIEW:
Write a code block to search, then wait for results:
```python
results = search("broad topic overview", 5)
print(results)
```

After seeing results, save a note:
```python
note("Key findings: ...")
```

SEARCHES 2-5: Continue the same pattern for specific aspects. Stop after at most 5 searches.

After completing your searches (4-5 total), call finish() with your final synthesis:

```python
finish("""
Executive summary: ...

Key findings:
- ...

Notable sources:
- ...

Remaining uncertainties:
- ...
""")
```

Remember: Write ONE code block, wait for output, then continue. After at most 5 searches, you MUST call finish() with your final answer."#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("EXA_API_KEY").is_err() {
        eprintln!("Error: EXA_API_KEY environment variable not set");
        std::process::exit(1);
    }

    let config = AgentConfig::new("moonshotai/kimi-k2-instruct-0905")
        .max_iterations(25)
        .system(DEEP_RESEARCH_SYSTEM);

    let mut agent = Agent::new(config);

    // Register search tool
    let search_info = ToolInfo::new("search", "Search the web for information")
        .arg_required("query", "str", "The search query")
        .arg_optional("num_results", "int", "Number of results (1-10, default 5)")
        .returns("list");

    agent.register_tool(search_info, |args| {
        let query = args
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let num_results = args.get(1).and_then(|v| v.as_int()).unwrap_or(5);
        search_web(query, num_results)
    });

    // Register a notes tool for the agent to track its research
    let notes_info = ToolInfo::new("note", "Save a research note for later reference")
        .arg_required("content", "str", "The note content")
        .returns("str");

    let notes: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let notes_clone = notes.clone();

    agent.register_tool(notes_info, move |args| {
        let content = args
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let Ok(mut notes) = notes_clone.lock() {
            // Avoid duplicate notes
            if notes.last().map(|n| n == &content).unwrap_or(false) {
                PyValue::Str(format!("Note already saved. Total notes: {}", notes.len()))
            } else {
                notes.push(content.clone());
                PyValue::Str(format!("Note saved. Total notes: {}", notes.len()))
            }
        } else {
            PyValue::Str("Error saving note".to_string())
        }
    });

    let get_notes_info = ToolInfo::new("get_notes", "Retrieve all saved research notes")
        .returns("list");

    let notes_clone2 = notes.clone();
    agent.register_tool(get_notes_info, move |_args| {
        if let Ok(notes) = notes_clone2.lock() {
            PyValue::List(notes.iter().map(|n| PyValue::Str(n.clone())).collect())
        } else {
            PyValue::List(vec![])
        }
    });

    // Get task from command line or use default
    let task = env::args().nth(1).unwrap_or_else(|| {
        r#"Research the current state of WebAssembly (WASM) in 2024-2025:
- What are the major recent developments and new features?
- How is it being used in production (web, serverless, edge computing)?
- What are the main competing/complementary technologies?
- What does the future roadmap look like?"#.to_string()
    });

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    DEEP RESEARCH TASK                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("{}\n", &task);
    println!("Starting deep research...\n");
    println!("────────────────────────────────────────────────────────────────\n");

    match agent.run::<String>(&task).await {
        Ok(result) => {
            // Show the research process
            print_research_steps(agent.messages());

            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                    FINAL RESEARCH REPORT                     ║");
            println!("╚══════════════════════════════════════════════════════════════╝\n");
            println!("{}", result);

            // Show collected notes if any
            if let Ok(notes) = notes.lock() {
                if !notes.is_empty() {
                    println!("\n────────────────────────────────────────────────────────────────");
                    println!("📝 Research Notes Collected: {}", notes.len());
                    for (i, note) in notes.iter().enumerate() {
                        println!("  {}. {}", i + 1, truncate(note, 80));
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Agent error: {}", e);
            print_research_steps(agent.messages());
        }
    }

    Ok(())
}

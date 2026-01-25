//! Multi-agent deep research example.
//!
//! Uses a planner agent to outline sections, then a research agent to fill each section.
//!
//! Run with:
//!   EXA_API_KEY=your_key GROQ_API_KEY=your_key cargo run --example multi_agent_research "topic"

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

fn register_search_tool(agent: &mut Agent) {
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
}

const PLANNER_SYSTEM: &str = r#"You are a research planner. Your job is to:
1. Do a broad search on the given topic
2. Identify 3-4 key sections that should be researched in depth
3. Return a structured plan as a dictionary

FORMAT: Write Python code in ```python blocks.

Process:
1. Search for broad overview of the topic
2. Based on results, identify key themes/sections
3. Call finish() with a dictionary of sections

Example output format:
```python
results = search("topic overview", 5)
print(results)
```

Then after seeing results:
```python
finish({
    "section1_title": "Brief description of what to research",
    "section2_title": "Brief description of what to research",
    "section3_title": "Brief description of what to research"
})
```

Keep section titles concise (3-6 words). Descriptions should guide the research agent.
Do exactly 1-2 searches, then call finish() with your sections dictionary."#;

const RESEARCHER_SYSTEM: &str = r#"You are a research specialist. You will research a specific section of a larger report.

CRITICAL: You MUST do exactly 2 searches before calling finish(). Do NOT skip searches.

FORMAT: Write Python code in ```python blocks. ONE action per code block.

REQUIRED STEPS (follow exactly):

STEP 1 - First search:
```python
results = search("your first query about the section", 5)
print(results)
```

STEP 2 - Save note from first search:
```python
note("Key facts and figures from search results")
```

STEP 3 - Second search (different angle):
```python
results = search("different aspect or more specific detail", 5)
print(results)
```

STEP 4 - Save note and finish with structured output:
```python
note("Additional facts from second search")
finish({
    "content": "Your 2-3 paragraph section content here. Include specific facts and figures but do NOT include URLs in the content.",
    "sources": [
        "https://example.com/source1 - Description of source",
        "https://example.com/source2 - Description of source"
    ]
})
```

IMPORTANT:
- ONE action per code block
- Make second search query DIFFERENT from first
- finish() must be a dict with "content" and "sources" keys
- Put URLs ONLY in sources list, not in content
- Do NOT repeat information from previous sections"#;

fn create_planner_agent() -> Agent {
    let config = AgentConfig::new("llama-3.3-70b-versatile")
        .max_iterations(8)
        .system(PLANNER_SYSTEM);

    let mut agent = Agent::new(config);
    register_search_tool(&mut agent);

    // Custom finish tool that expects a dictionary of sections
    let finish_info = ToolInfo::new("finish", "Return the research sections as a structured plan")
        .arg_required("sections", "dict", "Dictionary mapping section titles to research descriptions")
        .returns("dict");

    agent.register_finish(finish_info, |args| {
        // Return the dict as-is
        args.get(0).cloned().unwrap_or(PyValue::None)
    });

    agent
}

fn create_researcher_agent() -> Agent {
    let config = AgentConfig::new("llama-3.3-70b-versatile")
        .max_iterations(12)
        .system(RESEARCHER_SYSTEM);

    let mut agent = Agent::new(config);
    register_search_tool(&mut agent);

    // Add notes tool
    let notes_info = ToolInfo::new("note", "Save a research note")
        .arg_required("content", "str", "The note content")
        .returns("str");

    let notes: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    agent.register_tool(notes_info, move |args| {
        let content = args
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let Ok(mut n) = notes.lock() {
            n.push(content);
            PyValue::Str(format!("Note saved. Total: {}", n.len()))
        } else {
            PyValue::Str("Error saving note".to_string())
        }
    });

    // Custom finish tool expecting structured output with content and sources
    let finish_info = ToolInfo::new("finish", "Return section content with sources")
        .arg_required("result", "dict", "Dict with 'content' (str) and 'sources' (list of str)")
        .returns("dict");

    agent.register_finish(finish_info, |args| {
        args.get(0).cloned().unwrap_or(PyValue::None)
    });

    agent
}

/// Parsed section result with content and sources separated
struct SectionResult {
    title: String,
    content: String,
    sources: Vec<String>,
}

/// Extract content and sources from a PyValue (expected to be a dict with "content" and "sources" keys)
fn extract_section_from_pyvalue(value: &PyValue) -> (String, Vec<String>) {
    match value {
        PyValue::Dict(pairs) => {
            let mut content = String::new();
            let mut sources = Vec::new();

            for (key, val) in pairs {
                match key.as_str() {
                    "content" => {
                        if let PyValue::Str(s) = val {
                            content = s.clone();
                        }
                    }
                    "sources" => {
                        if let PyValue::List(items) = val {
                            for item in items {
                                if let PyValue::Str(s) = item {
                                    sources.push(s.clone());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            (content, sources)
        }
        PyValue::Str(s) => {
            // Fallback for plain string output
            (s.clone(), vec![])
        }
        _ => (String::new(), vec![])
    }
}

fn print_separator(title: &str) {
    println!("\n{}", "═".repeat(70));
    println!("  {}", title);
    println!("{}\n", "═".repeat(70));
}

fn print_subseparator(title: &str) {
    println!("\n{}", "─".repeat(60));
    println!("  {}", title);
    println!("{}\n", "─".repeat(60));
}

fn extract_code_block(text: &str) -> Option<String> {
    if let Some(start) = text.find("```") {
        let after_start = &text[start + 3..];
        if let Some(newline) = after_start.find('\n') {
            let code_start = &after_start[newline + 1..];
            if let Some(end) = code_start.find("```") {
                return Some(code_start[..end].trim().to_string());
            }
        }
    }
    None
}

fn print_agent_step(agent_name: &str, step_num: &mut usize, content: &str) {
    if let Some(code) = extract_code_block(content) {
        *step_num += 1;

        // Extract what the agent is doing
        if code.contains("search(") {
            if let Some(query_start) = code.find('"') {
                let after_quote = &code[query_start + 1..];
                if let Some(query_end) = after_quote.find('"') {
                    let query = &after_quote[..query_end];
                    println!("  [{}] Step {}: 🔍 SEARCH \"{}\"", agent_name, step_num, query);
                    return;
                }
            }
            if let Some(query_start) = code.find('\'') {
                let after_quote = &code[query_start + 1..];
                if let Some(query_end) = after_quote.find('\'') {
                    let query = &after_quote[..query_end];
                    println!("  [{}] Step {}: 🔍 SEARCH \"{}\"", agent_name, step_num, query);
                    return;
                }
            }
        }

        if code.contains("note(") {
            println!("  [{}] Step {}: 📝 SAVING NOTE", agent_name, step_num);
            return;
        }

        if code.contains("finish(") {
            println!("  [{}] Step {}: ✅ FINISHING", agent_name, step_num);
            return;
        }

        let truncated = if code.len() > 50 { &code[..50] } else { &code };
        println!("  [{}] Step {}: ⚡ {}", agent_name, step_num, truncated);
    }
}

fn parse_sections(result: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();

    // Try format 1: JSON-style {"title": "description", ...}
    let mut remaining = result;
    while let Some(key_start) = remaining.find('"') {
        let after_key_start = &remaining[key_start + 1..];
        if let Some(key_end) = after_key_start.find('"') {
            let key = &after_key_start[..key_end];
            let after_key = &after_key_start[key_end + 1..];

            if let Some(val_start) = after_key.find('"') {
                let after_val_start = &after_key[val_start + 1..];
                if let Some(val_end) = after_val_start.find('"') {
                    let value = &after_val_start[..val_end];
                    sections.push((key.to_string(), value.to_string()));
                    remaining = &after_val_start[val_end + 1..];
                    continue;
                }
            }
        }
        break;
    }

    // If JSON parsing didn't work, try format 2: "Title: Description" per line
    if sections.is_empty() {
        for line in result.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Look for "Title: Description" pattern
            if let Some(colon_pos) = line.find(':') {
                let title = line[..colon_pos].trim();
                let desc = line[colon_pos + 1..].trim();
                // Skip if title is too long (probably not a section header)
                if !title.is_empty() && !desc.is_empty() && title.len() < 50 {
                    sections.push((title.to_string(), desc.to_string()));
                }
            }
        }
    }

    sections
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("EXA_API_KEY").is_err() {
        eprintln!("Error: EXA_API_KEY environment variable not set");
        std::process::exit(1);
    }

    let topic = env::args().nth(1).unwrap_or_else(|| {
        "The current state and future of quantum computing".to_string()
    });

    print_separator(&format!("MULTI-AGENT DEEP RESEARCH: {}", topic));

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: PLANNER AGENT
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 1: PLANNER AGENT");
    println!("The planner will research the topic and identify key sections to explore.\n");

    let mut planner = create_planner_agent();
    let planner_task = format!(
        "Research this topic and identify 3-4 key sections for a comprehensive report: {}",
        topic
    );

    println!("📋 Task: {}\n", planner_task);

    let planner_result = match planner.run::<String>(&planner_task).await {
        Ok(result) => {
            // Print intermediate steps
            let mut step_num = 0;
            for msg in planner.messages().iter().skip(1) {
                let role = format!("{:?}", msg.role);
                if role == "Assistant" {
                    print_agent_step("Planner", &mut step_num, &msg.content);
                }
            }

            println!("\n📊 Planner Output:");
            println!("{}", result);
            result
        }
        Err(e) => {
            eprintln!("Planner error: {}", e);
            // Print what we have
            let mut step_num = 0;
            for msg in planner.messages().iter().skip(1) {
                let role = format!("{:?}", msg.role);
                if role == "Assistant" {
                    print_agent_step("Planner", &mut step_num, &msg.content);
                }
            }
            return Err(e.into());
        }
    };

    // Parse the sections from planner output
    let sections = parse_sections(&planner_result);

    if sections.is_empty() {
        eprintln!("Error: Could not parse sections from planner output");
        eprintln!("Raw output: {}", planner_result);
        return Ok(());
    }

    println!("\n📑 Identified {} sections to research:", sections.len());
    for (i, (title, desc)) in sections.iter().enumerate() {
        println!("  {}. {} - {}", i + 1, title, desc);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: RESEARCH AGENTS (one per section)
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 2: RESEARCH AGENTS");

    let mut section_results: Vec<SectionResult> = Vec::new();

    for (i, (section_title, section_desc)) in sections.iter().enumerate() {
        print_subseparator(&format!("Section {}/{}: {}", i + 1, sections.len(), section_title));

        let mut researcher = create_researcher_agent();

        // Build context from previous section if available (content only, not sources)
        let previous_context = if let Some(prev) = section_results.last() {
            format!(
                "\n\nPrevious section for context:\n## {}\n{}\n\nBuild on this context but focus on your assigned section.",
                prev.title, prev.content
            )
        } else {
            String::new()
        };

        let research_task = format!(
            "Topic: {}\n\nSection to research: {}\nGuidance: {}{}\n\nResearch this section thoroughly and provide detailed, well-sourced content.",
            topic, section_title, section_desc, previous_context
        );

        println!("📋 Research Task: {} - {}\n", section_title, section_desc);

        match researcher.run::<String>(&research_task).await {
            Ok(result) => {
                // Print intermediate steps
                let mut step_num = 0;
                for msg in researcher.messages().iter().skip(1) {
                    let role = format!("{:?}", msg.role);
                    if role == "Assistant" {
                        print_agent_step(&format!("Researcher-{}", i + 1), &mut step_num, &msg.content);
                    }
                }

                // Get structured output directly from finish_value()
                let (content, sources) = if let Some(value) = researcher.finish_value() {
                    extract_section_from_pyvalue(&value)
                } else {
                    // Fallback to string result
                    (result, vec![])
                };

                println!("\n📄 Section Content Preview:");
                let preview: String = content.chars().take(300).collect();
                println!("{}...\n", preview);

                if !sources.is_empty() {
                    println!("📚 Sources: {}", sources.len());
                }

                section_results.push(SectionResult {
                    title: section_title.clone(),
                    content,
                    sources,
                });
            }
            Err(e) => {
                eprintln!("Research error for section '{}': {}", section_title, e);
                // Print what we have
                let mut step_num = 0;
                for msg in researcher.messages().iter().skip(1) {
                    let role = format!("{:?}", msg.role);
                    if role == "Assistant" {
                        print_agent_step(&format!("Researcher-{}", i + 1), &mut step_num, &msg.content);
                    }
                }
                section_results.push(SectionResult {
                    title: section_title.clone(),
                    content: format!("[Research incomplete: {}]", e),
                    sources: vec![],
                });
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3: FINAL REPORT
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("FINAL RESEARCH REPORT");

    println!("# {}\n", topic);

    // Print each section's content
    for section in &section_results {
        println!("## {}\n", section.title);
        println!("{}\n", section.content);
        println!("{}\n", "─".repeat(50));
    }

    // Collect and print all sources at the end
    let all_sources: Vec<&String> = section_results
        .iter()
        .flat_map(|s| &s.sources)
        .collect();

    if !all_sources.is_empty() {
        println!("\n## Sources\n");
        for (i, source) in all_sources.iter().enumerate() {
            println!("{}. {}", i + 1, source);
        }
        println!();
    }

    print_separator("RESEARCH COMPLETE");
    println!("Generated {} sections with {} sources for topic: {}", section_results.len(), all_sources.len(), topic);

    Ok(())
}

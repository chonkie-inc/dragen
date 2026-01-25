//! Deep Searcher - Collects high-quality sources on a topic adaptively.
//!
//! Uses Cerebras GPT-OSS-120B to iteratively search, review, and accumulate
//! relevant sources for downstream deep research pipelines.
//!
//! Run with:
//!   cargo run --example deep_searcher
//!   cargo run --example deep_searcher "Your custom topic here"
//!
//! The searcher adapts to query complexity:
//! - Narrow/specific queries: fewer sources (15-25)
//! - Broad/open-ended queries: more sources (40-50)

use dragen::{Agent, AgentConfig, AgentEvent};
use littrs::{PyValue, ToolInfo};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    #[serde(default)]
    published_date: Option<String>,
    #[serde(default)]
    author: Option<String>,
}

/// Search the web using Exa API
fn search_web(query: String, num_results: i64, search_count: Arc<AtomicUsize>) -> PyValue {
    let api_key = match env::var("EXA_API_KEY") {
        Ok(key) => key,
        Err(_) => return PyValue::Str("Error: EXA_API_KEY not set".to_string()),
    };

    // Increment search counter
    let count = search_count.fetch_add(1, Ordering::SeqCst) + 1;

    let start = Instant::now();

    let request = ExaSearchRequest {
        query: query.clone(),
        num_results: num_results.max(1).min(10) as u32,
        search_type: "auto".to_string(),
        contents: ExaContents { text: true },
    };

    let response = ureq::post("https://api.exa.ai/search")
        .header("x-api-key", &api_key)
        .header("Content-Type", "application/json")
        .send_json(&request);

    let elapsed = start.elapsed();

    match response {
        Ok(mut resp) => match resp.body_mut().read_json::<ExaSearchResponse>() {
            Ok(data) => {
                let result_count = data.results.len();
                println!(
                    "    🔍 [{}] \"{}\" → {} results ({:.1}s)",
                    count,
                    if query.len() > 50 {
                        format!("{}...", &query[..50])
                    } else {
                        query
                    },
                    result_count,
                    elapsed.as_secs_f64()
                );

                let results: Vec<PyValue> = data
                    .results
                    .into_iter()
                    .map(|r| {
                        PyValue::Dict(vec![
                            ("title".to_string(), PyValue::Str(r.title)),
                            ("url".to_string(), PyValue::Str(r.url)),
                            (
                                "snippet".to_string(),
                                PyValue::Str(
                                    r.text.unwrap_or_default().chars().take(800).collect(),
                                ),
                            ),
                            (
                                "date".to_string(),
                                PyValue::Str(r.published_date.unwrap_or_default()),
                            ),
                            (
                                "author".to_string(),
                                PyValue::Str(r.author.unwrap_or_default()),
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
        Err(e) => PyValue::Str(format!("Request error: {:?}", e)),
    }
}

const SYSTEM_PROMPT: &str = r#"<role>
You are a Deep Research Source Collector. Your task is to gather high-quality, diverse sources on a given topic through iterative web searching.
</role>

<objective>
Collect sources adaptively based on topic complexity:
- Narrow/specific topics (e.g., "Python 3.12 pattern matching syntax"): 15-25 sources
- Moderate topics (e.g., "best practices for microservices"): 25-35 sources
- Broad/open-ended topics (e.g., "future of AI agents"): 35-50 sources
</objective>

<workflow>
Work in cycles. Each cycle: ASSESS → SEARCH → REVIEW → DECIDE

<step name="assess">
Before each search round, consider:
- What aspects of the topic haven't been covered yet?
- What types of sources are missing? (academic, industry, tutorials, case studies)
- What search angles would yield complementary results?
</step>

<step name="search">
Execute 2-4 searches with diverse queries in a single code block:
```python
results_a = search("specific technical query", 8)
results_b = search("different angle or perspective", 8)
results_c = search("practical applications or case studies", 8)
```
</step>

<step name="review">
For each result, evaluate:
- Is it directly relevant to the topic?
- Is it from a credible source?
- Does it add unique information not covered by existing sources?

Filter and add good sources to collected_sources:
```python
for r in results_a:
    if is_relevant_and_quality(r):
        collected_sources.append({
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"][:200],
            "relevance": "Brief note on value"
        })
```
</step>

<step name="decide">
After reviewing, decide:
- If coverage is comprehensive for the topic's scope → finish
- If gaps remain → continue with targeted searches

Print your current count and reasoning:
```python
print(f"Collected {len(collected_sources)} sources so far")
print(f"Gaps remaining: ...")
```
</step>
</workflow>

<tools>
- search(query: str, num_results: int) → list[dict]
  Returns: [{title, url, snippet, date, author}, ...]

- finish(result: dict) → Complete the task
  Call when you have sufficient coverage for the topic's complexity
</tools>

<variables>
- collected_sources: list - Accumulate your approved sources here
</variables>

<output_format>
When finished, call:
```python
finish({
    "topic": "The research topic",
    "complexity": "narrow|moderate|broad",
    "total_sources": len(collected_sources),
    "sources": collected_sources,
    "coverage_summary": "What aspects are covered",
    "search_rounds": number_of_rounds_performed
})
```
</output_format>

<rules>
- Quality over quantity: reject duplicates, irrelevant, or low-quality sources
- Diversity matters: mix academic papers, blog posts, documentation, news, case studies
- Be adaptive: simple topics need fewer sources, complex topics need thorough coverage
- Show your reasoning: explain what you're searching for and why
- Track gaps: note what aspects still need coverage after each round
</rules>

<constraints>
IMPORTANT: The Python sandbox has limited features. You MUST NOT use:
- def (no function definitions) - use inline code instead
- try/except (no exception handling) - assume operations succeed
- globals() (not available) - just use variables directly
- {k: v for ...} (no dict comprehensions) - use explicit loops or list comprehensions
- import (no imports available)
- class (no class definitions)

SUPPORTED features:
- Variables, lists, dicts, strings, ints, floats, bools
- for loops, while loops, if/elif/else
- List comprehensions: [x for x in items if condition]
- f-strings: f"text {variable}"
- Tuple unpacking: for i, item in enumerate(items)
- List methods: append, extend, pop, clear, insert, remove
- String methods: lower, upper, strip, split, join, replace, startswith, endswith
- Dict methods: get, keys, values, items
- Builtins: len, str, int, float, bool, list, range, abs, min, max, sum, print, enumerate, zip, sorted, reversed, any, all
</constraints>"#;

/// Output structure for collected sources
#[derive(Debug, Serialize, Deserialize)]
struct DeepSearchResult {
    topic: String,
    complexity: String,
    total_sources: usize,
    sources: Vec<Source>,
    coverage_summary: String,
    #[serde(default)]
    search_rounds: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Source {
    title: String,
    url: String,
    snippet: String,
    relevance: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for API keys
    if env::var("EXA_API_KEY").is_err() {
        eprintln!("Error: EXA_API_KEY environment variable not set");
        std::process::exit(1);
    }
    if env::var("CEREBRAS_API_KEY").is_err() {
        eprintln!("Error: CEREBRAS_API_KEY environment variable not set");
        std::process::exit(1);
    }

    // Track metrics
    let search_count = Arc::new(AtomicUsize::new(0));
    let search_count_clone = search_count.clone();
    let start_time = Instant::now();

    // Configure agent with Cerebras GPT-OSS-120B
    let config = AgentConfig::new("cerebras:gpt-oss-120b")
        .max_iterations(20) // Allow enough iterations for adaptive search
        .system(SYSTEM_PROMPT);

    let mut agent = Agent::new(config)
        // Show thinking/reasoning
        .on_iteration_start(|event| {
            if let AgentEvent::IterationStart { iteration, max_iterations } = event {
                println!("\n┌─ Iteration {}/{} ─────────────────────────────────────────────",
                    iteration, max_iterations);
            }
        })
        .on_llm_response(|event| {
            if let AgentEvent::LLMResponse { content, .. } = event {
                // Extract and display the thinking/reasoning part (non-code text)
                let lines: Vec<&str> = content.lines().collect();
                let mut in_code_block = false;
                let mut thinking_lines: Vec<&str> = Vec::new();

                for line in lines {
                    if line.trim().starts_with("```") {
                        in_code_block = !in_code_block;
                        continue;
                    }
                    if !in_code_block && !line.trim().is_empty() {
                        thinking_lines.push(line);
                    }
                }

                if !thinking_lines.is_empty() {
                    println!("│");
                    println!("│ 💭 Thinking:");
                    for line in thinking_lines.iter().take(15) {
                        println!("│    {}", line);
                    }
                    if thinking_lines.len() > 15 {
                        println!("│    ... ({} more lines)", thinking_lines.len() - 15);
                    }
                }
            }
        })
        .on_code_generated(|event| {
            if let AgentEvent::CodeGenerated { code } = event {
                let line_count = code.lines().count();
                println!("│");
                println!("│ 📝 Code ({} lines):", line_count);
                for line in code.lines().take(10) {
                    println!("│    {}", line);
                }
                if line_count > 10 {
                    println!("│    ... ({} more lines)", line_count - 10);
                }
            }
        })
        .on_code_executed(|event| {
            if let AgentEvent::CodeExecuted { output, success, .. } = event {
                if !output.is_empty() {
                    println!("│");
                    println!("│ 📤 Output ({}):", if *success { "ok" } else { "error" });
                    for line in output.lines().take(10) {
                        println!("│    {}", line);
                    }
                    if output.lines().count() > 10 {
                        println!("│    ... ({} more lines)", output.lines().count() - 10);
                    }
                }
            }
        });

    // Initialize the collected_sources list in the sandbox
    agent.set_variable("collected_sources", PyValue::List(vec![]));

    // Register the search tool with timing
    let search_info = ToolInfo::new("search", "Search the web using Exa")
        .arg_required("query", "str", "The search query")
        .arg_optional("num_results", "int", "Number of results (1-10, default 8)")
        .returns("list[dict]");

    agent.register_tool(search_info, move |args| {
        let query = args
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let num_results = args.get(1).and_then(|v| v.as_int()).unwrap_or(8);
        search_web(query, num_results, search_count_clone.clone())
    });

    // Get topic from command line or use default
    let topic = env::args()
        .nth(1)
        .unwrap_or_else(|| "Recent advances in AI agents and agentic workflows".to_string());

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                       DEEP SEARCHER                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Topic: {}", topic);
    println!("  Model: cerebras:gpt-oss-120b");
    println!("  Mode:  Adaptive (complexity-based source targeting)");
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");

    let task = format!(
        "Collect sources on the following topic. First assess its complexity (narrow/moderate/broad), then search adaptively:\n\n{}",
        topic
    );

    match agent.run::<DeepSearchResult>(&task).await {
        Ok(result) => {
            let elapsed = start_time.elapsed();
            let total_searches = search_count.load(Ordering::SeqCst);

            println!("\n└──────────────────────────────────────────────────────────────────");
            println!();
            println!("╔═══════════════════════════════════════════════════════════════════╗");
            println!("║                     COLLECTION COMPLETE                           ║");
            println!("╚═══════════════════════════════════════════════════════════════════╝");
            println!();
            println!("  📊 Metrics:");
            println!("     Total time:     {:.1}s", elapsed.as_secs_f64());
            println!("     Search calls:   {}", total_searches);
            println!("     Sources found:  {}", result.total_sources);
            println!("     Complexity:     {}", result.complexity);
            if result.search_rounds > 0 {
                println!("     Search rounds:  {}", result.search_rounds);
            }
            println!();
            println!("  📋 Coverage: {}", result.coverage_summary);
            println!();
            println!("───────────────────────────────────────────────────────────────────────");
            println!("                            SOURCES                                   ");
            println!("───────────────────────────────────────────────────────────────────────");
            println!();

            for (i, source) in result.sources.iter().enumerate() {
                println!("  {}. {}", i + 1, source.title);
                println!("     {}", source.url);
                println!("     └─ {}", source.relevance);
                println!();
            }
        }
        Err(e) => {
            eprintln!("Agent error: {}", e);
        }
    }

    Ok(())
}

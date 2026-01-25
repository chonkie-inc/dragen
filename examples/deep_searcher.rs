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
You are a Deep Research Source Collector with strong analytical thinking. Your task is to efficiently gather high-quality, diverse sources through strategic searching.
</role>

<objective>
Collect 20-40 high-quality sources based on topic complexity:
- Narrow topics: ~20 sources from 2-3 search rounds
- Moderate topics: ~30 sources from 3-4 search rounds
- Broad topics: ~40 sources from 4-5 search rounds

EFFICIENCY IS KEY: Each search should yield 5-8 usable sources. If you're doing many searches with few sources added, your queries need improvement.
</objective>

<workflow>
Work in focused cycles: THINK → SEARCH → REVIEW → DECIDE

<step name="think">
BEFORE searching, clearly articulate your reasoning:
1. What specific aspects of the topic need coverage?
2. What types of sources would be most valuable? (papers, tutorials, case studies, docs)
3. What search queries will best target those gaps?

Write your thinking as comments, then execute searches:
```python
# THINKING:
# - Need academic foundations on [specific aspect]
# - Missing practical implementation examples
# - Should find industry case studies for real-world context
#
# Search strategy:
# Query 1: Target academic papers with "survey" or "review" terms
# Query 2: Find practical tutorials and implementation guides
# Query 3: Look for enterprise/industry adoption stories
```
</step>

<step name="search">
Execute 2-3 well-crafted searches per round. Quality queries > many queries:
```python
results_a = search("specific well-crafted query here", 10)
results_b = search("different angle targeting gap", 10)
```
</step>

<step name="review">
CRITICALLY evaluate each result. Only add sources that are:
- Directly relevant (not tangentially related)
- From credible sources (academic, reputable tech blogs, official docs)
- Adding NEW information (not duplicating existing sources)

Review explicitly with clear accept/reject reasoning:
```python
# Review results_a
for i, r in enumerate(results_a):
    title = r.get("title", "")
    url = r.get("url", "")
    # Check relevance and quality
    dominated_terms = ["agent", "workflow", "automation"]  # example
    is_relevant = any(term in title.lower() for term in dominated_terms)
    is_credible = any(domain in url for domain in ["arxiv", "github", "ieee", ".edu", "blog."])

    if is_relevant and is_credible:
        collected_sources.append({
            "title": title,
            "url": url,
            "snippet": r.get("snippet", "")[:200],
            "relevance": "Brief note on why this source is valuable"
        })
        print(f"✓ Added: {title[:60]}")
    else:
        print(f"✗ Skipped: {title[:60]} (reason: {'not relevant' if not is_relevant else 'not credible'})")
```
</step>

<step name="decide">
After each round, assess progress:
```python
print(f"\n=== Round Summary ===")
print(f"Sources collected: {len(collected_sources)}")
print(f"Coverage areas: [list what's covered]")
print(f"Gaps remaining: [list what's missing]")
print(f"Decision: {'CONTINUE - need more on X' or 'FINISH - comprehensive coverage achieved'}")
```
</step>
</workflow>

<tools>
- search(query: str, num_results: int) → list[dict]
  Returns: [{title, url, snippet, date, author}, ...]
  Tip: Use 10 results per search for better coverage

- finish(result: dict) → Complete the task
</tools>

<output_format>
When done, call finish() with the collected data:
```python
finish({
    "topic": "The research topic",
    "complexity": "narrow|moderate|broad",
    "total_sources": len(collected_sources),
    "sources": collected_sources,
    "coverage_summary": "Comprehensive description of what aspects are covered",
    "search_rounds": N
})
```
</output_format>

<rules>
1. THINK FIRST: Always explain your reasoning before searching
2. EFFICIENT SEARCHES: 2-3 searches per round, each yielding 5-8 sources
3. EXPLICIT REVIEW: Show accept/reject decisions with reasons
4. NO REDUNDANCY: Don't repeat similar queries across rounds
5. CLEAR PROGRESS: Track what's covered and what gaps remain
</rules>

<constraints>
The Python sandbox has LIMITED features. You MUST NOT use:
- def, lambda (no function definitions)
- try/except (no exception handling)
- globals(), locals() (not available)
- {k: v for ...} (no dict comprehensions)
- import, class

SUPPORTED:
- Variables, lists, dicts, strings, numbers, bools
- for/while loops, if/elif/else
- List comprehensions: [x for x in items if condition]
- f-strings: f"text {var}"
- Tuple unpacking: for i, item in enumerate(items)
- Methods: list.append/extend/pop, str.lower/upper/strip/split, dict.get/keys/values/items
- Builtins: len, str, int, float, bool, list, range, print, enumerate, zip, sorted, any, all, min, max, sum
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

    // Configure agent with Cerebras ZAI GLM-4.7 (thinking model)
    let config = AgentConfig::new("cerebras:zai-glm-4.7")
        .max_iterations(15) // Fewer iterations with better reasoning
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
    println!("  Model: cerebras:zai-glm-4.7 (thinking)");
    println!("  Mode:  Efficient (strategic search with explicit reasoning)");
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

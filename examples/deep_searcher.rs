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

// Cerebras Chat Completion API structs
#[derive(Serialize)]
struct CerebrasRequest {
    model: String,
    messages: Vec<CerebrasMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct CerebrasMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CerebrasResponse {
    choices: Vec<CerebrasChoice>,
}

#[derive(Deserialize)]
struct CerebrasChoice {
    message: CerebrasMessageContent,
}

#[derive(Deserialize)]
struct CerebrasMessageContent {
    content: String,
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

/// Review search results using Cerebras llama-3.3-70b for fast batch filtering
fn review_sources(results: &[PyValue], topic: &str, review_count: Arc<AtomicUsize>) -> PyValue {
    let api_key = match env::var("CEREBRAS_API_KEY") {
        Ok(key) => key,
        Err(_) => return PyValue::Str("Error: CEREBRAS_API_KEY not set".to_string()),
    };

    // Extract source info from PyValue list
    let mut sources_text = String::new();
    for (i, result) in results.iter().enumerate() {
        if let PyValue::Dict(fields) = result {
            let title = fields
                .iter()
                .find(|(k, _)| k == "title")
                .and_then(|(_, v)| v.as_str())
                .unwrap_or("Unknown");
            let url = fields
                .iter()
                .find(|(k, _)| k == "url")
                .and_then(|(_, v)| v.as_str())
                .unwrap_or("");
            let snippet = fields
                .iter()
                .find(|(k, _)| k == "snippet")
                .and_then(|(_, v)| v.as_str())
                .unwrap_or("");

            sources_text.push_str(&format!(
                "[{}] Title: {}\nURL: {}\nSnippet: {}\n\n",
                i, title, url, &snippet.chars().take(400).collect::<String>()
            ));
        }
    }

    let count = review_count.fetch_add(1, Ordering::SeqCst) + 1;
    let start = Instant::now();

    let prompt = format!(
        r#"You are a research relevance evaluator. Review these search results for the topic: "{}"

For each source, determine if it's RELEVANT or NOT RELEVANT to the research topic.

Sources to review:
{}

Respond with a JSON array. For each source, include:
- "index": the source number
- "relevant": true or false
- "reason": brief explanation (10-20 words)

Example response:
[
  {{"index": 0, "relevant": true, "reason": "Directly discusses AI agent architectures and design patterns"}},
  {{"index": 1, "relevant": false, "reason": "About general machine learning, not specifically agents"}}
]

Respond ONLY with the JSON array, no other text."#,
        topic, sources_text
    );

    let request = CerebrasRequest {
        model: "llama-3.3-70b".to_string(),
        messages: vec![CerebrasMessage {
            role: "user".to_string(),
            content: prompt,
        }],
        temperature: 0.1,
        max_tokens: 2048,
    };

    let response = ureq::post("https://api.cerebras.ai/v1/chat/completions")
        .header("Authorization", &format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .send_json(&request);

    let elapsed = start.elapsed();

    match response {
        Ok(mut resp) => match resp.body_mut().read_json::<CerebrasResponse>() {
            Ok(data) => {
                let content = &data.choices.first().map(|c| c.message.content.clone()).unwrap_or_default();

                // Parse the JSON response
                let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(content);

                match parsed {
                    Ok(reviews) => {
                        let mut relevant_sources = Vec::new();
                        let mut relevant_count = 0;
                        let mut rejected_count = 0;
                        let mut rejected_titles: Vec<String> = Vec::new();

                        for review in &reviews {
                            let index = review.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let is_relevant = review.get("relevant").and_then(|v| v.as_bool()).unwrap_or(false);
                            let reason = review.get("reason").and_then(|v| v.as_str()).unwrap_or("");

                            if let Some(PyValue::Dict(fields)) = results.get(index) {
                                let title = fields
                                    .iter()
                                    .find(|(k, _)| k == "title")
                                    .and_then(|(_, v)| v.as_str())
                                    .unwrap_or("Unknown");
                                let url = fields
                                    .iter()
                                    .find(|(k, _)| k == "url")
                                    .and_then(|(_, v)| v.as_str())
                                    .unwrap_or("");
                                let snippet = fields
                                    .iter()
                                    .find(|(k, _)| k == "snippet")
                                    .and_then(|(_, v)| v.as_str())
                                    .unwrap_or("");

                                if is_relevant {
                                    relevant_sources.push(PyValue::Dict(vec![
                                        ("title".to_string(), PyValue::Str(title.to_string())),
                                        ("url".to_string(), PyValue::Str(url.to_string())),
                                        ("snippet".to_string(), PyValue::Str(snippet.chars().take(200).collect())),
                                        ("relevance".to_string(), PyValue::Str(reason.to_string())),
                                    ]));
                                    relevant_count += 1;
                                } else {
                                    rejected_titles.push(format!("{}... ({})", &title.chars().take(50).collect::<String>(), reason));
                                    rejected_count += 1;
                                }
                            }
                        }

                        println!(
                            "    📋 [Review {}] {} sources → {} relevant, {} rejected ({:.1}s)",
                            count,
                            results.len(),
                            relevant_count,
                            rejected_count,
                            elapsed.as_secs_f64()
                        );

                        // Show rejected sources for debugging
                        for rejected in &rejected_titles {
                            println!("       ✗ {}", rejected);
                        }

                        PyValue::List(relevant_sources)
                    }
                    Err(e) => {
                        println!("    ⚠️  Review parse error: {}", e);
                        // Return original results if parsing fails
                        PyValue::List(results.to_vec())
                    }
                }
            }
            Err(e) => PyValue::Str(format!("Error parsing Cerebras response: {}", e)),
        },
        Err(ureq::Error::StatusCode(code)) => {
            PyValue::Str(format!("Cerebras HTTP error {}", code))
        }
        Err(e) => PyValue::Str(format!("Cerebras request error: {:?}", e)),
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
CRITICAL: Complete ALL steps in a SINGLE code block per round. Do NOT split across iterations.

Each round = ONE code execution with: INTENT → SEARCH → REVIEW → DECIDE

```python
# === ROUND N (all in one code block) ===
intent("What you're searching for and why")

# Search
results = search("query 1", 10) + search("query 2", 10)

# Review (uses fast LLM to filter)
reviewed = review(results, "your research topic")
collected_sources.extend(reviewed)

# Decide
print(f"Added {len(reviewed)} from {len(results)} results")
print(f"Total: {len(collected_sources)} sources")
print(f"Gaps: [what's missing]")
# If gaps remain, continue to next round. If comprehensive, call finish()
```

EFFICIENCY RULES:
- 2-3 search rounds total for most topics
- Each round: 2 searches + 1 review + decide (all in ONE code block)
- Never split search and review into separate iterations
</workflow>

<tools>
- intent(message: str) → None
  Declare your search intent before each round. REQUIRED.

- search(query: str, num_results: int) → list[dict]
  Returns: [{title, url, snippet, date, author}, ...]
  Tip: Use 10 results per search for better coverage

- review(results: list, topic: str) → list[dict]
  Uses fast LLM to filter sources. Returns only relevant ones with:
  [{title, url, snippet, relevance}, ...]
  The 'relevance' field explains why each source is valuable.

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
    let review_count = Arc::new(AtomicUsize::new(0));
    let review_count_clone = review_count.clone();
    let start_time = Instant::now();

    // Configure agent with Cerebras ZAI GLM-4.7 (thinking model)
    let config = AgentConfig::new("cerebras:zai-glm-4.7")
        .max_iterations(15) // Fewer iterations with better reasoning
        .system(SYSTEM_PROMPT);

    let mut agent = Agent::new(config)
        .on_iteration_start(|event| {
            if let AgentEvent::IterationStart { iteration, max_iterations } = event {
                println!("\n┌─ Iteration {}/{} ─────────────────────────────────────────────",
                    iteration, max_iterations);
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

    // Register the intent tool for declaring search intentions
    let intent_info = ToolInfo::new("intent", "Declare your intent before searching")
        .arg_required("message", "str", "Brief description of what you're searching for and why")
        .returns("None");

    agent.register_tool(intent_info, |args| {
        let message = args
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!("│");
        println!("│ 💭 {}", message);
        PyValue::None
    });

    // Register the review tool for batch filtering with llama-3.3-70b
    let review_info = ToolInfo::new("review", "Review and filter search results for relevance")
        .arg_required("results", "list", "List of search results to review")
        .arg_required("topic", "str", "The research topic for relevance evaluation")
        .returns("list[dict]");

    agent.register_tool(review_info, move |args| {
        let results = args.get(0).and_then(|v| {
            if let PyValue::List(items) = v {
                Some(items.clone())
            } else {
                None
            }
        }).unwrap_or_default();
        let topic = args
            .get(1)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        review_sources(&results, &topic, review_count_clone.clone())
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
            let total_reviews = review_count.load(Ordering::SeqCst);

            println!("\n└──────────────────────────────────────────────────────────────────");
            println!();
            println!("╔═══════════════════════════════════════════════════════════════════╗");
            println!("║                     COLLECTION COMPLETE                           ║");
            println!("╚═══════════════════════════════════════════════════════════════════╝");
            println!();
            println!("  📊 Metrics:");
            println!("     Total time:     {:.1}s", elapsed.as_secs_f64());
            println!("     Search calls:   {}", total_searches);
            println!("     Review calls:   {}", total_reviews);
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

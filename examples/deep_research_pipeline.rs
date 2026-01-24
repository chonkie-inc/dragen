//! Deep research pipeline example.
//!
//! Mimics the labs-deepresearch-service architecture with:
//! - Planner Agent: Creates research plan and outline
//! - Executor Agent: Generates content for each section
//! - Summary Agent: Creates executive summary
//!
//! This example demonstrates key dragen features:
//! - **`<finish>` blocks**: Agents return structured JSON directly without code execution
//! - **Typed outputs**: Results are automatically deserialized to Rust structs
//! - **Error feedback**: If JSON is malformed, the LLM gets feedback to self-correct
//! - **Tool registration**: The search tool shows how to expose functions to the sandbox
//!
//! Run with:
//!   EXA_API_KEY=your_key GROQ_API_KEY=your_key cargo run --example deep_research_pipeline "topic"

use dragen::{Agent, AgentConfig, Context};
use futures::future::join_all;
use litter::{PyValue, ToolInfo};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::{Arc, Mutex};

// ═══════════════════════════════════════════════════════════════════════════
// EXA SEARCH TOOL (shared across agents)
// ═══════════════════════════════════════════════════════════════════════════

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

#[derive(Deserialize, Clone)]
struct ExaResult {
    title: String,
    url: String,
    text: Option<String>,
}

/// Captured search result for sharing between agents
#[derive(Clone)]
struct CapturedSearch {
    query: String,
    results: Vec<ExaResult>,
}

/// Shared search log type
type SearchLog = Arc<Mutex<Vec<CapturedSearch>>>;

/// Search the web and optionally capture results to a shared log
fn search_web(query: String, num_results: i64, search_log: Option<SearchLog>) -> PyValue {
    let api_key = match env::var("EXA_API_KEY") {
        Ok(key) => key,
        Err(_) => return PyValue::Str("Error: EXA_API_KEY not set".to_string()),
    };

    let request = ExaSearchRequest {
        query: query.clone(),
        num_results: num_results.max(1).min(10) as u32,
        search_type: "auto".to_string(),
        contents: ExaContents {
            text: ExaTextConfig { max_characters: 8000 },  // Increased from 2000
        },
    };

    let response = ureq::post("https://api.exa.ai/search")
        .header("x-api-key", &api_key)
        .header("Content-Type", "application/json")
        .send_json(&request);

    match response {
        Ok(mut resp) => match resp.body_mut().read_json::<ExaSearchResponse>() {
            Ok(data) => {
                // Capture results to shared log if provided
                if let Some(log) = search_log {
                    if let Ok(mut log) = log.lock() {
                        log.push(CapturedSearch {
                            query: query.clone(),
                            results: data.results.clone(),
                        });
                    }
                }

                let results: Vec<PyValue> = data
                    .results
                    .into_iter()
                    .map(|r| {
                        PyValue::Dict(vec![
                            ("title".to_string(), PyValue::Str(r.title)),
                            ("url".to_string(), PyValue::Str(r.url)),
                            ("text".to_string(), PyValue::Str(r.text.unwrap_or_default())),
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

/// Register search tool without capturing results
fn register_search_tool(agent: &mut Agent) {
    register_search_tool_with_log(agent, None);
}

/// Register search tool with optional result capture
fn register_search_tool_with_log(agent: &mut Agent, search_log: Option<SearchLog>) {
    let search_info = ToolInfo::new("search", "Search the web for information on a topic")
        .arg_required("query", "str", "The search query")
        .arg_optional("limit", "int", "Number of results (1-10, default 5)")
        .returns("list");

    agent.register_tool(search_info, move |args| {
        let query = args.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let limit = args.get(1).and_then(|v| v.as_int()).unwrap_or(5);
        search_web(query, limit, search_log.clone())
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// PLANNER AGENT
// ═══════════════════════════════════════════════════════════════════════════

const PLANNER_SYSTEM: &str = r#"You are an expert research planner creating comprehensive market research reports.

Your job is to:
1. Search to understand the topic landscape (3-4 searches)
2. Create a detailed research plan with specific questions to answer
3. Design a comprehensive outline with sections AND subsections

FORMAT: Write Python code in ```python blocks. ONE action per block.

SEARCH STRATEGY:
- Search 1: Market overview and size
- Search 2: Key players and competitors
- Search 3: Technology trends and innovations
- Search 4: Market dynamics and future outlook

WHEN READY TO FINISH, use a <finish> block with valid JSON:

<finish>
{
    "plan": "RESEARCH OBJECTIVES:\n1. [Question 1]\n2. [Question 2]\n\nFOR EACH SECTION:\n- Section Name: [search queries]\n- Key data: [metrics, companies, trends]",
    "outline": {
        "title": "Comprehensive Report Title",
        "sections": [
            {
                "title": "Executive Summary and Market Overview",
                "description": "Market size, growth rates, key trends, and major players overview",
                "subsections": ["Market Size and Projections", "Key Growth Drivers", "Market Segmentation"]
            },
            {
                "title": "Competitive Landscape Analysis",
                "description": "Detailed analysis of market leaders, their offerings, and positioning",
                "subsections": ["Tier 1 Market Leaders", "Emerging Players", "Competitive Positioning"]
            },
            {
                "title": "Technology and Innovation Trends",
                "description": "Current technologies, AI/ML integration, and future developments",
                "subsections": ["Current Technology Stack", "AI and Automation", "Emerging Innovations"]
            },
            {
                "title": "Market Outlook and Recommendations",
                "description": "Future projections, opportunities, challenges, and strategic recommendations",
                "subsections": ["Growth Projections", "Opportunities and Challenges", "Strategic Recommendations"]
            }
        ]
    }
}
</finish>

IMPORTANT: Use valid JSON format with \n for newlines in strings. Do NOT use Python triple-quotes.

REQUIREMENTS:
- Create 4-6 comprehensive sections
- Each section MUST have 2-4 subsections
- Subsections guide the executor on specific topics to cover
- Plan should include specific search queries for each section
- Include what metrics and data points to look for"#;

fn create_planner_agent(search_log: Option<SearchLog>) -> Agent {
    // Claude 4.5 Opus for high-quality planning and research strategy
    let config = AgentConfig::new("claude-opus-4-5-20251101")
        .max_iterations(15)
        .system(PLANNER_SYSTEM);

    let mut agent = Agent::new(config);
    register_search_tool_with_log(&mut agent, search_log);

    // Note: No register_finish() needed - <finish> blocks are handled directly by the framework
    // and deserialized to the typed PlannerOutput struct

    agent
}

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTOR AGENT
// ═══════════════════════════════════════════════════════════════════════════

const EXECUTOR_SYSTEM: &str = r#"You are an expert research analyst writing comprehensive market research reports.

You will receive:
- A research plan with specific questions to answer
- The current section and its subsections to write
- Previous section content (to avoid repetition)

Your job is to produce DETAILED, DATA-RICH content for ONE section.

FORMAT: Write Python code in ```python blocks. ONE action per block.

RESEARCH PROCESS (3-4 searches required):
1. Search for primary data (market size, key players)
2. Search for specific details (features, pricing, comparisons)
3. Search for trends and analysis
4. Search for additional perspectives if needed

CONTENT REQUIREMENTS:
- Write 400-600 words per section (NOT 2-3 paragraphs - write COMPREHENSIVE content)
- Include ALL subsections as ### headers
- Include specific numbers: market sizes, growth rates, percentages, pricing
- Name specific companies, products, and technologies
- Include comparisons and analysis, not just descriptions
- Add bullet points for lists of features, players, or trends
- Include a comparison table or matrix where appropriate using markdown

WHEN READY TO FINISH, use a <finish> block with valid JSON:

<finish>
{
    "content": "The competitive intelligence market has experienced remarkable growth...\n\n### Market Size and Projections\n\nThe global market was valued at $X billion in 2024...\n\n### Key Growth Drivers\n\n1. **Digital Transformation** - Organizations increasingly require...\n2. **AI Integration** - Machine learning capabilities have...",
    "sources": [
        "https://example.com - Market size report 2024",
        "https://example2.com - Industry analysis"
    ]
}
</finish>

CRITICAL RULES:
- Do NOT include the main section title (## header) - it's added automatically
- DO include subsection headers (### headers) for each subsection in the outline
- Start with substantive content, not the section title
- Write MUCH more than 2-3 paragraphs - aim for comprehensive coverage
- Include specific data points from your research
- Do NOT repeat information from previous sections
- Always cite sources with URLs
- Use \n for newlines in the JSON content field"#;

fn create_executor_agent() -> Agent {
    // Gemini 3 Flash for fast, cost-effective section generation
    let config = AgentConfig::new("gemini-3-flash-preview")
        .max_iterations(15)  // Allow more iterations for thorough research
        .system(EXECUTOR_SYSTEM);

    let mut agent = Agent::new(config);
    register_search_tool(&mut agent);

    // Note: No register_finish() needed - <finish> blocks are handled directly by the framework
    // and deserialized to the typed ExecutorOutput struct

    agent
}

// ═══════════════════════════════════════════════════════════════════════════
// SUMMARY AGENT
// ═══════════════════════════════════════════════════════════════════════════

const SUMMARY_SYSTEM: &str = r#"You are a research summarizer. You will be given the full report content.

Your job is to create an executive summary with key insights.

WHEN READY TO FINISH, use a <finish> block with valid JSON:

<finish>
{
    "key_takeaways": [
        "First major insight from the research",
        "Second major insight",
        "Third major insight"
    ],
    "key_metrics": [
        {"metric": "Market Size", "value": "$X billion", "context": "Brief context"},
        {"metric": "Growth Rate", "value": "X%", "context": "Brief context"}
    ],
    "opportunities": [
        "First opportunity identified",
        "Second opportunity"
    ],
    "risks": [
        "First risk or challenge",
        "Second risk"
    ]
}
</finish>

IMPORTANT:
- Extract the most important insights from the report
- Include specific numbers and metrics where available
- Be concise but comprehensive
- Focus on actionable insights"#;

fn create_summary_agent() -> Agent {
    // Gemini 3 Flash for fast summarization
    let config = AgentConfig::new("gemini-3-flash-preview")
        .max_iterations(5)
        .system(SUMMARY_SYSTEM);

    // Note: No tools needed for summary agent - it just reads the report and outputs structured data
    // No register_finish() needed - <finish> blocks are handled directly by the framework
    // and deserialized to the typed SummaryOutput struct

    Agent::new(config)
}

// ═══════════════════════════════════════════════════════════════════════════
// REVIEWER AGENT
// ═══════════════════════════════════════════════════════════════════════════

const REVIEWER_SYSTEM: &str = r#"<role>
You are an expert research editor. You review report sections and make targeted edits to improve coherence and remove redundancy.
</role>

<tools>
You have access to the following tool:

edit(section, action, text=None, old=None, new=None)
  - section: int, the section number (1-based)
  - action: str, one of "prepend", "append", "remove", "replace"
  - text: str, the text to add (for prepend/append) or remove (for remove)
  - old: str, the text to find (for replace)
  - new: str, the replacement text (for replace)

Examples:
  edit(2, "prepend", text="Building on the market analysis above, ")
  edit(3, "remove", text="The market is projected to reach $47 billion by 2030.")
  edit(4, "replace", old="2024", new="2025")
  edit(5, "append", text="This sets the stage for the challenges ahead.")
</tools>

<instructions>
1. Read all sections carefully
2. Write ONE Python code block with ALL your edit() calls AND finish() at the end
3. Focus on:
   - Adding transitions to sections 2+ (prepend a sentence connecting to previous section)
   - Removing redundant facts that appear in multiple sections
4. IMPORTANT: Always end your code block with finish("summary of changes made")
</instructions>

<rules>
- Make MINIMAL edits - preserve original content
- Transitions: 1-2 sentences connecting to previous section's theme
- Only remove TRULY redundant content (exact same facts repeated)
- For "remove" and "replace", text must match EXACTLY
- Put ALL edits for this pass in ONE code block
</rules>"#;

fn create_reviewer_agent(sections: Arc<Mutex<Vec<SectionResult>>>) -> Agent {
    let config = AgentConfig::new("claude-opus-4-5-20251101")
        .max_iterations(10)  // Allow retries if some edits fail
        .system(REVIEWER_SYSTEM);

    let mut agent = Agent::new(config);

    // Register the edit tool
    let sections_clone = Arc::clone(&sections);
    let edit_info = ToolInfo::new("edit", "Edit a section of the report")
        .arg_required("section", "int", "Section number (1-based)")
        .arg_required("action", "str", "One of: prepend, append, remove, replace")
        .arg_optional("text", "str", "Text to add (prepend/append) or remove")
        .arg_optional("old", "str", "Text to find (replace)")
        .arg_optional("new", "str", "Replacement text (replace)")
        .returns("str");

    agent.register_tool(edit_info, move |args| {
        let section_num = args.get(0).and_then(|v| v.as_int()).unwrap_or(0) as usize;
        let action = args.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let text = args.get(2).and_then(|v| v.as_str()).map(|s| s.to_string());
        let old = args.get(3).and_then(|v| v.as_str()).map(|s| s.to_string());
        let new = args.get(4).and_then(|v| v.as_str()).map(|s| s.to_string());

        let mut sections = sections_clone.lock().unwrap();

        if section_num < 1 || section_num > sections.len() {
            return PyValue::Str(format!("Error: Invalid section {}", section_num));
        }

        let section = &mut sections[section_num - 1];

        match action.as_str() {
            "prepend" => {
                if let Some(t) = text {
                    section.content = format!("{}\n\n{}", t, section.content);
                    PyValue::Str(format!("✓ Prepended to section {}", section_num))
                } else {
                    PyValue::Str("Error: 'text' required for prepend".to_string())
                }
            }
            "append" => {
                if let Some(t) = text {
                    section.content = format!("{}\n\n{}", section.content, t);
                    PyValue::Str(format!("✓ Appended to section {}", section_num))
                } else {
                    PyValue::Str("Error: 'text' required for append".to_string())
                }
            }
            "remove" => {
                if let Some(t) = text {
                    if section.content.contains(&t) {
                        section.content = section.content.replace(&t, "");
                        PyValue::Str(format!("✓ Removed from section {}", section_num))
                    } else {
                        PyValue::Str(format!("Warning: Text not found in section {}", section_num))
                    }
                } else {
                    PyValue::Str("Error: 'text' required for remove".to_string())
                }
            }
            "replace" => {
                if let (Some(o), Some(n)) = (old, new) {
                    if section.content.contains(&o) {
                        section.content = section.content.replace(&o, &n);
                        PyValue::Str(format!("✓ Replaced in section {}", section_num))
                    } else {
                        PyValue::Str(format!("Warning: Text not found in section {}", section_num))
                    }
                } else {
                    PyValue::Str("Error: 'old' and 'new' required for replace".to_string())
                }
            }
            _ => PyValue::Str(format!("Error: Unknown action '{}'", action))
        }
    });

    agent
}

// ═══════════════════════════════════════════════════════════════════════════
// TYPED OUTPUT STRUCTS (using serde for automatic deserialization)
// ═══════════════════════════════════════════════════════════════════════════

/// Planner agent output
#[derive(Debug, Deserialize, Serialize)]
struct PlannerOutput {
    plan: String,
    outline: Outline,
}

#[derive(Debug, Deserialize, Serialize)]
struct Outline {
    title: String,
    sections: Vec<Section>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Section {
    title: String,
    description: String,
    #[serde(default)]
    subsections: Vec<String>,
}

/// Executor agent output
#[derive(Debug, Deserialize, Serialize)]
struct ExecutorOutput {
    content: String,
    #[serde(default)]
    sources: Vec<String>,
}

/// Summary agent output
#[derive(Debug, Default, Deserialize, Serialize)]
struct SummaryOutput {
    #[serde(default)]
    key_takeaways: Vec<String>,
    #[serde(default)]
    key_metrics: Vec<KeyMetric>,
    #[serde(default)]
    opportunities: Vec<String>,
    #[serde(default)]
    risks: Vec<String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct KeyMetric {
    #[serde(default)]
    metric: String,
    #[serde(default)]
    value: String,
    #[serde(default)]
    context: String,
}

// Reviewer returns a simple string summary via finish()
// (edits are applied via the edit() tool during execution)

/// Internal struct for collecting section results
#[derive(Clone)]
struct SectionResult {
    title: String,
    content: String,
    sources: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

fn print_separator(title: &str) {
    println!("\n{}", "═".repeat(70));
    println!("  {}", title);
    println!("{}\n", "═".repeat(70));
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN PIPELINE
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("EXA_API_KEY").is_err() {
        eprintln!("Error: EXA_API_KEY environment variable not set");
        std::process::exit(1);
    }

    let query = env::args().nth(1).unwrap_or_else(|| {
        "What are the key trends in AI agents and agentic frameworks in 2025?".to_string()
    });

    println!("\n{}", "═".repeat(70));
    println!("  DEEP RESEARCH PIPELINE");
    println!("  Query: {}", query);
    println!("{}\n", "═".repeat(70));

    // Create shared context for passing data between agents
    let ctx = Context::new();

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: PLANNER
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 1: PLANNER AGENT");

    // Create shared search log to capture planner's research
    let search_log: SearchLog = Arc::new(Mutex::new(Vec::new()));

    let mut planner = create_planner_agent(Some(search_log.clone()))
        .to_context(&ctx, "plan");  // Output automatically saved to context
    let planner_task = format!(
        "Create a research plan and outline for this query: {}",
        query
    );

    println!("📋 Task: {}\n", planner_task);

    // Typed output - automatically saved to context via to_context()
    let planner_output: PlannerOutput = match planner.run::<PlannerOutput>(&planner_task).await {
        Ok(output) => {
            println!("✅ Planner completed\n");
            println!("📄 Report Title: {}", output.outline.title);
            println!("📝 Plan: {}...", &output.plan.chars().take(200).collect::<String>());
            println!("\n📑 Outline ({} sections):", output.outline.sections.len());
            for (i, section) in output.outline.sections.iter().enumerate() {
                println!("  {}. {}", i + 1, section.title);
                println!("     {}", section.description);
                for subsection in &section.subsections {
                    println!("       • {}", subsection);
                }
            }
            output
        }
        Err(e) => {
            eprintln!("❌ Planner error: {}", e);
            return Ok(());
        }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: PARALLEL EXECUTOR AGENTS
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 2: PARALLEL EXECUTOR AGENTS");

    // Extract captured research from planner and save to context
    let total_results: usize = {
        let log = search_log.lock().unwrap();
        let total = log.iter().map(|s| s.results.len()).sum();

        if !log.is_empty() {
            let mut research = String::from("═══ RESEARCH FROM PLANNER (use this data, search only if needed) ═══\n\n");
            for (i, search) in log.iter().enumerate() {
                research.push_str(&format!("── Search {}: \"{}\" ──\n", i + 1, search.query));
                for result in &search.results {
                    research.push_str(&format!("\n📄 {}\n", result.title));
                    research.push_str(&format!("🔗 {}\n", result.url));
                    if let Some(text) = &result.text {
                        // Truncate very long texts for context window efficiency
                        let truncated: String = text.chars().take(4000).collect();
                        let truncated = if text.len() > truncated.len() {
                            format!("{}...", truncated)
                        } else {
                            truncated
                        };
                        research.push_str(&format!("{}\n", truncated));
                    }
                    research.push_str("\n");
                }
            }
            // Save research to context for executors to read
            ctx.set("search_log", &research);
        }

        total
    };

    println!("📚 Passing {} search results from planner to executors via Context", total_results);
    println!("🚀 Launching {} executor agents in parallel...\n", planner_output.outline.sections.len());

    // Create futures for all sections to run in parallel
    // Each executor reads from shared context (plan + search_log)
    let executor_futures: Vec<_> = planner_output.outline.sections
        .iter()
        .enumerate()
        .map(|(i, section)| {
            let ctx = ctx.clone();  // Cheap clone (Arc-based)
            let section_title = section.title.clone();
            let section_description = section.description.clone();
            let subsections = section.subsections.clone();
            let section_num = i + 1;
            let total_sections = planner_output.outline.sections.len();

            async move {
                println!("  ▶ Starting section {}/{}: {}", section_num, total_sections, section_title);

                // Executor reads plan and search_log from context (auto-injected into prompt)
                let mut executor = create_executor_agent()
                    .from_context(&ctx, "plan")
                    .from_context(&ctx, "search_log");

                // Format subsections for the task
                let subsections_str = if subsections.is_empty() {
                    String::new()
                } else {
                    format!("\nSubsections to cover:\n{}",
                        subsections.iter()
                            .map(|s| format!("  - {}", s))
                            .collect::<Vec<_>>()
                            .join("\n"))
                };

                // Task is now simpler - context (plan + research) is auto-injected
                let executor_task = format!(
                    "CURRENT SECTION TO WRITE:\nTitle: {}\nDescription: {}{}\n\nIMPORTANT: Use the research data from the context above. Only search if you need additional specific information not covered.\n\nWrite comprehensive content covering ALL subsections. Use ### headers for each subsection.",
                    section_title, section_description, subsections_str
                );

                // Typed output - no manual extraction needed!
                match executor.run::<ExecutorOutput>(&executor_task).await {
                    Ok(output) => {
                        println!("  ✅ Section {}/{} complete: {}", section_num, total_sections, section_title);
                        SectionResult {
                            title: section_title,
                            content: output.content,
                            sources: output.sources,
                        }
                    }
                    Err(e) => {
                        eprintln!("  ⚠️ Section {}/{} error: {}", section_num, total_sections, e);
                        SectionResult {
                            title: section_title,
                            content: format!("[Error: {}]", e),
                            sources: vec![],
                        }
                    }
                }
            }
        })
        .collect();

    // Run all executors in parallel
    let raw_section_results: Vec<SectionResult> = join_all(executor_futures).await;

    println!("\n✅ All {} sections generated in parallel", raw_section_results.len());

    // Collect all sources before review
    let all_sources: Vec<String> = raw_section_results
        .iter()
        .flat_map(|s| s.sources.clone())
        .collect();

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2.5: REVIEWER AGENT (edit tool with retry capability)
    // ═══════════════════════════════════════════════════════════════════════

    // Check if reviewer is enabled (set SKIP_REVIEWER=1 to disable)
    let skip_reviewer = env::var("SKIP_REVIEWER").is_ok();

    let section_results: Vec<SectionResult> = if skip_reviewer {
        println!("⏭️  Skipping reviewer (SKIP_REVIEWER is set)\n");
        raw_section_results
    } else {
        print_separator("PHASE 2.5: REVIEWER AGENT (OPUS)");

        // Shared sections that the reviewer can edit via the edit() tool
        let shared_sections: Arc<Mutex<Vec<SectionResult>>> = Arc::new(Mutex::new(raw_section_results));

        // Build the content for the reviewer to see
        let sections_content: String = {
            let sections = shared_sections.lock().unwrap();
            sections
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    format!(
                        "=== SECTION {} ===\nTitle: {}\n\n{}\n",
                        i + 1, s.title, s.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let reviewer_task = format!(
            "Review this research report titled: \"{}\"\n\n{}\n\nUse edit() to add transitions to sections 2+ and remove redundant content. Put all edits in ONE code block. When done, call finish(\"summary\").",
            planner_output.outline.title, sections_content
        );

        println!("📝 Reviewing {} sections...\n", shared_sections.lock().unwrap().len());

        let mut reviewer = create_reviewer_agent(Arc::clone(&shared_sections));

        match reviewer.run::<String>(&reviewer_task).await {
            Ok(summary) => {
                println!("✅ Review complete: {}", summary);
            }
            Err(e) => {
                eprintln!("⚠️ Reviewer error: {}. Using sections as-is.", e);
            }
        }

        // Extract the edited sections
        let sections = shared_sections.lock().unwrap();
        sections.clone()
    };

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3: SUMMARY
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 3: SUMMARY AGENT");

    // Combine all section content for summary
    let full_report: String = section_results
        .iter()
        .map(|s| format!("## {}\n\n{}", s.title, s.content))
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut summary_agent = create_summary_agent();
    let summary_task = format!(
        "Create an executive summary for this research report:\n\n# {}\n\n{}",
        planner_output.outline.title, full_report
    );

    // Typed output - no manual extraction needed!
    let summary: Option<SummaryOutput> = match summary_agent.run::<SummaryOutput>(&summary_task).await {
        Ok(output) => Some(output),
        Err(e) => {
            eprintln!("⚠️ Summary error: {}", e);
            None
        }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // FINAL OUTPUT
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("FINAL RESEARCH REPORT");

    // Report Title
    println!("# {}\n", planner_output.outline.title);

    // Executive Summary (at the top, after title)
    if let Some(ref summary) = summary {
        println!("## Executive Summary\n");

        if !summary.key_takeaways.is_empty() {
            println!("**Key Takeaways:**");
            for (i, takeaway) in summary.key_takeaways.iter().enumerate() {
                println!("{}. {}", i + 1, takeaway);
            }
            println!();
        }

        if !summary.key_metrics.is_empty() {
            println!("**Key Metrics:**");
            for metric in &summary.key_metrics {
                println!("- **{}**: {} ({})", metric.metric, metric.value, metric.context);
            }
            println!();
        }

        if !summary.opportunities.is_empty() {
            println!("**Opportunities:**");
            for opp in &summary.opportunities {
                println!("- {}", opp);
            }
            println!();
        }

        if !summary.risks.is_empty() {
            println!("**Risks & Challenges:**");
            for risk in &summary.risks {
                println!("- {}", risk);
            }
            println!();
        }

        println!("{}\n", "─".repeat(50));
    }

    // Section Content
    for section in &section_results {
        println!("## {}\n", section.title);
        println!("{}\n", section.content);
    }

    // Sources (deduplicated by URL)
    if !all_sources.is_empty() {
        println!("\n## Sources\n");
        let mut seen_urls: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut unique_sources: Vec<&String> = Vec::new();

        for source in &all_sources {
            // Extract URL (first part before " - " or the whole thing)
            let url = source.split(" - ").next().unwrap_or(source).trim();
            if seen_urls.insert(url.to_string()) {
                unique_sources.push(source);
            }
        }

        for (i, source) in unique_sources.iter().enumerate() {
            println!("{}. {}", i + 1, source);
        }
    }

    // Count unique sources for final message
    let unique_source_count = {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for source in &all_sources {
            let url = source.split(" - ").next().unwrap_or(source).trim();
            seen.insert(url.to_string());
        }
        seen.len()
    };

    print_separator("PIPELINE COMPLETE");
    println!(
        "Generated {} sections with {} unique sources",
        section_results.len(),
        unique_source_count
    );

    Ok(())
}

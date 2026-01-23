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

use dragen::{Agent, AgentConfig};
use litter::{PyValue, ToolInfo};
use serde::{Deserialize, Serialize};
use std::env;

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
            text: ExaTextConfig { max_characters: 2000 },
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

fn register_search_tool(agent: &mut Agent) {
    let search_info = ToolInfo::new("search", "Search the web for information on a topic")
        .arg_required("query", "str", "The search query")
        .arg_optional("limit", "int", "Number of results (1-10, default 5)")
        .returns("list");

    agent.register_tool(search_info, |args| {
        let query = args.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let limit = args.get(1).and_then(|v| v.as_int()).unwrap_or(5);
        search_web(query, limit)
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

fn create_planner_agent() -> Agent {
    let config = AgentConfig::new("moonshotai/kimi-k2-instruct-0905")
        .max_iterations(15)
        .system(PLANNER_SYSTEM);

    let mut agent = Agent::new(config);
    register_search_tool(&mut agent);

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
    let config = AgentConfig::new("moonshotai/kimi-k2-instruct-0905")
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
    let config = AgentConfig::new("moonshotai/kimi-k2-instruct-0905")
        .max_iterations(5)
        .system(SUMMARY_SYSTEM);

    // Note: No tools needed for summary agent - it just reads the report and outputs structured data
    // No register_finish() needed - <finish> blocks are handled directly by the framework
    // and deserialized to the typed SummaryOutput struct

    Agent::new(config)
}

// ═══════════════════════════════════════════════════════════════════════════
// TYPED OUTPUT STRUCTS (using serde for automatic deserialization)
// ═══════════════════════════════════════════════════════════════════════════

/// Planner agent output
#[derive(Debug, Deserialize)]
struct PlannerOutput {
    plan: String,
    outline: Outline,
}

#[derive(Debug, Deserialize)]
struct Outline {
    title: String,
    sections: Vec<Section>,
}

#[derive(Debug, Deserialize)]
struct Section {
    title: String,
    description: String,
    #[serde(default)]
    subsections: Vec<String>,
}

/// Executor agent output
#[derive(Debug, Deserialize)]
struct ExecutorOutput {
    content: String,
    #[serde(default)]
    sources: Vec<String>,
}

/// Summary agent output
#[derive(Debug, Default, Deserialize)]
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

#[derive(Debug, Default, Deserialize)]
struct KeyMetric {
    #[serde(default)]
    metric: String,
    #[serde(default)]
    value: String,
    #[serde(default)]
    context: String,
}

/// Internal struct for collecting section results
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

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: PLANNER
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 1: PLANNER AGENT");

    let mut planner = create_planner_agent();
    let planner_task = format!(
        "Create a research plan and outline for this query: {}",
        query
    );

    println!("📋 Task: {}\n", planner_task);

    // Typed output - no manual extraction needed!
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
    // PHASE 2: EXECUTOR (per section)
    // ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 2: EXECUTOR AGENTS");

    let mut section_results: Vec<SectionResult> = Vec::new();
    let mut all_sources: Vec<String> = Vec::new();

    for (i, section) in planner_output.outline.sections.iter().enumerate() {
        println!("\n{}", "─".repeat(60));
        println!("  Section {}/{}: {}", i + 1, planner_output.outline.sections.len(), section.title);
        println!("{}\n", "─".repeat(60));

        let mut executor = create_executor_agent();

        // Build context from previous section
        let previous_context = if let Some(prev) = section_results.last() {
            format!(
                "\n\nPREVIOUS SECTION (for context, do not repeat):\n## {}\n{}",
                prev.title, prev.content
            )
        } else {
            String::new()
        };

        // Format subsections for the task
        let subsections_str = if section.subsections.is_empty() {
            String::new()
        } else {
            format!("\nSubsections to cover:\n{}",
                section.subsections.iter()
                    .map(|s| format!("  - {}", s))
                    .collect::<Vec<_>>()
                    .join("\n"))
        };

        let executor_task = format!(
            "RESEARCH PLAN:\n{}\n\nCURRENT SECTION TO WRITE:\nTitle: {}\nDescription: {}{}{}\n\nWrite comprehensive content covering ALL subsections. Use ### headers for each subsection.",
            planner_output.plan, section.title, section.description, subsections_str, previous_context
        );

        // Typed output - no manual extraction needed!
        match executor.run::<ExecutorOutput>(&executor_task).await {
            Ok(output) => {
                println!("✅ Section complete");
                println!("📄 Preview: {}...", output.content.chars().take(200).collect::<String>());
                println!("📚 Sources: {}", output.sources.len());

                all_sources.extend(output.sources.clone());
                section_results.push(SectionResult {
                    title: section.title.clone(),
                    content: output.content,
                    sources: output.sources,
                });
            }
            Err(e) => {
                eprintln!("⚠️ Executor error: {}", e);
                section_results.push(SectionResult {
                    title: section.title.clone(),
                    content: format!("[Error: {}]", e),
                    sources: vec![],
                });
            }
        }
    }

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

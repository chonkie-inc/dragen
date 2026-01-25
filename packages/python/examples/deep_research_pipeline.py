#!/usr/bin/env python3
"""
Deep research pipeline example - Python version.

Mimics the labs-deepresearch-service architecture with:
- Planner Agent: Creates research plan and outline
- Executor Agent: Generates content for each section (in parallel)
- Summary Agent: Creates executive summary

Run with:
    EXA_API_KEY=your_key ANTHROPIC_API_KEY=your_key python deep_research_pipeline.py "topic"
"""

import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from dragen import Agent, Context


# ═══════════════════════════════════════════════════════════════════════════
# EXA SEARCH TOOL
# ═══════════════════════════════════════════════════════════════════════════

search_counter = 0
captured_searches = []  # Captures planner searches to pass to executors


def search(query: str, limit: int = 5) -> list:
    """Search the web for information using Exa API."""
    global search_counter
    search_counter += 1

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return [{"error": "EXA_API_KEY not set"}]

    try:
        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "numResults": min(max(int(limit), 1), 10),
                "type": "auto",
                "contents": {
                    "text": {"maxCharacters": 8000}
                }
            }
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "text": r.get("text", "")
            })

        # Capture for passing to executors via Context
        captured_searches.append({"query": query, "results": results})
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = '''You are an expert research planner creating comprehensive market research reports.

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
    "plan": "RESEARCH OBJECTIVES:\\n1. [Question 1]\\n2. [Question 2]\\n\\nFOR EACH SECTION:\\n- Section Name: [search queries]\\n- Key data: [metrics, companies, trends]",
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

IMPORTANT: Use valid JSON format with \\n for newlines in strings. Do NOT use Python triple-quotes.

REQUIREMENTS:
- Create 4-6 comprehensive sections
- Each section MUST have 2-4 subsections
- Subsections guide the executor on specific topics to cover
- Plan should include specific search queries for each section
- Include what metrics and data points to look for'''

EXECUTOR_SYSTEM = '''You are an expert research analyst writing comprehensive market research reports.

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
    "content": "The competitive intelligence market has experienced remarkable growth...\\n\\n### Market Size and Projections\\n\\nThe global market was valued at $X billion in 2024...\\n\\n### Key Growth Drivers\\n\\n1. **Digital Transformation** - Organizations increasingly require...\\n2. **AI Integration** - Machine learning capabilities have...",
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
- Use \\n for newlines in the JSON content field'''

SUMMARY_SYSTEM = '''You are a research summarizer. You will be given the full report content.

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
- Focus on actionable insights'''

REVIEWER_SYSTEM = '''<role>
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
</rules>'''


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def print_separator(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")


@dataclass
class SectionResult:
    title: str
    content: str
    sources: list = field(default_factory=list)


def execute_section(section_info: dict, ctx: Context, section_num: int, total: int) -> SectionResult:
    """Execute a single section (run in thread pool)."""
    title = section_info["title"]
    description = section_info["description"]
    subsections = section_info.get("subsections", [])

    print(f"  ▶ Starting section {section_num}/{total}: {title}")

    # Create executor agent
    agent = Agent(
        "gemini-3-flash-preview",
        max_iterations=15,
        system=EXECUTOR_SYSTEM
    )
    agent.tool(search)  # Register shared search tool

    # Read from context
    agent.from_context(ctx, "plan")
    agent.from_context(ctx, "search_log")

    # Format subsections
    subsections_str = ""
    if subsections:
        subsections_str = "\nSubsections to cover:\n" + "\n".join(f"  - {s}" for s in subsections)

    task = f"""CURRENT SECTION TO WRITE:
Title: {title}
Description: {description}{subsections_str}

IMPORTANT: Use the research data from the context above. Only search if you need additional specific information not covered.

Write comprehensive content covering ALL subsections. Use ### headers for each subsection."""

    try:
        result = agent.run(task)
        print(f"  ✅ Section {section_num}/{total} complete: {title}")

        return SectionResult(
            title=title,
            content=result.get("content", str(result)),
            sources=result.get("sources", [])
        )
    except Exception as e:
        print(f"  ⚠️ Section {section_num}/{total} error: {e}")
        return SectionResult(
            title=title,
            content=f"[Error: {e}]",
            sources=[]
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global search_counter

    # Check for API keys
    if not os.environ.get("EXA_API_KEY"):
        print("Error: EXA_API_KEY environment variable not set")
        sys.exit(1)

    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else \
        "What are the key trends in AI agents and agentic frameworks in 2025?"

    print(f"\n{'═' * 70}")
    print("  DEEP RESEARCH PIPELINE (Python)")
    print(f"  Query: {query}")
    print(f"{'═' * 70}\n")

    # Create shared context
    ctx = Context()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: PLANNER
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 1: PLANNER AGENT")

    # Clear captured searches from any previous run
    captured_searches.clear()

    # Create planner agent with real-time callbacks
    planner = Agent(
        "claude-opus-4-5-20251101",
        max_iterations=15,
        system=PLANNER_SYSTEM
    )
    planner.tool(search)  # Register shared search tool

    # Real-time callbacks for observability
    @planner.on("iteration_start")
    def on_iter(event):
        print(f"  🔄 Iteration {event['iteration']}/{event['max_iterations']}")

    @planner.on("tool_call")
    def on_tool(event):
        args_preview = str(event['args'])[:50] + "..." if len(str(event['args'])) > 50 else str(event['args'])
        print(f"  🔧 Tool: {event['name']}({args_preview})")

    @planner.on("code_executed")
    def on_code(event):
        status = "✓" if event['success'] else "✗"
        output_preview = event['output'][:80].replace('\n', ' ') if event['output'] else ""
        if output_preview:
            print(f"  {status} Output: {output_preview}...")

    # Save output to context
    planner.to_context(ctx, "plan")

    planner_task = f"Create a research plan and outline for this query: {query}"
    print(f"📋 Task: {planner_task}\n")

    try:
        planner_output = planner.run(planner_task)
        print("✅ Planner completed\n")

        # Extract outline
        outline = planner_output.get("outline", {})
        print(f"📄 Report Title: {outline.get('title', 'Untitled')}")
        print(f"📝 Plan: {planner_output.get('plan', '')[:200]}...")

        sections = outline.get("sections", [])
        print(f"\n📑 Outline ({len(sections)} sections):")
        for i, section in enumerate(sections):
            print(f"  {i + 1}. {section.get('title', 'Untitled')}")
            print(f"     {section.get('description', '')}")
            for subsection in section.get("subsections", []):
                print(f"       • {subsection}")
    except Exception as e:
        print(f"❌ Planner error: {e}")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: PARALLEL EXECUTOR AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 2: PARALLEL EXECUTOR AGENTS")

    # Save captured research to context
    if captured_searches:
        research = "═══ RESEARCH FROM PLANNER (use this data, search only if needed) ═══\n\n"
        total_results = 0
        for i, search_item in enumerate(captured_searches):
            research += f"── Search {i + 1}: \"{search_item['query']}\" ──\n"
            for result in search_item["results"]:
                if "error" not in result:
                    total_results += 1
                    research += f"\n📄 {result.get('title', '')}\n"
                    research += f"🔗 {result.get('url', '')}\n"
                    text = result.get("text", "")
                    if text:
                        truncated = text[:4000] + "..." if len(text) > 4000 else text
                        research += f"{truncated}\n"
                    research += "\n"

        ctx.set("search_log", research)
        print(f"📚 Passing {total_results} search results from planner to executors via Context")

    print(f"🚀 Launching {len(sections)} executor agents in parallel...\n")

    # Run executors in parallel using thread pool
    section_results = []
    with ThreadPoolExecutor(max_workers=min(len(sections), 4)) as executor:
        futures = {
            executor.submit(execute_section, section, ctx, i + 1, len(sections)): i
            for i, section in enumerate(sections)
        }

        # Collect results in order
        results_by_index = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_index[idx] = future.result()
            except Exception as e:
                results_by_index[idx] = SectionResult(
                    title=sections[idx].get("title", "Unknown"),
                    content=f"[Error: {e}]"
                )

        # Sort by index
        section_results = [results_by_index[i] for i in range(len(sections))]

    print(f"\n✅ All {len(section_results)} sections generated")

    # Collect all sources
    all_sources = []
    for s in section_results:
        all_sources.extend(s.sources)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2.5: REVIEWER AGENT
    # ═══════════════════════════════════════════════════════════════════════
    skip_reviewer = os.environ.get("SKIP_REVIEWER")

    if skip_reviewer:
        print("⏭️  Skipping reviewer (SKIP_REVIEWER is set)\n")
    else:
        print_separator("PHASE 2.5: REVIEWER AGENT (OPUS)")

        # Create mutable sections list for the edit tool
        mutable_sections = list(section_results)

        reviewer_agent = Agent(
            "claude-opus-4-5-20251101",
            max_iterations=10,
            system=REVIEWER_SYSTEM
        )

        @reviewer_agent.tool
        def edit(section: int, action: str, text: str = None, old: str = None, new: str = None) -> str:
            """Edit a section of the report."""
            if section < 1 or section > len(mutable_sections):
                return f"Error: Invalid section {section}"

            sec = mutable_sections[section - 1]

            if action == "prepend":
                if text:
                    sec.content = f"{text}\n\n{sec.content}"
                    return f"✓ Prepended to section {section}"
                return "Error: 'text' required for prepend"

            elif action == "append":
                if text:
                    sec.content = f"{sec.content}\n\n{text}"
                    return f"✓ Appended to section {section}"
                return "Error: 'text' required for append"

            elif action == "remove":
                if text:
                    if text in sec.content:
                        sec.content = sec.content.replace(text, "")
                        return f"✓ Removed from section {section}"
                    return f"Warning: Text not found in section {section}"
                return "Error: 'text' required for remove"

            elif action == "replace":
                if old and new:
                    if old in sec.content:
                        sec.content = sec.content.replace(old, new)
                        return f"✓ Replaced in section {section}"
                    return f"Warning: Text not found in section {section}"
                return "Error: 'old' and 'new' required for replace"

            return f"Error: Unknown action '{action}'"

        # Build content for reviewer
        review_content = ""
        for i, s in enumerate(mutable_sections):
            review_content += f"\n\n{'=' * 60}\nSECTION {i + 1}: {s.title}\n{'=' * 60}\n\n{s.content}"

        reviewer_task = f"""Review this research report titled: "{outline.get('title', 'Research Report')}"

{review_content}

Use edit() to add transitions to sections 2+ and remove redundant content. Put all edits in ONE code block. When done, call finish("summary")."""

        print(f"📝 Reviewing {len(mutable_sections)} sections...\n")

        try:
            review_summary = reviewer_agent.run(reviewer_task)
            print(f"✅ Review complete: {review_summary}")
            # Update section_results with edited content
            section_results = mutable_sections
        except Exception as e:
            print(f"⚠️ Reviewer error: {e}. Using sections as-is.")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("PHASE 3: SUMMARY AGENT")

    # Combine all section content
    full_report = "\n\n".join(
        f"## {s.title}\n\n{s.content}" for s in section_results
    )

    summary_agent = Agent(
        "gemini-3-flash-preview",
        max_iterations=5,
        system=SUMMARY_SYSTEM
    )

    summary_task = f"""Create an executive summary for this research report:

# {outline.get('title', 'Research Report')}

{full_report}"""

    summary = None
    try:
        summary = summary_agent.run(summary_task)
        print("✅ Summary completed")
    except Exception as e:
        print(f"⚠️ Summary error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    print_separator("FINAL RESEARCH REPORT")

    markdown = f"# {outline.get('title', 'Research Report')}\n\n"

    # Executive Summary
    if summary:
        markdown += "## Executive Summary\n\n"

        takeaways = summary.get("key_takeaways", [])
        if takeaways:
            markdown += "**Key Takeaways:**\n"
            for i, t in enumerate(takeaways):
                markdown += f"{i + 1}. {t}\n"
            markdown += "\n"

        metrics = summary.get("key_metrics", [])
        if metrics:
            markdown += "**Key Metrics:**\n"
            for m in metrics:
                markdown += f"- **{m.get('metric', '')}**: {m.get('value', '')} ({m.get('context', '')})\n"
            markdown += "\n"

        opportunities = summary.get("opportunities", [])
        if opportunities:
            markdown += "**Opportunities:**\n"
            for o in opportunities:
                markdown += f"- {o}\n"
            markdown += "\n"

        risks = summary.get("risks", [])
        if risks:
            markdown += "**Risks & Challenges:**\n"
            for r in risks:
                markdown += f"- {r}\n"
            markdown += "\n"

        markdown += "---\n\n"

    # Section Content
    for section in section_results:
        markdown += f"## {section.title}\n\n"
        markdown += f"{section.content}\n\n"

    # Sources (deduplicated)
    if all_sources:
        markdown += "## Sources\n\n"
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            url = source.split(" - ")[0].strip()
            if url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)

        for i, source in enumerate(unique_sources):
            markdown += f"{i + 1}. {source}\n"

    # Write to file
    output_file = "report.md"
    with open(output_file, "w") as f:
        f.write(markdown)
    print(f"📄 Report written to: {output_file}")

    # Print report
    print(f"\n{markdown}")

    print_separator("PIPELINE COMPLETE")

    print(f"Generated {len(section_results)} sections with {len(set(all_sources))} unique sources")
    print(f"\n📊 Cost Estimate:")
    print(f"   Exa searches: {search_counter} × $0.005 = ${search_counter * 0.005:.3f}")
    print("   (LLM token costs not tracked)")


if __name__ == "__main__":
    main()

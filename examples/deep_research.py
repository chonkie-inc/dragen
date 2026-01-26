#!/usr/bin/env python3
"""
Deep Research - Complete research pipeline with source collection.

Combines deep searcher (source collection) with research pipeline (planning + writing):

1. Search Agent (gemini-3-flash-preview) - Collects high-quality sources
2. Planner Agent (claude-opus-4-5-20251101) - Creates outline with source assignments
3. Writer Agents (gemini-3-flash-preview) - Write sections in parallel
4. Reviewer Agent (claude-opus-4-5-20251101) - Reviews and improves coherence

Run with:
    python examples/deep_research.py "Your research topic"
"""

import os
import sys
import time
import json
import requests
from dataclasses import dataclass, field

from pydantic import BaseModel
from dragen import Agent, Context, ToolInfo


# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (for schema validation)
# ═══════════════════════════════════════════════════════════════════════════

class Source(BaseModel):
    """A single source from search results."""
    title: str
    url: str
    snippet: str = ""
    relevance: str = ""


class SearchOutput(BaseModel):
    """Output schema for the search agent."""
    topic: str
    total_sources: int
    sources: list[Source]


class Section(BaseModel):
    """A single section in the research outline."""
    title: str
    description: str
    source_urls: list[str]
    key_points: list[str] = []


class PlannerOutput(BaseModel):
    """Output schema for the planner agent."""
    title: str
    sections: list[Section]


class WriterOutput(BaseModel):
    """Output schema for the writer agent."""
    content: str
    sources_used: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# METRICS & STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    search_count: int = 0
    review_count: int = 0
    start_time: float = field(default_factory=time.time)

metrics = Metrics()


# ═══════════════════════════════════════════════════════════════════════════
# SEARCH TOOLS (for Search Agent)
# ═══════════════════════════════════════════════════════════════════════════

def search_web(query: str, num_results: int = 10) -> list:
    """Search the web using Exa API."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return [{"error": "EXA_API_KEY not set"}]

    metrics.search_count += 1
    count = metrics.search_count
    start = time.time()

    try:
        response = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": min(max(int(num_results), 1), 10),
                "type": "auto",
                "contents": {"text": True}
            }
        )
        response.raise_for_status()
        data = response.json()

        elapsed = time.time() - start
        results = []
        for r in data.get("results", []):
            text = r.get("text", "")
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": text[:1500] if text else "",
                "date": r.get("publishedDate", ""),
            })

        query_preview = query[:50] + "..." if len(query) > 50 else query
        print(f'    🔍 [{count}] "{query_preview}" → {len(results)} results ({elapsed:.1f}s)')
        return results

    except Exception as e:
        return [{"error": str(e)}]


def review_sources(results: list, topic: str) -> list:
    """Review search results using Groq with Kimi K2."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return results

    metrics.review_count += 1
    count = metrics.review_count
    start = time.time()

    sources_text = ""
    for i, result in enumerate(results):
        if isinstance(result, dict) and "error" not in result:
            title = result.get("title", "Unknown")
            url = result.get("url", "")
            snippet = result.get("snippet", "")[:400]
            sources_text += f"[{i}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n\n"

    prompt = f'''Review these search results for: "{topic}"

For each source, determine if it's RELEVANT or NOT RELEVANT.

Sources:
{sources_text}

Respond with JSON array:
[{{"index": 0, "relevant": true, "reason": "Why relevant"}}, ...]

ONLY output the JSON array.'''

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "moonshotai/kimi-k2-instruct-0905",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        reviews = json.loads(content)

        elapsed = time.time() - start
        relevant_sources = []
        rejected_count = 0

        for review in reviews:
            index = review.get("index", 0)
            is_relevant = review.get("relevant", False)
            reason = review.get("reason", "")

            if index < len(results) and is_relevant:
                result = results[index]
                if isinstance(result, dict) and "error" not in result:
                    relevant_sources.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", ""),
                        "relevance": reason
                    })
            elif not is_relevant:
                rejected_count += 1

        print(f"    📋 [Review {count}] {len(results)} → {len(relevant_sources)} relevant, {rejected_count} rejected ({elapsed:.1f}s)")
        return relevant_sources

    except Exception as e:
        print(f"    ⚠️  Review error: {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

SEARCH_AGENT_PROMPT = """<role>
You are an adaptive research source collector. Gather high-quality, diverse sources through intelligent searching.
</role>

<tools>
- intent(message: str) → None  # Declare what you're looking for
- search(query: str, num_results: int) → list[dict]  # Search the web
- review(results: list, topic: str) → list[dict]  # Filter for relevance
- finish(result: dict)  # Complete with final output
</tools>

<strategy>
Adapt your search strategy based on the topic:

**Narrow/specific topics** → Start small, expand if needed
**Broad/complex topics** → Cast a wider net initially

After each round, assess: What's covered? What's missing? Enough diversity?
</strategy>

<example topic="Python asyncio best practices">
# Narrow topic - start with 1 focused search
intent("Starting with a focused search on asyncio best practices")
results = search("Python asyncio best practices 2024", 10)
reviewed = review(results, "Python asyncio best practices")
collected_sources.extend(reviewed)
print(f"Added {len(reviewed)}, Total: {len(collected_sources)}")
# Got 8 relevant sources covering basics - good for narrow topic
</example>

<example topic="Python asyncio best practices" round="2">
# After reviewing round 1, I see gaps in error handling patterns
intent("Expanding to cover error handling which was missing")
results = search("asyncio exception handling patterns", 8)
reviewed = review(results, "Python asyncio best practices")
collected_sources.extend(reviewed)
print(f"Added {len(reviewed)}, Total: {len(collected_sources)}")
# Now have 12 sources - sufficient for this narrow topic
finish({"topic": "Python asyncio best practices", "total_sources": len(collected_sources), "sources": collected_sources})
</example>

<example topic="Impact of climate change on global agriculture">
# Broad topic with many facets - need wider initial search
intent("Broad topic requires multiple angles: crops, regions, economics, adaptation")
results = search("climate change agriculture impact 2024", 10)
results = results + search("crop yields climate change research", 10)
results = results + search("agricultural adaptation climate strategies", 10)
reviewed = review(results, "climate change agriculture")
collected_sources.extend(reviewed)
print(f"Added {len(reviewed)}, Total: {len(collected_sources)}")
# Got 22 sources but mostly crop yields - missing economic impact and regional data
</example>

<example topic="Impact of climate change on global agriculture" round="2">
# Filling gaps identified in round 1
intent("Need economic impact and regional perspectives")
results = search("climate change farming economic impact developing countries", 10)
results = results + search("regional agriculture climate vulnerability Africa Asia", 10)
reviewed = review(results, "climate change agriculture")
collected_sources.extend(reviewed)
print(f"Added {len(reviewed)}, Total: {len(collected_sources)}")
# Now have 35 sources with good coverage - done
finish({"topic": "Impact of climate change on global agriculture", "total_sources": len(collected_sources), "sources": collected_sources})
</example>

<output_format>
finish({
    "topic": "the research topic",
    "total_sources": len(collected_sources),
    "sources": collected_sources
})
</output_format>

<constraints>
NO: def, lambda, try/except, dict comprehensions, import, class
YES: variables, loops, list comprehensions, f-strings, builtins
</constraints>"""


PLANNER_PROMPT = """<role>
You are a research planner. Given collected sources, create a comprehensive report outline.
</role>

<context>
You have access to pre-collected sources via the context. Use ONLY these sources.
DO NOT search - sources are already provided.
</context>

<task>
1. Review the provided sources
2. Identify key themes and topics
3. Create a 4-6 section outline
4. Assign relevant sources to each section (by URL)

<output_format>
Call finish() with:
```python
finish({
    "title": "Report Title",
    "sections": [
        {
            "title": "Section Title",
            "description": "What this section covers",
            "source_urls": ["url1", "url2", ...],
            "key_points": ["point1", "point2", ...]
        },
        ...
    ]
})
```
</output_format>

<rules>
- Each section should have 3-6 assigned sources
- Sources can be used in multiple sections if relevant
- Focus on logical flow and comprehensive coverage
- NO searching - use only provided sources
</rules>

<constraints>
NO: def, lambda, try/except, dict comprehensions, import, class
YES: variables, loops, list comprehensions, f-strings, builtins
</constraints>"""


WRITER_PROMPT = """<role>
You are a research writer. Write comprehensive content for ONE section using provided sources.
</role>

<context>
You have access to:
- Section info with title, description, key points
- Assigned sources with NUMBER, title, and content
</context>

<requirements>
- Write 400-600 words of substantive content
- Include specific facts, numbers, and quotes from sources
- CITATIONS: Add citation numbers in brackets at the END of each paragraph or bullet point
  - Format: [1], [2], [1, 3], etc. using the source numbers provided
  - Place citations AFTER the period at paragraph/bullet end, not mid-sentence
  - Example: "AI coding assistants have seen rapid adoption in 2025. [1, 3]"
  - Example bullet: "- Cursor 2.0 introduces multi-agent architecture [2]"
- You can mention source titles inline AND add citation numbers: "According to **Source Title**, ... [1]"
- Use markdown formatting (headers, bullets, bold)
- DO NOT include the section title (## header) - it's added automatically
- Start directly with content
</requirements>

<output_format>
finish({
    "content": "Your markdown content here with [N] citations...",
    "sources_used": ["url1", "url2", ...]
})
</output_format>

<constraints>
NO: def, lambda, try/except, dict comprehensions, import, class
YES: variables, loops, list comprehensions, f-strings, builtins
</constraints>"""


REVIEWER_PROMPT = """<role>
You are an expert editor reviewing a research report for coherence and quality.
</role>

<tools>
edit(section: int, action: str, text: str = None, old: str = None, new: str = None)
  - section: 1-based section number
  - action: "prepend", "append", "remove", "replace"
  - For prepend/append: provide text
  - For replace: provide old and new
  - For remove: provide text to remove
</tools>

<task>
1. Read all sections carefully
2. Add transitions to sections 2+ (prepend connecting sentences)
3. Remove redundant content that appears in multiple sections
4. Fix any inconsistencies
5. Call finish("summary of changes") when done
</task>

<rules>
- Make MINIMAL edits - preserve original content
- Transitions: 1-2 sentences connecting to previous section
- Only remove TRULY redundant content
- Put ALL edits in ONE code block ending with finish()
</rules>

<constraints>
NO: def, lambda, try/except, dict comprehensions, import, class
YES: variables, loops, list comprehensions, f-strings, builtins
</constraints>"""


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════

def run_search_agent(topic: str) -> dict:
    """Stage 1: Collect sources using deep searcher."""
    print("\n" + "═" * 70)
    print("  STAGE 1: SOURCE COLLECTION (gemini-3-flash-preview)")
    print("═" * 70 + "\n")

    # Track actual search calls to prevent hallucinated sources
    search_call_count = 0

    agent = Agent(
        "gemini-3-flash-preview",
        max_iterations=12,
        system=SEARCH_AGENT_PROMPT
    )

    @agent.on("iteration_start")
    def on_iter(e):
        print(f"  ┌─ Iteration {e['iteration']}/{e['max_iterations']} ─────────────────")

    @agent.on("code_executed")
    def on_code(e):
        if e["output"] and e["success"]:
            for line in e["output"].split("\n")[:5]:
                print(f"  │ {line}")

    @agent.tool
    def intent(message: str) -> None:
        """Declare search intent."""
        print(f"  │ 💭 {message}")

    @agent.tool
    def search(query: str, num_results: int = 10) -> list:
        """Search the web."""
        nonlocal search_call_count
        search_call_count += 1
        return search_web(query, num_results)

    @agent.tool
    def review(results: list, topic: str) -> list:
        """Review and filter sources."""
        return review_sources(results, topic)

    @agent.tool(finish=True)
    def finish(result: dict) -> dict:
        """Complete the search with collected sources. Only call after using search() and review()."""
        nonlocal search_call_count
        sources = result.get("sources", []) if isinstance(result, dict) else []

        if search_call_count == 0 and len(sources) > 0:
            raise ValueError("You cannot finish with sources without calling search() first. Do not make up sources - use the search() tool to find real sources, then review() to filter them.")

        if search_call_count == 0:
            raise ValueError("collected_sources is empty. You must call search() at least once before finishing.")

        return result

    task = f"Collect comprehensive sources on: {topic}"
    result = agent.run(task, schema=SearchOutput.model_json_schema())

    print(f"\n  ✅ Collected {result.get('total_sources', 0)} sources")
    return result


def run_planner_agent(topic: str, sources: list, ctx: Context) -> dict:
    """Stage 2: Create outline with source assignments."""
    print("\n" + "═" * 70)
    print("  STAGE 2: PLANNING (claude-opus-4-5-20251101)")
    print("═" * 70 + "\n")

    # Store sources in context
    sources_text = "COLLECTED SOURCES:\n\n"
    for i, s in enumerate(sources):
        sources_text += f"[{i+1}] {s.get('title', 'Untitled')}\n"
        sources_text += f"    URL: {s.get('url', '')}\n"
        sources_text += f"    Relevance: {s.get('relevance', '')}\n"
        sources_text += f"    Content: {s.get('snippet', '')[:500]}...\n\n"

    ctx.set("sources", sources_text)
    ctx.set("sources_list", sources)

    agent = Agent(
        "claude-opus-4-5-20251101",
        max_iterations=5,
        system=PLANNER_PROMPT
    )
    agent.from_context(ctx, "sources")

    @agent.on("iteration_start")
    def on_iter(e):
        print(f"\n  ┌─ Planning iteration {e['iteration']}/{e['max_iterations']} ─────────────────")

    @agent.on("code_generated")
    def on_code_gen(e):
        code = e.get("code", "")
        print(f"  │ 📝 Generated code:")
        for line in code.split("\n")[:15]:
            print(f"  │   {line}")
        if code.count("\n") > 15:
            print(f"  │   ... ({code.count(chr(10)) - 15} more lines)")

    @agent.on("code_executed")
    def on_code_exec(e):
        output = e.get("output", "")
        success = e.get("success", False)
        status = "✅" if success else "❌"
        if output:
            print(f"  │ {status} Output:")
            for line in output.split("\n")[:10]:
                print(f"  │   {line}")
        else:
            print(f"  │ {status} (no output)")

    task = f"Create a research outline for: {topic}\n\nUse the sources provided in context."
    result = agent.run(task, schema=PlannerOutput.model_json_schema())

    sections = result.get("sections", [])
    print(f"\n  📋 Outline: {result.get('title', 'Untitled')}")
    for i, sec in enumerate(sections):
        print(f"     {i+1}. {sec.get('title', '')} ({len(sec.get('source_urls', []))} sources)")

    return result


def build_writer_task(section: dict, all_sources: list, url_to_num: dict) -> str:
    """Build a task string for the writer agent with numbered sources."""
    title = section.get("title", "Untitled")
    description = section.get("description", "")
    key_points = section.get("key_points", [])

    # Get assigned sources with their global numbers
    assigned_urls = set(section.get("source_urls", []))

    task = f"""Write content for this section:

SECTION: {title}
DESCRIPTION: {description}
KEY POINTS: {', '.join(key_points) if key_points else 'None specified'}

ASSIGNED SOURCES (use these citation numbers [N] at the end of paragraphs/bullets):

"""
    for url in assigned_urls:
        # Find the source in all_sources
        for s in all_sources:
            if s.get("url") == url:
                num = url_to_num.get(url, 0)
                task += f"[{num}] **{s.get('title', 'Untitled')}**\n"
                task += f"    URL: {url}\n"
                task += f"    Content: {s.get('snippet', '')[:800]}\n\n"
                break

    task += "\nWrite comprehensive content using these sources. Add citation numbers like [1], [2, 3] at the END of each paragraph or bullet point."
    return task


def run_writers_parallel(sections: list, all_sources: list) -> list:
    """Stage 3: Write all sections in parallel using agent.map()."""
    print("\n" + "═" * 70)
    print("  STAGE 3: PARALLEL WRITING (gemini-3-flash-preview)")
    print("═" * 70 + "\n")

    # Build URL -> source number mapping (1-indexed for citations)
    url_to_num = {}
    for i, s in enumerate(all_sources):
        url = s.get("url", "")
        if url and url not in url_to_num:
            url_to_num[url] = len(url_to_num) + 1

    # Build tasks for each section
    tasks = []
    for i, section in enumerate(sections):
        task = build_writer_task(section, all_sources, url_to_num)
        tasks.append(task)
        print(f"  ▶ Queued section {i+1}/{len(sections)}: {section.get('title', 'Untitled')}")

    # Create writer agent
    agent = Agent(
        "gemini-3-flash-preview",
        max_iterations=8,
        system=WRITER_PROMPT
    )

    @agent.on("finish")
    def on_finish(e):
        print(f"  ✅ Section completed")

    @agent.on("error")
    def on_error(e):
        print(f"  ⚠️ Error: {e.get('message', '')[:80]}...")

    # Run all sections in parallel with schema validation
    print(f"\n  Running {len(tasks)} writers in parallel...")
    try:
        results = agent.map(tasks, schema=WriterOutput.model_json_schema())
    except Exception as e:
        print(f"  ❌ Parallel execution failed: {e}")
        # Return empty results for all sections
        return [{"title": s.get("title", ""), "content": f"[Error: {e}]", "sources_used": []} for s in sections]

    # Convert results to expected format
    written_sections = []
    for i, (section, result) in enumerate(zip(sections, results)):
        if isinstance(result, dict):
            written_sections.append({
                "title": section.get("title", "Untitled"),
                "content": result.get("content", ""),
                "sources_used": result.get("sources_used", [])
            })
        else:
            written_sections.append({
                "title": section.get("title", "Untitled"),
                "content": str(result),
                "sources_used": []
            })

    print(f"\n  ✅ All {len(sections)} sections written")
    return written_sections


def run_reviewer_agent(sections: list, report_title: str) -> list:
    """Stage 4: Review and improve coherence."""
    print("\n" + "═" * 70)
    print("  STAGE 4: REVIEW (claude-opus-4-5-20251101)")
    print("═" * 70 + "\n")

    mutable_sections = list(sections)

    agent = Agent(
        "claude-opus-4-5-20251101",
        max_iterations=8,
        system=REVIEWER_PROMPT
    )

    @agent.tool
    def edit(section: int, action: str, text: str = None, old: str = None, new: str = None) -> str:
        """Edit a section."""
        if section < 1 or section > len(mutable_sections):
            return f"Error: Invalid section {section}"

        sec = mutable_sections[section - 1]
        content = sec.get("content", "")

        if action == "prepend" and text:
            sec["content"] = f"{text}\n\n{content}"
            return f"✓ Prepended to section {section}"
        elif action == "append" and text:
            sec["content"] = f"{content}\n\n{text}"
            return f"✓ Appended to section {section}"
        elif action == "remove" and text:
            if text in content:
                sec["content"] = content.replace(text, "")
                return f"✓ Removed from section {section}"
            return f"Text not found in section {section}"
        elif action == "replace" and old and new:
            if old in content:
                sec["content"] = content.replace(old, new)
                return f"✓ Replaced in section {section}"
            return f"Text not found in section {section}"

        return f"Error: Invalid action '{action}'"

    # Build review content
    review_content = f"REPORT: {report_title}\n\n"
    for i, sec in enumerate(mutable_sections):
        review_content += f"{'='*60}\nSECTION {i+1}: {sec.get('title', '')}\n{'='*60}\n\n"
        review_content += sec.get("content", "") + "\n\n"

    task = f"""Review this research report and improve coherence:

{review_content}

Add transitions to sections 2+, remove redundancy, then call finish("summary")."""

    try:
        result = agent.run(task)
        print(f"  ✅ Review complete: {result}")
    except Exception as e:
        print(f"  ⚠️ Review error: {e}")

    return mutable_sections


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Check API keys (GEMINI_API_KEY is an alternative to GOOGLE_API_KEY)
    required_keys = ["EXA_API_KEY", "CEREBRAS_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [k for k in required_keys if not os.environ.get(k)]
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    topic = sys.argv[1] if len(sys.argv) > 1 else "Recent advances in AI agents and agentic workflows"

    print("\n" + "═" * 70)
    print("  DEEP RESEARCH PIPELINE")
    print("═" * 70)
    print(f"\n  Topic: {topic}")
    print(f"  Stages: Search → Plan → Write (parallel) → Review")
    print()

    ctx = Context()
    start_time = time.time()

    # Stage 1: Collect sources
    search_result = run_search_agent(topic)
    sources = search_result.get("sources", [])

    if not sources:
        print("❌ No sources collected. Exiting.")
        return

    # Stage 2: Create plan
    plan = run_planner_agent(topic, sources, ctx)
    sections = plan.get("sections", [])

    if not sections:
        print("❌ No sections planned. Exiting.")
        return

    # Stage 3: Write sections in parallel
    written_sections = run_writers_parallel(sections, sources)

    # Stage 4: Review
    reviewed_sections = run_reviewer_agent(written_sections, plan.get("title", topic))

    # Build final report
    elapsed = time.time() - start_time

    print("\n" + "═" * 70)
    print("  FINAL REPORT")
    print("═" * 70 + "\n")

    markdown = f"# {plan.get('title', topic)}\n\n"

    # Executive summary
    markdown += "## Executive Summary\n\n"
    markdown += f"This report covers {len(reviewed_sections)} key areas based on analysis of {len(sources)} sources.\n\n"
    markdown += "---\n\n"

    # Sections
    for sec in reviewed_sections:
        markdown += f"## {sec.get('title', 'Untitled')}\n\n"
        markdown += sec.get("content", "") + "\n\n"

    # Sources
    markdown += "## Sources\n\n"
    seen_urls = set()
    for i, s in enumerate(sources):
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            markdown += f"{len(seen_urls)}. [{s.get('title', 'Untitled')}]({url})\n"

    # Write to file
    output_file = "report.md"
    with open(output_file, "w") as f:
        f.write(markdown)

    print(markdown)

    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)
    print(f"\n  📊 Metrics:")
    print(f"     Total time:     {elapsed:.1f}s")
    print(f"     Search calls:   {metrics.search_count}")
    print(f"     Review calls:   {metrics.review_count}")
    print(f"     Sources:        {len(sources)}")
    print(f"     Sections:       {len(reviewed_sections)}")
    print(f"\n  📄 Report saved to: {output_file}")


if __name__ == "__main__":
    main()

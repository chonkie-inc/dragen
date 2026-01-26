#!/usr/bin/env python3
"""
Deep Searcher - Collects high-quality sources on a topic adaptively.

Uses Cerebras ZAI-GLM-4.7 to iteratively search, review, and accumulate
relevant sources for downstream deep research pipelines.

Run with:
    python examples/deep_searcher.py
    python examples/deep_searcher.py "Your custom topic here"
"""

import os
import sys
import time
import json
import requests
from dataclasses import dataclass, field

# Add parent to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'python'))

from dragen import Agent

# ═══════════════════════════════════════════════════════════════════════════
# METRICS TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    search_count: int = 0
    review_count: int = 0
    start_time: float = field(default_factory=time.time)

metrics = Metrics()

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """<role>
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
</constraints>"""

# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
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
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
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
                "snippet": text[:800] if text else "",
                "date": r.get("publishedDate", ""),
                "author": r.get("author", "")
            })

        query_preview = query[:50] + "..." if len(query) > 50 else query
        print(f'    🔍 [{count}] "{query_preview}" → {len(results)} results ({elapsed:.1f}s)')
        return results

    except Exception as e:
        return [{"error": str(e)}]


def review_sources(results: list, topic: str) -> list:
    """Review search results using Cerebras llama-3.3-70b for fast batch filtering."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        return results  # Return unfiltered if no API key

    metrics.review_count += 1
    count = metrics.review_count
    start = time.time()

    # Build sources text for review
    sources_text = ""
    for i, result in enumerate(results):
        if isinstance(result, dict) and "error" not in result:
            title = result.get("title", "Unknown")
            url = result.get("url", "")
            snippet = result.get("snippet", "")[:400]
            sources_text += f"[{i}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n\n"

    prompt = f'''You are a research relevance evaluator. Review these search results for the topic: "{topic}"

For each source, determine if it's RELEVANT or NOT RELEVANT to the research topic.

Sources to review:
{sources_text}

Respond with a JSON array. For each source, include:
- "index": the source number
- "relevant": true or false
- "reason": brief explanation (10-20 words)

Example response:
[
  {{"index": 0, "relevant": true, "reason": "Directly discusses AI agent architectures and design patterns"}},
  {{"index": 1, "relevant": false, "reason": "About general machine learning, not specifically agents"}}
]

Respond ONLY with the JSON array, no other text.'''

    try:
        response = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048
            }
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        reviews = json.loads(content)

        elapsed = time.time() - start

        relevant_sources = []
        rejected_titles = []

        for review in reviews:
            index = review.get("index", 0)
            is_relevant = review.get("relevant", False)
            reason = review.get("reason", "")

            if index < len(results):
                result = results[index]
                if isinstance(result, dict) and "error" not in result:
                    title = result.get("title", "Unknown")

                    if is_relevant:
                        relevant_sources.append({
                            "title": title,
                            "url": result.get("url", ""),
                            "snippet": result.get("snippet", "")[:200],
                            "relevance": reason
                        })
                    else:
                        rejected_titles.append(f"{title[:50]}... ({reason})")

        relevant_count = len(relevant_sources)
        rejected_count = len(rejected_titles)

        print(f"    📋 [Review {count}] {len(results)} sources → {relevant_count} relevant, {rejected_count} rejected ({elapsed:.1f}s)")

        for rejected in rejected_titles:
            print(f"       ✗ {rejected}")

        return relevant_sources

    except Exception as e:
        print(f"    ⚠️  Review error: {e}")
        return results  # Return unfiltered on error


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Check for API keys
    if not os.environ.get("EXA_API_KEY"):
        print("Error: EXA_API_KEY environment variable not set")
        sys.exit(1)
    if not os.environ.get("CEREBRAS_API_KEY"):
        print("Error: CEREBRAS_API_KEY environment variable not set")
        sys.exit(1)

    # Get topic from command line or use default
    topic = sys.argv[1] if len(sys.argv) > 1 else "Recent advances in AI agents and agentic workflows"

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                    DEEP SEARCHER (Python)                         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Topic: {topic}")
    print("  Model: cerebras:zai-glm-4.7 (thinking)")
    print("  Review: cerebras:llama-3.3-70b (fast)")
    print("  Mode:  Efficient (strategic search with review filtering)")
    print()
    print("═══════════════════════════════════════════════════════════════════════")

    # Create agent
    agent = Agent(
        "cerebras:zai-glm-4.7",
        max_iterations=15,
        system=SYSTEM_PROMPT
    )

    # ─────────────────────────────────────────────────────────────────────
    # EVENT CALLBACKS
    # ─────────────────────────────────────────────────────────────────────

    @agent.on("iteration_start")
    def on_iteration(event):
        print(f"\n┌─ Iteration {event['iteration']}/{event['max_iterations']} ─────────────────────────────────────────────")

    @agent.on("code_generated")
    def on_code(event):
        code = event["code"]
        lines = code.split("\n")
        print("│")
        print(f"│ 📝 Code ({len(lines)} lines):")
        for line in lines[:10]:
            print(f"│    {line}")
        if len(lines) > 10:
            print(f"│    ... ({len(lines) - 10} more lines)")

    @agent.on("code_executed")
    def on_executed(event):
        output = event["output"]
        success = event["success"]
        if output:
            print("│")
            print(f"│ 📤 Output ({'ok' if success else 'error'}):")
            for line in output.split("\n")[:10]:
                print(f"│    {line}")
            if len(output.split("\n")) > 10:
                print(f"│    ... ({len(output.split(chr(10))) - 10} more lines)")

    # ─────────────────────────────────────────────────────────────────────
    # REGISTER TOOLS
    # ─────────────────────────────────────────────────────────────────────

    @agent.tool
    def intent(message: str) -> None:
        """Declare your intent before searching."""
        print("│")
        print(f"│ 💭 {message}")

    @agent.tool
    def search(query: str, num_results: int = 10) -> list:
        """Search the web using Exa."""
        return search_web(query, num_results)

    @agent.tool
    def review(results: list, topic: str) -> list:
        """Review and filter search results for relevance."""
        return review_sources(results, topic)

    # ─────────────────────────────────────────────────────────────────────
    # RUN
    # ─────────────────────────────────────────────────────────────────────

    task = f"Collect sources on the following topic. First assess its complexity (narrow/moderate/broad), then search adaptively:\n\n{topic}"

    try:
        result = agent.run(task)

        elapsed = time.time() - metrics.start_time

        print("\n└──────────────────────────────────────────────────────────────────")
        print()
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║                     COLLECTION COMPLETE                           ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print()
        print("  📊 Metrics:")
        print(f"     Total time:     {elapsed:.1f}s")
        print(f"     Search calls:   {metrics.search_count}")
        print(f"     Review calls:   {metrics.review_count}")
        print(f"     Sources found:  {result.get('total_sources', 0)}")
        print(f"     Complexity:     {result.get('complexity', 'unknown')}")
        if result.get('search_rounds', 0) > 0:
            print(f"     Search rounds:  {result.get('search_rounds', 0)}")
        print()
        print(f"  📋 Coverage: {result.get('coverage_summary', 'N/A')}")
        print()
        print("───────────────────────────────────────────────────────────────────────")
        print("                            SOURCES                                   ")
        print("───────────────────────────────────────────────────────────────────────")
        print()

        sources = result.get("sources", [])
        for i, source in enumerate(sources):
            print(f"  {i + 1}. {source.get('title', 'Unknown')}")
            print(f"     {source.get('url', '')}")
            print(f"     └─ {source.get('relevance', '')}")
            print()

    except Exception as e:
        print(f"Agent error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

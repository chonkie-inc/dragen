#!/usr/bin/env python3
"""
Local deep research script using smolagents.

Mimics the labs-deepresearch-service architecture but uses Exa web search
for fair comparison with the dragen Rust implementation.

Usage:
    EXA_API_KEY=xxx GROQ_API_KEY=xxx python smolagents_research.py "Your research query"
"""

import os
import sys
import json
from typing import Dict, Optional, Any

from smolagents import Tool, CodeAgent, LiteLLMModel


# ═══════════════════════════════════════════════════════════════════════════════
# EXA SEARCH TOOL
# ═══════════════════════════════════════════════════════════════════════════════

class ExaSearchTool(Tool):
    """Web search tool using Exa API."""

    name = "search"
    description = "Search the web for information on a topic. Returns relevant snippets from web pages."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Be specific and descriptive for better results."
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (1-10).",
            "default": 5,
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable not set")

    def forward(self, query: str, num_results: int = 5) -> str:
        """Execute the search."""
        import httpx

        num_results = max(1, min(10, num_results or 5))

        request_data = {
            "query": query,
            "numResults": num_results,
            "type": "auto",
            "contents": {
                "text": {"maxCharacters": 2000}
            }
        }

        try:
            response = httpx.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return "No results found."

            # Format results for the agent
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                text = r.get("text", "")[:1500]  # Truncate
                formatted.append(f"Result {i}:\nTitle: {title}\nURL: {url}\nContent: {text}\n")

            return "\n---\n".join(formatted)

        except Exception as e:
            return f"Search error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# PLANNER AGENT
# ═══════════════════════════════════════════════════════════════════════════════

PLANNER_PROMPT = """<task> Based on the available tools, generate a plan on how you could go about generating a report answering the question: '{{QUERY}}'. Also generate the outline of the report.

To get to the plan and outline, you'll need to use the search tool. You can keep searching till you have enough information to generate the plan and outline. When you're done, use the final_answer tool to generate the final answer.

Note that there's a hard limit of 5 steps for search, so you wouldn't be able to search more than 5 times. You can use the final_answer tool to generate the final answer after you've done the search.
</task>

<format>
The final output should have a plan and outline.

Your plan should contain instructions on how the executor should go about finding proper details for all the sections in the outline.

Your outline should contain a JSON parsable outline, with the following structure:
  - title: string
  - sections: array of objects
    - section: string
    - subsections: array of strings
</format>
"""


class PlannerFinalAnswerTool(Tool):
    """Planner final answer tool."""

    name = "final_answer"
    description = "This tool is used to provide the final answer to the user. The Final Answer would be a plan for the report (string) and an outline_json for the report. Keep the outline_json in the format as deemed fit by the user."
    inputs = {
        "plan": {"type": "string", "description": "The plan for the report."},
        "outline_json": {"type": "object", "description": "The outline_json for the report."}
    }
    output_type = "any"

    def forward(self, plan: str, outline_json: Dict) -> Any:
        """Forward the final answer tool."""
        return {"plan": plan, "outline_json": outline_json}


def get_planner_prompt(query: str) -> str:
    """Get the planner prompt with query substituted."""
    return PLANNER_PROMPT.replace("{{QUERY}}", query)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

EXECUTOR_PROMPT_WITH_PREVIOUS = """<task> You are a Researching Agent, whose final goal is to create a well-formated, markdown report on a given user query. You have been given a plan and an outline to follow, along with a current section to focus on.

You also have access to tools that would be essential for research. Make sure to use the tools properly.
</task>

<plan>
{{PLAN}}
</plan>

<outline>
{{OUTLINE}}
</outline>

<previous_section>
{{PREVIOUS_SECTION}}
</previous_section>

<focus>
Your current focus is on the following section: {{CURRENT_SECTION}}
</focus>

<format>
The final output should be a well-formated, markdown report. When generating a section, only use h2 for the section title and h3 for the subsections. Do not use h1 heading as that is reserved for the title of the report.
</format>
"""

EXECUTOR_PROMPT_NO_PREVIOUS = """<task> You are a Researching Agent, whose final goal is to create a well-formated, markdown report on a given user query. You have been given a plan and an outline to follow, along with a current section to focus on.

You also have access to tools that would be essential for research. Make sure to use the tools properly.
</task>

<plan>
{{PLAN}}
</plan>

<outline>
{{OUTLINE}}
</outline>

<focus>
Your current focus is on the following section: {{CURRENT_SECTION}}
</focus>

<format>
The final output should be a well-formated, markdown report. When generating a section, only use h2 for the section title and h3 for the subsections. Do not use h1 heading as that is reserved for the title of the report.
</format>
"""


class ExecutorFinalAnswerTool(Tool):
    """Executor final answer tool."""

    name = "final_answer"
    description = "This tool is used to provide the final answer to the user. The Final Answer would be a well-formated, markdown report section. Keep the section in the format as deemed fit by the user."
    inputs = {
        "section": {"type": "string", "description": "The markdown content for this section."}
    }
    output_type = "string"

    def forward(self, section: str) -> str:
        """Forward the final answer tool."""
        return section


def get_executor_prompt(plan: str, outline: Dict, previous_section: Optional[str], current_section: str) -> str:
    """Create the executor prompt."""
    outline_str = json.dumps(outline, indent=2)

    if previous_section:
        return (EXECUTOR_PROMPT_WITH_PREVIOUS
            .replace("{{PLAN}}", plan)
            .replace("{{OUTLINE}}", outline_str)
            .replace("{{PREVIOUS_SECTION}}", previous_section)
            .replace("{{CURRENT_SECTION}}", current_section)
        )
    else:
        return (EXECUTOR_PROMPT_NO_PREVIOUS
            .replace("{{PLAN}}", plan)
            .replace("{{OUTLINE}}", outline_str)
            .replace("{{CURRENT_SECTION}}", current_section)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(plan: str, outline: Dict, search_tool: Tool, model) -> str:
    """Generate a markdown report.

    Args:
        plan: Research plan from planner
        outline: Report outline with sections
        search_tool: Search tool instance
        model: LLM model instance

    Returns:
        Markdown content as string
    """
    # Initialize executor agent
    executor_final_answer_tool = ExecutorFinalAnswerTool()

    executor_agent = CodeAgent(
        tools=[search_tool, executor_final_answer_tool],
        model=model,
        max_steps=10,
    )

    # Generate report sections
    title = outline.get('title', 'Research Report')
    report = f"# {title}\n\n"
    previous_section: Optional[str] = None

    sections = outline.get('sections', [])
    for idx, section in enumerate(sections):
        section_name = section.get('section', f'Section {idx + 1}')
        print(f"\n{'─' * 60}")
        print(f"  Section {idx + 1}/{len(sections)}: {section_name}")
        print(f"{'─' * 60}\n")

        # Create executor prompt
        executor_prompt = get_executor_prompt(
            plan=plan,
            outline=outline,
            previous_section=previous_section,
            current_section=section_name
        )

        # Run executor agent
        try:
            executor_output = executor_agent.run(executor_prompt)
            print(f"✅ Section complete")

            # Add section content to report
            if executor_output:
                report += str(executor_output) + "\n\n"
                previous_section = str(executor_output)

        except Exception as e:
            print(f"⚠️ Error generating section: {e}")
            report += f"## {section_name}\n\n[Error generating this section]\n\n"

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "Who are the biggest players in the competitor intelligence market?"

    print("\n" + "═" * 70)
    print("  SMOLAGENTS DEEP RESEARCH")
    print(f"  Query: {query}")
    print("═" * 70 + "\n")

    # Check environment variables
    if not os.environ.get("EXA_API_KEY"):
        print("Error: EXA_API_KEY environment variable not set")
        sys.exit(1)
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize model (same as labs-deepresearch-service)
    model = LiteLLMModel(
        model_id="groq/moonshotai/kimi-k2-instruct-0905",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    print("✓ Initialized model: groq/moonshotai/kimi-k2-instruct-0905")

    # Initialize search tool
    search_tool = ExaSearchTool()
    print("✓ Initialized Exa search tool")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: PLANNER
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PHASE 1: PLANNER AGENT")
    print("═" * 70 + "\n")

    planner_final_answer_tool = PlannerFinalAnswerTool()
    planner_agent = CodeAgent(
        tools=[search_tool, planner_final_answer_tool],
        model=model,
        max_steps=10,
    )

    planner_prompt = get_planner_prompt(query)
    print("Running planner agent...")

    try:
        planner_output = planner_agent.run(planner_prompt)

        if isinstance(planner_output, dict):
            plan = planner_output.get('plan', '')
            outline = planner_output.get('outline_json', {})
        else:
            print(f"⚠️ Unexpected planner output type: {type(planner_output)}")
            print(f"Output: {planner_output}")
            sys.exit(1)

        print("\n✅ Planner completed\n")
        print(f"📄 Report Title: {outline.get('title', 'Unknown')}")
        print(f"📝 Plan: {plan[:200]}...")
        print(f"\n📑 Outline ({len(outline.get('sections', []))} sections):")
        for i, section in enumerate(outline.get('sections', []), 1):
            section_name = section.get('section', f'Section {i}')
            subsections = section.get('subsections', [])
            print(f"  {i}. {section_name}")
            for sub in subsections[:3]:  # Show first 3 subsections
                print(f"     - {sub}")

    except Exception as e:
        print(f"❌ Planner error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: EXECUTOR
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PHASE 2: EXECUTOR AGENTS")
    print("═" * 70)

    report = generate_report(
        plan=plan,
        outline=outline,
        search_tool=search_tool,
        model=model
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  FINAL RESEARCH REPORT")
    print("═" * 70 + "\n")

    print(report)

    print("\n" + "═" * 70)
    print("  SMOLAGENTS PIPELINE COMPLETE")
    print("═" * 70 + "\n")

    # Save to file for comparison
    output_file = "smolagents_report.md"
    with open(output_file, "w") as f:
        f.write(report)
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()

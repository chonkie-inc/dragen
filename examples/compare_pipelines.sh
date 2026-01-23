#!/bin/bash
# Compare dragen (Rust) vs smolagents (Python) deep research pipelines
#
# Usage:
#   ./compare_pipelines.sh "Your research query"
#
# Requirements:
#   - EXA_API_KEY environment variable
#   - GROQ_API_KEY environment variable
#   - Python with smolagents installed
#   - Rust/cargo for dragen

set -e

QUERY="${1:-Who are the biggest players in the competitor intelligence market?}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="comparison_${TIMESTAMP}"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPARISON"
echo "  Query: $QUERY"
echo "  Output directory: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════════════════"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check environment variables
if [ -z "$EXA_API_KEY" ]; then
    echo "Error: EXA_API_KEY not set"
    exit 1
fi
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY not set"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# RUN DRAGEN (RUST)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  RUNNING DRAGEN (RUST) PIPELINE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

START_DRAGEN=$(date +%s)

cd "$(dirname "$0")/.."
cargo run --example deep_research_pipeline "$QUERY" 2>&1 | tee "$OUTPUT_DIR/dragen_output.txt"

END_DRAGEN=$(date +%s)
DRAGEN_TIME=$((END_DRAGEN - START_DRAGEN))

echo ""
echo "Dragen completed in ${DRAGEN_TIME} seconds"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN SMOLAGENTS (PYTHON)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  RUNNING SMOLAGENTS (PYTHON) PIPELINE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

START_SMOLAGENTS=$(date +%s)

# Activate venv if it exists
if [ -d "/Users/bhavnick/Workspace/labs-deepresearch-service/.venv" ]; then
    source /Users/bhavnick/Workspace/labs-deepresearch-service/.venv/bin/activate
fi

cd "$(dirname "$0")"
python smolagents_research.py "$QUERY" 2>&1 | tee "$OUTPUT_DIR/smolagents_output.txt"

# Copy the markdown report
if [ -f "smolagents_report.md" ]; then
    mv smolagents_report.md "$OUTPUT_DIR/"
fi

END_SMOLAGENTS=$(date +%s)
SMOLAGENTS_TIME=$((END_SMOLAGENTS - START_SMOLAGENTS))

echo ""
echo "Smolagents completed in ${SMOLAGENTS_TIME} seconds"

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Query: $QUERY"
echo ""
echo "Timing:"
echo "  - Dragen (Rust):     ${DRAGEN_TIME}s"
echo "  - Smolagents (Python): ${SMOLAGENTS_TIME}s"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/dragen_output.txt"
echo "  - $OUTPUT_DIR/smolagents_output.txt"
echo "  - $OUTPUT_DIR/smolagents_report.md"
echo ""
echo "To compare reports side by side, check the output files above."
echo ""

# Create a simple comparison summary
cat > "$OUTPUT_DIR/summary.txt" << EOF
Pipeline Comparison Summary
===========================

Query: $QUERY
Date: $(date)

Timing:
- Dragen (Rust):       ${DRAGEN_TIME}s
- Smolagents (Python): ${SMOLAGENTS_TIME}s

Files:
- dragen_output.txt:     Full console output from Rust pipeline
- smolagents_output.txt: Full console output from Python pipeline
- smolagents_report.md:  Final markdown report from Python pipeline

Notes:
- Both pipelines use the same model: llama-3.3-70b-versatile (Groq)
- Both pipelines use Exa for web search
- Dragen uses CodeAct pattern (Python in sandbox)
- Smolagents uses CodeAgent with native Python tools
EOF

echo "Summary saved to: $OUTPUT_DIR/summary.txt"

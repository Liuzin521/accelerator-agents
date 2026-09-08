# auto_agent/subagents/profiling/prompts/analyze_profile_prompt.py
"""Prompt for analyzing profiling results using offline XProf tools."""

PROMPT = """
Your goal is to provide the results from the profiling execution and perform deep analysis.
Your response should have three parts:
1) A summary of the profiling results.
2) Deep analysis using the available offline XProf tools.
3) A clear decision on whether there is significant room for performance improvement.

For context, here are the profiling results (might contain the xplane.pb path or direct output):
{profiling_results}

Attributes of a good analysis:
*   Observe the "DMAs_and_memory_transfers_ratio and compute_ratio".
*   Use the `load_xplane_and_query` tool to explore the profiling data if you have an xplane.pb file path.
    *   Table schemas:
        *   planes (id, name)
        *   lines (id, plane_id, display_id, name, timestamp_ns)
        *   events (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)
*   Look for top ops by duration (sum(duration_ps)).
*   **Per-namespace breakdown (REQUIRED)**: the kernel wraps its logical phases
    in `jax.named_scope(...)` and names its `pl.pallas_call`. In the trace these
    appear (a) as a device line named "Framework Name Scope" whose events are the
    scope names with per-phase durations, and (b) as scope-path prefixes on op
    names (e.g. `jit(computation)/preprocess/mul`). Query the events table,
    group `sum(duration_ps)` by scope, and report a table of
    namespace -> time -> share of device time. Time spent OUTSIDE the
    pallas_call scope is XLA glue around the kernel — call it out explicitly
    when it exceeds ~10% (typical fixes: fold casts/relayouts into the kernel
    or into BlockSpec.index_map).
*   Use `get_hlo_dump` to check for specific HLO instructions if needed.
*   Use `get_overview_page_metrics` to retrieve high-level metrics (e.g., duty cycle, step time) for the summary.
*   Use `create_chart_from_xplane` to visualize distributions.
*   Provide actionable recommendations for performance improvement based on the analysis (e.g., specific HLO ops to optimize, memory bandwidth issues, or low duty cycle causes). Every recommendation MUST name the specific namespace/phase it applies to (e.g. "preprocess: fold the f32 cast into the kernel"), not the program as a whole.
*   Use `vertex_ai_rag_tool` to find relevant optimization guides or similar HLO patterns in the knowledge base.

If the profiling results contain a path to an `xplane.pb` file, prioritize using the tools to get more insights.

At the very end of your response, you MUST include a section formatted EXACTLY as follows:
DECISION: NEEDS_IMPROVEMENT = [True/False]

Use True if there is significant room for improvement, and False otherwise.
"""

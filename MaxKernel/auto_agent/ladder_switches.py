"""Environment switches for the expert-knowledge ladder experiments.

LADDER_DEEP_TRACE (default "on") is the single switch between "MK-Ours" and
"upstream MaxKernel behaviour". With it off:
  * the profiling script does NOT set the libtpu custom-call-tracing flag
    (assemble_profiling_script.py), and
  * the four prompts that ask for / analyse jax.named_scope annotations fall
    back to the upstream (215915f) wording verbatim (kernel planning, kernel
    implementation, profiling-script generation, profile analysis).
So a LADDER_DEEP_TRACE=off run is upstream MaxKernel plus only the
robustness fixes (profiling timeout, deterministic script assembly, jit
naming, RSS watchdog, retrieval_tool stub) and the jax 0.11 venv.
"""
import os


def deep_trace_enabled() -> bool:
  return os.environ.get("LADDER_DEEP_TRACE", "on").lower() != "off"

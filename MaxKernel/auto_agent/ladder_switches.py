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


def ladder_cards_text() -> str:
  """Knowledge-gap experiment: LADDER_CARDS="/path/a.md,/path/b.md" appends the
  given problem-agnostic "platform usage note" cards to the Pallas docs that
  every sub-agent sees in every iteration (state["pallas_docs"]). Delivered
  here rather than in the plan because the agent rewrites the plan between
  iterations and dropped the card in 1/2 seeds (10-04 L1+dma batch)."""
  spec = os.environ.get("LADDER_CARDS", "").strip()
  if not spec:
    return ""
  parts = []
  for p in [s.strip() for s in spec.split(",") if s.strip()]:
    with open(p) as f:
      parts.append(f.read().strip())
  if not parts:
    return ""
  return ("\n\n# Platform usage notes (TPU Pallas, not specific to any kernel)\n\n"
          "The following notes describe the only correct way to use a few Pallas TPU "
          "APIs. They take precedence over anything above that disagrees.\n\n"
          + "\n\n".join(parts) + "\n")

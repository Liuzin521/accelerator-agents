"""Deterministic profiling-script assembly (no LLM).

Why: the LLM-written profiling script (GenerateProfilingScriptAgent) is handed
only optimized_kernel.py, which has no inputs, so it invents toy shapes. For a
kernel that hard-codes benchmark shapes (7p ragged paged attention) that
crashes before any trace is taken (ladder batch 2026-09-11, L0+ x2). The
harness test file already contains the benchmark inputs (`get_inputs()`), so
the profiling script is now assembled verbatim from:

    LIBTPU deep-tracing env  +  optimized_kernel.py  +  inputs from the test
    file  +  a fixed profiler epilogue (warm-up outside the trace, N traced
    runs of jax.jit(computation)).

`assemble()` is a pure function so scripts/ladder_profiler_gate.py in the
TPU-project repo can build the exact same script for the expert kernel.
"""

from __future__ import annotations

import ast
import logging
import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

TRACE_ITERS_DEFAULT = 3

_HEADER_DEEP = '''# Deep kernel tracing — must precede any jax import
import os
os.environ["LIBTPU_INIT_ARGS"] = "--xla_xprof_enable_custom_call_tracing=true"
'''
_HEADER_SHALLOW = '''# Deep kernel tracing DISABLED for this script (LADDER_DEEP_TRACE=off)
import os
'''

_EPILOGUE = '''

# ---------------------------------------------------------------------------
# Profiling epilogue (assembled by AssembleProfilingScript; not LLM-written)
# ---------------------------------------------------------------------------
import functools as _functools
import time as _time

import jax as _jax


def _ladder_pick_fn():
    g = globals()
    if "computation" in g:
        return g["computation"]
    if "workload" in g:
        return g["workload"]
    raise RuntimeError("profiling script: neither computation() nor workload() defined")


def _ladder_pick_inputs():
    g = globals()
    if "get_inputs" in g:                      # harness test file: [(args, kwargs), ...]
        cases = g["get_inputs"]()
        args, kwargs = cases[0]
        return list(args), dict(kwargs)
    if "create_inputs" in g:                   # JAXBench reference: tuple of arrays
        out = g["create_inputs"]()
        if isinstance(out, dict):
            return [], out
        return list(out), {}
    raise RuntimeError("profiling script: neither get_inputs() nor create_inputs() defined")


def _ladder_rss_watchdog(cap_gb):
    """Abort the process before the host swaps itself to death: deep tracing
    of a large kernel accumulates the trace in host RAM (2026-09-16: a 7p
    expert-kernel profile made the v5e VM unreachable). Polls RSS every 2 s."""
    import threading

    def _rss_gb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1e6
        except OSError:
            pass
        return 0.0

    def _loop():
        while True:
            _time.sleep(2)
            rss = _rss_gb()
            if rss > cap_gb:
                print(f"[ladder-profile] RSS {rss:.1f} GB > cap {cap_gb:.1f} GB — aborting "
                      "(trace too large for host memory)", flush=True)
                os._exit(97)

    threading.Thread(target=_loop, daemon=True).start()


def _ladder_mem_total_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def _ladder_profile_main():
    cap = float(os.environ.get("LADDER_PROFILE_RSS_CAP_GB", 0) or 0)
    if cap <= 0:
        cap = 0.6 * _ladder_mem_total_gb() or 64.0
    _ladder_rss_watchdog(cap)
    print(f"[ladder-profile] host MemTotal {_ladder_mem_total_gb():.0f} GB, RSS cap {cap:.0f} GB", flush=True)
    _picked = _ladder_pick_fn()
    # tools/analyze_profile.py keys on device events named `jit_computation`;
    # JAXBench kernels define workload(), which would trace as `jit_workload`.
    def computation(*a, **kw):
        return _picked(*a, **kw)
    fn = _jax.jit(computation)
    args, kwargs = _ladder_pick_inputs()
    t0 = _time.perf_counter()
    _jax.block_until_ready(fn(*args, **kwargs))      # compile + warm-up, outside the trace
    print(f"[ladder-profile] compile+warmup {_time.perf_counter() - t0:.1f}s", flush=True)

    options = _jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

    t0 = _time.perf_counter()
    _jax.profiler.start_trace("jax_trace", profiler_options=options)
    for _ in range(__TRACE_ITERS__):
        _jax.block_until_ready(fn(*args, **kwargs))
    _jax.profiler.stop_trace()
    print(f"[ladder-profile] traced {__TRACE_ITERS__} runs in {_time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    _ladder_profile_main()
'''


def _strip_main_guard(src: str) -> str:
  """Drop every top-level `if __name__ == "__main__":` block."""
  try:
    tree = ast.parse(src)
  except SyntaxError:
    return src
  drop = []
  for node in tree.body:
    if isinstance(node, ast.If):
      t = node.test
      if (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name)
        and t.left.id == "__name__"
      ):
        drop.append((node.lineno, node.end_lineno))
  if not drop:
    return src
  lines = src.splitlines(keepends=True)
  keep = []
  for i, line in enumerate(lines, 1):
    if any(a <= i <= b for a, b in drop):
      continue
    keep.append(line)
  return "".join(keep)


# Top-level defs that an inputs file (harness test file or JAXBench
# baseline.py) may carry but which must NOT shadow the kernel's own.
_INPUTS_DROP_DEFS = {"workload", "computation", "kernel", "benchmark", "main", "get_flops"}


def _strip_inputs_file(src: str) -> str:
  """Keep only what the inputs file contributes: imports, constants and the
  input builder. Drop (a) the mock-execution `base_mod` try-block of harness
  test files, (b) any top-level function that would shadow the kernel's
  computation()/workload() (JAXBench baseline.py defines the BASELINE
  workload), (c) top-level statements that call those functions."""
  try:
    tree = ast.parse(src)
  except SyntaxError:
    return src
  drop = []
  for node in tree.body:
    seg = ast.get_source_segment(src, node) or ""
    if isinstance(node, ast.Try) and "base_mod" in seg:
      drop.append((node.lineno, node.end_lineno))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _INPUTS_DROP_DEFS:
      drop.append((node.lineno, node.end_lineno))
    elif isinstance(node, ast.Expr) and any(
      isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in _INPUTS_DROP_DEFS
      for c in ast.walk(node)
    ):
      drop.append((node.lineno, node.end_lineno))
  if not drop:
    return src
  lines = src.splitlines(keepends=True)
  return "".join(
    l for i, l in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop)
  )


def assemble(
  kernel_src: str,
  inputs_src: str,
  trace_iters: int = TRACE_ITERS_DEFAULT,
  deep_trace: bool = True,
) -> str:
  """Pure function: kernel source + inputs source -> profiling script.
  deep_trace=False drops the LIBTPU custom-call-tracing flag (no kernel-internal
  named_scope events, but a much smaller trace) — used by the gate's light probe."""
  kernel = _strip_main_guard(kernel_src)
  inputs = _strip_inputs_file(_strip_main_guard(inputs_src))
  return (
    (_HEADER_DEEP if deep_trace else _HEADER_SHALLOW)
    + "\n# ---- kernel (verbatim: optimized_kernel.py) ----\n"
    + kernel
    + "\n\n# ---- benchmark inputs (verbatim: harness test file) ----\n"
    + inputs
    + _EPILOGUE.replace("__TRACE_ITERS__", str(int(trace_iters)))
  )


class AssembleProfilingScript(BaseAgent):
  """Writes profiling_script_path and sets state['profiling_script']."""

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    st = ctx.session.state
    kernel_path = st.get("optimized_kernel_path")
    inputs_path = st.get("test_file_path")
    out_path = st.get("profiling_script_path")
    trace_iters = int(os.environ.get("LADDER_PROFILE_TRACE_ITERS", TRACE_ITERS_DEFAULT))
    deep = os.environ.get("LADDER_DEEP_TRACE", "on").lower() != "off"
    delta = {"profiling_failed": False, "profiling_failure_reason": ""}
    try:
      with open(kernel_path) as f:
        kernel_src = f.read()
      with open(inputs_path) as f:
        inputs_src = f.read()
      script = assemble(kernel_src, inputs_src, trace_iters, deep_trace=deep)
      with open(out_path, "w") as f:
        f.write(script)
      delta["profiling_script"] = script
      logging.info(
        f"[{self.name}] Assembled profiling script -> {out_path} "
        f"(kernel {len(kernel_src)} B + inputs {len(inputs_src)} B, "
        f"trace_iters={trace_iters}, deep_trace={deep})"
      )
    except Exception as e:  # noqa: BLE001
      msg = f"could not assemble profiling script: {e}"
      logging.error(f"[{self.name}] {msg}")
      delta.update(
        profiling_script=None,
        profiling_failed=True,
        profiling_failure_reason=msg,
      )
    yield Event(author=self.name, actions=EventActions(state_delta=delta))

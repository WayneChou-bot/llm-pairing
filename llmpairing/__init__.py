"""LLM pairing — match your machine to the local LLMs it can actually run.

T-002 core: pure-function memory budget (budget/) and roofline throughput
prediction (predict/). No I/O of any kind is permitted inside those packages
(SPEC §1.3).
"""
from llmpairing.types import DemandBreakdown, Measured, Source, Tier, Verdict

__all__ = [
    "DemandBreakdown",
    "Measured",
    "Source",
    "Tier",
    "Verdict",
]

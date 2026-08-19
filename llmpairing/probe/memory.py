"""System memory detection (T-001 SPEC §3.1).

T1-P04: `available` is psutil's OS-informed reallocatable estimate — the
`free` figure severely undercounts on cached systems and is banned.
Bandwidth is spec_db territory (S5); until then it is honestly UNKNOWN.
"""
from __future__ import annotations

from typing import Any

from llmpairing.probe.runner import ProbeError
from llmpairing.schemas import SystemMemory
from llmpairing.types import Source

_SENTINEL = object()


def parse_memory(total: int, available: int, diags: list[str]) -> SystemMemory:
    if available > total:
        raise ProbeError(
            f"memory: available ({available}) exceeds total ({total}) — "
            f"corrupted reading, refusing to pass it on"
        )
    return SystemMemory(
        total_bytes=total,
        available_bytes=available,  # P-04: point-in-time snapshot semantics
        bandwidth_bytes_per_s=None,
        bandwidth_source=Source.UNKNOWN,  # spec_db lookup arrives in S5
    )


def collect_memory(diags: list[str], _psutil: Any = _SENTINEL) -> SystemMemory:
    """Thin I/O layer. psutil is a declared runtime dependency; its absence
    is a broken install and fails loudly (N-T1-08), never silently."""
    mod: Any
    if _psutil is _SENTINEL:
        try:
            import psutil
            mod = psutil
        except ImportError:
            mod = None
    else:
        mod = _psutil
    if mod is None:
        raise ProbeError(
            "memory: psutil is unavailable — install the package "
            "(declared runtime dependency); refusing to emit RAM figures "
            "from a degraded source"
        )
    vm = mod.virtual_memory()
    return parse_memory(int(vm.total), int(vm.available), diags)

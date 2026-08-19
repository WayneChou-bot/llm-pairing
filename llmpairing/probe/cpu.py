"""CPU detection (T-001 SPEC §3.1).

parse_cpu() is pure assembly from injected inputs (tested against F-01);
collect_cpu() is the thin I/O layer gathering those inputs.

T1-P03: platform.processor() returns a family string ("Intel64 Family 6
Model 189...") on Windows — the CIM marketing name is preferred for the
model field and the family string is preserved as a diagnostic.
"""
from __future__ import annotations

import platform
from typing import Any

from llmpairing.probe import cim
from llmpairing.probe.runner import ProbeError
from llmpairing.schemas import CpuInfo


def parse_cpu(cim_cpu: dict[str, Any] | None, family_str: str,
              physical: int | None, logical: int | None,
              diags: list[str]) -> CpuInfo:
    """Assemble CpuInfo from injected raw inputs. Never guesses (R-T1-4)."""
    if cim_cpu and cim_cpu.get("Name"):
        model = str(cim_cpu["Name"]).strip()
        if family_str and family_str != model:
            diags.append(f"cpu: family string retained as diagnostic: {family_str}")
    else:
        model = family_str
        diags.append("cpu: CIM name unavailable — falling back to family string (T1-P03)")

    if physical is None and cim_cpu and cim_cpu.get("NumberOfCores"):
        physical = int(cim_cpu["NumberOfCores"])
        diags.append("cpu: physical cores from CIM fallback")
    if logical is None and cim_cpu and cim_cpu.get("NumberOfLogicalProcessors"):
        logical = int(cim_cpu["NumberOfLogicalProcessors"])
        diags.append("cpu: logical cores from CIM fallback")

    if not model:
        raise ProbeError("cpu: no model string from any source")
    if physical is None or logical is None:
        raise ProbeError(
            "cpu: core counts unavailable from psutil and CIM — refusing to "
            "guess (R-T1-4); mandatory schema field"
        )
    return CpuInfo(model=model, physical_cores=physical, logical_cores=logical)


def collect_cpu(diags: list[str]) -> CpuInfo:
    """Thin I/O layer: gather inputs, delegate to parse_cpu."""
    physical: int | None = None
    logical: int | None = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except ImportError:
        diags.append("cpu: psutil unavailable — CIM fallback only")

    cim_cpu = None
    if platform.system().lower() == "windows":
        got = cim.run_cim("cpu", diags)
        if isinstance(got, dict):
            cim_cpu = got
        elif isinstance(got, list) and got:  # multi-socket returns a list
            cim_cpu = got[0]
            diags.append(f"cpu: {len(got)} processor entries; using the first")
    return parse_cpu(cim_cpu, platform.processor() or platform.machine(),
                     physical, logical, diags)

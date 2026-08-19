"""Frozen Windows CIM query templates (T-001 SPEC §3.2, R-T1-2/R-T1-3).

powershell is a whitelisted command, but its -Command argument is an
arbitrary-code escape hatch — so probe modules may NEVER assemble
powershell strings. The ONLY sanctioned invocations are the literal
templates below (read-only Get-CimInstance | Select-Object | ConvertTo-Json
pipelines, no chaining, no invocation operators). test_probe_cpu_memory.py
lints both the templates and the rest of probe/ for violations.
"""
from __future__ import annotations

import json
from typing import Any, Final

from llmpairing.probe.runner import run_readonly

_PS_PREFIX: Final[list[str]] = [
    "powershell", "-NoProfile", "-NonInteractive", "-Command"
]

#: name -> full argv. PURE LITERALS — never build these dynamically.
QUERIES: Final[dict[str, list[str]]] = {
    "cpu": _PS_PREFIX + [
        "Get-CimInstance Win32_Processor | Select-Object "
        "Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Depth 4"
    ],
    "gpu": _PS_PREFIX + [
        "Get-CimInstance Win32_VideoController | Select-Object "
        "Name,AdapterRAM,DriverVersion,VideoProcessor,PNPDeviceID "
        "| ConvertTo-Json -Depth 4"
    ],
    "system": _PS_PREFIX + [
        "Get-CimInstance Win32_ComputerSystem | Select-Object "
        "TotalPhysicalMemory,Model,Manufacturer | ConvertTo-Json -Depth 4"
    ],
    "enclosure": _PS_PREFIX + [
        "Get-CimInstance Win32_SystemEnclosure | Select-Object "
        "ChassisTypes | ConvertTo-Json -Depth 4"
    ],
}


def run_cim(name: str, diags: list[str]) -> Any:
    """Run a registered CIM query; return parsed JSON or None.

    None means "unavailable on this system" (non-Windows, PowerShell
    missing, query failed) — callers degrade to their fallback path.
    """
    argv = QUERIES[name]  # KeyError on unknown name is a programming bug
    raw = run_readonly(argv, diags=diags)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        diags.append(f"cim[{name}]: undecodable JSON ({exc})")
        return None

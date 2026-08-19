"""Read-only subprocess wrapper + fixture utilities (T-001 SPEC §1.3, R-T1-2).

Every external tool invocation in probe/ MUST go through run_readonly():
- the command must be on the read-only whitelist (bare name, normalized)
- the timeout is mandatory and capped at MAX_TIMEOUT_S
- every failure mode (absent tool, timeout, nonzero exit, undecodable
  output) returns None and records a diagnostic — exceptions never escape

Direct subprocess usage anywhere else in probe/ is a bug (enforced by the
probe lint in tests/test_lint.py).
"""
from __future__ import annotations

import subprocess  # noqa: S404 — this module IS the sanctioned wrapper
from pathlib import PurePath
from typing import Any

class ProbeError(Exception):
    """A probe could not produce a schema-legal value AND the field is
    mandatory — fail loudly with a human-readable reason (N-T1-08/09).
    Optional fields never raise; they degrade to UNKNOWN (R-T1-4)."""


#: Commands probe/ may execute. All are read-only information queries.
#: Adding an entry requires owner review (BRIEF §3: readonly whitelist).
READONLY_COMMANDS: frozenset[str] = frozenset({
    "nvidia-smi",        # NVIDIA VRAM/driver query (csv output only, T1-P07)
    "powershell",        # Windows CIM queries via Get-CimInstance
    "pwsh",              # PowerShell 7 variant
    "sysctl",            # macOS memory / iogpu readouts
    "system_profiler",   # macOS display info
})

#: T1-P05: no probe may block the tool for long; hard cap per invocation.
MAX_TIMEOUT_S: float = 10.0


def _normalize_cmd_name(cmd0: str) -> str:
    """'C:\\...\\NVIDIA-SMI.EXE' -> 'nvidia-smi' (T1-P07-adjacent hygiene)."""
    name = PurePath(cmd0.replace("\\", "/")).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def run_readonly(cmd: list[str], *, timeout_s: float = MAX_TIMEOUT_S,
                 diags: list[str] | None = None) -> str | None:
    """Run a whitelisted read-only command; return stripped stdout or None.

    None ALWAYS means "this information is unavailable" — callers translate
    it to UNKNOWN fields, never to guesses (R-T1-4).
    """
    if timeout_s > MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout_s {timeout_s} exceeds the {MAX_TIMEOUT_S}s cap (T1-P05)"
        )
    sink = diags if diags is not None else []
    if not cmd:
        sink.append("empty command")
        return None
    name = _normalize_cmd_name(cmd[0])
    if name not in READONLY_COMMANDS:
        sink.append(f"refused: '{name}' not on the read-only whitelist (R-T1-2)")
        return None
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        sink.append(f"{name}: timeout after {timeout_s}s (T1-P05)")
        return None
    except FileNotFoundError:
        sink.append(f"{name}: binary not found")
        return None
    except OSError as exc:
        sink.append(f"{name}: OS error {exc!r}")
        return None
    if result.returncode != 0:
        sink.append(
            f"{name}: exit code {result.returncode}; "
            f"stderr={result.stderr.strip()[:200]!r}"
        )
        return None
    return result.stdout.strip()


def resolve_runtime_sentinels(expected: dict[str, Any],
                              values: dict[str, Any]) -> dict[str, Any]:
    """Substitute '__RUNTIME__' placeholders in a fixture's expected profile.

    Keys are matched by field name anywhere in the tree. Any sentinel left
    unsubstituted raises — a fixture must never silently pass with holes.
    """
    def walk(node: Any, key: str | None = None) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, key) for v in node]
        if node == "__RUNTIME__":
            if key is None or key not in values:
                raise ValueError(
                    f"unsubstituted __RUNTIME__ sentinel for field {key!r}"
                )
            return values[key]
        return node

    result = walk(expected)
    assert isinstance(result, dict)
    return result

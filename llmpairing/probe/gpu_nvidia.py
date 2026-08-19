"""NVIDIA VRAM via nvidia-smi (T-001 S4 — owner-ratified compromise).

STATUS (2026-08-19): no NVIDIA machine has run this parser. It is
anchored to NVIDIA's documented --query-gpu CSV contract:

    nvidia-smi --query-gpu=name,memory.total,memory.free
               --format=csv,noheader,nounits

with memory figures in MiB (per nvidia-smi --help-query-gpu). Until a
real card verifies the format in the field, every profile enriched here
carries NVIDIA_SMI_PARSER_UNVERIFIED_ON_REAL_HW in probe_notes — the
flag propagates through classify into every verdict (schema 1.1). The
flag, this notice, and the fixture strings retire together once a real
NVIDIA run confirms the format.

Honesty rules:
- [N/A] or non-numeric memory: entry skipped with a diagnostic, never
  guessed (R-T1-4)
- CIM-count vs nvidia-smi-count mismatch: pairing would be a guess —
  no merge, diagnosed
- laptop detection: chassis signal (already on the Accelerator) OR the
  documented name markers (Laptop / Mobile / Max-Q), P-15
"""
from __future__ import annotations

import re

from llmpairing.probe.runner import run_readonly
from llmpairing.schemas import Accelerator

_QUERY = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
          "--format=csv,noheader,nounits"]
_MIB = 1024 ** 2
_LAPTOP_NAME_RE = re.compile(r"\b(laptop|mobile|max-q)\b", re.IGNORECASE)

UNVERIFIED_NOTE = "NVIDIA_SMI_PARSER_UNVERIFIED_ON_REAL_HW"


def parse_nvidia_smi_csv(text: str, diags: list[str]) -> list[dict[str, object]]:
    """Documented-format CSV -> [{name, vram_total_bytes, vram_free_bytes}]."""
    out: list[dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            diags.append(f"nvidia-smi: unparseable line {line!r} — skipped")
            continue
        name, total, free = parts
        if not name or not total.isdigit() or not free.isdigit():
            diags.append(
                f"nvidia-smi: non-numeric memory for {name or line!r} — "
                f"entry skipped, never guessed (R-T1-4)")
            continue
        out.append({
            "name": name,
            "vram_total_bytes": int(total) * _MIB,
            "vram_free_bytes": int(free) * _MIB,
        })
    return out


def enrich_nvidia(accelerators: list[Accelerator], diags: list[str], *,
                  smi_text: str | None = None,
                  ) -> tuple[list[Accelerator], list[str]]:
    """Fill VRAM figures on CIM-enumerated NVIDIA entries.

    smi_text is injectable for tests; None means "query live" (only
    attempted when an NVIDIA adapter is present). Returns the (possibly
    updated) accelerator list plus probe_notes to stamp on the profile.
    """
    nv_idx = [i for i, a in enumerate(accelerators) if a.vendor == "nvidia"]
    if not nv_idx:
        return accelerators, []
    if smi_text is None:
        smi_text = run_readonly(_QUERY, diags=diags)
        if smi_text is None:
            diags.append("nvidia-smi unavailable — NVIDIA VRAM stays UNKNOWN "
                         "(data-less card degrades in classify)")
            return accelerators, []
    gpus = parse_nvidia_smi_csv(smi_text, diags)
    if not gpus:
        return accelerators, []
    if len(gpus) != len(nv_idx):
        diags.append(
            f"nvidia-smi reports {len(gpus)} GPUs but CIM enumerated "
            f"{len(nv_idx)} NVIDIA adapters — counts differ, pairing would "
            f"be a guess: no merge")
        return accelerators, []
    out = list(accelerators)
    for i, g in zip(nv_idx, gpus):
        a = out[i]
        name = str(g["name"])
        laptop = a.is_laptop_variant or bool(_LAPTOP_NAME_RE.search(name))
        out[i] = a.model_copy(update={
            "vram_total_bytes": g["vram_total_bytes"],
            "vram_free_bytes": g["vram_free_bytes"],
            "is_laptop_variant": laptop,
        })
        diags.append(f"nvidia-smi: '{name}' VRAM filled "
                     f"(UNVERIFIED parser — see {UNVERIFIED_NOTE})")
    return out, [UNVERIFIED_NOTE]

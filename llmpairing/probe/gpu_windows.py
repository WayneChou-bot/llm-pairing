"""Windows GPU enumeration + iGPU discrimination (T-001 SPEC §3.2).

parse_video_controllers() is pure assembly from injected CIM output
(anchored to the F-01 recording); collect_video_controllers() is the thin
I/O layer.

Hard rules:
- CIM AdapterRAM NEVER reaches the profile. It is a uint32-capped garbage
  figure (F-01 records 2^31-4096 for a shared-memory iGPU) — P-06 for
  shared topologies, and even for discrete cards the real number belongs
  to nvidia-smi/pynvml (S4).
- probe enumerates, it does not adjudicate (T1-P11): every physical
  adapter is reported in a stable order; classify owns topology decisions.
- unknown vendor patterns classify INTEGRATED_SHARED — the conservative
  direction (forces T0 + the smaller shared-memory budget downstream:
  false-rejects only, never false-accepts).
"""
from __future__ import annotations

import re
from typing import Any

from llmpairing.probe import cim
from llmpairing.schemas import Accelerator
from llmpairing.types import Source

#: substrings identifying non-physical display adapters (N-T1-04)
_VIRTUAL_MARKERS = (
    "microsoft basic display", "microsoft basic render",
    "microsoft remote display", "microsoft hyper-v video",
    "displaylink", "virtual", "vmware", "virtualbox", "parallels",
    "citrix", "teamviewer", "spacedesk", "idd ",
)

#: SMBIOS chassis types that imply a portable machine (P-15 signal)
_LAPTOP_CHASSIS = {8, 9, 10, 11, 12, 14, 18, 21, 30, 31, 32}

#: Intel discrete Arc: A-series (Alchemist) / B-series (Battlemage) —
#: "Arc(TM) A770", "Arc(TM) B580". Lunar-Lake iGPUs are "Arc(TM) 130V/140V"
#: (numeric + V) and are NOT discrete.
_INTEL_DISCRETE_RE = re.compile(r"\bArc\(TM\)\s+[AB]\d{3}\b", re.IGNORECASE)

_PNP_ID_RE = re.compile(r"(VEN_[0-9A-F]{4})&(DEV_[0-9A-F]{4})", re.IGNORECASE)


def _vendor(name: str, pnp: str) -> str:
    p = pnp.upper()
    if "VEN_10DE" in p:
        return "nvidia"
    if "VEN_1002" in p or "VEN_1022" in p:
        return "amd"
    if "VEN_8086" in p:
        return "intel"
    n = name.lower()
    if "nvidia" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    return "none"


def _topology(vendor: str, name: str, diags: list[str]) -> str:
    if vendor == "nvidia":
        return "DISCRETE"
    if vendor == "intel":
        if _INTEL_DISCRETE_RE.search(name):
            return "DISCRETE"
        if not re.search(r"\b(HD|UHD|Iris|Arc)\b", name, re.IGNORECASE):
            diags.append(
                f"gpu: unrecognized intel pattern '{name}' — conservative "
                f"INTEGRATED_SHARED classification"
            )
        return "INTEGRATED_SHARED"
    if vendor == "amd":
        # discrete Radeon carries a model number (RX 7800 XT etc.); the bare
        # "Radeon(TM) Graphics" is the APU iGPU
        if re.search(r"\bRX\s*\d{3,4}\b|\bRadeon\s+(PRO|VII)\b", name, re.IGNORECASE):
            return "DISCRETE"
        return "INTEGRATED_SHARED"
    diags.append(f"gpu: unknown vendor for '{name}' — conservative INTEGRATED_SHARED")
    return "INTEGRATED_SHARED"


def _stable_id(pnp: str, index: int) -> str:
    m = _PNP_ID_RE.search(pnp or "")
    if m:
        return f"pci-{m.group(1).upper()}-{m.group(2).upper()}"
    return f"gpu-{index}"


def parse_video_controllers(cim_gpu: Any, *, chassis_types: list[int] | None,
                            diags: list[str]) -> list[Accelerator]:
    """Assemble Accelerator entries from injected Win32_VideoController data."""
    if cim_gpu is None:
        return []
    entries = cim_gpu if isinstance(cim_gpu, list) else [cim_gpu]
    is_laptop = bool(set(chassis_types or []) & _LAPTOP_CHASSIS)

    physical: list[dict[str, Any]] = []
    for e in entries:
        name = str(e.get("Name") or "").strip()
        if not name or any(marker in name.lower() for marker in _VIRTUAL_MARKERS):
            diags.append(f"gpu: filtered virtual/unnamed adapter '{name}' (N-T1-04)")
            continue
        physical.append(e)

    # T1-P11: stable order regardless of CIM enumeration order
    physical.sort(key=lambda e: (_stable_id(str(e.get("PNPDeviceID") or ""), 0),
                                 str(e.get("Name") or "")))

    accelerators: list[Accelerator] = []
    for i, e in enumerate(physical):
        name = str(e["Name"]).strip()
        pnp = str(e.get("PNPDeviceID") or "")
        vendor = _vendor(name, pnp)
        topology = _topology(vendor, name, diags)
        if e.get("AdapterRAM") is not None:
            diags.append(
                f"gpu: AdapterRAM={e['AdapterRAM']} for '{name}' discarded — "
                f"uint32-capped/driver-fiction figure never enters the "
                f"profile (P-06); real VRAM comes from nvidia-smi (S4)"
            )
        accelerators.append(Accelerator(
            id=_stable_id(pnp, i),
            vendor=vendor,  # type: ignore[arg-type]
            name=name,  # raw driver string, deliberately not normalized
            is_laptop_variant=is_laptop,
            topology=topology,  # type: ignore[arg-type]
            # undetermined which adapter drives the display -> conservative
            # True (budget shrinks: FR direction). SPEC §3.2.
            drives_display=True,
            vram_total_bytes=None,
            vram_free_bytes=None,
            bandwidth_bytes_per_s=None,
            bandwidth_source=Source.UNKNOWN,  # spec_db arrives in S5
            peak_flops_fp16=None,
            flops_source=Source.UNKNOWN,
        ))
    return accelerators


def collect_video_controllers(diags: list[str]) -> list[Accelerator]:
    """Thin I/O layer: CIM gpu + enclosure queries, then pure assembly."""
    gpu_raw = cim.run_cim("gpu", diags)
    chassis: list[int] | None = None
    enc = cim.run_cim("enclosure", diags)
    if isinstance(enc, dict):
        got = enc.get("ChassisTypes")
        if isinstance(got, list):
            chassis = [int(c) for c in got]
    return parse_video_controllers(gpu_raw, chassis_types=chassis, diags=diags)

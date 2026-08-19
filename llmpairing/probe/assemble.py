"""Profile assembly (T-001 SPEC §1.3): gather collectors, validate, emit.

The ONLY module that composes a HardwareProfile from live collectors. A
profile that cannot be made schema-legal raises ProbeError — schema
violations are T-001 bugs, never downstream surprises.
"""
from __future__ import annotations

from llmpairing.probe.cpu import collect_cpu
from llmpairing.probe.gpu_nvidia import enrich_nvidia
from llmpairing.probe.gpu_windows import collect_video_controllers
from llmpairing.probe.memory import collect_memory
from llmpairing.probe.runner import ProbeError
from llmpairing.schemas import HardwareProfile
from llmpairing.types import Tier

_PLATFORMS = ("windows", "darwin", "linux")


def build_profile(diags: list[str], *, platform_name: str,
                  measured_at_unix: int) -> HardwareProfile:
    """Assemble and schema-validate a HardwareProfile.

    probe_tier is T0 until a bandwidth microbenchmark (S7) or T-003 deep
    scan upgrades it — spec-table lookups and API readouts never rate
    higher (SPEC §2.1).
    """
    if platform_name not in _PLATFORMS:
        raise ProbeError(f"unsupported platform '{platform_name}'")
    if platform_name == "darwin":
        # S6 (Apple branch) is unimplemented. A darwin HardwareProfile
        # REQUIRES the apple memory block (schema §2.1) and we refuse to
        # fabricate one (R-T1-4).
        raise ProbeError(
            "darwin probing requires the Apple branch (T-001 S6, not yet "
            "implemented) — refusing to fabricate the apple memory block"
        )

    probe_notes: list[str] = []
    if platform_name == "windows":
        accelerators = collect_video_controllers(diags)
        # T-001 S4 (compromise): fill NVIDIA VRAM from nvidia-smi; the
        # parser is field-UNVERIFIED and stamps probe_notes accordingly
        accelerators, probe_notes = enrich_nvidia(accelerators, diags)
    else:
        accelerators = []
        diags.append(
            f"gpu: {platform_name} GPU enumeration not implemented "
            f"(T-001 S4/S6) — no accelerators reported"
        )

    return HardwareProfile(
        schema_version="1.1" if probe_notes else "1.0",
        probe_tier=Tier.T0,
        platform=platform_name,  # type: ignore[arg-type]
        measured_at_unix=measured_at_unix,
        cpu=collect_cpu(diags),
        system_memory=collect_memory(diags),
        accelerators=accelerators,
        apple=None,
        probe_notes=probe_notes,
    )

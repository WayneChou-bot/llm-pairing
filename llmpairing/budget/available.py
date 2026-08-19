"""Available memory budget B (T-002 SPEC §4.1) — three topology branches.

PURE FUNCTIONS ONLY (SPEC §1.3): no I/O, no clock, no randomness.

Interpretation notes (recorded per BRIEF §4; flagged in the S3 report):
- SPEC §4.1 lists R_safety inside B while §4.4 steps 5-6 subtract safety
  again. Read together with the TIGHT definition in §3.2 ("fits but
  headroom < safety margin"), the consistent semantics are:
      FITS  : D <= B_no_safety - safety
      TIGHT : D <= B_no_safety
  compute_budget() returns the SPEC §4.1 value (WITH safety subtracted);
  safety_margin_bytes() exposes the margin so classify can form the band.
- SPEC §4.1b's literal formula does not subtract R_driver_ctx for UNIFIED
  even though the platform table lists Metal 0.2 GiB. OWNER DECISION
  (2026-08-14): subtract it — conservative-first (§6.1), FR-only direction.
- Safety margin: OWNER DECISION (2026-08-14): interpretation A — subtracted
  exactly once; TIGHT band per §3.2 (see classify.py).
- Ratio bases: (b) and (c) apply frag/safety to the post-min() base.
"""
from __future__ import annotations

import math

from llmpairing.budget.constants import (
    APPLE_DEFAULT_WIRED_RATIO,
    APPLE_WIRED_HEADROOM_RATIO,
    IGPU_AVAILABLE_RATIO,
    R_DISPLAY_BYTES,
    R_DRIVER_CTX_BYTES,
    R_FRAG_RATIO,
)
from llmpairing.schemas import Accelerator, HardwareProfile, Workload
from llmpairing.tier import TierTracker
from llmpairing.types import Measured, Source, Tier

MIB = 1024 * 1024


def _ceil_ratio(base: int, ratio: float) -> int:
    """ratio -> bytes, always rounded UP (reserves grow: conservative, §6.1)."""
    return math.ceil(base * ratio)


def safety_margin_bytes(hw: HardwareProfile, wl: Workload) -> int:
    """R_safety for the pool this hardware classifies into (same base as B)."""
    base, _, _, _ = _pool(hw)
    return _ceil_ratio(base, wl.safety_margin_ratio)


def _sole_accelerator(hw: HardwareProfile) -> Accelerator | None:
    if len(hw.accelerators) == 0:
        return None
    if len(hw.accelerators) > 1:
        raise ValueError("multi-accelerator topologies are classified upstream (N-08)")
    return hw.accelerators[0]


def _pool(hw: HardwareProfile) -> tuple[int, int, Tier, list[str]]:
    """Return (ratio_base, fixed_reserves, tier_cap, notes) for the pool.

    ratio_base is what frag/safety ratios apply to; fixed_reserves are the
    flat R_display / R_driver_ctx subtractions.
    """
    acc = _sole_accelerator(hw)
    notes: list[str] = []

    if acc is None:
        # RAM_ONLY pool (SPEC §4.4 step 4): system memory, same formula shape.
        # OS reserve is already reflected in `available_bytes`; no display/
        # driver reserves apply. Interpretation recorded in S3 report.
        notes.append("RAM_ONLY_POOL")
        return hw.system_memory.available_bytes, 0, hw.probe_tier, notes

    if acc.topology == "DISCRETE":
        if acc.vram_free_bytes is None:
            raise ValueError(
                "DISCRETE accelerator without vram_free_bytes: input contract "
                "violation — T-001 must supply it (R-2: we do not guess)"
            )
        fixed = R_DRIVER_CTX_BYTES[hw.platform]
        if acc.drives_display:
            fixed += R_DISPLAY_BYTES[hw.platform]
        return acc.vram_free_bytes, fixed, hw.probe_tier, notes

    if acc.topology == "UNIFIED":
        apple = hw.apple
        if apple is None:
            raise ValueError("UNIFIED topology requires the apple block (schema enforces)")
        if apple.recommended_max_working_set_bytes is not None:
            ceiling = apple.recommended_max_working_set_bytes
        elif apple.iogpu_wired_limit_mb not in (None, 0):
            # 0 and None both mean "system default" (P-05) — handled below.
            assert apple.iogpu_wired_limit_mb is not None
            ceiling = apple.iogpu_wired_limit_mb * MIB
        else:
            ceiling = math.floor(apple.unified_memory_bytes * APPLE_DEFAULT_WIRED_RATIO)
        if ceiling < apple.unified_memory_bytes * APPLE_WIRED_HEADROOM_RATIO:
            notes.append("APPLE_WIRED_LIMIT_HEADROOM")
        base = min(ceiling, hw.system_memory.available_bytes)
        # Owner decision 2026-08-14: subtract Metal driver context reserve
        # (constants table value) despite §4.1b's literal formula omitting it.
        # R_display stays 0 on darwin — already inside the wired limit.
        return base, R_DRIVER_CTX_BYTES[hw.platform], hw.probe_tier, notes

    # INTEGRATED_SHARED (SPEC §4.1c, P-06): dedicated-VRAM figure is
    # physically meaningless; structural approximation, forced T0.
    notes.append("IGPU_ESTIMATE_LOW_CONFIDENCE")
    base = math.floor(hw.system_memory.available_bytes * IGPU_AVAILABLE_RATIO)
    fixed = R_DRIVER_CTX_BYTES[hw.platform]
    if acc.drives_display:
        fixed += R_DISPLAY_BYTES[hw.platform]
    return base, fixed, Tier.T0, notes


def compute_budget(hw: HardwareProfile, wl: Workload) -> Measured[int]:
    """SPEC §4.1 available budget B (WITH the safety margin subtracted).

    vram_free is a point-in-time snapshot (P-04) — the caller's report layer
    must surface measured_at alongside this number.
    """
    base, fixed, tier_cap, notes = _pool(hw)
    frag = _ceil_ratio(base, R_FRAG_RATIO)
    safety = _ceil_ratio(base, wl.safety_margin_ratio)
    value = base - fixed - frag - safety
    tier = TierTracker.combine(hw.probe_tier, tier_cap)
    return Measured[int](
        value=value,
        unit="bytes",
        tier=tier,
        source=Source.ESTIMATED,
        ci_low=None,
        ci_high=None,
        notes=notes,
    )

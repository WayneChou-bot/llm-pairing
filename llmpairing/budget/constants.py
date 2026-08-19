"""Platform constants for the memory budget model (T-002 SPEC §4.1).

EVERY value here is a PRIOR, not a law of physics (SPEC §4.1). T-003
calibration overwrites them with per-machine measurements, and any such
override must be recorded in the Measured.source of downstream values.
Changing any value requires a commit message with the new source and the
re-run verification results (SPEC §9, P-20).

All capacities are int bytes (R-4 / P-03). 1 GiB = 1024**3 = 1,073,741,824.
"""
from __future__ import annotations

from typing import Final

GIB: Final[int] = 1024**3

# --- OS / desktop compositor VRAM reserve (R_display) ------------------------
# Source: T-002 SPEC §4.1 platform constants table (owner-supplied priors,
# recorded 2026-08; to be overwritten by T-003 calibration).
# Applied only when the accelerator drives the display.
R_DISPLAY_BYTES: Final[dict[str, int]] = {
    "windows": 1 * GIB,          # DWM compositor typical resident footprint
    "darwin": 0,                 # already inside the Apple wired limit (§4.1b)
    "linux": 512 * 1024 * 1024,  # 0.5 GiB, X11/Wayland compositor
}

# --- driver context reserve (R_driver_ctx) -----------------------------------
# Source: same SPEC table. 0.4 GiB is not an integer byte count; we take
# ceil(0.4 GiB) = 429,496,730 — rounding UP makes the reserve larger, which
# is the conservative direction (SPEC §6.1).
R_DRIVER_CTX_BYTES: Final[dict[str, int]] = {
    "windows": 429_496_730,      # ceil(0.4 GiB), CUDA context
    "darwin": 214_748_365,       # ceil(0.2 GiB), Metal
    "linux": 429_496_730,        # ceil(0.4 GiB), CUDA context
}

# --- allocator fragmentation ratio -------------------------------------------
# Source: SPEC §4.1a — "R_frag = 0.03 x vram_free". Status: PRIOR.
R_FRAG_RATIO: Final[float] = 0.03

# --- Apple unified-memory GPU ceiling ----------------------------------------
# Source: SPEC §4.1b — system default approximation when neither
# recommendedMaxWorkingSetSize nor a non-zero iogpu.wired_limit_mb is
# available: 0.75 x unified memory. Status: PRIOR (macOS default behaviour).
APPLE_DEFAULT_WIRED_RATIO: Final[float] = 0.75
# SPEC §4.1b: ceiling below 0.70 x unified triggers APPLE_WIRED_LIMIT_HEADROOM.
APPLE_WIRED_HEADROOM_RATIO: Final[float] = 0.70

# --- Windows/Linux iGPU shared-memory budget ratio ---------------------------
# Source: SPEC §4.1c — "B = min(available x 0.5, driver ceiling)".
# The 0.5 is a structural approximation; the whole branch is forced to T0
# with IGPU_ESTIMATE_LOW_CONFIDENCE (P-06).
IGPU_AVAILABLE_RATIO: Final[float] = 0.5

# --- activation / compute buffer (A term, SPEC §4.2) -------------------------
# Source: SPEC §4.2 — A_base default 0.5 GiB; A_slope awaits T-003
# calibration. Pre-calibration (T0) the spec mandates using the UPPER bound
# of the [0.5 GiB, 1.5 GiB] interval (conservative-first, §6.3), with the
# interval preserved as the confidence bounds.
# Status: PRIOR — overwritten per-engine by T-003.
A_T0_LOWER_BYTES: Final[int] = 512 * 1024 * 1024    # 0.5 GiB
A_T0_UPPER_BYTES: Final[int] = 1_610_612_736        # 1.5 GiB

# --- weights fallback uncertainty (P-09) -------------------------------------
# Source: T-002 PITFALLS P-09 — reconstructing W from params x bpw for mixed
# precision quants (Q4_K_M) carries up to +/-8% error. Used only on the
# ESTIMATED fallback path; the ci is widened by this ratio.
WEIGHTS_FALLBACK_CI_RATIO: Final[float] = 0.08

# --- bytes per logit (L term, SPEC §4.2) -------------------------------------
# float32 logits buffer: 4 bytes per vocab entry. Structural constant of
# llama.cpp's output buffer, not a tunable.
LOGIT_BYTES: Final[int] = 4

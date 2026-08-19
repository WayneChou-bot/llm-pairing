"""Efficiency constants for the roofline throughput model (T-002 SPEC §5).

EVERY value here must carry its provenance (R-6). All are PRIORS awaiting
T-003 per-machine calibration; any change requires the re-run verification
results in the commit message (P-20).
"""
from __future__ import annotations

from typing import Final

# eta_decode — fraction of the roofline memory-bandwidth ceiling that real
# single-stream decode achieves.
# Source: roofline literature reports actual llama.cpp-class decode
# throughput at 50-80% of the theoretical BW/bytes ceiling; the gap is
# attention over the KV cache, kernel overhead and sampling (T-002 SPEC
# §5.2, owner-supplied range).
# Value: median of the range. ci = the full range.
# Status: PRIOR — T-003 overwrites with the machine's measured eta (T1),
# with ci from repeated-measurement dispersion.
ETA_DECODE_T0: Final[float] = 0.65
ETA_DECODE_T0_CI_LOW: Final[float] = 0.50
ETA_DECODE_T0_CI_HIGH: Final[float] = 0.80

# eta_prefill — fraction of peak FP16 FLOPS that real prompt processing
# achieves on consumer hardware.
# Source: T-002 SPEC §5.4 (owner-supplied): (1) quantized GEMM on consumer
# GPUs lands far below peak; (2) spec-sheet peak_flops_fp16 has no direct
# mapping to Q4 kernels; (3) flops_source is usually SPEC_DB. Hence a wide
# interval and a MANDATORY low-confidence flag downstream (P-14).
# Value: 0.35 with ci [0.15, 0.55] — deliberately wider (relative) than
# eta_decode's; a structural test pins that ordering.
# Status: PRIOR — T-003 overwrites per machine.
ETA_PREFILL_T0: Final[float] = 0.35
ETA_PREFILL_T0_CI_LOW: Final[float] = 0.15
ETA_PREFILL_T0_CI_HIGH: Final[float] = 0.55

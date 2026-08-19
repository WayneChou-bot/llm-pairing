"""Memory demand D = W + KV + A + L (T-002 SPEC §4.2).

PURE FUNCTIONS ONLY (SPEC §1.3). All byte quantities are int (R-4 / P-18);
float appears only in bpw / ratios, converted with math.ceil at the boundary.
"""
from __future__ import annotations

import math

from llmpairing.budget.constants import (
    A_T0_LOWER_BYTES,
    A_T0_UPPER_BYTES,
    LOGIT_BYTES,
    WEIGHTS_FALLBACK_CI_RATIO,
)
from llmpairing.budget.kv import kv_bytes_total
from llmpairing.schemas import ModelSpec, QuantVariant, Workload
from llmpairing.tier import TierTracker
from llmpairing.types import DemandBreakdown, Measured, Source, Tier


def _weights(quant: QuantVariant, model: ModelSpec) -> Measured[int]:
    """W — SPEC §4.2. file_bytes is authoritative (P-09); the params x bpw
    reconstruction is a flagged fallback with a +/-8% confidence interval."""
    if quant.file_bytes_source == "CATALOG_VERIFIED":
        assert quant.file_bytes is not None  # schema guarantees (P-09)
        return Measured[int](
            value=quant.file_bytes,
            unit="bytes",
            tier=TierTracker.combine(Tier.T2),  # exact byte count from catalog
            source=Source.CATALOG_VERIFIED,
            notes=[],
        )
    if quant.file_bytes is not None:
        # ESTIMATED source but a size present: treat as estimate, keep it.
        value = quant.file_bytes
    else:
        assert quant.bpw_effective is not None  # schema guarantees (P-09)
        value = math.ceil(model.n_params_total * quant.bpw_effective / 8)
    return Measured[int](
        value=value,
        unit="bytes",
        tier=TierTracker.combine(Tier.T0),
        source=Source.ESTIMATED,
        ci_low=math.floor(value * (1 - WEIGHTS_FALLBACK_CI_RATIO)),
        ci_high=math.ceil(value * (1 + WEIGHTS_FALLBACK_CI_RATIO)),
        notes=["WEIGHTS_ESTIMATED"],
    )


def _kv(model: ModelSpec, wl: Workload) -> Measured[int]:
    """KV — dispatched through budget/kv.py. May raise UnsupportedArchError;
    classify maps that to UNSUPPORTED_ARCH (never guessed here, P-16)."""
    notes: list[str] = []
    if model.arch_handler == "OVERRIDE":
        notes.append("KV_FROM_CATALOG_OVERRIDE")  # SPEC §4.3 mandated flag
    return Measured[int](
        value=kv_bytes_total(model, wl.kv_dtype, wl.ctx_target_tokens),
        unit="bytes",
        tier=TierTracker.combine(Tier.T2),  # exact formula over catalog params
        source=Source.CATALOG_VERIFIED,
        notes=notes,
    )


def _activation(model: ModelSpec, wl: Workload) -> Measured[int]:
    """A — SPEC §4.2. Pre-calibration (no T-003 data) the spec mandates the
    UPPER bound of the [0.5, 1.5] GiB interval, ci = the interval, tier T0."""
    return Measured[int](
        value=A_T0_UPPER_BYTES,
        unit="bytes",
        tier=TierTracker.combine(Tier.T0),
        source=Source.ESTIMATED,
        ci_low=A_T0_LOWER_BYTES,
        ci_high=A_T0_UPPER_BYTES,
        notes=["ACTIVATION_PRIOR_PENDING_T003"],
    )


def _logits(model: ModelSpec, wl: Workload) -> Measured[int]:
    """L — SPEC §4.2. v1 fixes logits_all=False, so L = vocab x 4 bytes.
    The term must exist explicitly in the breakdown even when small (P-11)."""
    return Measured[int](
        value=model.vocab_size * LOGIT_BYTES,
        unit="bytes",
        tier=TierTracker.combine(Tier.T2),
        source=Source.CATALOG_VERIFIED,
        notes=[],
    )


def compute_demand(model: ModelSpec, quant: QuantVariant, wl: Workload) -> DemandBreakdown:
    """SPEC §7 public API. Four mandatory components — never fewer (P-11).

    MoE note (P-07): weights/demand use n_params_total; n_params_active is
    ONLY for throughput (predict/decode.py) and is deliberately not imported
    here.
    """
    return DemandBreakdown(
        weights=_weights(quant, model),
        kv_cache=_kv(model, wl),
        activation=_activation(model, wl),
        logits=_logits(model, wl),
    )


def total_demand(bd: DemandBreakdown) -> Measured[int]:
    """Sum of the four components; V-P1 pins sum == total.

    Conservative ends: ci_high sums each component's ci_high (or value when
    no interval) so classification can take the demand-enlarging end (§6.1).
    """
    parts = (bd.weights, bd.kv_cache, bd.activation, bd.logits)
    values = [p.value for p in parts]
    if any(v is None for v in values):
        raise ValueError("demand components must all carry values (P-11)")
    total = sum(v for v in values if v is not None)
    hi = sum((p.ci_high if p.ci_high is not None else p.value) or 0 for p in parts)
    lo = sum((p.ci_low if p.ci_low is not None else p.value) or 0 for p in parts)
    notes = sorted({n for p in parts for n in p.notes})
    return Measured[int](
        value=total,
        unit="bytes",
        tier=TierTracker.combine(*parts),
        source=Source.ESTIMATED,
        ci_low=lo,
        ci_high=hi,
        notes=notes,
    )

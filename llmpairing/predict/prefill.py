"""Prefill throughput + TTFT — compute-bound roofline (T-002 SPEC §5.4).

    flops_per_token ~ 2 x n_params_active
                    + 2 x n_layers x n_attn_heads x head_dim x seq_len
    prefill_tps = eta_prefill x peak_flops_fp16 / flops_per_token
    ttft        = prompt_tokens / prefill_tps

seq_len in the attention quadratic term is the prompt being prefilled
(wl.prompt_tokens) — the sequence attention runs over during prompt
processing.

CONFIDENCE IS STRUCTURALLY LOW (P-14): the wide eta interval and the
mandatory PREFILL_LOW_CONFIDENCE marking have NO exception path — the
report layer must render prefill at reduced visual weight, and the flag
travels inside the Measured itself so no caller can drop it silently.

peak_flops_fp16 is None -> value None (R-2: no guessing).

PURE FUNCTIONS ONLY (SPEC §1.3).
"""
from __future__ import annotations

from llmpairing.predict.eta import (
    ETA_PREFILL_T0,
    ETA_PREFILL_T0_CI_HIGH,
    ETA_PREFILL_T0_CI_LOW,
)
from llmpairing.schemas import HardwareProfile, ModelSpec, QuantVariant, Workload
from llmpairing.tier import TierTracker
from llmpairing.types import Measured, Source, Tier

FLAG = "PREFILL_LOW_CONFIDENCE"


def _flops_per_token(model: ModelSpec, wl: Workload) -> int:
    linear = 2 * model.n_params_active  # P-07: active, not total
    attention = 2 * model.n_layers * model.n_attn_heads * model.head_dim * wl.prompt_tokens
    return linear + attention


def predict_prefill(hw: HardwareProfile, model: ModelSpec, quant: QuantVariant,
                    wl: Workload) -> Measured[float]:
    """SPEC §7 public API. Always carries PREFILL_LOW_CONFIDENCE in notes."""
    acc = hw.accelerators[0] if hw.accelerators else None
    if acc is None or acc.peak_flops_fp16 is None:
        return Measured[float](
            value=None, unit="tok/s", tier=hw.probe_tier, source=Source.UNKNOWN,
            notes=[FLAG, "peak_flops_fp16 unavailable — no claim (R-2)"],
        )
    flops = _flops_per_token(model, wl)
    v = ETA_PREFILL_T0 * acc.peak_flops_fp16 / flops
    # flops_source is typically SPEC_DB -> T0; eta is a T0 prior regardless
    tier = TierTracker.combine(hw.probe_tier, Tier.T0)
    return Measured[float](
        value=v, unit="tok/s", tier=tier, source=Source.ESTIMATED,
        ci_low=v / ETA_PREFILL_T0 * ETA_PREFILL_T0_CI_LOW,
        ci_high=v / ETA_PREFILL_T0 * ETA_PREFILL_T0_CI_HIGH,
        notes=[FLAG],
    )


def compute_ttft(prefill: Measured[float], wl: Workload) -> Measured[float]:
    """ttft = prompt_tokens / prefill_tps; the ci interval inverts."""
    if prefill.value is None:
        return Measured[float](
            value=None, unit="s", tier=prefill.tier, source=Source.UNKNOWN,
            notes=[FLAG, "no prefill prediction — no TTFT claim"],
        )
    return Measured[float](
        value=wl.prompt_tokens / prefill.value,
        unit="s",
        tier=prefill.tier,
        source=prefill.source,
        ci_low=(wl.prompt_tokens / prefill.ci_high) if prefill.ci_high else None,
        ci_high=(wl.prompt_tokens / prefill.ci_low) if prefill.ci_low else None,
        notes=[FLAG],
    )

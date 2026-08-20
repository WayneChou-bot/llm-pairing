"""Decode throughput — roofline model + context decay curve (T-002 SPEC §5.2-§5.6).

Single-stream decode is memory-bound: the ceiling is BW / bytes_per_token,
and real throughput lands at eta x ceiling. bytes_per_token grows linearly
with context (KV reads), so a single tok/s number is misleading (P-12) —
this module always returns two anchors plus the full decay curve.

PARTIAL_OFFLOAD uses the harmonic model (SPEC §5.6): the tokens/second the
user loses to the layers that fell off the GPU is the single most valuable
number this tool produces.

Refusal rules (R-2): bandwidth_source UNKNOWN -> every value is None with
NO_BANDWIDTH_NO_PREDICTION. No same-generation averages, no laptop factors.

PURE FUNCTIONS ONLY (SPEC §1.3).

MoE note (P-07): W_active_resident scales W by n_params_active /
n_params_total — memory demand elsewhere uses total; decode traffic uses
active. Golden vector G-07 and negative test N-05 land in S10.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from llmpairing.budget.classify import FitResult
from llmpairing.budget.kv import kv_bytes_total
from llmpairing.predict.calibration import MachineCalibration
from llmpairing.predict.eta import (
    ETA_DECODE_T0,
    ETA_DECODE_T0_CI_HIGH,
    ETA_DECODE_T0_CI_LOW,
)
from llmpairing.predict.prefill import FLAG as PREFILL_FLAG
from llmpairing.predict.prefill import compute_ttft, predict_prefill
from llmpairing.schemas import HardwareProfile, ModelSpec, QuantVariant, Workload
from llmpairing.tier import TierTracker
from llmpairing.types import Measured, Source, Tier, Verdict

_RUNNABLE = (Verdict.FITS, Verdict.TIGHT, Verdict.RAM_ONLY,
             Verdict.PARTIAL_OFFLOAD, Verdict.OOM_AT_CONTEXT)


class ThroughputResult(BaseModel):
    """SPEC §3.3 output contract (prefill fields arrive in S7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decode_tps_at_zero_ctx: Measured[float]
    decode_tps_at_target_ctx: Measured[float]
    decode_decay_curve: list[tuple[int, float]] = []
    prefill_tps: Measured[float]         # value=None until S7
    ttft_seconds: Measured[float]        # value=None until S7
    sustained_ratio: Measured[float]     # T2-measured ONLY; never fabricated (P-13)
    bound_by: Literal["MEMORY", "COMPUTE"]
    flags: list[str] = []


def _none_f(tier: Tier, unit: str, note: str) -> Measured[float]:
    return Measured[float](
        value=None, unit=unit, tier=tier, source=Source.UNKNOWN, notes=[note]
    )


def _sustained(hw: HardwareProfile) -> Measured[float]:
    # P-13 / V-P7: value only ever comes from a T2 measurement input, which
    # does not exist yet — so this is None with an honest note, NEVER a
    # laptop factor.
    return _none_f(hw.probe_tier, "ratio", "需 --deep 模式實測（T2）；不以經驗值填補")


def _refusal(hw: HardwareProfile, flags: list[str], note: str) -> ThroughputResult:
    t = hw.probe_tier
    return ThroughputResult(
        decode_tps_at_zero_ctx=_none_f(t, "tok/s", note),
        decode_tps_at_target_ctx=_none_f(t, "tok/s", note),
        decode_decay_curve=[],
        prefill_tps=_none_f(t, "tok/s", f"{PREFILL_FLAG}; {note}"),
        ttft_seconds=_none_f(t, "s", f"{PREFILL_FLAG}; {note}"),
        sustained_ratio=_sustained(hw),
        bound_by="MEMORY",  # theoretical for single-stream; no data to dispute it
        flags=flags,
    )


def _w_active_resident(quant: QuantVariant, model: ModelSpec) -> float:
    """SPEC §5.2: dense -> W; MoE -> W x active/total (P-07).

    A5: per-token-lookup params (gemma4 PLE tables) are read one row per
    token — effectively zero streaming traffic — and are excluded here.
    Memory demand elsewhere still uses the full file size.
    """
    w = quant.file_bytes
    if w is None:
        assert quant.bpw_effective is not None
        w = -(-int(model.n_params_total * quant.bpw_effective) // 8)
    streamed = model.n_params_active - model.n_params_per_token_lookup
    return w * (streamed / model.n_params_total)


def _bytes_per_token(model: ModelSpec, quant: QuantVariant, wl: Workload,
                     ctx: int) -> float:
    # decode reads the whole resident KV cache every token; for SWA/HYBRID
    # that cache saturates at the window, so totals (not a linear per-token
    # normalizer) are the correct traffic model
    return _w_active_resident(quant, model) + kv_bytes_total(model, wl.kv_dtype, ctx)


def _bound_by(model: ModelSpec, quant: QuantVariant, wl: Workload,
              hw: HardwareProfile, bw: int) -> Literal["MEMORY", "COMPUTE"]:
    """SPEC §5.5 self-consistency sentinel (V-P5)."""
    acc = hw.accelerators[0] if hw.accelerators else None
    if acc is None or acc.peak_flops_fp16 is None:
        return "MEMORY"  # single-stream decode is memory-bound by theory
    ai = (2 * model.n_params_active) / _bytes_per_token(model, quant, wl, wl.ctx_target_tokens)
    machine_balance = acc.peak_flops_fp16 / bw
    if ai > machine_balance:
        raise ValueError(
            "single-stream decode computed as COMPUTE-bound — input data is "
            "corrupt (SPEC §5.5 / V-P5 sentinel)"
        )
    return "MEMORY"


def _calibrated_cpu(hw: HardwareProfile, model: ModelSpec, quant: QuantVariant,
                    wl: Workload, fit: FitResult,
                    cal: MachineCalibration) -> ThroughputResult:
    """T-003 calibrated path: tps = measured product / bytes_per_token.

    Tier: combine(product T1, catalog-exact traffic T2) = T1. The A-prior
    and budget play no role in the traffic model, so they do not drag the
    tier down (§3.1: combine the inputs actually used).
    """
    tier = TierTracker.combine(Tier.T1, Tier.T2)
    notes = [f"calibrated on {cal.calibration_model} ({cal.calibration_quant}), "
             f"n={cal.n_runs}, {cal.backend}", *cal.notes]

    def tps(ctx: int) -> float:
        return cal.product_bytes_per_s / _bytes_per_token(model, quant, wl, ctx)

    def measured(ctx: int, extra: list[str]) -> Measured[float]:
        bpt = _bytes_per_token(model, quant, wl, ctx)
        return Measured[float](
            value=cal.product_bytes_per_s / bpt,
            unit="tok/s", tier=tier, source=Source.MEASURED,
            ci_low=cal.product_ci_low / bpt,
            ci_high=cal.product_ci_high / bpt,
            notes=notes + extra,
        )

    cap = fit.ctx_max_tokens.value or 0
    points = [0]
    step = 1024
    while step < cap:
        points.append(step)
        step *= 2
    if cap > 0:
        points.append(cap)
    curve = [(c, tps(c)) for c in points]
    target_ok = wl.ctx_target_tokens <= cap
    at_target = (measured(wl.ctx_target_tokens, [])
                 if target_ok
                 else _none_f(tier, "tok/s",
                              "target ctx exceeds feasible ctx_max — no claim"))

    prefill = predict_prefill(hw, model, quant, wl)  # calibration is decode-only
    return ThroughputResult(
        decode_tps_at_zero_ctx=measured(0, ["theoretical zero-context anchor"]),
        decode_tps_at_target_ctx=at_target,
        decode_decay_curve=curve,
        prefill_tps=prefill,
        ttft_seconds=compute_ttft(prefill, wl),
        sustained_ratio=_sustained(hw),  # V-P7: thermal stays T2-only
        bound_by="MEMORY",
        flags=(["MACHINE_CALIBRATED", PREFILL_FLAG]
               + (["LOOKUP_PARAMS_TRAFFIC_EXCLUDED"]
                  if model.n_params_per_token_lookup else [])),
    )


def _bandwidths(hw: HardwareProfile, fit: FitResult) -> tuple[int | None, int | None, Tier]:
    """(gpu_bw, ram_bw, bw_tier). RAM_ONLY machines use the RAM pool only."""
    ram_bw = hw.system_memory.bandwidth_bytes_per_s
    if not hw.accelerators:
        tier = Tier.T0 if hw.system_memory.bandwidth_source is Source.SPEC_DB else hw.probe_tier
        return None, ram_bw, tier
    acc = hw.accelerators[0]
    tier = Tier.T0 if acc.bandwidth_source is Source.SPEC_DB else hw.probe_tier
    return acc.bandwidth_bytes_per_s, ram_bw, tier


def predict_decode(hw: HardwareProfile, model: ModelSpec, quant: QuantVariant,
                   wl: Workload, fit: FitResult,
                   calibration: MachineCalibration | None = None) -> ThroughputResult:
    """SPEC §7 public API. Never called into by classify_fit (no cycles).

    T-003: a MachineCalibration whose pool matches this scenario replaces
    the spec-db roofline with the MEASURED BW x eta product — predictions
    become T1 with the measurement's own dispersion as ci.
    """
    if fit.verdict not in _RUNNABLE:
        return _refusal(hw, [], "not runnable — no throughput claim (R-2)")

    cal_flags: list[str] = []
    if calibration is not None:
        from llmpairing.predict.calibration import calibration_applies
        scenario_pool = "gpu" if hw.accelerators else "cpu"
        ok, refusal = calibration_applies(hw, calibration, scenario_pool)
        if ok and scenario_pool == "cpu":
            return _calibrated_cpu(hw, model, quant, wl, fit, calibration)
        cal_flags.append(refusal or "CALIBRATION_POOL_MISMATCH_IGNORED")

    gpu_bw, ram_bw, bw_tier = _bandwidths(hw, fit)
    partial = fit.verdict is Verdict.PARTIAL_OFFLOAD

    if hw.accelerators and gpu_bw is None:
        return _refusal(hw, cal_flags + ["NO_BANDWIDTH_NO_PREDICTION"],
                        "bandwidth_source UNKNOWN — 不猜測（§5.1）")
    if not hw.accelerators and ram_bw is None:
        return _refusal(hw, cal_flags + ["NO_BANDWIDTH_NO_PREDICTION"],
                        "RAM bandwidth UNKNOWN — 不猜測（§5.1）")
    if partial and ram_bw is None:
        return _refusal(hw, cal_flags + ["NO_RAM_BANDWIDTH"],
                        "partial offload 需 RAM 頻寬，UNKNOWN — 不猜測（§5.6）")

    flags: list[str] = list(cal_flags)
    if model.n_params_per_token_lookup:
        flags.append("LOOKUP_PARAMS_TRAFFIC_EXCLUDED")  # A5
    notes: list[str] = []
    g = n = 0
    if partial:
        assert fit.n_gpu_layers_max.value is not None
        g, n = fit.n_gpu_layers_max.value, model.n_layers
        notes.append(f"harmonic partial-offload model, g={g}/{n} (SPEC §5.6)")
        flags.append("PARTIAL_OFFLOAD_HARMONIC")

    def tps(ctx: int) -> float:
        bpt = _bytes_per_token(model, quant, wl, ctx)
        if partial:
            assert gpu_bw is not None and ram_bw is not None
            t = (bpt * g / n) / gpu_bw + (bpt * (n - g) / n) / ram_bw
            return ETA_DECODE_T0 / t
        bw = gpu_bw if gpu_bw is not None else ram_bw
        assert bw is not None
        return ETA_DECODE_T0 * bw / bpt

    # eta is a T0 prior -> every prediction is T0 until T-003 calibration
    tier = TierTracker.combine(bw_tier, Tier.T0, fit.headroom_bytes)

    def measured(ctx: int, extra: list[str]) -> Measured[float]:
        v = tps(ctx)
        return Measured[float](
            value=v, unit="tok/s", tier=tier, source=Source.ESTIMATED,
            ci_low=v / ETA_DECODE_T0 * ETA_DECODE_T0_CI_LOW,
            ci_high=v / ETA_DECODE_T0 * ETA_DECODE_T0_CI_HIGH,
            notes=notes + extra,
        )

    # decay curve cap: full-GPU feasible ctx_max; PARTIAL runs at the target
    # under the harmonic model instead
    cap = fit.ctx_max_tokens.value or 0
    if partial:
        cap = max(cap, wl.ctx_target_tokens)
    points = [0]
    step = 1024
    while step < cap:
        points.append(step)
        step *= 2
    if cap > 0:
        points.append(cap)
    curve = [(c, tps(c)) for c in points]

    target_ok = wl.ctx_target_tokens <= cap or partial
    at_target = (
        measured(wl.ctx_target_tokens, [])
        if target_ok
        else _none_f(tier, "tok/s", "target ctx exceeds feasible ctx_max — no claim")
    )

    prefill = predict_prefill(hw, model, quant, wl)
    ttft = compute_ttft(prefill, wl)
    flags.append(PREFILL_FLAG)  # mandatory, no exception path (P-14)

    return ThroughputResult(
        decode_tps_at_zero_ctx=measured(0, ["theoretical zero-context anchor"]),
        decode_tps_at_target_ctx=at_target,
        decode_decay_curve=curve,
        prefill_tps=prefill,
        ttft_seconds=ttft,
        sustained_ratio=_sustained(hw),
        bound_by=_bound_by(model, quant, wl, hw, gpu_bw or ram_bw or 1),
        flags=flags,
    )


def decode_bytes_per_token(model: ModelSpec, quant: QuantVariant,
                           wl: Workload, ctx: int) -> float:
    """Public read of the decode traffic model (bytes read per generated
    token). Used by the recommender for RELATIVE speed ordering when no
    bandwidth data exists — the ordering is sound under the roofline model
    even though absolute tok/s is unknown."""
    return _bytes_per_token(model, quant, wl, ctx)

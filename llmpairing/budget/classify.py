"""Fit classification decision chain (T-002 SPEC §4.4) — full S5 grain.

All nine chain steps are live: FITS / TIGHT / RAM_ONLY / OOM_AT_CONTEXT
(ctx_max + tradeoff curve) / PARTIAL_OFFLOAD (n_gpu_layers_max, conservative
all-KV-on-GPU approximation, SPEC §4.6) / OOM_AT_LOAD / UNSUPPORTED_*.

PURE FUNCTIONS ONLY (SPEC §1.3). classify_fit never calls predict_* (§7).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from llmpairing.budget.available import compute_budget, safety_margin_bytes
from llmpairing.budget.ctx_solver import (
    TRADEOFF_CTX_POINTS,
    _conservative_demand_at,
    solve_ctx_max,
)
from llmpairing.budget.demand import compute_demand, total_demand
from llmpairing.budget.kv import UnsupportedArchError
from llmpairing.budget.offload import (
    ram_spill_bytes,
    solve_n_gpu_layers_max,
    w_per_layer_bytes,
)
from llmpairing.budget.constants import A_T0_UPPER_BYTES, LOGIT_BYTES
from llmpairing.schemas import HardwareProfile, ModelSpec, QuantVariant, Workload
from llmpairing.tier import TierTracker
from llmpairing.types import DemandBreakdown, Measured, Source, Tier, Verdict

_CTX_MIN = 512  # SPEC §4.4 step 7 probe context
_PENDING_OFFLOAD_NOTE = "awaits S5 offload solver"


class FitResult(BaseModel):
    """SPEC §3.2 output contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Verdict
    budget_bytes: Measured[int]
    demand_bytes: Measured[int]
    demand_breakdown: DemandBreakdown
    headroom_bytes: Measured[int]
    ctx_max_tokens: Measured[int]
    n_gpu_layers_max: Measured[int]      # value=None until S5
    tradeoff_curve: list[tuple[int, Verdict]] = []
    flags: list[str] = []


def _none_measured(tier: Tier, note: str) -> Measured[int]:
    return Measured[int](
        value=None, unit="tokens", tier=tier, source=Source.UNKNOWN, notes=[note]
    )


def _unsupported(verdict: Verdict, hw: HardwareProfile) -> FitResult:
    """Short-circuit result for UNSUPPORTED_* — no numbers are claimed (R-2)."""
    empty = Measured[int](
        value=None, unit="bytes", tier=hw.probe_tier, source=Source.UNKNOWN, notes=[]
    )
    empty_bd = DemandBreakdown(
        weights=empty, kv_cache=empty, activation=empty, logits=empty
    )
    return FitResult(
        verdict=verdict,
        budget_bytes=empty,
        demand_bytes=empty,
        demand_breakdown=empty_bd,
        headroom_bytes=empty,
        ctx_max_tokens=_none_measured(hw.probe_tier, "unsupported input"),
        n_gpu_layers_max=_none_measured(hw.probe_tier, _PENDING_OFFLOAD_NOTE),
        tradeoff_curve=[],
        flags=[],
    )


def _band_verdict(demand_cons: int, b_with_safety: int, b_no_safety: int) -> Verdict:
    """FITS / TIGHT / (over budget) per owner-confirmed interpretation A."""
    if demand_cons <= b_with_safety:
        return Verdict.FITS
    if demand_cons <= b_no_safety:
        return Verdict.TIGHT
    return Verdict.OOM_AT_CONTEXT  # meaning: over budget AT THIS ctx


def _offload_gate_open(model: ModelSpec, quant: QuantVariant, b_ws: int) -> bool:
    """SPEC §4.4 step 8 (GPU side): W_single_layer + A + L <= B."""
    fixed = A_T0_UPPER_BYTES + model.vocab_size * LOGIT_BYTES
    return w_per_layer_bytes(model, quant) + fixed <= b_ws


def _ram_pool_budget(hw: HardwareProfile, wl: Workload) -> int:
    """Amendment A2: the RAM overflow pool, conservative end (safety off
    the top; available is already the OS's reallocatable estimate)."""
    return int(hw.system_memory.available_bytes * (1 - wl.safety_margin_ratio))


def _verdict_at(model: ModelSpec, quant: QuantVariant, wl: Workload, ctx: int,
                b_with_safety: int, b_no_safety: int,
                ram_only: bool, ram_budget: int,
                ) -> tuple[Verdict, int | None, int, list[str]]:
    """A6: THE single verdict-at-one-ctx path — the main verdict is this
    function at the target ctx, the trade-off curve is this function at
    each sampled ctx. Returns (verdict, g_layers, spill_bytes, notes).

    Order (owner ruling 2026-08-20, amendment A6): FITS / TIGHT ->
    PARTIAL_OFFLOAD (runnability first) -> OOM_AT_CONTEXT (not runnable
    at this ctx even with offload, but fully loadable at a short ctx)
    -> OOM_AT_LOAD.
    """
    # review #5 P1 (owner-ratified): a ctx the model cannot attend at is
    # refused BEFORE any memory math — "memory suffices" must never be
    # spoken as "it runs". Curve points are pre-filtered to <= model max,
    # so this fires only for an out-of-range target request.
    if ctx > model.max_position_embeddings:
        return (Verdict.CTX_EXCEEDS_MODEL_MAX, None, 0,
                ["CTX_TARGET_EXCEEDS_MODEL_MAX"])
    d = _conservative_demand_at(model, quant, wl, ctx)
    band = _band_verdict(d, b_with_safety, b_no_safety)
    if band is Verdict.FITS:
        return (Verdict.RAM_ONLY if ram_only else Verdict.FITS), None, 0, []
    if band is Verdict.TIGHT:
        if ram_only:
            return Verdict.RAM_ONLY, None, 0, ["RAM_ONLY_TIGHT_HEADROOM"]
        return Verdict.TIGHT, None, 0, []
    # over budget at this ctx — A6: try partial offload FIRST
    if not ram_only and _offload_gate_open(model, quant, b_with_safety):
        ctx_wl = wl.model_copy(update={"ctx_target_tokens": ctx})
        g, offload_notes = solve_n_gpu_layers_max(model, quant, ctx_wl,
                                                  b_with_safety)
        if g >= 1:
            spill = ram_spill_bytes(model, quant, g)
            if spill <= ram_budget:
                return Verdict.PARTIAL_OFFLOAD, g, spill, offload_notes
            # A2: the spilled weights have nowhere to live
            return (Verdict.OOM_AT_LOAD, None, spill,
                    ["RAM_SPILL_EXCEEDS_AVAILABLE"])
    # offload infeasible here: ctx problem or load problem?
    if _conservative_demand_at(model, quant, wl, _CTX_MIN) <= b_no_safety:
        return Verdict.OOM_AT_CONTEXT, None, 0, []
    return Verdict.OOM_AT_LOAD, None, 0, []


def _tradeoff_curve(model: ModelSpec, quant: QuantVariant, wl: Workload,
                    b_with_safety: int, b_no_safety: int,
                    ram_only: bool, ram_budget: int) -> list[tuple[int, Verdict]]:
    """SPEC §6.4 — the A6 shared path evaluated at each sampled ctx."""
    return [
        (ctx, _verdict_at(model, quant, wl, ctx, b_with_safety, b_no_safety,
                          ram_only, ram_budget)[0])
        for ctx in TRADEOFF_CTX_POINTS
        if ctx <= model.max_position_embeddings
    ]


def classify_fit(hw: HardwareProfile, model: ModelSpec,
                 quant: QuantVariant, wl: Workload) -> FitResult:
    """SPEC §4.4 decision chain (S4 grain)."""
    flags: list[str] = []

    # probe-level honesty notes ride along on every verdict (schema 1.1)
    flags.extend(n for n in hw.probe_notes if n not in flags)

    # 1. arch whitelist
    if not model.arch_supported:
        return _unsupported(Verdict.UNSUPPORTED_ARCH, hw)

    # A discrete GPU whose VRAM figures the probe could not obtain (T-001 S4
    # pending) is excluded from the pool rather than crashing the budget --
    # and this runs BEFORE the topology count (review #3 item 1: the common
    # iGPU + data-less dGPU laptop must degrade to the iGPU scenario, not
    # die at the topology gate). Conservative direction, flagged.
    usable = [a for a in hw.accelerators
              if not (a.topology == "DISCRETE" and a.vram_free_bytes is None)]
    if len(usable) != len(hw.accelerators):
        flags.append("DISCRETE_GPU_DATA_UNAVAILABLE_EXCLUDED")
        hw = hw.model_copy(update={"accelerators": usable})

    # 2. topology (P-19, N-08): >1 accelerator WITH usable data is a real
    # multi-GPU pool -- v1 refuses rather than silently picking one.
    if len(hw.accelerators) > 1:
        return _unsupported(Verdict.UNSUPPORTED_TOPOLOGY, hw)
    # 3. engine
    if wl.engine != "llama.cpp":
        return _unsupported(Verdict.UNSUPPORTED_ENGINE, hw)

    # honesty guard (review #3): wl.ubatch is not modeled yet -- a
    # non-default value must be visibly ignored, never silently accepted
    if wl.ubatch != type(wl).model_fields["ubatch"].default:
        flags.append("UBATCH_NOT_YET_MODELED")

    ram_only_mode = len(hw.accelerators) == 0  # step 4: RAM pool, same formulas

    try:
        bd = compute_demand(model, quant, wl)
    except UnsupportedArchError:
        return _unsupported(Verdict.UNSUPPORTED_ARCH, hw)

    budget = compute_budget(hw, wl)
    safety = safety_margin_bytes(hw, wl)
    demand = total_demand(bd)
    flags.extend(budget.notes)
    flags.extend(n for n in demand.notes if n not in flags)
    if wl.ctx_target_tokens > model.max_position_embeddings:
        flags.append("CTX_TARGET_EXCEEDS_MODEL_MAX")

    assert budget.value is not None and demand.value is not None
    # Conservative ends (SPEC §6.1): demand-enlarging, budget-shrinking.
    demand_cons = demand.ci_high if demand.ci_high is not None else demand.value
    b_with_safety = budget.value
    b_no_safety = budget.value + safety

    tier = TierTracker.combine(budget, demand)
    headroom = Measured[int](
        value=b_no_safety - demand_cons,
        unit="bytes",
        tier=tier,
        source=Source.ESTIMATED,
        notes=["headroom vs budget WITHOUT safety margin; TIGHT band = [0, safety)"],
    )

    # ctx_max is solved for every pool-classified result (SPEC §3.2 field)
    ctx_max_value = solve_ctx_max(model, quant, wl, b_no_safety)
    ctx_max = Measured[int](
        value=ctx_max_value,
        unit="tokens",
        tier=tier,
        source=Source.ESTIMATED,
        notes=["largest physically feasible ctx (budget without safety margin)"],
    )
    ram_budget = _ram_pool_budget(hw, wl)
    curve = _tradeoff_curve(model, quant, wl, b_with_safety, b_no_safety,
                            ram_only_mode, ram_budget)

    n_gpu_layers: Measured[int] = _none_measured(
        tier, "meaningful only for PARTIAL_OFFLOAD verdicts"
    )

    # A6: the main verdict IS the shared path at the target ctx — the
    # invariant verdict == tradeoff_curve[target] holds by construction
    verdict, g_val, spill, extra = _verdict_at(
        model, quant, wl, wl.ctx_target_tokens,
        b_with_safety, b_no_safety, ram_only_mode, ram_budget)
    flags.extend(n for n in extra if n not in flags)
    if verdict is Verdict.PARTIAL_OFFLOAD:
        assert g_val is not None
        flags.append("PARTIAL_OFFLOAD_KV_APPROX")  # SPEC §4.6 mandated
        n_gpu_layers = Measured[int](
            value=g_val,
            unit="layers",
            tier=tier,
            source=Source.ESTIMATED,
            notes=["conservative: assumes full KV cache resident on GPU",
                   f"RAM spill {spill} bytes within pool {ram_budget}"],
        )

    return FitResult(
        verdict=verdict,
        budget_bytes=budget,
        demand_bytes=demand,
        demand_breakdown=bd,
        headroom_bytes=headroom,
        ctx_max_tokens=ctx_max,
        n_gpu_layers_max=n_gpu_layers,
        tradeoff_curve=curve,
        flags=flags,
    )

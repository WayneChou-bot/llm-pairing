"""ctx_max solver (T-002 SPEC §4.5).

FULL attention: closed form. SLIDING_WINDOW / HYBRID (piecewise-linear KV,
S9): monotone bisection — the generic bisection is implemented and
cross-checked against the closed form on FULL already in S4.

ctx_max is solved against the budget WITHOUT the safety margin: it answers
"the largest physically feasible context" (SPEC §3.2 — 最大可行 ctx).
V-P10 pins the boundary: at ctx_max the verdict is FITS or TIGHT, at
ctx_max + 1 it is never FITS.

PURE FUNCTIONS ONLY (SPEC §1.3).
"""
from __future__ import annotations

from llmpairing.budget.demand import compute_demand, total_demand
from llmpairing.budget.kv import kv_bytes_per_token
from llmpairing.schemas import ModelSpec, QuantVariant, Workload

#: SPEC §4.5 — bisection convergence tolerance (documentation constant; the
#: integer bisection below converges exactly, tests keep the bound anyway).
BISECT_TOLERANCE_TOKENS = 128


def _conservative_demand_at(model: ModelSpec, quant: QuantVariant,
                            wl: Workload, ctx: int) -> int:
    """Demand at a given ctx, taking the demand-enlarging end (SPEC §6.1)."""
    probe_wl = wl.model_copy(update={"ctx_target_tokens": ctx})
    d = total_demand(compute_demand(model, quant, probe_wl))
    assert d.value is not None
    return d.ci_high if d.ci_high is not None else d.value


def solve_ctx_max(model: ModelSpec, quant: QuantVariant, wl: Workload,
                  budget_no_safety: int) -> int:
    """Closed form for FULL attention (SPEC §4.5):

        ctx_max = floor((B - W - A - L) / kv_bytes_per_token)

    clamped to [0, max_position_embeddings]. Non-FULL handlers delegate to
    bisection (their KV is piecewise linear in ctx).
    """
    if model.arch_handler != "FULL":
        return solve_ctx_max_bisect(model, quant, wl, budget_no_safety)
    fixed = _conservative_demand_at(model, quant, wl, 1) - kv_bytes_per_token(
        model, wl.kv_dtype
    )
    per_token = kv_bytes_per_token(model, wl.kv_dtype)
    if budget_no_safety <= fixed:
        return 0
    ctx = (budget_no_safety - fixed) // per_token
    return min(ctx, model.max_position_embeddings)


def solve_ctx_max_bisect(model: ModelSpec, quant: QuantVariant, wl: Workload,
                         budget_no_safety: int) -> int:
    """Monotone integer bisection: largest ctx with D(ctx) <= B.

    Premise (V-P2): D(ctx) is monotonically non-decreasing in ctx. Upper
    bound = max_position_embeddings (SPEC §4.5). Exact convergence.
    """
    lo, hi = 0, model.max_position_embeddings
    if _conservative_demand_at(model, quant, wl, 1) > budget_no_safety:
        return 0
    lo = 1
    if _conservative_demand_at(model, quant, wl, hi) <= budget_no_safety:
        return hi
    # invariant: D(lo) <= B < D(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _conservative_demand_at(model, quant, wl, mid) <= budget_no_safety:
            lo = mid
        else:
            hi = mid
    return lo


#: SPEC §6.4 sampling points for the tradeoff curve.
TRADEOFF_CTX_POINTS = (2_048, 4_096, 8_192, 16_384, 32_768, 131_072)

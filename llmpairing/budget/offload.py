"""Partial offload solver — n_gpu_layers_max (T-002 SPEC §4.6).

    W_per_layer = (W - W_embed) / n_layers
    g_max = floor((B - A - L - KV_gpu_portion) / W_per_layer), clamp [0, n_layers]

v1 conservative approximation (SPEC §4.6): KV is assumed to live ENTIRELY on
the GPU. This overestimates GPU demand, so errors are false-rejects only —
the caller must attach PARTIAL_OFFLOAD_KV_APPROX.

Direction analysis for every estimate in this module (SPEC §6.2 asymmetry):
- W_embed estimated from bpw_effective UNDER-estimates real embedding bytes
  (GGUF embeds are usually higher precision than the body average), which
  OVER-estimates W_per_layer -> fewer layers claimed -> FR only.
- bpw missing: W_embed = 0, maximally conservative, flagged.
- ceil on W_per_layer, floor on g_max: both FR direction.

g_max is solved against the budget WITH the safety margin: -ngl values are
launch configurations the user will actually run, the FA-critical case.

PURE FUNCTIONS ONLY (SPEC §1.3).
"""
from __future__ import annotations

import math

from llmpairing.budget.constants import A_T0_UPPER_BYTES, LOGIT_BYTES
from llmpairing.budget.kv import kv_bytes_total
from llmpairing.schemas import ModelSpec, QuantVariant, Workload


def _w_conservative(quant: QuantVariant, model: ModelSpec) -> int:
    """Weight bytes, demand-enlarging end (mirrors demand._weights policy)."""
    if quant.file_bytes is not None:
        if quant.file_bytes_source == "CATALOG_VERIFIED":
            return quant.file_bytes
        # estimated size: widen upward like demand.py's fallback ci does
        return math.ceil(quant.file_bytes * 1.08)
    assert quant.bpw_effective is not None  # schema guarantees (P-09)
    return math.ceil(math.ceil(model.n_params_total * quant.bpw_effective / 8) * 1.08)


def _w_embed_estimate(quant: QuantVariant, model: ModelSpec) -> tuple[int, list[str]]:
    """Embedding + lm_head bytes (not offloadable, SPEC §4.6).

    floor() and the bpw-average both bias LOW -> W_per_layer biases HIGH ->
    FR only. Without bpw we refuse to guess and use 0 (most conservative).
    """
    n_matrices = 1 if model.tie_word_embeddings else 2
    if quant.bpw_effective is None:
        return 0, ["OFFLOAD_EMBED_UNKNOWN_CONSERVATIVE"]
    embed_params = model.vocab_size * model.hidden_size * n_matrices
    return math.floor(embed_params * quant.bpw_effective / 8), []


def w_per_layer_bytes(model: ModelSpec, quant: QuantVariant) -> int:
    """Per-layer weight bytes for the offloadable stack (ceil: FR direction)."""
    w = _w_conservative(quant, model)
    w_embed, _ = _w_embed_estimate(quant, model)
    return -(-(w - w_embed) // model.n_layers)


def ram_spill_bytes(model: ModelSpec, quant: QuantVariant, g: int) -> int:
    """Weights that fall off the GPU at g layers (amendment A2).

    Conservative W end and ceil'd per-layer both bias the spill HIGH — the
    check errs FR-ward, never FA-ward."""
    w = _w_conservative(quant, model)
    return max(0, w - g * w_per_layer_bytes(model, quant))


def solve_n_gpu_layers_max(model: ModelSpec, quant: QuantVariant, wl: Workload,
                           budget_with_safety: int) -> tuple[int, list[str]]:
    """SPEC §4.6. Returns (g_max clamped to [0, n_layers], honesty notes)."""
    _, notes = _w_embed_estimate(quant, model)
    per_layer = w_per_layer_bytes(model, quant)
    kv_gpu = kv_bytes_total(model, wl.kv_dtype, wl.ctx_target_tokens)
    fixed = A_T0_UPPER_BYTES + model.vocab_size * LOGIT_BYTES
    numerator = budget_with_safety - fixed - kv_gpu
    if numerator <= 0 or per_layer <= 0:
        return 0, notes
    g = numerator // per_layer
    return max(0, min(model.n_layers, g)), notes

"""KV cache size dispatch (T-002 SPEC §4.3).

Handlers: FULL, SLIDING_WINDOW, HYBRID, OVERRIDE. Adding an architecture
means adding a handler + a golden vector, never modifying this core.

Two public quantities:
- kv_bytes_total(model, kv_dtype, ctx): the canonical KV cache size at a
  given context. SWA/HYBRID are piecewise-linear in ctx, so totals — not a
  per-token normalizer — are the general interface.
- kv_bytes_per_token(model, kv_dtype): the linear normalizer, valid ONLY
  for handlers whose KV is linear in ctx (FULL, OVERRIDE); others raise.
  Used by the closed-form ctx_max solver.

Catalog consistency guard (closes the S2 self-review hole): a model whose
attention carries kv_bytes_per_token_override but whose handler is not
OVERRIDE is inconsistent catalog data — refused, never silently ignored
(P-16 flavor). The OVERRIDE value is the catalog's own byte count and is
NOT rescaled by kv_dtype.

PURE FUNCTION IRON RULE (SPEC §1.3): no I/O of any kind in this module.
"""
from __future__ import annotations

from typing import Callable, Final

from llmpairing.schemas import ModelSpec
from llmpairing.types import KvDtype


class UnsupportedArchError(Exception):
    """Raised when a model's architecture cannot be handled.

    Callers (classify.py) map this to Verdict.UNSUPPORTED_ARCH.
    Guessing or falling back to FULL is forbidden (P-16, R-2).
    """


# bytes per KV cache element, expressed as EXACT integer sixteenths.
#
# Source: T-002 SPEC §4.3 table (llama.cpp KV quant formats, including
# block scale overhead — P-10):
#     f16  = 2.0    bytes/elem = 32/16
#     q8_0 = 1.0625 bytes/elem = 17/16   (32 weights + 2-byte scale per block)
#     q4_0 = 0.5625 bytes/elem =  9/16   (16 nibbles + 2-byte scale per block)
# Stored as x16 numerators so all arithmetic stays in int (R-4 / P-18).
# Status: FROZEN — these mirror llama.cpp storage formats, not tunables.
_KV_BYTES_PER_ELEM_X16: Final[dict[KvDtype, int]] = {
    "f16": 32,
    "q8_0": 17,
    "q4_0": 9,
}


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division (R-4: bytes are ints, rounded up)."""
    return -(-numerator // denominator)


def _layer_token_bytes(model: ModelSpec, kv_dtype: KvDtype) -> tuple[int, int]:
    """(numerator, 16) — exact bytes for one layer-token (K and V, x16)."""
    elements = 2 * model.n_kv_heads * model.head_dim  # P-01: kv heads, P-02: explicit head_dim
    return elements * _KV_BYTES_PER_ELEM_X16[kv_dtype], 16


def _kv_full(model: ModelSpec, kv_dtype: KvDtype, ctx: int) -> int:
    num, den = _layer_token_bytes(model, kv_dtype)
    return _ceil_div(model.n_layers * ctx * num, den)


def _kv_sliding_window(model: ModelSpec, kv_dtype: KvDtype, ctx: int) -> int:
    assert model.attention.window_size is not None  # schema guarantees
    per_layer_tokens = min(ctx, model.attention.window_size)
    num, den = _layer_token_bytes(model, kv_dtype)
    return _ceil_div(model.n_layers * per_layer_tokens * num, den)


def _kv_hybrid(model: ModelSpec, kv_dtype: KvDtype, ctx: int) -> int:
    """Hybrid sliding/full attention, since A3 possibly heterogeneous.

    A3 (gemma4): full layers may carry their own kv-head count / head_dim
    (global_kv_heads / global_head_dim, falling back to the model-level
    geometry), and the last kv_shared_trailing_layers layers store no KV —
    they reuse the last non-shared layer's tensors (HF gemma4 blog).
    Only global_layer_indices BELOW the non-shared prefix count as
    KV-bearing full layers. Defaults (None/0) reproduce the gemma3
    homogeneous formula exactly.
    """
    att = model.attention
    assert att.window_size is not None  # schema guarantees
    assert att.global_layer_indices  # schema guarantees non-empty
    prefix = model.n_layers - att.kv_shared_trailing_layers
    if prefix < 1:
        raise UnsupportedArchError(
            f"{model.model_id}: kv_shared_trailing_layers >= n_layers — "
            f"corrupt catalog"
        )
    n_global = sum(1 for i in att.global_layer_indices if i < prefix)
    n_local = prefix - n_global
    if n_local < 0:
        raise UnsupportedArchError(
            f"{model.model_id}: more global layers than layers — corrupt catalog"
        )
    g_heads = att.global_kv_heads if att.global_kv_heads is not None else model.n_kv_heads
    g_dim = att.global_head_dim if att.global_head_dim is not None else model.head_dim
    elem_x16 = _KV_BYTES_PER_ELEM_X16[kv_dtype]
    # D3: K and V counted separately (factor 2) even when attention_k_eq_v
    # — conservative: over-demand errs FR-ward, never FA.
    num_global = 2 * g_heads * g_dim * elem_x16
    num_local = 2 * model.n_kv_heads * model.head_dim * elem_x16
    total_x16 = (n_global * ctx * num_global
                 + n_local * min(ctx, att.window_size) * num_local)
    return _ceil_div(total_x16, 16)


def _kv_linear_interval(model: ModelSpec, kv_dtype: KvDtype, ctx: int) -> int:
    """LINEAR_INTERVAL (amendment A1, qwen3_5 family): affine KV.

    Full-attention layers (global_layer_indices) grow linearly with ctx;
    linear-attention layers hold a CONSTANT recurrent state whose size the
    catalog computes from config fields. kv_dtype scales only the growing
    part — llama.cpp KV quantization does not touch the float32 SSM state.
    """
    assert model.attention.global_layer_indices  # schema guarantees
    assert model.attention.linear_state_bytes_per_layer is not None  # schema
    n_full = len(model.attention.global_layer_indices)
    n_linear = model.n_layers - n_full
    if n_linear < 0:
        raise UnsupportedArchError(
            f"{model.model_id}: more full layers than layers — corrupt catalog"
        )
    num, den = _layer_token_bytes(model, kv_dtype)
    growing = _ceil_div(n_full * ctx * num, den)
    return growing + n_linear * model.attention.linear_state_bytes_per_layer


def _kv_override(model: ModelSpec, kv_dtype: KvDtype, ctx: int) -> int:
    override = model.attention.kv_bytes_per_token_override
    if override is None:
        raise UnsupportedArchError(
            f"{model.model_id}: OVERRIDE handler requires "
            f"kv_bytes_per_token_override (SPEC §4.3)"
        )
    # The override is the catalog's authoritative byte count (MLA/DSA style
    # compressed KV) — kv_dtype does not apply to it.
    return override * ctx

_HANDLERS: Final[dict[str, Callable[[ModelSpec, KvDtype, int], int]]] = {
    "FULL": _kv_full,
    "SLIDING_WINDOW": _kv_sliding_window,
    "HYBRID": _kv_hybrid,
    "OVERRIDE": _kv_override,
    "LINEAR_INTERVAL": _kv_linear_interval,
}

#: handlers whose KV grows linearly in ctx (closed-form ctx_max is valid)
_LINEAR_HANDLERS: Final[frozenset[str]] = frozenset({"FULL", "OVERRIDE"})


def _dispatch(model: ModelSpec) -> Callable[[ModelSpec, KvDtype, int], int]:
    if not model.arch_supported:
        raise UnsupportedArchError(
            f"arch '{model.arch}' is outside the supported whitelist "
            f"(arch_supported=False for {model.model_id})"
        )
    if (model.attention.kv_bytes_per_token_override is not None
            and model.arch_handler != "OVERRIDE"):
        raise UnsupportedArchError(
            f"{model.model_id}: kv override present but handler is "
            f"'{model.arch_handler}' — inconsistent catalog data is refused, "
            f"not silently ignored (R-2)"
        )
    try:
        return _HANDLERS[model.arch_handler]  # P-16: indexing, NOT .get with default
    except KeyError as exc:
        raise UnsupportedArchError(
            f"no KV handler for arch_handler '{model.arch_handler}' "
            f"({model.model_id}); refusing to guess (R-2)"
        ) from exc


def kv_bytes_total(model: ModelSpec, kv_dtype: KvDtype, ctx: int) -> int:
    """KV cache size in bytes at context length ctx, dispatched by handler."""
    return _dispatch(model)(model, kv_dtype, ctx)


def kv_bytes_per_token(model: ModelSpec, kv_dtype: KvDtype) -> int:
    """Linear per-token normalizer — ONLY for linear handlers (SPEC §4.3).

    Piecewise-linear handlers (SWA/HYBRID) have no single per-token figure;
    callers must use kv_bytes_total (the ctx_max solver bisects instead).
    """
    handler = _dispatch(model)
    if model.arch_handler not in _LINEAR_HANDLERS:
        raise UnsupportedArchError(
            f"{model.model_id}: kv_bytes_per_token is undefined for "
            f"piecewise-linear handler '{model.arch_handler}' — use kv_bytes_total"
        )
    return handler(model, kv_dtype, 1)

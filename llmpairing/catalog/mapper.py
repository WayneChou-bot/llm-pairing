"""config.json -> ModelSpec normalization (T-005).

Honesty rules:
- arch whitelist; anything else -> skipped with a reason (R-2 / P-16)
- head_dim: explicit config value ALWAYS wins; derivation hidden/heads is
  allowed ONLY for _DERIVE_HEAD_DIM_ARCHES (whose reference implementation
  defines exactly that), flagged HEAD_DIM_DERIVED_PER_ARCH_DEFAULT
  (owner decision 2026-08-14 refining P-02)
- MoE active params: derived by config arithmetic for qwen3_moe only
  (formula mirrors the G-07 hand derivation); other MoE archs are skipped
- every default filled in (tie_word_embeddings) carries a note
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llmpairing.schemas import ModelSpec, QuantVariant

#: architectures the T-002 core models exactly, with their KV handler
_ARCH_HANDLERS = {
    "llama": "FULL",
    "qwen2": "FULL",
    "qwen3": "FULL",
    "qwen3_moe": "FULL",
    "mistral": "FULL",  # SLIDING_WINDOW when sliding_window is set
    "gemma3": "HYBRID",
    "gemma3_text": "HYBRID",
    # amendment A3 (2026-08-18): heterogeneous hybrid + shared trailing KV
    "gemma4": "HYBRID",
    "gemma4_text": "HYBRID",
    # amendment A1 (2026-08-17): linear attention + interval full layers
    "qwen3_5": "LINEAR_INTERVAL",
    "qwen3_5_moe": "LINEAR_INTERVAL",
}

#: archs whose reference implementation defines head_dim = hidden / heads
_DERIVE_HEAD_DIM_ARCHES = {"llama", "qwen2", "mistral"}

_QUANT_RE = re.compile(
    r"(?:^|[-._])((?:I?Q\d(?:_[A-Z0-9]+)*)|MXFP4(?:_[A-Z0-9]+)*|F16|F32|BF16)"
    r"(?:[-._]|$)", re.IGNORECASE
)
_MULTIPART_RE = re.compile(r"-\d{5}-of-\d{5}", re.IGNORECASE)
#: auxiliary GGUF artifacts that are NOT main-model weights (review item 2:
#: a vision projector named mmproj-…-f16.gguf was summed into the F16 quant)
_AUX_RE = re.compile(
    r"(?:^|[-._])(mmproj|projector|adapter|lora|draft)(?:[-._]|$)", re.IGNORECASE
)


def _label_bpw_range(label: str) -> tuple[float, float] | None:
    """Plausible effective-bpw band PER QUANT LABEL (tighter than the
    global physical [1, 40] band). Bands are deliberately generous —
    small models carry proportionally heavy f16 embeddings (observed up
    to ~1.35x the nominal bit width) and _XL variants keep some tensors
    at higher precision. The gate exists to catch gross mislabeling
    (an F16-sized file labeled Q4), not to discriminate neighbors."""
    if label == "F32":
        return (24.0, 40.0)
    if label in ("F16", "BF16"):
        return (12.0, 24.0)
    if label.startswith("MXFP4"):
        return (2.5, 8.0)  # nominal ~4.25 bpw + mixed-precision headroom
    m = re.match(r"I?Q(\d)", label)
    if m:
        d = int(m.group(1))
        if d == 1:
            # field data 2026-08-18: dynamic sub-2-bit quants (unsloth UD)
            # keep salient tensors at 4-8 bit — observed 2.24-2.54 bpw on
            # Qwen3 MoE. The naive 1.8x cap false-dropped real quants.
            return (0.55, 3.2)
        return (d * 0.55, d * 1.8)
    return None


@dataclass
class GgufFile:
    filename: str
    size: int


@dataclass
class MapResult:
    spec: ModelSpec | None
    notes: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    #: review #3 batch 2: per-quant-label chosen-artifact provenance
    #: ({label: {stem, filenames, total_bytes}}) — what the user must
    #: actually download; recorded by build_quants, shipped in meta
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


def parse_quant_label(filename: str) -> str | None:
    if not filename.lower().endswith(".gguf"):
        return None
    stem = _MULTIPART_RE.sub("", filename[:-5])
    m = _QUANT_RE.search(stem)
    return m.group(1).upper() if m else None


def build_quants(files: list[GgufFile], notes: list[str],
                 n_params_total: int | None = None,
                 artifacts_out: dict[str, dict[str, Any]] | None = None,
                 ) -> list[QuantVariant]:
    """GGUF files -> quant variants, grouped by ARTIFACT (review item 2).

    Redone 2026-08-18. The old version grouped by quant label alone and
    SUMMED everything sharing a label — a repo with several distinct
    same-quant files produced a phantom 34.04 GiB "quant", and vision
    projectors (mmproj-…-f16.gguf) were added into F16.

    Rules, in order (every exclusion leaves a note — no silent drops):
    1. an artifact = files sharing a stem after stripping the multipart
       suffix (-NNNNN-of-NNNNN); parts sum WITHIN an artifact only (P-09)
    2. auxiliary artifacts (mmproj/projector/adapter/lora/draft) are
       excluded: not main-model weights
    3. artifacts whose stem yields no quant label are excluded, noted
    4. with n_params_total known, each artifact's implied bpw must fall
       in its label's plausible band (_label_bpw_range) — mislabeled or
       partial-weights artifacts are dropped (IMPLAUSIBLE_QUANT_DROPPED)
    5. if several distinct artifacts survive for one label, the LARGEST
       is kept (conservative: over-demand errs FR-ward, never FA), noted
    6. survivors carry bpw_effective derived from file size + params
    """
    # 1-3: group by artifact stem
    artifacts: dict[str, dict[str, Any]] = {}
    for f in files:
        if not f.filename.lower().endswith(".gguf"):
            continue
        stem = _MULTIPART_RE.sub("", f.filename[:-5])
        a = artifacts.setdefault(stem.lower(),
                                 {"stem": stem, "bytes": 0, "files": []})
        a["bytes"] += f.size
        a["files"].append(f.filename)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for a in artifacts.values():
        stem = str(a["stem"])
        if _AUX_RE.search(stem):
            notes.append(f"AUX_GGUF_EXCLUDED:{stem}")
            continue
        label = parse_quant_label(stem + ".gguf")
        if label is None:
            notes.append(f"UNLABELED_GGUF_EXCLUDED:{stem}")
            continue
        by_label.setdefault(label, []).append(a)

    out: list[QuantVariant] = []
    for label in sorted(by_label):
        cands = by_label[label]
        # 4: per-label bpw band (only when the param count is known)
        if n_params_total is not None:
            band = _label_bpw_range(label) or (1.0, 40.0)
            kept = []
            for a in cands:
                bpw = int(a["bytes"]) * 8 / n_params_total
                if band[0] <= bpw <= band[1]:
                    kept.append(a)
                else:
                    notes.append(
                        f"IMPLAUSIBLE_QUANT_DROPPED:{label} ({a['stem']}: "
                        f"{a['bytes']} bytes -> {bpw:.2f} bpw on "
                        f"{n_params_total} params, band {band[0]:.2f}-{band[1]:.2f})"
                    )
            cands = kept
        if not cands:
            continue
        # 5: never sum distinct artifacts — keep the largest (FR-ward)
        if len(cands) > 1:
            cands.sort(key=lambda a: int(a["bytes"]), reverse=True)
            notes.append(
                f"MULTIPLE_ARTIFACTS_FOR_QUANT:{label} kept {cands[0]['stem']} "
                f"({len(cands)} candidates; largest kept, conservative)"
            )
        chosen = cands[0]
        if artifacts_out is not None:
            artifacts_out[label] = {
                "stem": chosen["stem"],
                "filenames": sorted(chosen["files"]),
                "total_bytes": int(chosen["bytes"]),
            }
        bpw_eff = (round(int(chosen["bytes"]) * 8 / n_params_total, 3)
                   if n_params_total else None)
        out.append(QuantVariant(
            label=label, file_bytes=int(chosen["bytes"]),
            file_bytes_source="CATALOG_VERIFIED", bpw_effective=bpw_eff,
        ))
    return out


def _linear_state_bytes(cfg: dict[str, Any]) -> int:
    """Constant recurrent-state bytes per linear-attention layer (A1 §2).

    SSM state (v_heads x d_k x d_v) + conv state ((kernel-1) x mixed qkv
    width), in the dtype the config itself declares (float32 observed).
    """
    v_heads = int(cfg["linear_num_value_heads"])
    k_heads = int(cfg["linear_num_key_heads"])
    d_k = int(cfg["linear_key_head_dim"])
    d_v = int(cfg["linear_value_head_dim"])
    kernel = int(cfg["linear_conv_kernel_dim"])
    dtype = str(cfg.get("mamba_ssm_dtype", "float32"))
    bytes_per = 4 if "32" in dtype else 2
    ssm = v_heads * d_k * d_v
    conv = (kernel - 1) * (2 * k_heads * d_k + v_heads * d_v)
    return (ssm + conv) * bytes_per


def _qwen3_moe_params(cfg: dict[str, Any], notes: list[str]) -> tuple[int, int]:
    """Total/active by config arithmetic — mirrors the G-07 derivation
    (tests pin exact equality with those constants)."""
    hidden = int(cfg["hidden_size"])
    vocab = int(cfg["vocab_size"])
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    n_exp = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    moe_inter = int(cfg["moe_intermediate_size"])
    tie = bool(cfg.get("tie_word_embeddings", True))

    emb = vocab * hidden * (1 if tie else 2)
    attn = hidden * heads * head_dim + 2 * hidden * kv * head_dim + heads * head_dim * hidden
    router = hidden * n_exp
    expert = 3 * hidden * moe_inter
    total = emb + layers * (attn + router + n_exp * expert)
    active = emb + layers * (attn + router + top_k * expert)
    notes.append("MOE_PARAMS_DERIVED_FROM_CONFIG")
    return total, active


def map_config(model_id: str, raw_config: dict[str, Any], *,
               ggufs: list[GgufFile], n_params_total: int | None) -> MapResult:
    """Normalize one model. Never raises for bad upstream data — returns a
    skip_reason instead (the catalog records what it could not model)."""
    notes: list[str] = []

    # gemma3 nests the text model under text_config
    cfg = dict(raw_config)
    arch = str(cfg.get("model_type") or "")
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        inner = dict(cfg["text_config"])
        inner["model_type"] = arch or inner.get("model_type", "")
        cfg = {**inner, "model_type": arch}

    if arch not in _ARCH_HANDLERS:
        return MapResult(None, notes,
                         f"arch '{arch}' outside the whitelist (R-2: not modeled)")
    # MoE-looking configs outside the archs with a defined active-param
    # strategy: refuse (never mis-modeled)
    if arch not in ("qwen3_moe", "qwen3_5_moe", "gemma4", "gemma4_text") and any(
        cfg.get(k) for k in ("num_local_experts", "num_experts")
    ):
        return MapResult(None, notes,
                         f"MoE arch '{arch}' without a defined active-param "
                         f"derivation — skipped, not mis-modeled")

    required = ("num_hidden_layers", "num_key_value_heads",
                "num_attention_heads", "hidden_size", "vocab_size",
                "max_position_embeddings")
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        return MapResult(None, notes, f"config missing {missing}")

    heads = int(cfg["num_attention_heads"])
    hidden = int(cfg["hidden_size"])
    if cfg.get("head_dim") is not None:
        head_dim = int(cfg["head_dim"])
    elif arch in _DERIVE_HEAD_DIM_ARCHES and hidden % heads == 0:
        head_dim = hidden // heads
        notes.append("HEAD_DIM_DERIVED_PER_ARCH_DEFAULT")
    else:
        return MapResult(None, notes,
                         "head_dim absent and derivation not defined for "
                         f"arch '{arch}' (P-02)")

    if cfg.get("tie_word_embeddings") is None:
        tie = True  # transformers PretrainedConfig default
        notes.append("TIE_WORD_EMBEDDINGS_DEFAULTED")
    else:
        tie = bool(cfg["tie_word_embeddings"])

    # handler + attention geometry
    lookup_params = 0  # A5: per-token-lookup params (gemma4 PLE)
    handler = _ARCH_HANDLERS[arch]
    window = cfg.get("sliding_window")
    attention: dict[str, Any] = {
        "kind": "FULL", "window_size": None,
        "global_layer_indices": None, "kv_bytes_per_token_override": None,
    }
    if arch == "mistral":
        if window:
            handler = "SLIDING_WINDOW"
            attention = {**attention, "kind": "SLIDING_WINDOW",
                         "window_size": int(window)}
    elif handler == "LINEAR_INTERVAL":
        layer_types = cfg.get("layer_types")
        if not isinstance(layer_types, list) or not layer_types:
            return MapResult(None, notes,
                             "qwen3_5-style arch without layer_types — "
                             "cannot locate the full-attention layers")
        full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        if not full_idx:
            return MapResult(None, notes, "layer_types contains no full_attention")
        try:
            state = _linear_state_bytes(cfg)
        except KeyError as exc:
            return MapResult(None, notes, f"linear-attention config missing {exc}")
        attention = {"kind": "LINEAR_INTERVAL", "window_size": None,
                     "global_layer_indices": full_idx,
                     "kv_bytes_per_token_override": None,
                     "linear_state_bytes_per_layer": state}
        notes.append("LINEAR_STATE_FROM_CONFIG")
        if cfg.get("mtp_num_hidden_layers"):
            notes.append("MTP_NOT_MODELED")
    elif handler == "HYBRID":
        layer_types = cfg.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            # gemma4-style: explicit per-layer type list (A3)
            if not window:
                return MapResult(None, notes,
                                 "gemma4-style arch without sliding_window")
            full_idx = [i for i, t in enumerate(layer_types)
                        if t == "full_attention"]
            if not full_idx:
                return MapResult(None, notes,
                                 "layer_types contains no full_attention")
            g_kv = cfg.get("num_global_key_value_heads")
            g_hd = cfg.get("global_head_dim")
            shared = int(cfg.get("num_kv_shared_layers") or 0)
            attention = {"kind": "HYBRID", "window_size": int(window),
                         "global_layer_indices": full_idx,
                         "kv_bytes_per_token_override": None,
                         "global_kv_heads": int(g_kv) if g_kv else None,
                         "global_head_dim": int(g_hd) if g_hd else None,
                         "kv_shared_trailing_layers": shared}
            if shared:
                notes.append("KV_SHARED_TRAILING_LAYERS_FROM_CONFIG")
            if cfg.get("attention_k_eq_v"):
                # A3 D3: counted 2x anyway (conservative); real use may be
                # up to half — noted, never assumed
                notes.append("GEMMA4_K_EQ_V_CONSERVATIVE_2X")
            # A5: PLE tables are per-token lookups -> zero decode traffic
            ple_hidden = int(cfg.get("hidden_size_per_layer_input") or 0)
            ple_vocab = int(cfg.get("vocab_size_per_layer_input") or 0)
            if ple_hidden and ple_vocab:
                lookup_params = (ple_vocab * ple_hidden
                                 * int(cfg["num_hidden_layers"]))
                notes.append("PLE_LOOKUP_PARAMS_FROM_CONFIG")
            elif ple_hidden:
                notes.append("GEMMA4_PLE_NOT_MODELED")  # vocab unknown
        else:
            pattern = cfg.get("sliding_window_pattern")
            if not window or not pattern:
                return MapResult(None, notes,
                                 "gemma3-style arch without sliding_window/"
                                 "pattern — cannot build HYBRID geometry")
            layers = int(cfg["num_hidden_layers"])
            indices = [i for i in range(layers) if (i + 1) % int(pattern) == 0]
            attention = {"kind": "HYBRID", "window_size": int(window),
                         "global_layer_indices": indices,
                         "kv_bytes_per_token_override": None}

    # params
    active: int
    if arch == "qwen3_moe":
        try:
            total, active = _qwen3_moe_params(cfg, notes)
        except KeyError as exc:
            return MapResult(None, notes, f"qwen3_moe config missing {exc}")
        if n_params_total is not None and abs(n_params_total - total) > total * 0.02:
            notes.append("PARAMS_DERIVED_DIVERGES_FROM_UPSTREAM_GT2PCT")
    elif arch == "qwen3_5_moe":
        # A1 decision D3: active by subtraction of inactive routed experts;
        # the always-on shared expert stays inside the total.
        if n_params_total is None:
            return MapResult(None, notes,
                             "qwen3_5_moe without an upstream param count — "
                             "subtraction strategy needs the total (D3)")
        try:
            inactive = ((int(cfg["num_experts"]) - int(cfg["num_experts_per_tok"]))
                        * int(cfg["num_hidden_layers"]) * 3
                        * int(cfg["hidden_size"]) * int(cfg["moe_intermediate_size"]))
        except KeyError as exc:
            return MapResult(None, notes, f"qwen3_5_moe config missing {exc}")
        total = n_params_total
        active = total - inactive
        if active <= 0:
            return MapResult(None, notes,
                             "subtraction produced non-positive active params "
                             "— upstream total implausible, skipped")
        notes.append("MOE_ACTIVE_BY_SUBTRACTION")
    elif arch in ("gemma4", "gemma4_text") and cfg.get("enable_moe_block"):
        # A3 D4: active by subtraction of inactive routed experts (same
        # strategy as qwen3_5_moe / A1 D3). The upstream total includes the
        # vision tower, which subtraction cannot remove — active is
        # slightly OVER-estimated (FR-ward), noted via the standard flag.
        if n_params_total is None:
            return MapResult(None, notes,
                             "gemma4 MoE without an upstream param count — "
                             "subtraction strategy needs the total (A3 D4)")
        try:
            inactive = ((int(cfg["num_experts"]) - int(cfg["top_k_experts"]))
                        * int(cfg["num_hidden_layers"]) * 3
                        * int(cfg["hidden_size"]) * int(cfg["moe_intermediate_size"]))
        except (KeyError, TypeError) as exc:
            return MapResult(None, notes, f"gemma4 MoE config missing {exc}")
        total = n_params_total
        active = total - inactive
        if active <= 0:
            return MapResult(None, notes,
                             "subtraction produced non-positive active params "
                             "— upstream total implausible, skipped")
        notes.append("MOE_ACTIVE_BY_SUBTRACTION")
    else:
        if n_params_total is None:
            return MapResult(None, notes,
                             "dense model without an upstream param count "
                             "(safetensors total) — skipped, not derived")
        total = active = n_params_total

    # The bpw plausibility gate (field lesson 2026-08-17, V-P5: an
    # MTP-head-only repo published a 0.63 GB "Q8_0" for a 27.8B model)
    # now lives inside build_quants as per-label bands.
    artifacts: dict[str, dict[str, Any]] = {}
    quants = build_quants(ggufs, notes, n_params_total=total,
                          artifacts_out=artifacts)
    if not quants:
        if any(n.startswith("IMPLAUSIBLE_QUANT_DROPPED") for n in notes):
            return MapResult(None, notes,
                             "every quant failed the bits-per-weight "
                             "plausibility gate (partial/auxiliary-weights "
                             "repo?)")
        return MapResult(None, notes, "no GGUF quant files found")

    spec = ModelSpec(
        schema_version="1.4",
        model_id=model_id,
        arch=arch,
        arch_supported=True,
        arch_handler=handler,
        n_params_total=total,
        n_params_active=active,
        n_layers=int(cfg["num_hidden_layers"]),
        n_kv_heads=int(cfg["num_key_value_heads"]),
        n_attn_heads=heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(cfg["vocab_size"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        tie_word_embeddings=tie,
        n_params_per_token_lookup=min(lookup_params, active),
        attention=attention,  # type: ignore[arg-type]
        quants=quants,
    )
    return MapResult(spec, notes, None, artifacts=artifacts)

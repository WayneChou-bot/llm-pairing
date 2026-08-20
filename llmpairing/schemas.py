"""Frozen input contracts for LLM pairing (T-002 SPEC §2).

S1 scope: schema + structural validation only — zero computation code.
Once merged these schemas are FROZEN; any change requires a schema_version
bump plus a migration (SPEC §9).

Validation philosophy (R-2): reject, never coerce, never derive.
- missing head_dim is rejected, not derived from hidden/n_heads (P-02)
- a darwin profile without the apple block is rejected (SPEC §2.1)
- CATALOG_VERIFIED without file_bytes is a contradiction and is rejected (P-09)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmpairing.types import KvDtype, Source, Tier

_Bytes = Field(ge=0)  # bytes are non-negative ints, never float (R-4 / P-18)


class CpuInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)


class SystemMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_bytes: int = _Bytes
    available_bytes: int = _Bytes  # snapshot at probe time — see P-04
    bandwidth_bytes_per_s: int | None = Field(default=None, ge=1)
    bandwidth_source: Source

    @model_validator(mode="after")
    def _bandwidth_consistency(self) -> "SystemMemory":
        if self.bandwidth_source is Source.UNKNOWN and self.bandwidth_bytes_per_s is not None:
            raise ValueError("bandwidth_source UNKNOWN must not carry a bandwidth value")
        if self.bandwidth_source is not Source.UNKNOWN and self.bandwidth_bytes_per_s is None:
            raise ValueError("a known bandwidth_source requires a bandwidth value")
        return self


class Accelerator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    vendor: Literal["nvidia", "amd", "apple", "intel", "none"]
    name: str  # raw driver string, deliberately NOT normalized (SPEC §2.1)
    is_laptop_variant: bool  # P-15
    topology: Literal["DISCRETE", "UNIFIED", "INTEGRATED_SHARED"]
    drives_display: bool
    vram_total_bytes: int | None = Field(default=None, ge=0)  # meaningless for INTEGRATED_SHARED (P-06)
    vram_free_bytes: int | None = Field(default=None, ge=0)
    bandwidth_bytes_per_s: int | None = Field(default=None, ge=1)
    bandwidth_source: Source
    peak_flops_fp16: int | None = Field(default=None, ge=1)
    flops_source: Source

    @model_validator(mode="after")
    def _source_consistency(self) -> "Accelerator":
        if self.bandwidth_source is Source.UNKNOWN and self.bandwidth_bytes_per_s is not None:
            raise ValueError("bandwidth_source UNKNOWN must not carry a bandwidth value")
        if self.bandwidth_source is not Source.UNKNOWN and self.bandwidth_bytes_per_s is None:
            raise ValueError("a known bandwidth_source requires a bandwidth value")
        if self.flops_source is Source.UNKNOWN and self.peak_flops_fp16 is not None:
            raise ValueError("flops_source UNKNOWN must not carry a flops value")
        if self.flops_source is not Source.UNKNOWN and self.peak_flops_fp16 is None:
            raise ValueError("a known flops_source requires a flops value")
        return self


class AppleMemoryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unified_memory_bytes: int = Field(ge=1)
    # 0 and None BOTH mean "system default" — see P-05. The distinction is
    # preserved here verbatim; mapping to the 0.75 default happens in budget/ (S3).
    iogpu_wired_limit_mb: int | None = Field(default=None, ge=0)
    recommended_max_working_set_bytes: int | None = Field(default=None, ge=1)


class HardwareProfile(BaseModel):
    """Output contract of T-001 probe (SPEC §2.1), consumed read-only here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 1.1 (S4): adds probe_notes; 1.2 (review #4): adds machine_id
    schema_version: Literal["1.0", "1.1", "1.2"]
    probe_tier: Tier
    platform: Literal["windows", "darwin", "linux"]
    measured_at_unix: int = Field(ge=0)  # display only — never used in computation

    cpu: CpuInfo
    system_memory: SystemMemory
    accelerators: list[Accelerator]
    apple: AppleMemoryInfo | None = None
    #: v1.1 (T-001 S4): probe-level honesty notes (e.g. the nvidia-smi
    #: parser's UNVERIFIED_ON_REAL_HW marker) — classify surfaces these
    #: into every verdict's flags. Additive; default keeps 1.0 semantics.
    probe_notes: list[str] = Field(default_factory=list)
    #: v1.2: one-way hardware fingerprint (probe/fingerprint.py) — lets a
    #: MachineCalibration prove it was measured on THIS machine. None on
    #: old profiles: identity unverifiable -> calibration refuses to apply.
    machine_id: str | None = None

    @model_validator(mode="after")
    def _apple_required_on_darwin(self) -> "HardwareProfile":
        if self.platform == "darwin" and self.apple is None:
            raise ValueError("platform darwin requires the apple memory block (SPEC §2.1)")
        if self.platform != "darwin" and self.apple is not None:
            raise ValueError("apple memory block is only valid on darwin")
        return self


class AttentionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FULL", "SLIDING_WINDOW", "HYBRID", "LINEAR_INTERVAL"]
    window_size: int | None = Field(default=None, ge=1)
    global_layer_indices: list[int] | None = None
    kv_bytes_per_token_override: int | None = Field(default=None, ge=1)  # MLA/DSA escape hatch
    # v1.2 (A1): constant recurrent-state bytes per linear-attention layer
    # (computed by the catalog from config fields; LINEAR_INTERVAL only)
    linear_state_bytes_per_layer: int | None = Field(default=None, ge=1)
    # v1.3 (A3, gemma4): heterogeneous HYBRID — full-attention layers may
    # use their own kv-head count / head_dim (D1: explicit config values;
    # None falls back to the model-level geometry), and the last
    # kv_shared_trailing_layers layers store NO KV (D2: they reuse the
    # last non-shared layer's tensors — HF gemma4 blog semantics).
    global_kv_heads: int | None = Field(default=None, ge=1)
    global_head_dim: int | None = Field(default=None, ge=1)
    kv_shared_trailing_layers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _kind_requirements(self) -> "AttentionSpec":
        if self.kind in ("SLIDING_WINDOW", "HYBRID") and self.window_size is None:
            raise ValueError(f"attention kind {self.kind} requires window_size")
        if self.kind in ("HYBRID", "LINEAR_INTERVAL") and not self.global_layer_indices:
            raise ValueError(f"attention kind {self.kind} requires non-empty global_layer_indices")
        if self.kind == "FULL" and self.window_size is not None:
            raise ValueError("attention kind FULL must not carry window_size")
        if self.kind == "LINEAR_INTERVAL":
            if self.window_size is not None:
                raise ValueError("LINEAR_INTERVAL must not carry window_size")
            if self.linear_state_bytes_per_layer is None:
                raise ValueError("LINEAR_INTERVAL requires linear_state_bytes_per_layer")
        elif self.linear_state_bytes_per_layer is not None:
            raise ValueError("linear_state_bytes_per_layer is LINEAR_INTERVAL-only")
        if self.kind != "HYBRID" and (
            self.global_kv_heads is not None
            or self.global_head_dim is not None
            or self.kv_shared_trailing_layers != 0
        ):
            raise ValueError(
                "global_kv_heads / global_head_dim / kv_shared_trailing_layers "
                "are HYBRID-only (schema 1.3, amendment A3)"
            )
        return self


class QuantVariant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    # file_bytes is the AUTHORITATIVE weight size when present (P-09).
    # bpw_effective exists ONLY as fallback when file_bytes is absent.
    file_bytes: int | None = Field(default=None, ge=1)
    file_bytes_source: Literal["CATALOG_VERIFIED", "ESTIMATED"]
    bpw_effective: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _weight_info_present(self) -> "QuantVariant":
        if self.file_bytes_source == "CATALOG_VERIFIED" and self.file_bytes is None:
            raise ValueError("CATALOG_VERIFIED requires an actual file_bytes value (P-09)")
        if self.file_bytes is None and self.bpw_effective is None:
            raise ValueError("a quant needs file_bytes or bpw_effective; neither given (P-09)")
        return self


class ModelSpec(BaseModel):
    """Output contract of T-005 catalog (SPEC §2.2), consumed read-only here.

    v1.1 (owner decision 2026-08-14): adds max_position_embeddings, required
    by the S4 ctx_max solver (SPEC §4.5) and missing from v1.0.
    Migration 1.0 -> 1.1: supply max_position_embeddings from config.json's
    max_position_embeddings field; no other change.

    v1.2 (amendment A1, 2026-08-17): AttentionSpec gains kind
    LINEAR_INTERVAL + linear_state_bytes_per_layer for the qwen3_5 family
    (linear attention with interval full layers). Additive — 1.1 documents
    remain valid.

    v1.3 (amendment A3, 2026-08-18): AttentionSpec gains HYBRID-only
    global_kv_heads / global_head_dim / kv_shared_trailing_layers for the
    gemma4 family (heterogeneous hybrid attention + shared trailing KV
    layers). Additive — defaults reproduce homogeneous HYBRID exactly.

    v1.4 (amendment A5, 2026-08-19): n_params_per_token_lookup — params
    accessed as per-token table lookups (gemma4 PLE), excluded from
    decode streaming traffic. Memory demand is unaffected. Additive —
    default 0 reproduces prior behavior.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.1", "1.2", "1.3", "1.4"]
    model_id: str
    arch: str  # raw model_type from config.json
    arch_supported: bool  # False → downstream MUST short-circuit to UNSUPPORTED_ARCH
    arch_handler: str  # dispatch key, SPEC §4.2

    n_params_total: int = Field(ge=1)
    n_params_active: int = Field(ge=1)  # == total for dense; < total for MoE (P-07)

    n_layers: int = Field(ge=1)
    n_kv_heads: int = Field(ge=1)  # GQA KV heads, NOT attention heads (P-01)
    n_attn_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)  # explicit config.json value ONLY — never derived (P-02)
    hidden_size: int = Field(ge=1)
    vocab_size: int = Field(ge=1)
    max_position_embeddings: int = Field(ge=1)  # v1.1: ctx_max upper bound (SPEC §4.5)
    tie_word_embeddings: bool

    # v1.4 (A5): per-token-lookup params (PLE tables) — zero decode traffic
    n_params_per_token_lookup: int = Field(default=0, ge=0)

    attention: AttentionSpec
    quants: list[QuantVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def _params_consistency(self) -> "ModelSpec":
        if self.n_params_active > self.n_params_total:
            raise ValueError("n_params_active cannot exceed n_params_total")
        if self.n_params_per_token_lookup > self.n_params_active:
            raise ValueError(
                "n_params_per_token_lookup cannot exceed n_params_active (A5)")
        if self.n_kv_heads > self.n_attn_heads:
            raise ValueError("n_kv_heads cannot exceed n_attn_heads (GQA groups attention heads)")
        return self


class Workload(BaseModel):
    """User intent (SPEC §2.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ctx_target_tokens: int = Field(ge=1)
    prompt_tokens: int = Field(default=512, ge=1)
    kv_dtype: KvDtype = "f16"
    engine: Literal["llama.cpp", "mlx", "vllm"] = "llama.cpp"  # v1 implements llama.cpp only
    ubatch: int = Field(default=512, ge=1)
    safety_margin_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)  # policy knob — SPEC §6.3

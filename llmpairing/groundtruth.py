"""Ground-truth scoring (VERIFICATION §4 / review item 6) — PURE.

Turns one harness cell (what actually happened on the owner's machine)
plus the T-002 pipeline's opinion (what we would have told the user) into
a confusion-matrix score:

    TA — we said runnable, it ran          (true accept)
    FA — we said runnable, it failed       (THE bad one; budget <= 2%)
    FR — we said not runnable, it ran      (honest cost; budget <= 15%)
    TR — we said not runnable, it failed   (true reject)

and, when the verdict is runnable and a speed prediction exists, a
measured-vs-predicted tok/s comparison (ratio + inside-CI check).

Catalog resolution is fingerprint-based, never name-based: an ollama tag
("gemma4:e4b") is matched to a catalog entry by (architecture, upstream
parameter count within 2%, quant label) — all three from the GGUF's own
metadata. No match -> honestly unscored, never guessed (R-2).

PURE FUNCTION IRON RULE (SPEC §1.3): no I/O in this module.
"""
from __future__ import annotations

from typing import Any

from llmpairing.budget.classify import classify_fit
from llmpairing.predict.calibration import MachineCalibration
from llmpairing.predict.decode import predict_decode
from llmpairing.schemas import HardwareProfile, ModelSpec, QuantVariant, Workload
from llmpairing.types import Verdict

#: verdicts that tell the user "this will run (possibly slowly)"
RUNNABLE_VERDICTS = frozenset({
    Verdict.FITS, Verdict.TIGHT, Verdict.RAM_ONLY, Verdict.PARTIAL_OFFLOAD,
})

#: llama.cpp GGUF arch names -> HF model_type (the catalog's key space).
#: Field catch 2026-08-19: GGUF metadata said "qwen35" while the catalog
#: (and mapper whitelist) key on config.json's "qwen3_5" — the owner's
#: qwen3.5:4b harness cells were mis-scored UNSUPPORTED/FR. Aliases are
#: added ON EVIDENCE ONLY (R-2): qwen35 from the owner's harness data;
#: qwen3moe from llama.cpp's LLM_ARCH name table. Unknown names pass
#: through unchanged and simply fail the whitelist (honest refusal).
GGUF_ARCH_ALIASES: dict[str, str] = {
    "qwen35": "qwen3_5",
    "qwen35moe": "qwen3_5_moe",
    "qwen3moe": "qwen3_moe",
}


def normalize_gguf_arch(arch: str | None) -> str | None:
    """GGUF general.architecture -> catalog arch key (identity if unknown)."""
    if arch is None:
        return None
    return GGUF_ARCH_ALIASES.get(arch, arch)


def resolve_entry(
    arch: str | None,
    param_count: int | None,
    quant_label: str | None,
    specs: list[ModelSpec],
) -> tuple[ModelSpec, QuantVariant] | None:
    """GGUF-metadata fingerprint -> catalog (spec, quant), or None."""
    if not arch or not param_count or not quant_label:
        return None
    arch = normalize_gguf_arch(arch)
    for spec in specs:
        if spec.arch != arch:
            continue
        if abs(spec.n_params_total - param_count) > spec.n_params_total * 0.02:
            continue
        for q in spec.quants:
            if q.label.upper() == quant_label.upper():
                return spec, q
    return None


def score_cell(
    hw: HardwareProfile,
    model: ModelSpec,
    quant: QuantVariant,
    ctx: int,
    outcome: str | None,
    decode_tps_median: float | None,
    calibration: MachineCalibration | None = None,
) -> dict[str, Any]:
    """Score one harness cell against the pipeline. Pure."""
    wl = Workload(ctx_target_tokens=ctx)
    fit = classify_fit(hw, model, quant, wl)
    predicted_runnable = fit.verdict in RUNNABLE_VERDICTS
    ran = str(outcome or "").startswith("SUCCESS")
    score = ("TA" if predicted_runnable and ran
             else "FA" if predicted_runnable
             else "FR" if ran
             else "TR")
    out: dict[str, Any] = {
        "predicted_verdict": fit.verdict.value,
        "predicted_runnable": predicted_runnable,
        "actual_ran": ran,
        "score": score,
        "fit_tier": fit.headroom_bytes.tier.value,
        "fit_flags": fit.flags,
    }
    if not predicted_runnable:
        return out
    try:
        t = predict_decode(hw, model, quant, wl, fit, calibration=calibration)
    except ValueError:
        out["tps_note"] = "V-P5 sentinel refusal (inconsistent input data)"
        return out
    tv = t.decode_tps_at_target_ctx
    if tv.value is None:
        out["tps_note"] = ("no bandwidth data and no calibration — "
                           "speed honestly not scored")
        return out
    out["tps_predicted"] = round(tv.value, 2)
    out["tps_ci"] = [round(tv.ci_low or 0.0, 2), round(tv.ci_high or 0.0, 2)]
    out["tps_tier"] = tv.tier.value
    if decode_tps_median is not None:
        out["tps_measured"] = round(decode_tps_median, 3)
        out["tps_ratio_measured_over_predicted"] = round(
            decode_tps_median / tv.value, 3
        )
        out["tps_within_ci"] = bool(
            tv.ci_low is not None and tv.ci_high is not None
            and tv.ci_low <= decode_tps_median <= tv.ci_high
        )
    return out

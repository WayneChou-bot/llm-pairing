"""Analyze ground-truth JSONL files and score them against the predictor.

Usage:
    python tools/groundtruth/analyze.py data/groundtruth/<machine>/<run>.jsonl

Produces a per-cell summary (median + IQR — never mean, VERIFICATION §4.2)
and appends a scoring entry to docs/validation_log.md.

Honesty rules applied here:
- prefill numbers from cache-hit runs (prompt_eval faster than 1,000 tok/s
  on non-datacenter hardware) are marked CACHE_ARTIFACT and excluded; the
  warmup run carries the only real prefill for harness runs collected
  before the prompt-variation fix.
- when the model's architecture is outside the whitelist, the predicted
  verdict is UNSUPPORTED_ARCH and the comparison is scored as a
  FALSE-REJECT (FR) if the model actually ran. Never silently mapped to
  the nearest supported architecture (R-2).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

#: single source of truth — the catalog mapper's handler registry
#: (reviewer item 6: a second hardcoded whitelist here had already gone
#: stale within three days of the qwen3_5 addition)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from llmpairing.catalog.mapper import _ARCH_HANDLERS  # noqa: E402
from llmpairing.groundtruth import (  # noqa: E402
    normalize_gguf_arch,
    resolve_entry,
    score_cell,
)
from llmpairing.predict.calibration import MachineCalibration  # noqa: E402
from llmpairing.schemas import HardwareProfile, ModelSpec  # noqa: E402

SUPPORTED_ARCHES = frozenset(_ARCH_HANDLERS)

PREFILL_CACHE_THRESHOLD_TPS = 1_000.0

REPO = _Path(__file__).resolve().parents[2]


def _load_catalog_specs() -> tuple[str, list[ModelSpec]]:
    """Newest snapshot, sidecar-verified (same refusal rule as the demo)."""
    import hashlib
    snaps = sorted((REPO / "catalog").glob("catalog-*.json"))
    if not snaps:
        return "none", []
    raw = snaps[-1].read_bytes()
    sidecar = snaps[-1].with_suffix(".sha256")
    if sidecar.exists():
        want = sidecar.read_text(encoding="utf-8").strip()
        got = hashlib.sha256(raw).hexdigest()
        if want != got:
            raise SystemExit(
                f"catalog integrity FAILURE: {snaps[-1].name} sha256 "
                f"{got[:12]}… != sidecar {want[:12]}… — refusing to score "
                f"against corrupt data")
    data = json.loads(raw.decode("utf-8"))
    return snaps[-1].name, [
        ModelSpec.model_validate(e["spec"]) for e in data.get("entries", [])
    ]


def _load_profile() -> HardwareProfile | None:
    p = REPO / "my_profile.json"
    if not p.exists():
        return None
    return HardwareProfile.model_validate(
        json.loads(p.read_text(encoding="utf-8")))


def _load_calibration(hw: HardwareProfile | None) -> MachineCalibration | None:
    from llmpairing.predict.calibration import pick_calibration
    cals = [MachineCalibration.model_validate_json(
                f.read_text(encoding="utf-8"))
            for f in sorted((REPO / "calibration").glob("*.json"))]
    if hw is None:
        return cals[-1] if cals else None
    return pick_calibration(hw, cals)


def _median_iqr(values: list[float]) -> tuple[float, float]:
    med = statistics.median(values)
    if len(values) < 2:
        return med, 0.0
    qs = statistics.quantiles(values, n=4)
    return med, qs[2] - qs[0]


def analyze(path: Path, hw: HardwareProfile | None = None,
            specs: list[ModelSpec] | None = None,
            calibration: MachineCalibration | None = None) -> dict[str, Any]:
    machine: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] == "machine":
            machine = rec
        else:
            cells.append(rec)

    out: dict[str, Any] = {"machine": machine, "cells": []}
    for cell in cells:
        measures = [r for r in cell.get("runs", []) if r.get("kind") == "measure"]
        decode = [r["decode_tps"] for r in measures if r.get("decode_tps")]
        prefill_real = [
            r["prefill_tps"] for r in cell.get("runs", [])
            if r.get("prefill_tps")
            and r["prefill_tps"] < PREFILL_CACHE_THRESHOLD_TPS
        ]
        cache_hits = [
            r["prefill_tps"] for r in measures
            if r.get("prefill_tps")
            and r["prefill_tps"] >= PREFILL_CACHE_THRESHOLD_TPS
        ]
        arch_raw = (cell.get("model_info") or {}).get("general.architecture")
        arch = normalize_gguf_arch(arch_raw)  # GGUF name -> catalog key
        summary: dict[str, Any] = {
            "model": cell["model"],
            "num_ctx": cell["num_ctx"],
            "outcome": cell.get("outcome"),
            "arch": arch_raw,
            "arch_supported": arch in SUPPORTED_ARCHES,
            "resident_bytes": next(
                (p.get("size") for p in cell.get("ps_after", [])
                 if p.get("name") == cell["model"]), None),
            "vram_fraction": cell.get("vram_fraction"),
        }
        if decode:
            med, iqr = _median_iqr(decode)
            summary["decode_tps_median"] = round(med, 3)
            summary["decode_tps_iqr"] = round(iqr, 3)
        if prefill_real:
            med, iqr = _median_iqr(prefill_real)
            summary["prefill_tps_median"] = round(med, 3)
            summary["prefill_tps_n"] = len(prefill_real)
        if cache_hits:
            summary["prefill_cache_artifacts_excluded"] = len(cache_hits)
        # implied effective-bandwidth PRODUCT (BW_eff x eta x active-ratio):
        # bytes-read-per-token is unknown for unsupported arches, so only
        # the resident-size upper-bound product is reported, clearly labeled
        if decode and summary["resident_bytes"]:
            med, _ = _median_iqr(decode)
            summary["implied_bw_product_upper_gb_s"] = round(
                med * summary["resident_bytes"] / 1e9, 1
            )
            summary["implied_bw_note"] = (
                "median_tps x resident_bytes; true value scales by the "
                "arch's actual bytes-read-per-token (UNVERIFIED for "
                "unsupported arches)"
            )
        # scoring
        ran = str(summary["outcome"] or "").startswith("SUCCESS")
        if not summary["arch_supported"]:
            summary["predicted_verdict"] = "UNSUPPORTED_ARCH"
            summary["score"] = "FR" if ran else "TR"  # review #3: TR, not TN
            summary["score_note"] = (
                "arch outside whitelist -> honest refusal; counted as a "
                "false-reject in the FA/FR ledger (FR budget <= 15%), "
                "never an FA"
            )
        elif hw is not None and specs:
            # full-pipeline scoring (review item 6): fingerprint-resolve
            # the catalog entry from the GGUF's own metadata, then run
            # classify + predict exactly as the user-facing path does
            info = cell.get("model_info") or {}
            details = cell.get("model_details") or {}
            hit = resolve_entry(
                arch,
                info.get("general.parameter_count"),
                details.get("quantization_level"),
                specs,
            )
            if hit is None:
                summary["score_note"] = (
                    "arch supported but no catalog entry matches the GGUF "
                    "fingerprint — honestly unscored (rebuild the catalog?)")
            elif (cell.get("vram_fraction") or 0.0) > 0.0:
                summary["score_note"] = (
                    "GPU/mixed placement not scored yet (CPU-scenario "
                    "scoring only in this version)")
            else:
                spec, quant = hit
                cpu_hw = hw.model_copy(update={"accelerators": []})
                summary.update(score_cell(
                    cpu_hw, spec, quant, int(cell["num_ctx"]),
                    summary["outcome"],
                    summary.get("decode_tps_median"),
                    calibration=calibration,
                ))
                summary["catalog_model_id"] = spec.model_id
                summary["score_note"] = (
                    "availability is the probe-time snapshot, not "
                    "harness-time (P-04) — near-zero margins are soft")
        else:
            summary["score_note"] = ("no profile/catalog available — "
                                     "verdict not scored")
        out["cells"].append(summary)
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    snap_name, specs = _load_catalog_specs()
    hw = _load_profile()
    calibration = _load_calibration(hw)
    result = analyze(path, hw=hw, specs=specs, calibration=calibration)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    repo = Path(__file__).resolve().parents[2]
    log = repo / "docs" / "validation_log.md"
    m = result["machine"]
    lines = ["", f"## scored rerun: {path.stem} "
                 f"(catalog {snap_name}; "
                 f"calibration {'yes' if calibration else 'no'}; "
                 f"profile {'yes' if hw else 'no'})", ""]
    lines.append("| model | ctx | actual | predicted | score "
                 "| tok/s measured | tok/s predicted (ci) | ratio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in result["cells"]:
        ci = c.get("tps_ci")
        pred = (f"{c['tps_predicted']} ({ci[0]}–{ci[1]})"
                if c.get("tps_predicted") is not None and ci else "—")
        lines.append(
            f"| {c['model']} | {c['num_ctx']} | {c['outcome']} "
            f"| {c.get('predicted_verdict', 'n/a')} | {c.get('score', 'n/a')} "
            f"| {c.get('decode_tps_median', '—')} | {pred} "
            f"| {c.get('tps_ratio_measured_over_predicted', '—')} |"
        )
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended {len(result['cells'])} scored rows to {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

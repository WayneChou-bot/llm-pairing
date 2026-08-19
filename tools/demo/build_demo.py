"""Build the interactive demo HTML (v3): real machine x real catalog.

v3: --profile loads a probed HardwareProfile (from `llmpairing-probe
--json`); the newest catalog/catalog-*.json supplies real models. iGPU
machines get the T1-P08 dual view; two synthetic reference machines with
known bandwidth illustrate the tok/s model (the probed machine shows
honest "no bandwidth data" until spec_db/S7 lands). Falls back to the
synthetic fixtures when no profile/catalog exists.

Usage:  python tools/demo/build_demo.py [--profile my_profile.json]
Output: demo/llm-pairing-demo.html  (self-contained, opens via file://)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from llmpairing.budget.available import compute_budget, safety_margin_bytes
from llmpairing.budget.classify import classify_fit
from llmpairing.predict.calibration import MachineCalibration
from llmpairing.predict.decode import decode_bytes_per_token, predict_decode
from llmpairing.recommend import RecCandidate, recommend
from llmpairing.schemas import HardwareProfile, ModelSpec, Workload

REPO = Path(__file__).resolve().parents[2]
#: reference machines ship WITH the demo tool (the public repo carries no
#: tests/); these two files are copies of the canonical test fixtures
REF_HW = Path(__file__).parent / "reference_hw"
FIXTURES = REPO / "tests" / "fixtures"  # synthetic-model fallback (dev repo only)

CTX_OPTIONS = [2048, 8192, 32768]
KV_OPTIONS = ["f16", "q8_0", "q4_0"]
PREVIEW_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")

REFERENCE_HW = [
    ("hw_apple_unified_64gb", "參考機：Apple 統一記憶體 64GB"),
    ("hw_discrete_nvidia_8gb", "參考機：桌機獨顯 8GB"),
]


def _hw_fixture(name: str) -> HardwareProfile:
    return HardwareProfile.model_validate(
        json.loads((REF_HW / f"{name}.json").read_text(encoding="utf-8"))
    )


def load_machines(profile_path: str | None) -> list[dict[str, Any]]:
    machines: list[dict[str, Any]] = []
    if profile_path and Path(profile_path).exists():
        prof = HardwareProfile.model_validate(
            json.loads(Path(profile_path).read_text(encoding="utf-8"))
        )
        igpu = any(a.topology == "INTEGRATED_SHARED" for a in prof.accelerators)
        if igpu:
            machines.append({"id": "you-igpu", "label": "你的機器（若引擎用 iGPU）",
                             "hw": prof, "real": True})
            machines.append({"id": "you-cpu", "label": "你的機器（純 CPU，如 Ollama）",
                             "hw": prof.model_copy(update={"accelerators": []}),
                             "real": True})
        else:
            machines.append({"id": "you", "label": "你的機器", "hw": prof, "real": True})
    for name, label in REFERENCE_HW:
        machines.append({"id": name, "label": label, "hw": _hw_fixture(name),
                         "real": False})
    return machines


def load_models() -> tuple[str, str, list[tuple[ModelSpec, dict[str, Any]]]]:
    snaps = sorted((REPO / "catalog").glob("catalog-*.json"))
    if snaps:
        # Reviewer item 3: verify the sidecar hash before trusting a snapshot
        raw = snaps[-1].read_bytes()
        sidecar = snaps[-1].with_suffix(".sha256")
        if sidecar.exists():
            import hashlib
            want = sidecar.read_text(encoding="utf-8").strip()
            got = hashlib.sha256(raw).hexdigest()
            if want != got:
                raise SystemExit(
                    f"catalog integrity FAILURE: {snaps[-1].name} sha256 {got[:12]}… "
                    f"!= sidecar {want[:12]}… — rebuild the snapshot; refusing to "
                    f"bake corrupt data into the demo"
                )
        data = json.loads(raw.decode("utf-8"))
        out = []
        for e in data.get("entries", []):
            spec = ModelSpec.model_validate(e["spec"])
            out.append((spec, e.get("meta", {})))
        return snaps[-1].name, str(data.get("generated_at") or ""), out
    fixtures = sorted((FIXTURES / "models").glob("*.json"))  # absent in the
    # public repo: there the catalog snapshot is the only model source
    return "synthetic", "", [
        (ModelSpec.model_validate(json.loads(f.read_text(encoding="utf-8"))), {})
        for f in fixtures
    ]


def _m(v: Any) -> Any:
    return None if v is None else v


def load_calibration() -> MachineCalibration | None:
    cals = sorted((REPO / "calibration").glob("*.json"))
    if not cals:
        return None
    return MachineCalibration.model_validate_json(
        cals[-1].read_text(encoding="utf-8")
    )


def build(profile_path: str | None) -> dict[str, Any]:
    machines = load_machines(profile_path)
    snap_name, generated_at, model_entries = load_models()
    calibration = load_calibration()

    columns = []
    for spec, meta in model_entries:
        shown = [q for q in spec.quants if q.label in PREVIEW_QUANTS] or spec.quants[:2]
        for q in shown:
            short = spec.model_id.split("/")[-1]
            art = (meta.get("artifacts") or {}).get(q.label) or {}
            columns.append({
                "id": f"{spec.model_id}::{q.label}",
                "label": short,
                "quant": q.label,
                "repo": meta.get("gguf_repo"),
                "files": art.get("filenames"),
                "weights_gib": (round(q.file_bytes / 1024**3, 2)
                                if q.file_bytes else None),
                "params_b": round(spec.n_params_total / 1e9, 1),
                "active_b": round(spec.n_params_active / 1e9, 1),
                "license": meta.get("license"),
                "arch": spec.arch,
                "max_ctx": spec.max_position_embeddings,
            })

    hardware = []
    for m in machines:
        hw = m["hw"]
        acc = hw.accelerators[0] if hw.accelerators else None
        hardware.append({
            "id": m["id"], "label": m["label"], "real": m["real"],
            "topology": acc.topology if acc else "RAM_ONLY",
            "ram_gib": round(hw.system_memory.total_bytes / 1024**3, 1),
            "avail_gib": round(hw.system_memory.available_bytes / 1024**3, 1),
            "gpu": acc.name if acc else None,
            "measured_at": hw.measured_at_unix if m["real"] else None,
        })

    results: dict[str, Any] = {}
    for m in machines:
        hw = m["hw"]
        for spec, meta in model_entries:
            shown = [q for q in spec.quants if q.label in PREVIEW_QUANTS] or spec.quants[:2]
            for q in shown:
                for ctx in CTX_OPTIONS:
                    for kv in KV_OPTIONS:
                        wl = Workload(ctx_target_tokens=ctx, kv_dtype=kv)  # type: ignore[arg-type]
                        r = classify_fit(hw, spec, q, wl)
                        key = f"{m['id']}|{spec.model_id}::{q.label}|{ctx}|{kv}"
                        entry: dict[str, Any] = {
                            "v": r.verdict.value,
                            "tier": r.headroom_bytes.tier.value,
                            "flags": r.flags,
                        }
                        if r.budget_bytes.value is not None:
                            b = compute_budget(hw, wl)
                            assert b.value is not None
                            entry.update({
                                "budget": b.value,
                                "safety": safety_margin_bytes(hw, wl),
                                "w": _m(r.demand_breakdown.weights.value),
                                "kv": _m(r.demand_breakdown.kv_cache.value),
                                "act": _m(r.demand_breakdown.activation.value),
                                "logits": _m(r.demand_breakdown.logits.value),
                                "demand": _m(r.demand_bytes.ci_high or r.demand_bytes.value),
                                "headroom": _m(r.headroom_bytes.value),
                                "ctx_max": _m(r.ctx_max_tokens.value),
                                "g": _m(r.n_gpu_layers_max.value),
                                "n_layers": spec.n_layers,
                                "curve": [[c, v.value] for c, v in r.tradeoff_curve],
                            })
                            try:
                                # calibration applies to real machines only;
                                # pool matching is enforced inside predict
                                t = predict_decode(
                                    hw, spec, q, wl, r,
                                    calibration=calibration if m["real"] else None,
                                )
                            except ValueError:
                                # V-P5 sentinel: corrupt catalog data for
                                # this cell — quarantine honestly, never
                                # kill the whole build
                                entry["tps_note"] = ("速度模型拒絕：目錄資料"
                                                     "不一致（V-P5 哨兵）")
                                results[key] = entry
                                continue
                            tv = t.decode_tps_at_target_ctx
                            if tv.value is not None:
                                entry["tps"] = round(tv.value, 1)
                                entry["tps_lo"] = round(tv.ci_low or 0, 1)
                                entry["tps_hi"] = round(tv.ci_high or 0, 1)
                                entry["tps_tier"] = tv.tier.value
                                entry["calibrated"] = "MACHINE_CALIBRATED" in t.flags
                                entry["decay"] = [[c, round(v, 1)]
                                                  for c, v in t.decode_decay_curve]
                            else:
                                entry["tps_note"] = "無頻寬資料（spec_db 未收錄此硬體）"
                        results[key] = entry

    # ---- recommendation-first view (review UI item): PURE ranking over the
    # cells computed above; policy + honesty contract live in
    # llmpairing.recommend, tested there. Real machines only.
    rec_ctx, rec_long = 8_192, 32_768
    recommendations: dict[str, Any] = {}
    col_of: dict[str, tuple[ModelSpec, Any]] = {}
    for spec, meta in model_entries:
        shown = [q for q in spec.quants if q.label in PREVIEW_QUANTS] or spec.quants[:2]
        for q in shown:
            col_of[f"{spec.model_id}::{q.label}"] = (spec, q)
    for m in machines:
        if not m["real"]:
            continue
        cands: list[RecCandidate] = []
        for col_id, (spec, q) in col_of.items():
            for ctx in (rec_ctx, rec_long):
                key = f"{m['id']}|{col_id}|{ctx}|f16"
                r0 = results.get(key)
                if r0 is None:
                    continue
                wl = Workload(ctx_target_tokens=ctx, kv_dtype="f16")
                try:
                    bpt: float | None = decode_bytes_per_token(spec, q, wl, ctx)
                except Exception:
                    bpt = None  # unrankable, never guessed (R-2)
                cands.append(RecCandidate(
                    model_id=spec.model_id,
                    label=spec.model_id.split("/")[-1],
                    quant=q.label,
                    params_active=spec.n_params_active,
                    params_total=spec.n_params_total,
                    weights_bytes=q.file_bytes,
                    verdict=r0["v"],
                    ctx=ctx,
                    ctx_max=r0.get("ctx_max"),
                    bytes_per_token=bpt,
                    tps=r0.get("tps"),
                    tps_tier=r0.get("tps_tier"),
                ))
        rr = recommend(cands, target_ctx=rec_ctx, long_ctx=rec_long)
        recommendations[m["id"]] = {
            "picks": [{
                "kind": p.kind,
                "reason": p.reason,
                "caveats": p.caveats,
                "key": f"{m['id']}|{p.candidate.model_id}::{p.candidate.quant}"
                       f"|{p.candidate.ctx}|f16",
                "label": p.candidate.label,
                "quant": p.candidate.quant,
                "params_active_b": round(p.candidate.params_active / 1e9, 1),
                "weights_gib": (round(p.candidate.weights_bytes / 1024**3, 2)
                                if p.candidate.weights_bytes else None),
                "verdict": p.candidate.verdict,
                "ctx": p.candidate.ctx,
                "tps": p.candidate.tps,
                "tps_tier": p.candidate.tps_tier,
            } for p in rr.picks],
            "notes": rr.notes,
        }

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, check=True,
                                ).stdout.strip()
    except Exception:
        commit = "unknown"

    return {
        "hardware": hardware,
        "models": columns,
        "ctx_options": CTX_OPTIONS,
        "kv_options": KV_OPTIONS,
        "catalog": snap_name,
        "catalog_generated_at": generated_at,
        "commit": commit,
        "combo_count": len(results),
        "results": results,
        "recommend": recommendations,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="HardwareProfile JSON from llmpairing-probe")
    ap.add_argument("--no-profile", action="store_true",
                    help="force the reference-machines-only showcase build "
                         "(what the public GitHub Pages demo uses)")
    ap.add_argument("--out", help="output path (default demo/llm-pairing-demo.html)")
    args = ap.parse_args()
    default_profile = REPO / "my_profile.json"
    if args.no_profile:
        args.profile = None
        default_profile = Path("/nonexistent")
    if args.profile and not Path(args.profile).exists():
        # honesty guard: silently building the reference-machines-only demo
        # while printing "profile YES" hid a missing profile (field catch
        # 2026-08-18: cloud rebuild showed 450 combos vs the owner's 900)
        raise SystemExit(f"--profile {args.profile} does not exist")
    profile = args.profile or (str(default_profile) if default_profile.exists() else None)

    data = build(profile)
    template = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = template.replace("/*__DATA__*/null", payload)
    out = Path(args.out) if args.out else REPO / "demo" / "llm-pairing-demo.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KiB, "
          f"{data['combo_count']} combos, catalog {data['catalog']}, "
          f"profile {'YES' if profile else 'no'}, commit {data['commit']})")


if __name__ == "__main__":
    main()

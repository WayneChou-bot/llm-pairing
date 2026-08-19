"""Preview probe — first real-machine scan (NOT the official CLI; that is
T-001 S8). Assembles a HardwareProfile from the S2/S3 collectors, validates
it against the frozen schema, prints a source-annotated summary, and runs
the T-002 core against the bundled synthetic models as a pairing preview.

Read-only. Writes exactly one file: probe_preview.json (the profile), next
to the repo root. Nothing is uploaded anywhere.

Usage:  python tools\\preview_probe.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from llmpairing.budget.classify import classify_fit  # noqa: E402
from llmpairing.probe.cpu import collect_cpu  # noqa: E402
from llmpairing.probe.gpu_windows import collect_video_controllers  # noqa: E402
from llmpairing.probe.memory import collect_memory  # noqa: E402
from llmpairing.schemas import HardwareProfile, ModelSpec, QuantVariant, Workload  # noqa: E402

GIB = 1024**3


def build_profile(diags: list[str]) -> HardwareProfile:
    system = platform.system().lower()
    plat = system if system in ("windows", "darwin", "linux") else "linux"
    accelerators = collect_video_controllers(diags) if plat == "windows" else []
    return HardwareProfile(
        schema_version="1.0",
        probe_tier="T0",  # type: ignore[arg-type]
        platform=plat,  # type: ignore[arg-type]
        measured_at_unix=int(time.time()),
        cpu=collect_cpu(diags),
        system_memory=collect_memory(diags),
        accelerators=accelerators,
        apple=None,
    )


def summarize(profile: HardwareProfile) -> None:
    m = profile.system_memory
    print("=" * 62)
    print("LLM pairing — 硬體掃描預覽（T0 估算・唯讀・零上傳）")
    print("=" * 62)
    print(f"平台        : {profile.platform}  (probe tier {profile.probe_tier.value})")
    print(f"CPU         : {profile.cpu.model}")
    print(f"              {profile.cpu.physical_cores}C/{profile.cpu.logical_cores}T  [來源: CIM/psutil]")
    print(f"RAM 總量    : {m.total_bytes / GIB:.1f} GiB  [來源: psutil, 讀數]")
    print(f"RAM 可用    : {m.available_bytes / GIB:.1f} GiB  [量測當下快照 P-04]")
    print(f"RAM 頻寬    : 未知  [spec_db 於 S5 落地前誠實 UNKNOWN]")
    if profile.accelerators:
        for a in profile.accelerators:
            print(f"加速器      : {a.name}")
            print(f"              vendor={a.vendor}  topology={a.topology}"
                  f"  laptop={a.is_laptop_variant}")
            print(f"              VRAM=未知（{'共享記憶體，驅動數字無意義 P-06' if a.topology == 'INTEGRATED_SHARED' else '待 S4 nvidia-smi'}）")
    else:
        print("加速器      : 無（純 CPU 推論）")


#: representative quants shown per model (full list lives in the snapshot)
_PREVIEW_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")


def _load_catalog() -> tuple[str, list[tuple[ModelSpec, dict[str, object]]]] | None:
    """Newest versioned snapshot under catalog/, or None."""
    snaps = sorted((REPO / "catalog").glob("catalog-*.json"))
    if not snaps:
        return None
    snap = snaps[-1]
    data = json.loads(snap.read_text(encoding="utf-8"))
    entries = [
        (ModelSpec.model_validate(e["spec"]), e.get("meta", {}))
        for e in data.get("entries", [])
    ]
    return snap.name, entries


def _verdict_line(hw: HardwareProfile, model: ModelSpec, q: QuantVariant,
                  wl: Workload, name: str, note: str = "") -> None:
    r = classify_fit(hw, model, q, wl)
    extra = ""
    if r.n_gpu_layers_max.value is not None:
        extra = f"  GPU {r.n_gpu_layers_max.value}/{model.n_layers} 層"
    elif r.verdict.value in ("FITS", "TIGHT", "RAM_ONLY") and r.headroom_bytes.value:
        extra = f"  餘裕 {r.headroom_bytes.value / GIB:.1f} GiB"
    elif r.verdict.value == "OOM_AT_CONTEXT" and r.ctx_max_tokens.value:
        extra = f"  ctx 上限 {r.ctx_max_tokens.value:,}"
    print(f"  {name:<42} {q.label:<7} -> {r.verdict.value}{extra}{note}")


def pairing_preview(profile: HardwareProfile) -> None:
    wl = Workload(ctx_target_tokens=8192)
    catalog = _load_catalog()
    print("-" * 62)
    igpu = any(a.topology == "INTEGRATED_SHARED" for a in profile.accelerators)
    scenarios: list[tuple[str, HardwareProfile]] = [("", profile)]
    if igpu:
        # T1-P08: the engine may ignore the iGPU entirely (Ollama on this
        # machine runs pure CPU) — show both worlds side by side.
        cpu_only = profile.model_copy(update={"accelerators": []})
        scenarios = [("若引擎使用 iGPU", profile), ("若純 CPU（如 Ollama 實測）", cpu_only)]

    if catalog:
        snap_name, entries = catalog
        print(f"配對 @ 8K context（真實模型目錄 {snap_name}・T0 估算）")
        for label, hw in scenarios:
            if label:
                print(f"\n[{label}]")
            for model, meta in entries:
                lic = meta.get("license") or "?"
                shown = [q for q in model.quants if q.label in _PREVIEW_QUANTS]
                for q in shown or model.quants[:2]:
                    _verdict_line(hw, model, q, wl, model.model_id,
                                  note=f"  [{lic}]")
    else:
        fixtures = REPO / "tests" / "fixtures" / "models"
        print("配對預覽 @ 8K context（合成模型目錄・T0 估算）")
        for label, hw in scenarios:
            if label:
                print(f"\n[{label}]")
            for f in sorted(fixtures.glob("*.json")):
                model = ModelSpec.model_validate(json.loads(f.read_text(encoding="utf-8")))
                for q in model.quants:
                    _verdict_line(hw, model, q, wl, model.model_id)


def main() -> int:
    diags: list[str] = []
    profile = build_profile(diags)
    summarize(profile)
    pairing_preview(profile)
    out = REPO / "probe_preview.json"
    out.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    print("-" * 62)
    print(f"完整 profile 已存: {out.name}（通過 HardwareProfile v1.0 schema 驗證）")
    if diags:
        print(f"\n診斷紀錄（{len(diags)} 條）:")
        for d in diags:
            print(f"  - {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

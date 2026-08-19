"""Official probe CLI (T-001 SPEC §4): `llmpairing probe`.

Exit codes:
    0  — a schema-legal HardwareProfile was produced (honest degradation
         with UNKNOWN fields is success, not failure)
    1  — no schema-legal profile could be assembled (reason on stderr)

The ONLY file this writes is the --json output the user asked for.
"""
from __future__ import annotations

import argparse
import platform as _platform
import sys
import time

from llmpairing.probe import assemble
from llmpairing.probe.runner import ProbeError
from llmpairing.schemas import HardwareProfile

GIB = 1024**3


def _current_platform() -> str:
    return _platform.system().lower()


def _summary(profile: HardwareProfile) -> str:
    m = profile.system_memory
    lines = [
        "LLM pairing — 硬體掃描（T0 估算・唯讀・零上傳）",
        f"平台     : {profile.platform}  (probe tier {profile.probe_tier.value})",
        f"CPU      : {profile.cpu.model}  "
        f"{profile.cpu.physical_cores}C/{profile.cpu.logical_cores}T",
        f"RAM      : 總量 {m.total_bytes / GIB:.1f} GiB・"
        f"可用 {m.available_bytes / GIB:.1f} GiB（量測當下快照 P-04）",
        "RAM 頻寬 : 未知（spec_db 待 S5；不猜測）",
    ]
    if profile.accelerators:
        for a in profile.accelerators:
            vram = ("未知（共享記憶體 P-06）" if a.topology == "INTEGRATED_SHARED"
                    else "未知（待 S4 nvidia-smi）"
                    if a.vram_total_bytes is None
                    else f"{a.vram_total_bytes / GIB:.1f} GiB")
            lines.append(
                f"加速器   : {a.name}  [{a.vendor}/{a.topology}"
                f"{'/laptop' if a.is_laptop_variant else ''}]  VRAM {vram}"
            )
    else:
        lines.append("加速器   : 無（純 CPU 推論）")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llmpairing probe",
        description="Read-only hardware probe producing a HardwareProfile",
    )
    ap.add_argument("--json", metavar="PATH",
                    help="write the full HardwareProfile JSON here")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-probe diagnostics")
    args = ap.parse_args(argv)

    diags: list[str] = []
    try:
        profile = assemble.build_profile(
            diags, platform_name=_current_platform(),
            measured_at_unix=int(time.time()),
        )
    except ProbeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1

    print(_summary(profile))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))
        print(f"profile 已寫入: {args.json}")
    if args.verbose and diags:
        print(f"\n診斷紀錄（{len(diags)} 條）:")
        for d in diags:
            print(f"  - {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
